"""Round-robin pool over multiple Tavily API keys.

Goals:
- Distribute call volume as evenly as possible within a single run (least-used selection).
- Skip keys automatically when quota is exhausted / invalidated, without blocking other keys.
- Cross-run fairness: rotate initial counts by a random offset at construction, avoiding always
  burning keys[0] first.

Failure handling:
- "auth" (401/403)        - permanent blacklist; the key is not used again in this process.
- "rate_limit" (429/quota) - temporary blacklist for cooldown_s seconds.
- "other"                  - no blacklist; handled by search.py backoff retry logic.

Concurrency safety relies on a single asyncio.Lock; critical sections only do O(N) dict operations,
N is typically <10.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Literal

from loguru import logger


FailureKind = Literal["auth", "rate_limit", "other"]


class AllKeysExhausted(RuntimeError):
    """All keys in the pool are currently unavailable (permanently blacklisted or still in cooldown).

    `tavily_search` catches this and falls through to SearchResult.error_kind="tavily_error", not
    propagating outside the ReAct loop, so the LLM still sees a tool_result and can continue.
    """


@dataclass
class _KeyState:
    key: str
    used: int = 0          # cumulative acquire count (including initial random offset)
    blacklisted: bool = False
    cooldown_until: float = 0.0  # monotonic timestamp


@dataclass
class TavilyKeyPool:
    """Least-used key picker with cooldown + permanent blacklist.

    Construct via `from_keys` (random starting offset enabled by default); direct construction is
    only for tests injecting deterministic state.
    """

    states: list[_KeyState]
    cooldown_s: float = 60.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Injectable clock; tests can replace it with a controllable function. Default monotonic avoids
    # system-time jumps.
    _now: callable = time.monotonic  # type: ignore[assignment]

    @classmethod
    def from_keys(
        cls,
        keys: list[str],
        *,
        cooldown_s: float = 60.0,
        rng: random.Random | None = None,
    ) -> "TavilyKeyPool":
        if not keys:
            raise ValueError("TavilyKeyPool requires at least one key")
        # Dedupe while preserving order: writing the same equivalent key twice should not
        # multiply its weight at selection time.
        seen: set[str] = set()
        deduped: list[str] = []
        for k in keys:
            if k in seen:
                logger.warning("TavilyKeyPool: duplicate key {} ignored", _short(k))
                continue
            seen.add(k)
            deduped.append(k)

        states = [_KeyState(key=k) for k in deduped]
        # Random starting point: give each key an initial used offset in [0, len) so least-used
        # selection naturally starts from a random position; cross-run decorrelation comes from the
        # rng seed. Offset uses len(states) rather than a larger value to ensure initial count
        # differences are smaller than one round of calls.
        n = len(states)
        if n > 1:
            r = rng or random.Random()
            offset = r.randrange(n)
            for i, st in enumerate(states):
                st.used = (i + offset) % n
        return cls(states=states, cooldown_s=cooldown_s)

    def __post_init__(self) -> None:
        if self.cooldown_s < 0:
            raise ValueError(f"cooldown_s must be >= 0, got {self.cooldown_s}")

    async def acquire(self) -> str:
        """Return the healthiest least-used key, or raise AllKeysExhausted.

        Healthy = not permanently blacklisted and cooldown has elapsed. Tie-breaker: with equal used,
        take the first in states order (deterministic, easy to debug; cross-run fairness relies on
        the initial offset).
        """
        async with self._lock:
            now = self._now()
            best: _KeyState | None = None
            for st in self.states:
                if st.blacklisted:
                    continue
                if st.cooldown_until > now:
                    continue
                if best is None or st.used < best.used:
                    best = st
            if best is None:
                raise AllKeysExhausted(self._exhausted_reason(now))
            best.used += 1
            return best.key

    async def report_ok(self, key: str) -> None:
        # Success path requires no state mutation; counter was incremented during acquire.
        # Leaving a hook here in case EWMA or a success counter is introduced later.
        return None

    async def report_failure(self, key: str, kind: FailureKind) -> None:
        async with self._lock:
            st = self._find(key)
            if st is None:
                # key not in the pool (theoretically impossible; defensive log).
                logger.warning("TavilyKeyPool: report_failure for unknown key {}", _short(key))
                return
            if kind == "auth":
                st.blacklisted = True
                logger.warning(
                    "TavilyKeyPool: key {} permanently blacklisted (auth failure)",
                    _short(key),
                )
            elif kind == "rate_limit":
                st.cooldown_until = self._now() + self.cooldown_s
                logger.info(
                    "TavilyKeyPool: key {} cooldown for {}s (rate limit)",
                    _short(key),
                    self.cooldown_s,
                )
            # "other": network / 5xx are not blacklisted; handled by search.py backoff.

    def _find(self, key: str) -> _KeyState | None:
        for st in self.states:
            if st.key == key:
                return st
        return None

    def _exhausted_reason(self, now: float) -> str:
        n_total = len(self.states)
        n_black = sum(1 for s in self.states if s.blacklisted)
        n_cool = sum(1 for s in self.states if not s.blacklisted and s.cooldown_until > now)
        return (
            f"all {n_total} Tavily keys unavailable: "
            f"{n_black} blacklisted (auth), {n_cool} in cooldown"
        )

    def snapshot(self) -> list[dict[str, object]]:
        """Diagnostic only — current per-key state for logs / tests."""
        now = self._now()
        return [
            {
                "key_prefix": _short(s.key),
                "used": s.used,
                "blacklisted": s.blacklisted,
                "cooldown_remaining_s": max(0.0, s.cooldown_until - now),
            }
            for s in self.states
        ]


def _short(key: str) -> str:
    """Log-safe key fragment: prefix + last 4 chars only."""
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}…{key[-4:]}"


# ---------- Process-wide pool cache ----------
# Grid dispatcher uses settings.model_copy() to create cell-local sub-views; the sub-view's
# TAVILY_API_KEY list contents are the same as the parent settings, but model_copy clones it as a
# new list instance. We reuse the pool by tuple(keys) so all cells + all workers share a single pool;
# usage counts then truly aggregate (rather than each cell starting independently from 0).
_pool_cache: dict[tuple[str, ...], TavilyKeyPool] = {}


def get_pool(keys: list[str], cooldown_s: float) -> TavilyKeyPool:
    """Return the process-wide pool for `keys`, creating it on first use.

    Cache key depends only on `tuple(keys)` - cooldown_s generally does not change within the same
    process; if it does, the value at first call is taken (restart the process to refresh). This is
    an intentional simplification, consistent with grid model_copy's immutable semantics.
    """
    cache_key = tuple(keys)
    pool = _pool_cache.get(cache_key)
    if pool is None:
        pool = TavilyKeyPool.from_keys(keys, cooldown_s=cooldown_s)
        _pool_cache[cache_key] = pool
    return pool


def reset_pool_cache() -> None:
    """Test helper — drop all cached pools so each test starts clean."""
    _pool_cache.clear()
