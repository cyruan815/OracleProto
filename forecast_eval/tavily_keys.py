"""Round-robin pool over multiple Tavily API keys.

Goals:
- 在一次 run 内尽量平均分配调用量 (least-used 选择).
- key 配额耗尽 / 失效时自动跳过, 不阻塞其他 key.
- 跨 run 公平性: 构造时按随机偏移轮换初始计数, 避免每次 run 都先烧 keys[0].

Failure handling:
- "auth" (401/403)        — 永久拉黑, 该 key 本次 process 不再使用.
- "rate_limit" (429/配额)  — 临时拉黑 cooldown_s 秒.
- "other"                  — 不拉黑, 由 search.py 的 backoff 重试逻辑处理.

并发安全靠单一 asyncio.Lock; 临界区只有 O(N) 字典操作, N 通常 <10.
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
    """池中所有 key 当前都不可用 (永久拉黑或仍在 cooldown).

    `tavily_search` 捕获后落到 SearchResult.error_kind="tavily_error", 不抛到
    ReAct 循环外, 让 LLM 仍能看到 tool_result 并继续。
    """


@dataclass
class _KeyState:
    key: str
    used: int = 0          # 累计 acquire 计数 (含初始随机偏移)
    blacklisted: bool = False
    cooldown_until: float = 0.0  # monotonic timestamp


@dataclass
class TavilyKeyPool:
    """Least-used key picker with cooldown + permanent blacklist.

    构造请走 `from_keys` (默认开启随机起点偏移); 直接构造仅供测试注入确定性状态.
    """

    states: list[_KeyState]
    cooldown_s: float = 60.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # 注入式时钟, 测试可替换为可控函数; 默认 monotonic 避免系统时间跳变.
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
        # 去重保持顺序: 用户多写一份等价 key 不应在选择阶段乘上权重.
        seen: set[str] = set()
        deduped: list[str] = []
        for k in keys:
            if k in seen:
                logger.warning("TavilyKeyPool: duplicate key {} ignored", _short(k))
                continue
            seen.add(k)
            deduped.append(k)

        states = [_KeyState(key=k) for k in deduped]
        # 随机起点: 给每个 key 一个 [0, len) 内的初始 used 偏移, 让 least-used
        # 选择天然从随机位置开始; 多 run 之间通过 rng 的 seed 自然解相关.
        # 偏移用 len(states) 而非更大值, 保证初始计数差距小于一轮调用.
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

        Healthy = 未永久拉黑且 cooldown 已过. Tie-breaker: 同 used 取 states 顺序
        的第一个 (deterministic, 便于调试; 跨 run 公平性靠初始 offset).
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
        # 成功路径无需修改状态; 计数已在 acquire 阶段加. 这里留个 hook 以备
        # 未来引入 EWMA 或 success counter.
        return None

    async def report_failure(self, key: str, kind: FailureKind) -> None:
        async with self._lock:
            st = self._find(key)
            if st is None:
                # key 不在池里 (理论上不可能, 防御性 log).
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
            # "other": 网络/5xx 不拉黑, 由 search.py backoff 处理.

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
# Grid dispatcher 用 settings.model_copy() 创建 cell-local 子视图; 子视图的
# TAVILY_API_KEY 列表内容与父 settings 相同, 但 model_copy 会复制为新 list 实例.
# 这里把池按 tuple(keys) 复用, 让所有 cell + 所有 worker 共享同一个 pool,
# 用量计数才会真正聚合 (而不是每个 cell 独立从 0 开始统计).
_pool_cache: dict[tuple[str, ...], TavilyKeyPool] = {}


def get_pool(keys: list[str], cooldown_s: float) -> TavilyKeyPool:
    """Return the process-wide pool for `keys`, creating it on first use.

    Cache key 只依赖 `tuple(keys)` — cooldown_s 在同一 process 内一般不会变;
    若变了, 取首次调用时的值 (再起 process 即可刷新). 这是有意的简化,
    与 grid model_copy 的不可变语义一致.
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
