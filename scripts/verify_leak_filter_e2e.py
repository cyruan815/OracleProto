"""End-to-end verification of search-leak-filter-v1 real-world filtering capability.

Calls real Tavily exactly once, then:
  Step 1: Print the raw 5 results (equivalent to filter=OFF path) for manual review
  Step 2: deepcopy the same results and feed them to leak_filter.filter_search_result;
          collect per-item verdict + audit metadata
  Step 3: Align raw_content with verdict item-by-item; let a human judge whether
          the filtering decision is reasonable

The query is deliberately chosen so that the event only truly happens after the
cutoff (q.end_time + offset = 2026-04-26) — under end_date filtering Tavily can
only return pre-cutoff prediction/analysis articles, raw_content should contain
abundant forward-looking descriptions, and the detector should drop most of them.

Usage:
    python scripts/verify_leak_filter_e2e.py
"""
from __future__ import annotations

import asyncio
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecast_eval import leak_filter  # noqa: E402
from forecast_eval.config import Settings  # noqa: E402
from forecast_eval.search import tavily_search  # noqa: E402


QUERY = "2026 Brazil presidential election candidates Lula Bolsonaro outcome"
END_DATE = "2026-04-26"
SNIPPET_CHARS = 240


def _snippet(text: str | None, n: int = SNIPPET_CHARS) -> str:
    if not text:
        return "(empty)"
    text = " ".join(text.split())
    if len(text) <= n:
        return text
    return text[:n].rstrip() + "..."


async def main() -> int:
    try:
        raw = Settings()
    except Exception as exc:
        print(f"[verify] Settings boot failed: {exc}", file=sys.stderr)
        return 2

    # Collapse grid axes (TAVILY_MAX_RESULTS / REACT_MAX_SEARCH_CALLS are list[int];
    # the dispatcher collapses them to a single int when deriving a real cell —
    # we do the equivalent manually here).
    R = raw.TAVILY_MAX_RESULTS[0] if raw.TAVILY_MAX_RESULTS else 5
    C = raw.REACT_MAX_SEARCH_CALLS[0] if raw.REACT_MAX_SEARCH_CALLS else 8
    base = raw.model_copy(update={
        "TAVILY_MAX_RESULTS": R,
        "REACT_MAX_SEARCH_CALLS": C,
    })

    print(f"[setup] query={QUERY!r}")
    print(f"[setup] end_date={END_DATE}")
    print(f"[setup] LEAK_DETECTOR_MODEL={base.LEAK_DETECTOR_MODEL}")
    print(f"[setup] LEAK_DETECTOR_FAIL_ACTION={base.LEAK_DETECTOR_FAIL_ACTION}")
    print(f"[setup] prompt_hash={leak_filter._compute_prompt_hash()}")
    print()

    # ─────────── Step 1: single real Tavily call (filter OFF) ───────────
    off = base.model_copy(update={"ENABLE_SEARCH_LEAK_FILTER": False})
    print("=" * 80)
    print("[Step 1] Real Tavily (filter=OFF, detector not invoked)")
    print("=" * 80)
    raw_res = await tavily_search(query=QUERY, end_date=END_DATE, settings=off)
    if raw_res.error_message:
        print(f"  ERROR: {raw_res.error_message}")
        return 3
    if not raw_res.results:
        print(f"  Tavily returned 0 results, try another query")
        return 3
    print(f"  audit={raw_res.audit}  (expect: None)")
    print(f"  n={len(raw_res.results)}")
    for i, it in enumerate(raw_res.results):
        print(f"\n  [{i}] {it.title!r}")
        print(f"      url={it.url}")
        print(f"      published={it.published_date}")
        print(f"      content[:{SNIPPET_CHARS}]={_snippet(it.content)}")
        print(f"      raw[:{SNIPPET_CHARS}]={_snippet(it.raw_content)}")
    print()

    # ─────────── Step 2: feed the same results to the detector (filter=ON) ───────────
    print("=" * 80)
    print("[Step 2] Reuse the same results through leak_filter.filter_search_result")
    print("=" * 80)
    raw_copy = copy.deepcopy(raw_res)
    filtered = await leak_filter.filter_search_result(
        raw_copy, end_date=END_DATE, settings=base
    )
    audit = filtered.audit or {}
    print(
        f"  audit: n_raw={audit.get('n_results_raw')} "
        f"n_kept={audit.get('n_results_kept')} "
        f"latency_ms={audit.get('detector_latency_ms')} "
        f"err={audit.get('detector_error_kind')}"
    )
    verdicts = list(audit.get("detector_verdicts", []))

    # ─────────── Step 3: strict alignment of raw items × verdicts ───────────
    print()
    print("=" * 80)
    print("[Step 3] verdict-by-result (raw items × verdicts strictly aligned)")
    print("=" * 80)
    kept = dropped = 0
    for i, (it, v) in enumerate(zip(raw_res.results, verdicts)):
        mark = "✓ KEEP" if v == "keep" else f"✗ {v.upper()}"
        print(f"\n  [{i}] {mark}")
        print(f"       title={it.title!r}")
        print(f"       url={it.url}")
        print(f"       published={it.published_date}")
        print(f"       content[:{SNIPPET_CHARS}]={_snippet(it.content)}")
        print(f"       raw[:{SNIPPET_CHARS}]={_snippet(it.raw_content)}")
        if v == "keep":
            kept += 1
        else:
            dropped += 1
    print(f"\n  ─── summary ──────────────────────────────────────────────────")
    print(f"  kept={kept}  dropped={dropped}  total={kept + dropped}")

    print(f"\n  filtered.results (the items kept after the detector):")
    for i, it in enumerate(filtered.results):
        print(f"    [{i}] {it.title[:80]!r}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
