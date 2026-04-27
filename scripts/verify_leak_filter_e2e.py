"""端到端验证 search-leak-filter-v1 真实拦截能力.

只调一次真实 Tavily, 然后:
  Step 1: 打印原始 5 条 result (filter=OFF 等价路径) 供人工核对
  Step 2: 把同一份 result deepcopy 后丢给 leak_filter.filter_search_result
          得到每条的 verdict + audit 元数据
  Step 3: 逐条对齐 raw_content 与 verdict, 让人工判断拦截是否合理

查询故意选 cutoff (q.end_time + offset = 2026-04-26) 后才会真正发生的
事件 — Tavily 受 end_date 过滤只能拿到 pre-cutoff 的预测/分析文章,
raw_content 中应大量包含 forward-looking 描述, detector 应将其多数 drop.

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

    # Collapse grid axes (TAVILY_MAX_RESULTS / REACT_MAX_SEARCH_CALLS 是 list[int],
    # dispatcher 真实派生 cell 时会 collapse 为单 int; 我们这里手动等价).
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

    # ─────────── Step 1: 单次真实 Tavily 调用 (filter OFF) ───────────
    off = base.model_copy(update={"ENABLE_SEARCH_LEAK_FILTER": False})
    print("=" * 80)
    print("[Step 1] 真实 Tavily (filter=OFF, 不调 detector)")
    print("=" * 80)
    raw_res = await tavily_search(query=QUERY, end_date=END_DATE, settings=off)
    if raw_res.error_message:
        print(f"  ERROR: {raw_res.error_message}")
        return 3
    if not raw_res.results:
        print(f"  Tavily 返回 0 条 result, 换 query 再试")
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

    # ─────────── Step 2: 同一份 result 喂给 detector (filter=ON) ───────────
    print("=" * 80)
    print("[Step 2] 复用同一份 result 走 leak_filter.filter_search_result")
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

    # ─────────── Step 3: 严格对齐 raw items × verdicts ───────────
    print()
    print("=" * 80)
    print("[Step 3] verdict-by-result (raw items × verdicts 严格对齐)")
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

    print(f"\n  filtered.results (经 detector 后留下的):")
    for i, it in enumerate(filtered.results):
        print(f"    [{i}] {it.title[:80]!r}")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
