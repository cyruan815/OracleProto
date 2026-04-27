"""Manual smoke test for forecast_eval.leak_filter (search-leak-filter-v1).

Runs five hand-crafted synthetic SearchResultItems past the real detector LLM
and prints the resulting verdict / reason for each. Intended for human
inspection of prompt strictness — read the printed reasons and decide whether
the detector is calibrated correctly before kicking off a full run.

Usage:
    python scripts/smoke_leak_filter.py

Requires `.env` to define LEAK_DETECTOR_API_KEY / LEAK_DETECTOR_MODEL plus
the rest of the standard Settings (LLM_API_KEY etc.). Set
`ENABLE_SEARCH_LEAK_FILTER=true` so the singleton detector client is
configured by Settings._post_validate.

The five synthetic items deliberately span the four spec-defined leakage
classes plus one safe baseline:

    1. specific observed event AFTER cutoff  → expect drop
    2. scheduled future event AFTER cutoff   → expect drop
    3. speculative future prediction         → expect drop
    4. ambiguous wording                     → expect drop (per "暧昧 → drop" rule)
    5. fully pre-cutoff factual statement    → expect keep
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Make `forecast_eval` importable when the script is run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecast_eval import leak_filter  # noqa: E402
from forecast_eval.config import Settings  # noqa: E402
from forecast_eval.search import SearchResult, SearchResultItem  # noqa: E402


CUTOFF_DATE = "2026-04-27"


def _make_items() -> list[SearchResultItem]:
    return [
        SearchResultItem(
            title="2026 election final results",
            url="https://example.com/election",
            content="Detailed recap of the November 2026 vote.",
            published_date="2026-11-10",
            raw_content=(
                "On 2026-11-08 voters chose Candidate Smith 52% vs Candidate "
                "Jones 47%. Jones conceded the next morning."
            ),
        ),
        SearchResultItem(
            title="Big launch scheduled",
            url="https://example.com/launch",
            content="Press release about an upcoming product launch.",
            published_date="2026-04-25",
            raw_content=(
                "The launch is scheduled for 2026-08-15. Pre-orders open in "
                "July."
            ),
        ),
        SearchResultItem(
            title="Analyst forecast",
            url="https://example.com/forecast",
            content="An investment bank's outlook for late 2026.",
            published_date="2026-04-20",
            raw_content=(
                "Analysts at FooBank expect the merger to close in late 2026, "
                "though regulatory risks remain."
            ),
        ),
        SearchResultItem(
            title="Vague update",
            url="https://example.com/update",
            content="A short blog post with mixed timeline.",
            published_date="2026-04-26",
            raw_content=(
                "The team has been working hard. We will share more news soon. "
                "The roadmap continues to evolve as the year unfolds."
            ),
        ),
        SearchResultItem(
            title="2024 Q3 results",
            url="https://example.com/2024-q3",
            content="Earnings recap.",
            published_date="2024-11-01",
            raw_content=(
                "In 2024-Q3 the company reported revenue of $12B, a 15% YoY "
                "increase. CEO commented during the earnings call."
            ),
        ),
    ]


async def main() -> int:
    try:
        settings = Settings()
    except Exception as exc:  # surfaced to user
        print(f"[smoke] Settings boot failed: {exc}", file=sys.stderr)
        return 2
    if not settings.ENABLE_SEARCH_LEAK_FILTER:
        print(
            "[smoke] ENABLE_SEARCH_LEAK_FILTER is false in .env; flip to true.",
            file=sys.stderr,
        )
        return 2
    if not settings.LEAK_DETECTOR_API_KEY:
        print(
            "[smoke] LEAK_DETECTOR_API_KEY empty after boot; please configure .env first.",
            file=sys.stderr,
        )
        return 2

    items = _make_items()
    result = SearchResult(
        query="smoke",
        end_date=CUTOFF_DATE,
        answer=None,
        results=items,
    )
    print(f"[smoke] cutoff_date={CUTOFF_DATE} model={settings.LEAK_DETECTOR_MODEL}")
    print(f"[smoke] running detector on {len(items)} synthetic items...")
    out = await leak_filter.filter_search_result(
        result, end_date=CUTOFF_DATE, settings=settings
    )
    audit = out.audit or {}
    verdicts = audit.get("detector_verdicts", [])
    for i, (item, verdict) in enumerate(zip(items, verdicts)):
        print(f"\n[{i}] title={item.title!r}")
        print(f"    raw_content: {item.raw_content}")
        print(f"    verdict: {verdict}")
    print(
        f"\n[smoke] n_raw={audit.get('n_results_raw')} "
        f"n_kept={audit.get('n_results_kept')} "
        f"latency_ms={audit.get('detector_latency_ms')} "
        f"error_kind={audit.get('detector_error_kind')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
