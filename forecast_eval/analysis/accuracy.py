"""Accuracy-side metrics: pass@1 / pass_any@N / majority_vote / breakdowns.

These metrics are byte-identical to v3 — the v4 refactor only relocated them
from the monolithic `analysis.py`. The probabilistic family lives next door
in `proper_score.py`.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .flatten import SampleRow


def _mean(values: Iterable[float | int]) -> float | None:
    collected = [float(v) for v in values if v is not None]
    if not collected:
        return None
    return sum(collected) / len(collected)


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


@dataclass
class Aggregate:
    eligible_samples: int
    eligible_questions: int
    resolvable_samples: int

    cutoff_skip_samples: int
    cutoff_skip_rate: float | None

    pass_at_1_avg: float | None
    resolvable_rate: float | None

    pass_any_at_n: float | None
    at_least_majority_at_n: float | None
    at_least_all_at_n: float | None

    majority_vote_accuracy: float | None
    majority_vote_resolvable_rate: float | None

    parse_failure_rate: float | None
    error_rate: float | None

    avg_tool_calls: float | None
    avg_react_steps: float | None
    avg_latency_ms: float | None
    avg_prompt_tokens: float | None
    avg_completion_tokens: float | None
    avg_reasoning_tokens: float | None
    avg_nudges_used: float | None

    def as_ordered_dict(self) -> dict[str, Any]:
        return {
            "eligible_samples": self.eligible_samples,
            "eligible_questions": self.eligible_questions,
            "resolvable_samples": self.resolvable_samples,
            "cutoff_skip_samples": self.cutoff_skip_samples,
            "cutoff_skip_rate": _round(self.cutoff_skip_rate),
            "pass_at_1_avg": _round(self.pass_at_1_avg),
            "resolvable_rate": _round(self.resolvable_rate),
            "pass_any_at_n": _round(self.pass_any_at_n),
            "at_least_majority_at_n": _round(self.at_least_majority_at_n),
            "at_least_all_at_n": _round(self.at_least_all_at_n),
            "majority_vote_accuracy": _round(self.majority_vote_accuracy),
            "majority_vote_resolvable_rate": _round(self.majority_vote_resolvable_rate),
            "parse_failure_rate": _round(self.parse_failure_rate),
            "error_rate": _round(self.error_rate),
            "avg_tool_calls": _round(self.avg_tool_calls, 2),
            "avg_react_steps": _round(self.avg_react_steps, 2),
            "avg_latency_ms": _round(self.avg_latency_ms, 1),
            "avg_prompt_tokens": _round(self.avg_prompt_tokens, 1),
            "avg_completion_tokens": _round(self.avg_completion_tokens, 1),
            "avg_reasoning_tokens": _round(self.avg_reasoning_tokens, 1),
            "avg_nudges_used": _round(self.avg_nudges_used, 2),
        }


def _aggregate(
    samples: list[SampleRow],
    sampling_n: int,
    gt_map: dict[str, frozenset[str]] | None = None,
) -> Aggregate:
    total = len(samples)
    cutoff_samples = [s for s in samples if s.is_cutoff]
    eligible_samples = [s for s in samples if s.is_eligible]
    resolvable_samples = [s for s in eligible_samples if s.is_resolvable]

    eligible_questions = {s.question_id for s in eligible_samples}
    by_q_resolvable: dict[str, list[SampleRow]] = {}
    by_q_all_eligible: dict[str, list[SampleRow]] = {}
    for s in eligible_samples:
        by_q_all_eligible.setdefault(s.question_id, []).append(s)
        if s.is_resolvable:
            by_q_resolvable.setdefault(s.question_id, []).append(s)

    if resolvable_samples:
        pass_at_1 = sum(1 for s in resolvable_samples if s.correct == 1) / len(resolvable_samples)
    else:
        pass_at_1 = None

    resolvable_rate = (
        len(resolvable_samples) / len(eligible_samples) if eligible_samples else None
    )

    majority_threshold = math.ceil(sampling_n / 2)

    pass_any_hits: list[int] = []
    at_least_majority_hits: list[int] = []
    at_least_all_hits: list[int] = []
    for qid, rs in by_q_resolvable.items():
        n = len(rs)
        corrects = sum(1 for s in rs if s.correct == 1)
        pass_any_hits.append(1 if corrects >= 1 else 0)
        at_least_majority_hits.append(1 if corrects >= majority_threshold else 0)
        at_least_all_hits.append(1 if (n == sampling_n and corrects == n) else 0)

    pass_any_at_n = _mean(pass_any_hits) if pass_any_hits else None
    at_least_majority_at_n = _mean(at_least_majority_hits) if at_least_majority_hits else None
    at_least_all_at_n = _mean(at_least_all_hits) if at_least_all_hits else None

    mv_resolvable_hits: list[int] = []
    mv_correct_hits: list[int] = []
    if gt_map is not None:
        for qid, rs in by_q_all_eligible.items():
            gt = gt_map.get(qid)
            if gt is None:
                continue
            parsed = [s.parsed_letters for s in rs if s.parsed_letters is not None]
            if not parsed:
                continue
            counts = Counter(parsed)
            top_count = max(counts.values())
            winners = [k for k, v in counts.items() if v == top_count]
            if len(winners) != 1:
                continue
            mv_resolvable_hits.append(1)
            mv_correct_hits.append(1 if winners[0] == gt else 0)
        majority_vote_accuracy = _mean(mv_correct_hits) if mv_correct_hits else None
        majority_vote_resolvable_rate = (
            len(mv_resolvable_hits) / len(eligible_questions) if eligible_questions else None
        )
    else:
        majority_vote_accuracy = None
        majority_vote_resolvable_rate = None

    if eligible_samples:
        parse_failure_rate = sum(
            1 for s in eligible_samples if s.parse_ok == 0 and (s.error is None)
        ) / len(eligible_samples)
        error_rate = sum(
            1 for s in eligible_samples if s.error is not None
        ) / len(eligible_samples)
    else:
        parse_failure_rate = None
        error_rate = None

    avg_tool_calls = _mean(s.tool_calls_count for s in eligible_samples)
    avg_react_steps = _mean(s.react_steps for s in eligible_samples)
    avg_latency = _mean(s.latency_ms for s in eligible_samples)
    avg_ptok = _mean(s.prompt_tokens for s in eligible_samples)
    avg_ctok = _mean(s.completion_tokens for s in eligible_samples)
    avg_rtok = _mean(s.reasoning_tokens for s in eligible_samples)
    # Pre-v3 rows have nudges_used=NULL — _mean already filters those, so a
    # mid-run schema upgrade silently averages over the v3 rows only.
    avg_nudges = _mean(s.nudges_used for s in eligible_samples)

    return Aggregate(
        eligible_samples=len(eligible_samples),
        eligible_questions=len(eligible_questions),
        resolvable_samples=len(resolvable_samples),
        cutoff_skip_samples=len(cutoff_samples),
        cutoff_skip_rate=(len(cutoff_samples) / total) if total else None,
        pass_at_1_avg=pass_at_1,
        resolvable_rate=resolvable_rate,
        pass_any_at_n=pass_any_at_n,
        at_least_majority_at_n=at_least_majority_at_n,
        at_least_all_at_n=at_least_all_at_n,
        majority_vote_accuracy=majority_vote_accuracy,
        majority_vote_resolvable_rate=majority_vote_resolvable_rate,
        parse_failure_rate=parse_failure_rate,
        error_rate=error_rate,
        avg_tool_calls=avg_tool_calls,
        avg_react_steps=avg_react_steps,
        avg_latency_ms=avg_latency,
        avg_prompt_tokens=avg_ptok,
        avg_completion_tokens=avg_ctok,
        avg_reasoning_tokens=avg_rtok,
        avg_nudges_used=avg_nudges,
    )


def _slice_by(
    samples: list[SampleRow],
    key_fn: Callable[[SampleRow], str],
    sampling_n: int,
    gt_map: dict[str, frozenset[str]],
) -> dict[str, Aggregate]:
    buckets: dict[str, list[SampleRow]] = {}
    for s in samples:
        buckets.setdefault(key_fn(s), []).append(s)
    return {k: _aggregate(v, sampling_n, gt_map) for k, v in sorted(buckets.items())}


def _error_breakdown(samples: list[SampleRow]) -> Counter:
    """Count error codes across ALL samples (including cutoff)."""
    counter: Counter = Counter()
    for s in samples:
        if s.error is not None:
            counter[s.error] += 1
        else:
            counter["<ok>"] += 1
    return counter


def _finish_reason_breakdown(samples: list[SampleRow]) -> Counter:
    """Count `finish_reason` values across eligible samples (cutoff excluded).

    The cutoff path never invokes the LLM, so its `finish_reason` is always
    NULL — including those rows would just inflate the `<missing>` bucket.
    A NULL on an eligible sample (legacy v2 row, or pre-extraction failure)
    is rare but real, so we keep it as `<missing>` rather than silently dropping.
    """
    counter: Counter = Counter()
    for s in samples:
        if s.is_eligible:
            counter[s.finish_reason or "<missing>"] += 1
    return counter


__all__ = [
    "Aggregate",
    "_aggregate",
    "_slice_by",
    "_error_breakdown",
    "_finish_reason_breakdown",
    "_mean",
    "_round",
]
