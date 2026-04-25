"""CSV / Markdown / JSON serialisation for the analysis layer.

Phase 1 appends probabilistic columns at the end of `_SUMMARY_FIELDS`,
keeping the existing accuracy columns byte-identical to v3 (so
`per_model_summary.csv` regresses cleanly when an old run is re-analyzed
under v4 code). `error_breakdown.csv` / `finish_reason_breakdown.csv` are
not touched at all.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .accuracy import Aggregate
from .proper_score import ModelProbabilisticAggregate


# v3 accuracy columns — DO NOT reorder. Phase 1 only appends.
_SUMMARY_FIELDS_V3: tuple[str, ...] = (
    "model",
    "sampling_n",
    "eligible_samples",
    "eligible_questions",
    "resolvable_samples",
    "cutoff_skip_samples",
    "cutoff_skip_rate",
    "pass_at_1_avg",
    "resolvable_rate",
    "pass_any_at_n",
    "at_least_majority_at_n",
    "at_least_all_at_n",
    "majority_vote_accuracy",
    "majority_vote_resolvable_rate",
    "parse_failure_rate",
    "error_rate",
    "avg_tool_calls",
    "avg_react_steps",
    "avg_latency_ms",
    "avg_prompt_tokens",
    "avg_completion_tokens",
    "avg_reasoning_tokens",
    "avg_nudges_used",
)

# v4 probabilistic columns appended at the end. `bi_dec` only meaningful for
# single-choice subsets; on a global sheet it averages BI_dec over single
# questions only and skips multi rows (same NULL-on-multi convention as MBS).
_PROB_FIELDS_V4: tuple[str, ...] = (
    "bi",
    "bi_dec",
    "nll",
    "mbs",
    "abi_crowd",
    "abi_uniform",
    "fallback_share",
)

_SUMMARY_FIELDS: tuple[str, ...] = _SUMMARY_FIELDS_V3 + _PROB_FIELDS_V4


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _prob_dict(agg: ModelProbabilisticAggregate | None) -> dict[str, Any]:
    """Map a probabilistic aggregate onto the CSV/MD column names.

    Returns all-None when `agg is None` (caller didn't compute probabilistic
    metrics for this row, e.g. a slice with zero scoreable questions). This
    keeps the CSV row width constant.
    """
    if agg is None:
        return {k: None for k in _PROB_FIELDS_V4}
    return {
        "bi": _round(agg.bi),
        "bi_dec": _round(agg.bi_dec),
        "nll": _round(agg.nll),
        "mbs": _round(agg.mbs),
        "abi_crowd": _round(agg.abi_crowd),
        "abi_uniform": _round(agg.abi_uniform),
        "fallback_share": _round(agg.fallback_share),
    }


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)
    return path


def _write_per_model_summary_csv(
    path: Path,
    per_model: dict[str, tuple[int, Aggregate]],
    prob: dict[str, ModelProbabilisticAggregate] | None = None,
) -> Path:
    header = list(_SUMMARY_FIELDS)
    rows: list[list[Any]] = []
    for model, (sampling_n, agg) in per_model.items():
        prob_agg = prob.get(model) if prob else None
        row_dict = {
            "model": model,
            "sampling_n": sampling_n,
            **agg.as_ordered_dict(),
            **_prob_dict(prob_agg),
        }
        rows.append([row_dict.get(k) for k in header])
    return _write_csv(path, header, rows)


def _write_slice_csv(
    path: Path,
    slice_header_field: str,
    per_model: dict[str, tuple[int, dict[str, Aggregate]]],
    prob: dict[str, dict[str, ModelProbabilisticAggregate]] | None = None,
) -> Path:
    header = ["model", slice_header_field, "sampling_n", *[
        f for f in _SUMMARY_FIELDS if f not in ("model", "sampling_n")
    ]]
    rows: list[list[Any]] = []
    for model, (sampling_n, agg_map) in per_model.items():
        prob_for_model = prob.get(model) if prob else None
        for key, agg in agg_map.items():
            prob_agg = prob_for_model.get(key) if prob_for_model else None
            row_dict = {
                "model": model,
                slice_header_field: key,
                "sampling_n": sampling_n,
                **agg.as_ordered_dict(),
                **_prob_dict(prob_agg),
            }
            rows.append([row_dict.get(k) for k in header])
    return _write_csv(path, header, rows)


def _write_error_breakdown_csv(
    path: Path,
    per_model: dict[str, tuple[int, Counter]],
) -> Path:
    header = ["model", "error_kind", "count", "share_of_total_samples"]
    rows: list[list[Any]] = []
    for model, (total, counter) in per_model.items():
        for kind, count in sorted(counter.items()):
            share = count / total if total else 0.0
            rows.append([model, kind, count, round(share, 4)])
    return _write_csv(path, header, rows)


def _write_finish_reason_breakdown_csv(
    path: Path,
    per_model: dict[str, Counter],
) -> Path:
    """Write `finish_reason` distribution per model. The denominator is the
    eligible sample count for that model (= sum of the counter values, since
    `_finish_reason_breakdown` already filters cutoff rows out)."""
    header = ["model", "finish_reason", "count", "share_of_eligible"]
    rows: list[list[Any]] = []
    for model, counter in per_model.items():
        eligible_total = sum(counter.values())
        for reason, count in sorted(counter.items()):
            share = count / eligible_total if eligible_total else 0.0
            rows.append([model, reason, count, round(share, 4)])
    return _write_csv(path, header, rows)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_per_model_summary_md(
    path: Path,
    per_model: dict[str, tuple[int, Aggregate]],
    prob: dict[str, ModelProbabilisticAggregate] | None = None,
) -> Path:
    lines = ["# Per-model summary", ""]
    header = [
        "model", "N",
        "eligible_Q", "eligible_S", "cutoff_S",
        "pass@1", "pass_any@N", "≥majority", "≥all",
        "majority_acc", "parse_fail", "error_rate",
        "BI", "NLL", "MBS", "ABI_crowd", "ABI_unif", "fallback%",
        "avg_tool", "avg_steps", "avg_nudges", "avg_latency_ms",
        "avg_p/c/r_tokens",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for model, (sampling_n, agg) in per_model.items():
        row_dict = agg.as_ordered_dict()
        prob_dict = _prob_dict(prob.get(model) if prob else None)
        # Surface fallback share as a percentage so reviewers can spot
        # "this model never produced a parsed belief, take the metrics with
        # a grain of salt" without doing arithmetic in their head.
        fallback_pct = (
            f"{prob_dict['fallback_share'] * 100:.1f}%"
            if prob_dict.get("fallback_share") is not None
            else "—"
        )
        tok_cell = f"{_fmt(row_dict['avg_prompt_tokens'])} / {_fmt(row_dict['avg_completion_tokens'])} / {_fmt(row_dict['avg_reasoning_tokens'])}"
        cells = [
            model,
            str(sampling_n),
            _fmt(row_dict["eligible_questions"]),
            _fmt(row_dict["eligible_samples"]),
            _fmt(row_dict["cutoff_skip_samples"]),
            _fmt(row_dict["pass_at_1_avg"]),
            _fmt(row_dict["pass_any_at_n"]),
            _fmt(row_dict["at_least_majority_at_n"]),
            _fmt(row_dict["at_least_all_at_n"]),
            _fmt(row_dict["majority_vote_accuracy"]),
            _fmt(row_dict["parse_failure_rate"]),
            _fmt(row_dict["error_rate"]),
            _fmt(prob_dict["bi"]),
            _fmt(prob_dict["nll"]),
            _fmt(prob_dict["mbs"]),
            _fmt(prob_dict["abi_crowd"]),
            _fmt(prob_dict["abi_uniform"]),
            fallback_pct,
            _fmt(row_dict["avg_tool_calls"]),
            _fmt(row_dict["avg_react_steps"]),
            _fmt(row_dict["avg_nudges_used"]),
            _fmt(row_dict["avg_latency_ms"]),
            tok_cell,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_overall_json(
    path: Path,
    *,
    run_id: str,
    sampling_n_by_model: dict[str, int],
    per_model: dict[str, Aggregate],
    per_model_by_qtype: dict[str, dict[str, Aggregate]],
    per_model_by_ctype: dict[str, dict[str, Aggregate]],
    error_breakdown: dict[str, tuple[int, Counter]],
    probabilistic_per_model: dict[str, ModelProbabilisticAggregate] | None = None,
    probabilistic_by_qtype: (
        dict[str, dict[str, ModelProbabilisticAggregate]] | None
    ) = None,
    probabilistic_by_ctype: (
        dict[str, dict[str, ModelProbabilisticAggregate]] | None
    ) = None,
    analysis_schema: str | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "run_id": run_id,
        "sampling_n_by_model": sampling_n_by_model,
        "per_model": {m: agg.as_ordered_dict() for m, agg in per_model.items()},
        "per_model_by_question_type": {
            m: {k: agg.as_ordered_dict() for k, agg in by_k.items()}
            for m, by_k in per_model_by_qtype.items()
        },
        "per_model_by_choice_type": {
            m: {k: agg.as_ordered_dict() for k, agg in by_k.items()}
            for m, by_k in per_model_by_ctype.items()
        },
        "error_breakdown": {
            m: {"total_samples": total, "counts": dict(sorted(counter.items()))}
            for m, (total, counter) in error_breakdown.items()
        },
    }
    if probabilistic_per_model is not None:
        payload["probabilistic"] = {
            "per_model": {m: _prob_dict(agg) for m, agg in probabilistic_per_model.items()},
            "per_model_by_question_type": {
                m: {k: _prob_dict(agg) for k, agg in by_k.items()}
                for m, by_k in (probabilistic_by_qtype or {}).items()
            },
            "per_model_by_choice_type": {
                m: {k: _prob_dict(agg) for k, agg in by_k.items()}
                for m, by_k in (probabilistic_by_ctype or {}).items()
            },
        }
    if analysis_schema is not None:
        payload["analysis_schema"] = analysis_schema
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


__all__ = [
    "_SUMMARY_FIELDS",
    "_write_csv",
    "_write_per_model_summary_csv",
    "_write_slice_csv",
    "_write_error_breakdown_csv",
    "_write_finish_reason_breakdown_csv",
    "_write_per_model_summary_md",
    "_write_overall_json",
    "_fmt",
]
