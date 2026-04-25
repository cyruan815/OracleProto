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
from .aggregation import ShrinkageResult
from .calibration import (
    CalibrationBin,
    ModelCalibrationReport,
    MurphyDecomposition,
    PlattParams,
    TemperatureParams,
)
from .inference import ModelPairResult, PairedBootstrapResult
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
    cal: dict[str, ModelCalibrationReport] | None = None,
) -> Path:
    """Markdown table covering accuracy, probabilistic, and (Phase 2) calibration.

    `cal` overlays uncal/cal BI / NLL / ECE columns and the `cal*` warning
    marker per spec 20.8. When `cal` is None (no Phase 2 outputs available),
    the table degrades gracefully to v3+Phase-1 columns.
    """
    lines = ["# Per-model summary", ""]
    header = [
        "model", "N",
        "eligible_Q", "eligible_S", "cutoff_S",
        "pass@1", "pass_any@N", "≥majority", "≥all",
        "majority_acc", "parse_fail", "error_rate",
        "BI", "NLL", "MBS", "ABI_crowd", "ABI_unif", "fallback%",
    ]
    if cal is not None:
        header.extend(["BI_cal", "NLL_cal", "ECE_uncal", "ECE_cal"])
    header.extend([
        "avg_tool", "avg_steps", "avg_nudges", "avg_latency_ms",
        "avg_p/c/r_tokens",
    ])
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
        # Spec 20.8 哨兵: append `cal*` to the model cell when cal BI exceeds
        # uncal BI by 5 — that's the threshold for "calibration probably
        # overfit" per the design.
        model_cell = model
        if cal is not None and model in cal and cal[model].overfit_warning:
            model_cell = f"{model} cal*"
        tok_cell = f"{_fmt(row_dict['avg_prompt_tokens'])} / {_fmt(row_dict['avg_completion_tokens'])} / {_fmt(row_dict['avg_reasoning_tokens'])}"
        cells = [
            model_cell,
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
        ]
        if cal is not None:
            rep = cal.get(model)
            if rep is None:
                cells.extend(["—", "—", "—", "—"])
            else:
                cells.extend([
                    _fmt(_round(rep.cal_aggregate.bi)),
                    _fmt(_round(rep.cal_aggregate.nll)),
                    _fmt(_round(rep.ece_uncal)),
                    _fmt(_round(rep.ece_cal)),
                ])
        cells.extend([
            _fmt(row_dict["avg_tool_calls"]),
            _fmt(row_dict["avg_react_steps"]),
            _fmt(row_dict["avg_nudges_used"]),
            _fmt(row_dict["avg_latency_ms"]),
            tok_cell,
        ])
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


def _write_shrinkage_alpha_curve_csv(
    path: Path,
    per_model: dict[str, ShrinkageResult],
) -> Path:
    """Phase 2 task 19.5: per-model α grid scan with mean BS and BI."""
    header = ["model", "alpha", "mean_bs", "bi", "n_questions", "choice_type"]
    rows: list[list[Any]] = []
    for model, res in per_model.items():
        for alpha, mean_bs, bi in res.curve:
            rows.append(
                [model, round(alpha, 4), round(mean_bs, 6),
                 round(bi, 4), res.n_questions, res.choice_type]
            )
    return _write_csv(path, header, rows)


def _serialize_cell_params(params: Any) -> dict[str, Any]:
    """JSON-serializable form for a Platt or Temperature param bundle."""
    if isinstance(params, PlattParams):
        return {"method": "platt", "a": params.a, "b": params.b}
    if isinstance(params, TemperatureParams):
        return {"method": "temperature", "T": params.T}
    return {}


def _write_calibration_params_json(
    path: Path,
    reports: dict[str, ModelCalibrationReport],
) -> Path:
    """Phase 2 task 20.7: per-model per-cell calibration parameters (all-data fit).

    Layout matches design.md:
      `{model: {cell_name: {method, a/b/T, n_questions}}}`.
    """
    payload: dict[str, Any] = {}
    for model, rep in reports.items():
        cells_payload: dict[str, Any] = {}
        for name, cell in rep.cells.items():
            entry = _serialize_cell_params(cell.params)
            entry["n_questions"] = cell.n_questions
            entry["question_type"] = cell.cell.question_type
            entry["choice_type"] = cell.cell.choice_type
            cells_payload[name] = entry
        payload[model] = cells_payload
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


_CALIBRATED_SUMMARY_FIELDS: tuple[str, ...] = (
    "model",
    "n_questions",
    "fallback_share",
    "bi_uncal",
    "bi_cal",
    "nll_uncal",
    "nll_cal",
    "ece_uncal",
    "ece_cal",
    "abi_crowd_uncal",
    "abi_crowd_cal",
    "abi_uniform_uncal",
    "abi_uniform_cal",
    "overfit_warning",
)


def _write_per_model_summary_calibrated_csv(
    path: Path,
    reports: dict[str, ModelCalibrationReport],
) -> Path:
    """Phase 2 task 20.7: post-calibration headline metrics + uncal sanity."""
    header = list(_CALIBRATED_SUMMARY_FIELDS)
    rows: list[list[Any]] = []
    for model, rep in reports.items():
        rows.append([
            model,
            rep.uncal_aggregate.n_questions,
            _round(rep.uncal_aggregate.fallback_share),
            _round(rep.uncal_aggregate.bi),
            _round(rep.cal_aggregate.bi),
            _round(rep.uncal_aggregate.nll),
            _round(rep.cal_aggregate.nll),
            _round(rep.ece_uncal),
            _round(rep.ece_cal),
            _round(rep.uncal_aggregate.abi_crowd),
            _round(rep.cal_aggregate.abi_crowd),
            _round(rep.uncal_aggregate.abi_uniform),
            _round(rep.cal_aggregate.abi_uniform),
            int(rep.overfit_warning),
        ])
    return _write_csv(path, header, rows)


def _flat_pairs_for_reliability(
    rep: ModelCalibrationReport, *, calibrated: bool
) -> tuple[list[float], list[int]]:
    """Same flatten convention as `calibration._flat_pairs_for_ece`.

    For single-choice: top-1 confidence vs top-1 hit. For multi: per-(q, l)
    pairs. Imported into writers.py rather than `from calibration import`
    private helper so a future refactor of the helper doesn't break the
    JSON layout consumed by `scripts/plot_analysis.py`.
    """
    probs: list[float] = []
    obs: list[int] = []
    for cr in rep.calibrated_rows:
        r = cr.row
        p_vec = cr.cal_probs if calibrated else r.probs
        if r.choice_type == "single":
            best_i = 0
            best_p = p_vec[0]
            for i, p in enumerate(p_vec):
                if p > best_p:
                    best_p = p
                    best_i = i
            probs.append(best_p)
            obs.append(int(r.obs[best_i]))
        else:
            for p, o in zip(p_vec, r.obs):
                probs.append(p)
                obs.append(int(o))
    return probs, obs


def _bins_for_model_qtype(
    rep: ModelCalibrationReport, qtype: str, *, calibrated: bool, n_bins: int = 15
) -> list[CalibrationBin]:
    """Reliability bins restricted to a single question_type."""
    from .calibration import reliability_bins

    probs: list[float] = []
    obs: list[int] = []
    for cr in rep.calibrated_rows:
        r = cr.row
        if r.question_type != qtype:
            continue
        p_vec = cr.cal_probs if calibrated else r.probs
        if r.choice_type == "single":
            best_i = 0
            best_p = p_vec[0]
            for i, p in enumerate(p_vec):
                if p > best_p:
                    best_p = p
                    best_i = i
            probs.append(best_p)
            obs.append(int(r.obs[best_i]))
        else:
            for p, o in zip(p_vec, r.obs):
                probs.append(p)
                obs.append(int(o))
    return reliability_bins(probs, obs, n_bins)


def _write_reliability_data_json(
    path: Path,
    reports: dict[str, ModelCalibrationReport],
    *,
    calibrated: bool,
    n_bins: int = 15,
) -> Path:
    """Phase 2 task 20.7: per-(model, qtype) bins for reliability diagrams.

    Layout: `{model: {qtype: [{n, mean_p, mean_o, bin_lo, bin_hi}, ...]}}`.
    Empty bins are skipped (consistent with `reliability_bins`).
    """
    payload: dict[str, Any] = {}
    for model, rep in reports.items():
        per_qtype: dict[str, list[dict[str, Any]]] = {}
        qtypes = sorted({cr.row.question_type for cr in rep.calibrated_rows})
        for qt in qtypes:
            bins = _bins_for_model_qtype(rep, qt, calibrated=calibrated, n_bins=n_bins)
            per_qtype[qt] = [
                {
                    "bin_lo": round(b.bin_lo, 4),
                    "bin_hi": round(b.bin_hi, 4),
                    "n": b.n,
                    "mean_p": round(b.mean_p, 6),
                    "mean_o": round(b.mean_o, 6),
                }
                for b in bins
            ]
        payload[model] = per_qtype
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _write_brier_decomposition_csv(
    path: Path,
    reports: dict[str, ModelCalibrationReport],
) -> Path:
    """Phase 2 task 20.7: Murphy three-decomposition, uncal + cal."""
    header = [
        "model", "kind",
        "rel", "res", "unc", "total",
    ]
    rows: list[list[Any]] = []
    for model, rep in reports.items():
        for kind, decomp in (
            ("uncalibrated", rep.murphy_uncal),
            ("calibrated", rep.murphy_cal),
        ):
            if decomp is None:
                continue
            rows.append([
                model, kind,
                round(decomp.rel, 6),
                round(decomp.res, 6),
                round(decomp.unc, 6),
                round(decomp.total, 6),
            ])
    return _write_csv(path, header, rows)


def _bs_to_bi(mean_bs: float) -> float:
    """$100 \\cdot (1 - \\sqrt{\\overline{BS}})$ — the same convention `proper_score.brier_index` uses."""
    import math

    if mean_bs < 0:
        return 100.0 * (1.0 + math.sqrt(-mean_bs))
    return 100.0 * (1.0 - math.sqrt(mean_bs))


def _write_paired_delta_bi_csv(
    path: Path,
    pairs: list[ModelPairResult],
) -> Path:
    """Phase 2 task 21.6: pairwise mean ΔBS converted to ΔBI for readability."""
    header = [
        "model_a", "model_b", "n_questions",
        "delta_bs", "ci_low_bs", "ci_high_bs",
        "delta_bi_approx",
        "p_raw", "p_holm", "posterior_a_better",
    ]
    rows: list[list[Any]] = []
    for p in pairs:
        # ΔBI = BI_b - BI_a (positive means b is worse — i.e. a is better).
        # Approximated by linearising around the per-pair mean BS; not the
        # same as recomputing BI from each side but adequate as a column
        # alongside the BS-domain CIs.
        delta_bi_approx = -p.delta_bs_mean * 100.0
        rows.append([
            p.model_a, p.model_b, p.n_questions,
            round(p.delta_bs_mean, 6),
            round(p.ci_low, 6),
            round(p.ci_high, 6),
            round(delta_bi_approx, 4),
            round(p.p_raw, 6),
            round(p.p_holm, 6) if p.p_holm is not None else None,
            round(p.posterior_a_better, 6),
        ])
    return _write_csv(path, header, rows)


def _write_pairwise_significance_csv(
    path: Path,
    pairs: list[ModelPairResult],
    *,
    alpha: float = 0.05,
) -> Path:
    """Phase 2 task 21.6: significance flags at α=0.05 using Holm-adjusted p-values."""
    header = [
        "model_a", "model_b", "delta_bs", "p_raw", "p_holm",
        "is_significant_raw", "is_significant_holm",
    ]
    rows: list[list[Any]] = []
    for p in pairs:
        raw_sig = int(p.p_raw < alpha)
        holm_sig = int(p.p_holm is not None and p.p_holm < alpha)
        rows.append([
            p.model_a, p.model_b,
            round(p.delta_bs_mean, 6),
            round(p.p_raw, 6),
            round(p.p_holm, 6) if p.p_holm is not None else None,
            raw_sig, holm_sig,
        ])
    return _write_csv(path, header, rows)


def _write_posterior_pairwise_csv(
    path: Path,
    pairs: list[ModelPairResult],
) -> Path:
    """Phase 2 task 21.6: $\\Pr(\\mathrm{BI}_A > \\mathrm{BI}_B)$ from paired bootstrap."""
    header = ["model_a", "model_b", "n_questions", "prob_a_better"]
    rows: list[list[Any]] = []
    for p in pairs:
        rows.append([
            p.model_a, p.model_b, p.n_questions,
            round(p.posterior_a_better, 6),
        ])
    return _write_csv(path, header, rows)


def _write_per_model_by_difficulty_csv(
    path: Path,
    per_model_per_tier: dict[str, dict[str, ModelProbabilisticAggregate]],
) -> Path:
    """Phase 2 task 21.6: per-(model, tier) probabilistic aggregates."""
    header = [
        "model", "difficulty_tertile", "n_questions",
        "bi", "nll", "abi_crowd", "abi_uniform", "fallback_share",
    ]
    rows: list[list[Any]] = []
    tier_order = {"low": 0, "mid": 1, "high": 2}
    for model in sorted(per_model_per_tier.keys()):
        tiers = per_model_per_tier[model]
        for tier in sorted(tiers.keys(), key=lambda t: tier_order.get(t, 99)):
            agg = tiers[tier]
            rows.append([
                model, tier, agg.n_questions,
                _round(agg.bi),
                _round(agg.nll),
                _round(agg.abi_crowd),
                _round(agg.abi_uniform),
                _round(agg.fallback_share),
            ])
    return _write_csv(path, header, rows)


def _write_paired_delta_bi_by_difficulty_csv(
    path: Path,
    by_pair_per_tier: dict[tuple[str, str], dict[str, PairedBootstrapResult]],
) -> Path:
    """Phase 2 task 21.6: pairwise ΔBS within each difficulty tertile."""
    header = [
        "model_a", "model_b", "difficulty_tertile", "n_questions",
        "delta_bs", "ci_low_bs", "ci_high_bs",
        "delta_bi_approx", "p_two_sided",
    ]
    rows: list[list[Any]] = []
    tier_order = {"low": 0, "mid": 1, "high": 2}
    for (ma, mb) in sorted(by_pair_per_tier.keys()):
        tiers = by_pair_per_tier[(ma, mb)]
        for tier in sorted(tiers.keys(), key=lambda t: tier_order.get(t, 99)):
            res = tiers[tier]
            rows.append([
                ma, mb, tier, res.n_questions,
                round(res.delta_mean, 6),
                round(res.ci_low, 6),
                round(res.ci_high, 6),
                round(-res.delta_mean * 100.0, 4),
                round(res.p_two_sided, 6),
            ])
    return _write_csv(path, header, rows)


__all__ = [
    "_SUMMARY_FIELDS",
    "_CALIBRATED_SUMMARY_FIELDS",
    "_write_csv",
    "_write_per_model_summary_csv",
    "_write_slice_csv",
    "_write_error_breakdown_csv",
    "_write_finish_reason_breakdown_csv",
    "_write_per_model_summary_md",
    "_write_overall_json",
    "_write_shrinkage_alpha_curve_csv",
    "_write_calibration_params_json",
    "_write_per_model_summary_calibrated_csv",
    "_write_reliability_data_json",
    "_write_brier_decomposition_csv",
    "_write_paired_delta_bi_csv",
    "_write_pairwise_significance_csv",
    "_write_posterior_pairwise_csv",
    "_write_per_model_by_difficulty_csv",
    "_write_paired_delta_bi_by_difficulty_csv",
    "_fmt",
]
