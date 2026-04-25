"""Post-run statistics for one evaluation run.

Reads `RUNS_ROOT/{run_id}/db/*.db` (one SQLite per model), computes the metric
suite from FRAME.md §11, and writes the results as CSV/Markdown/JSON under
`RUNS_ROOT/{run_id}/analysis/`.

Pure read side: this module never mutates the per-model DBs.

Entry points:
    * `run_analysis(run_dir: Path) -> list[Path]` — programmatic entry used by
      `evaluation.py` at the end of each run.
    * `python -m forecast_eval.analysis RUNS_ROOT/{run_id}` — CLI to re-run
      analysis against existing DBs.

The v4 refactor (probabilistic-analysis-v4 task 12) split the original
single-file `analysis.py` into a package with focused modules:

* `flatten.py`     — `_flatten_db` pivot + `SampleRow` (incl. v4 `probabilities`).
* `accuracy.py`    — pass@1 / pass_any@N / majority_vote / breakdowns.
* `proper_score.py` — Phase 1 BS / NLL / MBS / BI / ABI (added in task 14).
* `aggregation.py` — Phase 2 K-trial aggregators + LOO shrinkage (task 19).
* `calibration.py` — Phase 2 Platt / temperature / ECE / Murphy + LOO (task 20).
* `inference.py`   — Phase 2 paired bootstrap + Holm + difficulty tertile + posterior (task 21).
* `writers.py`     — CSV / Markdown / JSON serialisation (incl. Phase 2 outputs).

The public surface area is intentionally tiny: external callers should only
import `run_analysis` from this module.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .. import db as dbmod
from .accuracy import (
    Aggregate,
    _aggregate,
    _error_breakdown,
    _finish_reason_breakdown,
    _slice_by,
)
from .aggregation import ShrinkageResult, loo_shrinkage
from .behavior import (
    build_belief_evolution_rows,
    confidence_calibration,
    confidence_conflict_models,
    numeric_confidence_calibration,
    reflection_ab_report,
    tool_usage_pdp,
)
from .calibration import ModelCalibrationReport, calibrate_run
from .flatten import (
    CUTOFF,
    SampleRow,
    _ANALYSIS_FIELDS,
    _answer_gt_for,
    _flatten_db,
    gt_vector,
)
from .inference import (
    DifficultyTertile,
    PairedBootstrapResult,
    difficulty_tertile,
    paired_bootstrap_by_difficulty,
    pairwise_paired_bootstrap,
)
from .probabilistic import (
    _QuestionProbabilityRow,
    _aggregate_for_subset,
    build_probabilistic_report,
)
from .proper_score import (
    ModelProbabilisticAggregate,
    brier_score_lab,
)
from .writers import (
    _SUMMARY_FIELDS,
    _write_belief_evolution_csv,
    _write_brier_decomposition_csv,
    _write_calibration_params_json,
    _write_confidence_calibration_csv,
    _write_error_breakdown_csv,
    _write_finish_reason_breakdown_csv,
    _write_numeric_confidence_calibration_csv,
    _write_overall_json,
    _write_paired_delta_bi_by_difficulty_csv,
    _write_paired_delta_bi_csv,
    _write_pairwise_significance_csv,
    _write_per_model_by_difficulty_csv,
    _write_per_model_summary_calibrated_csv,
    _write_per_model_summary_csv,
    _write_per_model_summary_md,
    _write_posterior_pairwise_csv,
    _write_reflection_ab_csv,
    _write_reliability_data_json,
    _write_shrinkage_alpha_curve_csv,
    _write_slice_csv,
    _write_tool_usage_pdp_csv,
)


# --------------------------------------------------------------------------- #
# Phase 2 helpers (live here so __init__.py owns orchestration)
# --------------------------------------------------------------------------- #


def _shrinkage_per_model_per_ctype(
    samples_by_model: dict[str, list[SampleRow]],
    gt_map: dict[str, frozenset[str]],
) -> dict[str, ShrinkageResult]:
    """Run `loo_shrinkage` per (model, choice_type).

    The shrinkage formula differs for single (softmax) vs multi (sigmoid),
    so we split each model's questions by choice_type and run shrinkage
    independently. Returns `{f"{model}__{ctype}": result}` so the writer
    can emit one row per (model, ctype) without nested keys.
    """
    out: dict[str, ShrinkageResult] = {}
    for model, samples in samples_by_model.items():
        # Group by question, keep all eligible probabilities.
        per_q: dict[str, list[SampleRow]] = {}
        for s in samples:
            if not s.is_eligible or s.probabilities is None:
                continue
            if s.question_id not in gt_map:
                continue
            per_q.setdefault(s.question_id, []).append(s)

        # Bucket by choice_type (canonical: take first sample's ctype).
        by_ctype: dict[str, list[tuple[list[list[float]], list[int]]]] = {}
        for qid, ss in per_q.items():
            gt = gt_map.get(qid)
            if gt is None or not ss[0].options:
                continue
            k = len(ss[0].options)
            obs = gt_vector(gt, k)
            preds = [s.probabilities for s in ss if s.probabilities is not None]
            if not preds:
                continue
            ctype = ss[0].choice_type
            by_ctype.setdefault(ctype, []).append((preds, obs))

        for ctype, items in by_ctype.items():
            preds_list = [it[0] for it in items]
            obs_list = [it[1] for it in items]
            try:
                res = loo_shrinkage(preds_list, obs_list, ctype)
            except ValueError:
                continue
            out[f"{model}__{ctype}"] = res
    return out


def _bs_by_model_qid_from_rows(
    rows_by_model: dict[str, list[_QuestionProbabilityRow]],
) -> dict[str, dict[str, float]]:
    """Per-(model, question) label-wise BS — input for paired bootstrap and posterior."""
    out: dict[str, dict[str, float]] = {}
    for model, rows in rows_by_model.items():
        per_q: dict[str, float] = {}
        for r in rows:
            per_q[r.question_id] = brier_score_lab(r.probs, r.obs)
        out[model] = per_q
    return out


def _gamma_uniform_per_qid(
    rows_by_model: dict[str, list[_QuestionProbabilityRow]],
) -> dict[str, float]:
    """Union of uniform $\\gamma$ per question across models.

    Uniform $\\gamma$ depends only on the observation vector — same across
    models on the same question. We take any model's value as authoritative.
    """
    from .proper_score import uniform_gamma_for

    out: dict[str, float] = {}
    for rows in rows_by_model.values():
        for r in rows:
            if r.question_id not in out:
                out[r.question_id] = uniform_gamma_for(r.obs)
    return out


def _per_model_per_tier_aggregates(
    rows_by_model: dict[str, list[_QuestionProbabilityRow]],
    tertile: DifficultyTertile,
    crowd_gammas_by_model: dict[str, dict[str, float | None]],
    uniform_gammas: dict[str, float],
) -> dict[str, dict[str, ModelProbabilisticAggregate]]:
    """Slice each model's rows by difficulty tier and re-aggregate proper scores."""
    out: dict[str, dict[str, ModelProbabilisticAggregate]] = {}
    for model, rows in rows_by_model.items():
        per_tier_rows: dict[str, list[_QuestionProbabilityRow]] = {
            "low": [], "mid": [], "high": [],
        }
        for r in rows:
            tier = tertile.by_question.get(r.question_id)
            if tier is None:
                continue
            per_tier_rows[tier].append(r)
        out[model] = {
            tier: _aggregate_for_subset(
                tier_rows,
                crowd_gammas=crowd_gammas_by_model.get(model),
                uniform_gammas=uniform_gammas,
            )
            for tier, tier_rows in per_tier_rows.items()
        }
    return out


def _paired_bootstrap_pairs_by_difficulty(
    bs_by_model_qid: dict[str, dict[str, float]],
    tertile: DifficultyTertile,
) -> dict[tuple[str, str], dict[str, PairedBootstrapResult]]:
    """For every ordered pair of models, run `paired_bootstrap_by_difficulty`."""
    out: dict[tuple[str, str], dict[str, PairedBootstrapResult]] = {}
    models = sorted(bs_by_model_qid.keys())
    for i, ma in enumerate(models):
        for mb in models[i + 1:]:
            common = sorted(
                set(bs_by_model_qid[ma]) & set(bs_by_model_qid[mb])
            )
            bs_a = {q: bs_by_model_qid[ma][q] for q in common}
            bs_b = {q: bs_by_model_qid[mb][q] for q in common}
            tertile_subset = DifficultyTertile(
                by_question={q: t for q, t in tertile.by_question.items() if q in common},
                threshold_low=tertile.threshold_low,
                threshold_high=tertile.threshold_high,
                n_low=sum(1 for q in common if tertile.by_question.get(q) == "low"),
                n_mid=sum(1 for q in common if tertile.by_question.get(q) == "mid"),
                n_high=sum(1 for q in common if tertile.by_question.get(q) == "high"),
            )
            out[(ma, mb)] = paired_bootstrap_by_difficulty(bs_a, bs_b, tertile_subset)
    return out


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #


def run_analysis(run_dir: Path) -> list[Path]:
    """Generate every analysis artefact for the run and return the file paths."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found under {run_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = manifest.get("run_id", run_dir.name)
    models: list[str] = manifest["models"]
    model_files: dict[str, str] = manifest["model_files"]
    sampling_n_top: int = manifest.get("sampling_n", 1)
    # `analysis_schema` was added in v4 manifests. v3 runs replayed under v4
    # code don't carry the field; treat that as "v3 fallback semantics".
    analysis_schema: str | None = manifest.get("analysis_schema")

    db_dir = run_dir / "db"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    per_model_agg: dict[str, Aggregate] = {}
    per_model_agg_qtype: dict[str, dict[str, Aggregate]] = {}
    per_model_agg_ctype: dict[str, dict[str, Aggregate]] = {}
    per_model_error: dict[str, tuple[int, Counter]] = {}
    per_model_finish_reason: dict[str, Counter] = {}
    sampling_n_by_model: dict[str, int] = {}
    samples_by_model: dict[str, list[SampleRow]] = {}
    gt_map_global: dict[str, frozenset[str]] = {}

    summary_payload: dict[str, tuple[int, Aggregate]] = {}
    slice_qtype_payload: dict[str, tuple[int, dict[str, Aggregate]]] = {}
    slice_ctype_payload: dict[str, tuple[int, dict[str, Aggregate]]] = {}

    for model in models:
        db_path = db_dir / model_files[model]
        if not db_path.exists():
            # Skip a missing DB rather than crash — user may have partial data.
            continue
        conn = dbmod.connect(db_path)
        try:
            meta_row = conn.execute(
                "SELECT sampling_n FROM run_meta ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            sampling_n = int(meta_row["sampling_n"]) if meta_row else sampling_n_top
            sampling_n_by_model[model] = sampling_n
            samples = _flatten_db(conn, sampling_n, model)
            gt_map = _answer_gt_for(conn)
            # Each per-model DB carries its own `questions` copy. Union them
            # so the probabilistic crowd baseline can use a question even if
            # one of the models skipped it (e.g. cutoff).
            for qid, gt in gt_map.items():
                gt_map_global.setdefault(qid, gt)
            agg = _aggregate(samples, sampling_n, gt_map=gt_map)
            agg_qtype = _slice_by(samples, lambda s: s.question_type, sampling_n, gt_map)
            agg_ctype = _slice_by(samples, lambda s: s.choice_type, sampling_n, gt_map)
            per_model_agg[model] = agg
            per_model_agg_qtype[model] = agg_qtype
            per_model_agg_ctype[model] = agg_ctype
            per_model_error[model] = (len(samples), _error_breakdown(samples))
            per_model_finish_reason[model] = _finish_reason_breakdown(samples)
            samples_by_model[model] = samples
            summary_payload[model] = (sampling_n, agg)
            slice_qtype_payload[model] = (sampling_n, agg_qtype)
            slice_ctype_payload[model] = (sampling_n, agg_ctype)
        finally:
            conn.close()

    prob_report = build_probabilistic_report(samples_by_model, gt_map_global)

    written: list[Path] = []
    # `per_model_summary.md` is written AFTER calibration so the markdown can
    # surface uncal/cal columns + the `cal*` overfit warning (spec 20.8). The
    # CSV is independent and can ship immediately.
    if summary_payload:
        written.append(_write_per_model_summary_csv(
            analysis_dir / "per_model_summary.csv",
            summary_payload,
            prob=prob_report.per_model,
        ))
    if slice_qtype_payload:
        written.append(_write_slice_csv(
            analysis_dir / "per_model_by_question_type.csv",
            "question_type",
            slice_qtype_payload,
            prob=prob_report.per_model_by_qtype,
        ))
    if slice_ctype_payload:
        written.append(_write_slice_csv(
            analysis_dir / "per_model_by_choice_type.csv",
            "choice_type",
            slice_ctype_payload,
            prob=prob_report.per_model_by_ctype,
        ))
    if per_model_error:
        written.append(_write_error_breakdown_csv(
            analysis_dir / "error_breakdown.csv", per_model_error,
        ))
    if per_model_finish_reason:
        written.append(_write_finish_reason_breakdown_csv(
            analysis_dir / "finish_reason_breakdown.csv", per_model_finish_reason,
        ))
    written.append(_write_overall_json(
        analysis_dir / "overall.json",
        run_id=run_id,
        sampling_n_by_model=sampling_n_by_model,
        per_model=per_model_agg,
        per_model_by_qtype=per_model_agg_qtype,
        per_model_by_ctype=per_model_agg_ctype,
        error_breakdown=per_model_error,
        probabilistic_per_model=prob_report.per_model,
        probabilistic_by_qtype=prob_report.per_model_by_qtype,
        probabilistic_by_ctype=prob_report.per_model_by_ctype,
        analysis_schema=analysis_schema,
    ))

    # ------------------------------------------------------------------ #
    # Phase 2 deliverables — only attempted when at least one model
    # produced probability vectors. Empty-input branches keep the analysis
    # idempotent on v3 fixtures with no boxed answers.
    # ------------------------------------------------------------------ #
    rows_by_model = prob_report.rows_by_model
    has_any_rows = any(rows for rows in rows_by_model.values())
    cal_reports: dict[str, ModelCalibrationReport] = {}
    if has_any_rows:
        # Aggregation: per-(model, choice_type) shrinkage curves.
        shrinkage_per_key = _shrinkage_per_model_per_ctype(
            samples_by_model, gt_map_global
        )
        if shrinkage_per_key:
            written.append(_write_shrinkage_alpha_curve_csv(
                analysis_dir / "shrinkage_alpha_curve.csv",
                shrinkage_per_key,
            ))

        # Calibration: per-(model, cell) Platt / Temperature with LOO.
        from .probabilistic import _build_crowd_gammas_per_model

        crowd_gammas_by_model = _build_crowd_gammas_per_model(rows_by_model)
        uniform_gammas_global = _gamma_uniform_per_qid(rows_by_model)
        cal_reports = calibrate_run(
            rows_by_model,
            crowd_gammas_by_model=crowd_gammas_by_model,
            uniform_gammas=uniform_gammas_global,
        )
        if cal_reports:
            written.append(_write_calibration_params_json(
                analysis_dir / "calibration_params.json", cal_reports,
            ))
            written.append(_write_per_model_summary_calibrated_csv(
                analysis_dir / "per_model_summary_calibrated.csv", cal_reports,
            ))
            written.append(_write_reliability_data_json(
                analysis_dir / "reliability_data.json",
                cal_reports, calibrated=False,
            ))
            written.append(_write_reliability_data_json(
                analysis_dir / "reliability_data_calibrated.json",
                cal_reports, calibrated=True,
            ))
            written.append(_write_brier_decomposition_csv(
                analysis_dir / "brier_decomposition.csv", cal_reports,
            ))

        # Inference: pairwise paired bootstrap + Holm + posterior.
        bs_by_model_qid = _bs_by_model_qid_from_rows(rows_by_model)
        if len(bs_by_model_qid) >= 2:
            pairs = pairwise_paired_bootstrap(bs_by_model_qid)
            if pairs:
                written.append(_write_paired_delta_bi_csv(
                    analysis_dir / "paired_delta_bi.csv", pairs,
                ))
                written.append(_write_pairwise_significance_csv(
                    analysis_dir / "pairwise_significance.csv", pairs,
                ))
                written.append(_write_posterior_pairwise_csv(
                    analysis_dir / "posterior_pairwise.csv", pairs,
                ))

        # Difficulty stratification: per-tier aggregates + paired bootstrap.
        if uniform_gammas_global:
            tertile = difficulty_tertile(uniform_gammas_global)
            per_model_per_tier = _per_model_per_tier_aggregates(
                rows_by_model, tertile, crowd_gammas_by_model, uniform_gammas_global,
            )
            if per_model_per_tier:
                written.append(_write_per_model_by_difficulty_csv(
                    analysis_dir / "per_model_by_difficulty.csv",
                    per_model_per_tier,
                ))
            if len(bs_by_model_qid) >= 2:
                by_pair_per_tier = _paired_bootstrap_pairs_by_difficulty(
                    bs_by_model_qid, tertile,
                )
                if by_pair_per_tier:
                    written.append(_write_paired_delta_bi_by_difficulty_csv(
                        analysis_dir / "paired_delta_bi_by_difficulty.csv",
                        by_pair_per_tier,
                    ))

    # ------------------------------------------------------------------ #
    # Phase 3 deliverables — behavior, reflection A/B, tool PDP, confidence.
    # All four blocks degrade gracefully on v3 fixtures where belief_trace
    # is uniformly NULL (empty CSVs / skipped writes).
    # ------------------------------------------------------------------ #
    behavior_rows = build_belief_evolution_rows(samples_by_model, gt_map_global)
    if behavior_rows:
        written.append(_write_belief_evolution_csv(
            analysis_dir / "belief_evolution.csv", behavior_rows,
        ))

    pdp_rows = tool_usage_pdp(samples_by_model, gt_map_global)
    if pdp_rows:
        written.append(_write_tool_usage_pdp_csv(
            analysis_dir / "tool_usage_pdp.csv", pdp_rows,
        ))

    confidence_rows = confidence_calibration(samples_by_model)
    numeric_confidence_rows = numeric_confidence_calibration(samples_by_model)
    # The confidence rows always have ≥1 model entry, but the values are
    # mostly None on v3 fixtures (no parsed beliefs). Suppress writing when
    # every numeric / hit_rate is None so the CSV doesn't add noise.
    if any(r.n_samples > 0 for r in confidence_rows):
        written.append(_write_confidence_calibration_csv(
            analysis_dir / "confidence_calibration.csv", confidence_rows,
        ))
    if any(r.n_samples > 0 for r in numeric_confidence_rows):
        written.append(_write_numeric_confidence_calibration_csv(
            analysis_dir / "numeric_confidence_calibration.csv",
            numeric_confidence_rows,
        ))

    # Reflection A/B requires sibling runs that match every fingerprint
    # except `reflection_protocol_hash`. We look one directory up from the
    # current run because pairs live as siblings under runs/.
    runs_root = run_dir.parent
    try:
        from .behavior import find_paired_runs

        pairs = find_paired_runs(runs_root)
        if pairs:
            ab_rows = reflection_ab_report(pairs)
            if ab_rows:
                written.append(_write_reflection_ab_csv(
                    analysis_dir / "reflection_ab.csv", ab_rows,
                ))
    except Exception:  # pragma: no cover — reflection A/B is best-effort
        pass

    conflict_models = confidence_conflict_models(confidence_rows)

    # `per_model_summary.md` is written last so it can include calibration
    # columns + the `cal*` overfit warning AND the Phase 3 `conflict*`
    # marker. When neither Phase 2 nor Phase 3 outputs are available, the
    # writer falls back to v3+Phase-1 columns automatically.
    if summary_payload:
        written.append(_write_per_model_summary_md(
            analysis_dir / "per_model_summary.md",
            summary_payload,
            prob=prob_report.per_model,
            cal=cal_reports if cal_reports else None,
            confidence_conflict_models=conflict_models or None,
        ))
    return written


__all__ = [
    "run_analysis",
    # Re-exported for tests / dogfooding tools that poke at internals.
    "Aggregate",
    "SampleRow",
    "CUTOFF",
    "_ANALYSIS_FIELDS",
    "_SUMMARY_FIELDS",
]
