"""Post-run statistics for one evaluation run (v5: discrete-native primary).

Reads `RUNS_ROOT/{run_id}/db/*.db` (one SQLite per model), computes the metric
suite from `ANALYSIS_DESIGN_v5.md`, and writes the results as CSV/Markdown/JSON
under `RUNS_ROOT/{run_id}/analysis/`.

Pure read side: this module never mutates the per-model DBs.

Entry points:
    * `run_analysis(run_dir: Path) -> list[Path]` — programmatic entry used by
      `evaluation.py` at the end of each run.
    * `python -m forecast_eval.analysis RUNS_ROOT/{run_id}` — CLI to re-run
      analysis against existing DBs.

v5 changes from v4:

* `calibration.py` deprecated and removed: at K=5 the empirical probability
  has only 6 discrete levels per label, making Reliability Diagram /
  Murphy decomposition / Platt scaling statistically meaningless. The 5
  associated outputs (`reliability_data*.json` / `brier_decomposition.csv`
  / `calibration_params.json` / `per_model_summary_calibrated.csv`) are no
  longer written.
* `consistency.py` is new: K-trial-only metrics — Fleiss' κ, predictive
  entropy, entropy-accuracy joint analysis (per-model tertile bucketing),
  VCI, MVG.
* `inference.py` extended with `metric_paired_bootstrap` for FSS / Acc /
  MV_Acc / Fleiss κ / EBI; the v4 BS-paired bootstrap is preserved
  (grid analysis depends on it).
* `accuracy.py` new functions: `tversky_score`, `fss`, `cohen_kappa`,
  `hamming_score`. FSS is the v5 main metric.

Module layout:

* `flatten.py`     — `_flatten_db` pivot + `SampleRow` (incl. v4 `probabilities`).
* `accuracy.py`    — pass@1 + v5 FSS / Cohen κ / Hamming.
* `consistency.py` — v5 K-trial Fleiss κ / entropy / entropy-Acc bins / VCI / MVG.
* `proper_score.py` — BS / NLL / MBS / BI / ABI (companion probabilistic).
* `aggregation.py` — K-trial aggregators + LOO shrinkage.
* `inference.py`   — v4 BS-paired bootstrap + v5 multi-metric paired bootstrap.
* `behavior.py`    — belief evolution + reflection A/B + tool PDP + confidence (v3 / v4 carry-over).
* `grid.py`        — grid (R, C) analysis (orthogonal to v5).
* `writers.py`     — CSV / Markdown / JSON serialisation.

External callers should only import `run_analysis` from this module.
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
    cohen_kappa,
    fss,
    hamming_score,
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
from .consistency import (
    ConsistencyReport,
    build_consistency_report,
    entropy_accuracy_bins,
)
from .flatten import (
    CUTOFF,
    SampleRow,
    _ANALYSIS_FIELDS,
    _answer_gt_for,
    _flatten_db,
    _question_options_for,
    gt_vector,
)
from .inference import (
    DEFAULT_METRIC_FNS,
    DifficultyTertile,
    PairedBootstrapResult,
    difficulty_tertile,
    paired_bootstrap_by_difficulty,
    pairwise_metric_bootstrap,
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
    _write_confidence_calibration_csv,
    _write_entropy_accuracy_bins_csv,
    _write_error_breakdown_csv,
    _write_finish_reason_breakdown_csv,
    _write_inter_trial_consistency_csv,
    _write_metric_pairwise_bootstrap_csv,
    _write_numeric_confidence_calibration_csv,
    _write_overall_json,
    _write_paired_delta_bi_by_difficulty_csv,
    _write_paired_delta_bi_csv,
    _write_pairwise_significance_csv,
    _write_per_model_by_difficulty_csv,
    _write_per_model_summary_csv,
    _write_per_model_summary_md,
    _write_posterior_pairwise_csv,
    _write_reflection_ab_csv,
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
    # `analysis_schema` was added in v4 manifests. v3 runs replayed under v4/v5
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
    options_map_global: dict[str, list[str]] = {}

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
            options_map = _question_options_for(conn)
            # Each per-model DB carries its own `questions` copy. Union them
            # so the probabilistic crowd baseline can use a question even if
            # one of the models skipped it (e.g. cutoff).
            for qid, gt in gt_map.items():
                gt_map_global.setdefault(qid, gt)
            for qid, opts in options_map.items():
                options_map_global.setdefault(qid, opts)
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

    # ------------------------------------------------------------------ #
    # v5 discrete-native family per model: FSS / Cohen κ / Hamming /
    # ConsistencyReport / per-model entropy-Acc bins. All return None /
    # empty-list on K=1 fixtures or missing data — graceful degradation.
    # ------------------------------------------------------------------ #
    fss_per_model: dict[str, dict] = {}
    cohen_kappa_per_model: dict[str, float | None] = {}
    hamming_per_model: dict[str, float | None] = {}
    consistency_per_model: dict[str, ConsistencyReport] = {}
    entropy_acc_per_model: dict[str, list[dict]] = {}
    samples_by_model_by_q: dict[str, dict[str, list[SampleRow]]] = {}
    for model, samples in samples_by_model.items():
        fss_per_model[model] = fss(samples, gt_map_global)
        # cohen_kappa needs samples grouped by question.
        by_q: dict[str, list[SampleRow]] = {}
        for s in samples:
            by_q.setdefault(s.question_id, []).append(s)
        samples_by_model_by_q[model] = by_q
        cohen_kappa_per_model[model] = cohen_kappa(by_q, gt_map_global)
        hamming_per_model[model] = hamming_score(samples, gt_map_global)
        consistency_per_model[model] = build_consistency_report(
            samples, gt_map_global, options_map_global,
        )
        entropy_acc_per_model[model] = entropy_accuracy_bins(
            by_q, gt_map_global, options_map_global,
        )

    written: list[Path] = []
    # v5: per_model_summary.csv now carries v3 + FSS + Consistency + v4
    # probabilistic columns; markdown synthesises them with the K=5
    # disclaimer footnote.
    if summary_payload:
        written.append(_write_per_model_summary_csv(
            analysis_dir / "per_model_summary.csv",
            summary_payload,
            prob=prob_report.per_model,
            fss_per_model=fss_per_model,
            cohen_kappa_per_model=cohen_kappa_per_model,
            hamming_per_model=hamming_per_model,
            consistency_per_model=consistency_per_model,
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
    # v5 K-trial consistency outputs — always produced; rows with K=1
    # carry NULL aggregates but the row is present so the writer schema
    # stays uniform.
    # ------------------------------------------------------------------ #
    if consistency_per_model:
        written.append(_write_inter_trial_consistency_csv(
            analysis_dir / "inter_trial_consistency.csv",
            consistency_per_model,
        ))
    # Skip writing entropy_accuracy_bins.csv when no model produced any
    # bucket (e.g. all-K=1 run) — empty file is more confusing than absent.
    if any(bins for bins in entropy_acc_per_model.values()):
        written.append(_write_entropy_accuracy_bins_csv(
            analysis_dir / "entropy_accuracy_bins.csv",
            entropy_acc_per_model,
        ))

    # ------------------------------------------------------------------ #
    # v5 multi-metric paired bootstrap on FSS / Acc / MV_Acc / Fleiss κ /
    # EBI. Skips metrics that return None on the data (e.g. EBI on
    # v3-style fixtures, Fleiss κ on K=1).
    # ------------------------------------------------------------------ #
    if len(samples_by_model_by_q) >= 2:
        metric_results = pairwise_metric_bootstrap(
            samples_by_model_by_q, gt_map_global, DEFAULT_METRIC_FNS,
        )
        if metric_results:
            written.append(_write_metric_pairwise_bootstrap_csv(
                analysis_dir / "pairwise_bootstrap.csv",
                metric_results,
            ))

    # ------------------------------------------------------------------ #
    # v4 probabilistic deliverables — kept (Decision 3) as companion
    # outputs; calibration / reliability / Murphy decomposition removed
    # (Decision 2).
    # ------------------------------------------------------------------ #
    rows_by_model = prob_report.rows_by_model
    has_any_rows = any(rows for rows in rows_by_model.values())
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

        # Inference: pairwise paired bootstrap + Holm + posterior (BS-based,
        # kept for grid.py dependency and the v4 probabilistic narrative).
        from .probabilistic import _build_crowd_gammas_per_model

        crowd_gammas_by_model = _build_crowd_gammas_per_model(rows_by_model)
        uniform_gammas_global = _gamma_uniform_per_qid(rows_by_model)
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

    # ------------------------------------------------------------------ #
    # Phase 1 of `react-tavily-grid-search` — grid CSVs over (R, C) cells.
    # Skipped (returns []) when manifest has no `grid` segment so legacy
    # v4 single-cell runs stay byte-identical. Wrapped best-effort to
    # mirror the reflection A/B pattern: a Phase-1 failure here MUST NOT
    # mask the v4 main flow's outputs.
    # ------------------------------------------------------------------ #
    try:
        from .grid import run_grid_analysis

        grid_artifacts = run_grid_analysis(
            run_dir=run_dir,
            manifest=manifest,
            samples_by_model=samples_by_model,
            gt_map_global=gt_map_global,
            rows_by_model=rows_by_model,
            analysis_dir=analysis_dir,
        )
        written.extend(grid_artifacts)
    except Exception:  # pragma: no cover — grid analysis is best-effort
        import logging

        logging.getLogger(__name__).exception(
            "grid analysis failed; continuing without grid_*.csv outputs"
        )

    # `per_model_summary.md` is written last so it includes the v5 discrete
    # family (FSS / Cohen κ / Fleiss κ / mean entropy / VCI / MVG) alongside
    # the v3 accuracy columns and v4 companion probabilistic columns. The
    # `conflict*` marker (Phase 3 confidence A/B) survives — the v5
    # `cal*` marker is gone (calibration deprecated).
    if summary_payload:
        written.append(_write_per_model_summary_md(
            analysis_dir / "per_model_summary.md",
            summary_payload,
            prob=prob_report.per_model,
            fss_per_model=fss_per_model,
            cohen_kappa_per_model=cohen_kappa_per_model,
            consistency_per_model=consistency_per_model,
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
