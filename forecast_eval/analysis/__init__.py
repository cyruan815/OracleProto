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
* `writers.py`     — CSV / Markdown / JSON serialisation.

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
from .flatten import (
    CUTOFF,
    SampleRow,
    _ANALYSIS_FIELDS,
    _answer_gt_for,
    _flatten_db,
)
from .probabilistic import build_probabilistic_report
from .writers import (
    _SUMMARY_FIELDS,
    _write_error_breakdown_csv,
    _write_finish_reason_breakdown_csv,
    _write_overall_json,
    _write_per_model_summary_csv,
    _write_per_model_summary_md,
    _write_slice_csv,
)


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
    if summary_payload:
        written.append(_write_per_model_summary_csv(
            analysis_dir / "per_model_summary.csv",
            summary_payload,
            prob=prob_report.per_model,
        ))
        written.append(_write_per_model_summary_md(
            analysis_dir / "per_model_summary.md",
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
