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

Metric definitions (see FRAME.md §11 for full rationale):

    eligible_sample        : error != 'skipped_training_cutoff'
    resolvable_sample      : eligible_sample AND correct IS NOT NULL
    pass_at_1_avg          : mean(correct==1) over resolvable samples
    pass_any_at_n          : per-question, 1 if any sample correct else 0,
                             averaged over questions with >=1 resolvable sample
    at_least_majority_at_n : per-question, 1 if sum(correct)>=ceil(n/2) else 0
    at_least_all_at_n      : per-question, 1 if all samples are correct else 0
    majority_vote_accuracy : per-question, take the modal answer letter-set
                             across samples (break ties = unresolved), 1 if
                             equals GT. Averaged over questions where at least
                             one resolvable sample exists.
    parse_failure_rate     : parse_ok==0 AND error IS NULL, over eligible
    error_rate             : error IS NOT NULL AND != cutoff, over eligible
    cutoff_skip_rate       : error==cutoff, over ALL samples (incl. cutoff)
    avg_*                  : only over samples that actually ran
                             (error != cutoff)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from . import db as dbmod


CUTOFF = "skipped_training_cutoff"


@dataclass
class SampleRow:
    """Flattened per-sample view of one cell in the wide run_results table."""

    model: str
    question_id: str
    question_type: str
    choice_type: str
    sample_idx: int

    correct: int | None
    parse_ok: int | None
    tool_calls_count: int | None
    react_steps: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    latency_ms: int | None
    final_answer_letters: str | None
    error: str | None
    created_at: str | None

    @property
    def is_cutoff(self) -> bool:
        return self.error == CUTOFF

    @property
    def is_eligible(self) -> bool:
        return not self.is_cutoff

    @property
    def is_resolvable(self) -> bool:
        return self.is_eligible and self.correct is not None

    @property
    def parsed_letters(self) -> frozenset[str] | None:
        if not self.final_answer_letters:
            return None
        try:
            return frozenset(json.loads(self.final_answer_letters))
        except (TypeError, ValueError):
            return None


def _flatten_db(conn: sqlite3.Connection, sampling_n: int, model: str) -> list[SampleRow]:
    """Pivot the wide run_results table into per-sample rows joined with question metadata."""
    cols: list[str] = ["q.id", "q.question_type", "q.choice_type"]
    for i in range(sampling_n):
        for name in (
            "correct", "parse_ok",
            "tool_calls_count", "react_steps",
            "prompt_tokens", "completion_tokens", "reasoning_tokens",
            "latency_ms",
            "final_answer_letters", "error", "created_at",
        ):
            cols.append(f"r.{dbmod.sample_col(i, name)}")
    sql = (
        "SELECT " + ", ".join(cols) + " "
        "FROM questions q LEFT JOIN run_results r ON q.id = r.question_id"
    )
    samples: list[SampleRow] = []
    for row in conn.execute(sql):
        qid, qtype, ctype = row[0], row[1], row[2]
        offset = 3
        step = 11
        for i in range(sampling_n):
            base = offset + step * i
            created = row[base + 10]
            if created is None:
                # Sample slot is empty — no record written. Skip; we can still
                # judge pass_any_at_n with the other samples. Counting absent
                # slots as "error=unknown" would inflate error rates unfairly.
                continue
            samples.append(
                SampleRow(
                    model=model,
                    question_id=qid,
                    question_type=qtype,
                    choice_type=ctype,
                    sample_idx=i,
                    correct=row[base + 0],
                    parse_ok=row[base + 1],
                    tool_calls_count=row[base + 2],
                    react_steps=row[base + 3],
                    prompt_tokens=row[base + 4],
                    completion_tokens=row[base + 5],
                    reasoning_tokens=row[base + 6],
                    latency_ms=row[base + 7],
                    final_answer_letters=row[base + 8],
                    error=row[base + 9],
                    created_at=created,
                )
            )
    return samples


def _group_by_question(samples: list[SampleRow]) -> dict[str, list[SampleRow]]:
    out: dict[str, list[SampleRow]] = {}
    for s in samples:
        out.setdefault(s.question_id, []).append(s)
    return out


def _answer_gt_for(conn: sqlite3.Connection) -> dict[str, frozenset[str]]:
    """Map question_id -> GT letter frozenset using the question-local parser."""
    from .parser import parse_gt
    rows = conn.execute("SELECT id, answer FROM questions").fetchall()
    out: dict[str, frozenset[str]] = {}
    for r in rows:
        try:
            out[r["id"]] = parse_gt(r["answer"])
        except ValueError:
            out[r["id"]] = frozenset()
    return out


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

    # pass@1 avg (sample-level), over resolvable samples
    if resolvable_samples:
        pass_at_1 = sum(1 for s in resolvable_samples if s.correct == 1) / len(resolvable_samples)
    else:
        pass_at_1 = None

    resolvable_rate = (
        len(resolvable_samples) / len(eligible_samples) if eligible_samples else None
    )

    # Per-question correct counts, only among resolvable samples
    majority_threshold = math.ceil(sampling_n / 2)

    pass_any_hits: list[int] = []
    at_least_majority_hits: list[int] = []
    at_least_all_hits: list[int] = []
    for qid, rs in by_q_resolvable.items():
        n = len(rs)
        corrects = sum(1 for s in rs if s.correct == 1)
        pass_any_hits.append(1 if corrects >= 1 else 0)
        at_least_majority_hits.append(1 if corrects >= majority_threshold else 0)
        # at_least_all_at_n: question must have had N resolvable samples AND all correct
        at_least_all_hits.append(1 if (n == sampling_n and corrects == n) else 0)

    pass_any_at_n = _mean(pass_any_hits) if pass_any_hits else None
    at_least_majority_at_n = _mean(at_least_majority_hits) if at_least_majority_hits else None
    at_least_all_at_n = _mean(at_least_all_hits) if at_least_all_hits else None

    # Majority vote accuracy
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
                # tie -> unresolved
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

    # parse failure + error rates
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

    # Averages over samples that actually ran (eligible_samples)
    avg_tool_calls = _mean(s.tool_calls_count for s in eligible_samples)
    avg_react_steps = _mean(s.react_steps for s in eligible_samples)
    avg_latency = _mean(s.latency_ms for s in eligible_samples)
    avg_ptok = _mean(s.prompt_tokens for s in eligible_samples)
    avg_ctok = _mean(s.completion_tokens for s in eligible_samples)
    avg_rtok = _mean(s.reasoning_tokens for s in eligible_samples)

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


# ---------- Writers ----------

_SUMMARY_FIELDS = (
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
)


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
) -> Path:
    header = list(_SUMMARY_FIELDS)
    rows: list[list[Any]] = []
    for model, (sampling_n, agg) in per_model.items():
        row_dict = {"model": model, "sampling_n": sampling_n, **agg.as_ordered_dict()}
        rows.append([row_dict.get(k) for k in header])
    return _write_csv(path, header, rows)


def _write_slice_csv(
    path: Path,
    slice_header_field: str,
    per_model: dict[str, tuple[int, dict[str, Aggregate]]],
) -> Path:
    header = ["model", slice_header_field, "sampling_n", *[
        f for f in _SUMMARY_FIELDS if f not in ("model", "sampling_n")
    ]]
    rows: list[list[Any]] = []
    for model, (sampling_n, agg_map) in per_model.items():
        for key, agg in agg_map.items():
            row_dict = {
                "model": model,
                slice_header_field: key,
                "sampling_n": sampling_n,
                **agg.as_ordered_dict(),
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


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_per_model_summary_md(
    path: Path,
    per_model: dict[str, tuple[int, Aggregate]],
) -> Path:
    lines = ["# Per-model summary", ""]
    header = [
        "model", "N",
        "eligible_Q", "eligible_S", "cutoff_S",
        "pass@1", "pass_any@N", "≥majority", "≥all",
        "majority_acc", "parse_fail", "error_rate",
        "avg_tool", "avg_steps", "avg_latency_ms",
        "avg_p/c/r_tokens",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for model, (sampling_n, agg) in per_model.items():
        row_dict = agg.as_ordered_dict()
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
            _fmt(row_dict["avg_tool_calls"]),
            _fmt(row_dict["avg_react_steps"]),
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
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


# ---------- Public entry points ----------

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

    db_dir = run_dir / "db"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    per_model_agg: dict[str, Aggregate] = {}
    per_model_agg_qtype: dict[str, dict[str, Aggregate]] = {}
    per_model_agg_ctype: dict[str, dict[str, Aggregate]] = {}
    per_model_error: dict[str, tuple[int, Counter]] = {}
    sampling_n_by_model: dict[str, int] = {}

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
            agg = _aggregate(samples, sampling_n, gt_map=gt_map)
            agg_qtype = _slice_by(samples, lambda s: s.question_type, sampling_n, gt_map)
            agg_ctype = _slice_by(samples, lambda s: s.choice_type, sampling_n, gt_map)
            per_model_agg[model] = agg
            per_model_agg_qtype[model] = agg_qtype
            per_model_agg_ctype[model] = agg_ctype
            per_model_error[model] = (len(samples), _error_breakdown(samples))
            summary_payload[model] = (sampling_n, agg)
            slice_qtype_payload[model] = (sampling_n, agg_qtype)
            slice_ctype_payload[model] = (sampling_n, agg_ctype)
        finally:
            conn.close()

    written: list[Path] = []
    if summary_payload:
        written.append(_write_per_model_summary_csv(
            analysis_dir / "per_model_summary.csv", summary_payload,
        ))
        written.append(_write_per_model_summary_md(
            analysis_dir / "per_model_summary.md", summary_payload,
        ))
    if slice_qtype_payload:
        written.append(_write_slice_csv(
            analysis_dir / "per_model_by_question_type.csv",
            "question_type",
            slice_qtype_payload,
        ))
    if slice_ctype_payload:
        written.append(_write_slice_csv(
            analysis_dir / "per_model_by_choice_type.csv",
            "choice_type",
            slice_ctype_payload,
        ))
    if per_model_error:
        written.append(_write_error_breakdown_csv(
            analysis_dir / "error_breakdown.csv", per_model_error,
        ))
    written.append(_write_overall_json(
        analysis_dir / "overall.json",
        run_id=run_id,
        sampling_n_by_model=sampling_n_by_model,
        per_model=per_model_agg,
        per_model_by_qtype=per_model_agg_qtype,
        per_model_by_ctype=per_model_agg_ctype,
        error_breakdown=per_model_error,
    ))
    return written


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="forecast_eval.analysis",
        description="Compute statistics for one completed evaluation run.",
    )
    parser.add_argument("run_dir", help="Path to a RUNS_ROOT/{run_id} directory.")
    args = parser.parse_args(argv)
    paths = run_analysis(Path(args.run_dir))
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
