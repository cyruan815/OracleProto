from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pytest

from forecast_eval import db as dbmod
from forecast_eval.runner import build_task_plan
from forecast_eval.types import Question


def _q(qid: str, end_time: str, qt: str = "yes_no") -> Question:
    return Question(
        id=qid,
        choice_type="single",
        question_type=qt,
        event="ev",
        options=json.dumps(["Yes", "No"]),
        answer="A",
        end_time=end_time,
    )


@dataclass
class _StubSettings:
    MODELS: list[str] = field(default_factory=lambda: ["m/cutoff", "m/free"])
    SAMPLING_N: int = 2
    MODEL_TRAINING_CUTOFFS: dict[str, date] = field(
        default_factory=lambda: {"m/cutoff": date(2025, 3, 1)}
    )


def test_cutoff_skips_produce_rows_without_running_llm() -> None:
    settings = _StubSettings()
    questions = [
        _q("q-early", "2025-01-15"),   # before cutoff for m/cutoff
        _q("q-on", "2025-03-01"),      # equals cutoff -> still skipped (<=)
        _q("q-late", "2025-06-01"),    # after cutoff -> runs normally
    ]
    todo, cutoff_rows, stats = build_task_plan(
        questions=questions,
        settings=settings,
        completed=set(),
        run_id="run-1",
    )

    total_expected = len(questions) * len(settings.MODELS) * settings.SAMPLING_N
    assert stats.total == total_expected
    # m/cutoff gets skipped for q-early + q-on (2 questions × 2 samples = 4)
    assert stats.skipped_cutoff == 4
    # m/free runs all three questions, and m/cutoff runs only q-late
    assert stats.planned == (3 * 2) + (1 * 2)

    cutoff_keys = {(r["question_id"], r["model"], r["sample_idx"]) for r in cutoff_rows}
    expected_cutoff = {
        ("q-early", "m/cutoff", 0), ("q-early", "m/cutoff", 1),
        ("q-on", "m/cutoff", 0), ("q-on", "m/cutoff", 1),
    }
    assert cutoff_keys == expected_cutoff

    # Rows must carry the required signal fields
    for row in cutoff_rows:
        assert row["error"] == "skipped_training_cutoff"
        assert row["parse_ok"] == 0
        assert row["correct"] is None
        assert row["final_answer_letters"] is None
        assert row["messages_trace"] is None
        assert row["search_calls"] is None

    # todo items cover exactly the non-cutoff combinations
    todo_keys = {(t.question.id, t.model, t.sample_idx) for t in todo}
    expected_todo = set()
    for q in questions:
        for m in settings.MODELS:
            cutoff = settings.MODEL_TRAINING_CUTOFFS.get(m)
            hit = cutoff is not None and date.fromisoformat(q.end_time) <= cutoff
            if hit:
                continue
            for s in range(settings.SAMPLING_N):
                expected_todo.add((q.id, m, s))
    assert todo_keys == expected_todo


def test_model_without_cutoff_runs_every_question() -> None:
    settings = _StubSettings(MODELS=["m/free"], MODEL_TRAINING_CUTOFFS={})
    questions = [_q("q1", "2024-01-01"), _q("q2", "2026-01-01")]
    todo, cutoff_rows, stats = build_task_plan(
        questions=questions,
        settings=settings,
        completed=set(),
        run_id="run-1",
    )
    assert stats.skipped_cutoff == 0
    assert not cutoff_rows
    assert stats.planned == len(questions) * settings.SAMPLING_N


def test_resume_wins_over_cutoff() -> None:
    """An already-completed row must NOT be re-written as skipped_cutoff."""
    settings = _StubSettings()
    questions = [_q("q-early", "2025-01-15")]
    completed = {("q-early", "m/cutoff", 0)}
    todo, cutoff_rows, stats = build_task_plan(
        questions=questions,
        settings=settings,
        completed=completed,
        run_id="run-1",
    )
    cutoff_keys = {(r["question_id"], r["model"], r["sample_idx"]) for r in cutoff_rows}
    assert ("q-early", "m/cutoff", 0) not in cutoff_keys
    # The other sample_idx for the same (q, model) still gets skipped
    assert ("q-early", "m/cutoff", 1) in cutoff_keys


def test_cutoff_rows_persist_through_writer(tmp_path: Path) -> None:
    """Smoke: actually flush a cutoff row into results.db via _insert_rows_sync."""
    conn = dbmod.connect(tmp_path / "r.db")
    dbmod.init_schema(conn)
    dbmod.register_run(
        conn,
        run_id="run-1",
        filters_snapshot={},
        config_snapshot={},
        source_db_hash="a" * 64,
        metadata_hash="b" * 64,
        prompt_templates_hash="c" * 64,
    )
    conn.execute(
        "INSERT INTO questions (id, choice_type, question_type, event, options, answer, end_time, imported_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        ("q-early", "single", "yes_no", "ev", json.dumps(["Yes", "No"]), "A", "2025-01-15", dbmod.utcnow_iso()),
    )

    settings = _StubSettings()
    _, cutoff_rows, _ = build_task_plan(
        questions=[_q("q-early", "2025-01-15")],
        settings=settings,
        completed=set(),
        run_id="run-1",
    )
    dbmod._insert_rows_sync(conn, cutoff_rows)
    completed = dbmod.load_completed(conn, "run-1")
    assert completed == {(row["question_id"], row["model"], row["sample_idx"]) for row in cutoff_rows}
