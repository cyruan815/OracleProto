from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pytest

from forecast_eval import db as dbmod
from forecast_eval.runner import build_task_plan
from forecast_eval.types import Question


def _q(qid: str, end_time: str = "2026-05-01") -> Question:
    return Question(
        id=qid,
        choice_type="single",
        question_type="yes_no",
        event="ev",
        options=json.dumps(["Yes", "No"]),
        answer="A",
        end_time=end_time,
    )


@dataclass
class _StubSettings:
    MODELS: list[str] = field(default_factory=lambda: ["m/a"])
    SAMPLING_N: int = 3
    MODEL_TRAINING_CUTOFFS: dict[str, date] = field(default_factory=dict)


def _seed_run(conn, run_id: str, rows: list[dict]) -> None:
    dbmod.register_run(
        conn,
        run_id=run_id,
        filters_snapshot={},
        config_snapshot={},
        source_db_hash="a" * 64,
        metadata_hash="b" * 64,
        prompt_templates_hash="c" * 64,
    )
    dbmod._insert_rows_sync(conn, rows)


def _row(run_id: str, qid: str, model: str, sample_idx: int, error: str | None) -> dict:
    return {
        "run_id": run_id,
        "question_id": qid,
        "model": model,
        "sample_idx": sample_idx,
        "final_answer_letters": None,
        "final_answer_raw": None,
        "correct": None,
        "parse_ok": 0,
        "tool_calls_count": 0,
        "react_steps": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "latency_ms": 0,
        "user_prompt": None,
        "messages_trace": None,
        "search_calls": None,
        "error": error,
        "created_at": dbmod.utcnow_iso(),
    }


def test_resume_skips_completed_and_cutoff_rows_only(tmp_path: Path) -> None:
    conn = dbmod.connect(tmp_path / "r.db")
    dbmod.init_schema(conn)
    conn.execute(
        "INSERT INTO questions (id, choice_type, question_type, event, options, answer, end_time, imported_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        ("q1", "single", "yes_no", "ev", json.dumps(["Yes", "No"]), "A", "2026-05-01", dbmod.utcnow_iso()),
    )

    run_id = "20260424-120000-a000"
    rows = [
        _row(run_id, "q1", "m/a", 0, None),                           # completed
        _row(run_id, "q1", "m/a", 1, "skipped_training_cutoff"),     # skipped cutoff
        _row(run_id, "q1", "m/a", 2, "network"),                      # retryable
    ]
    _seed_run(conn, run_id, rows)

    completed = dbmod.load_completed(conn, run_id)
    assert completed == {("q1", "m/a", 0), ("q1", "m/a", 1)}

    settings = _StubSettings()
    todo, cutoff_rows, stats = build_task_plan(
        questions=[_q("q1")],
        settings=settings,
        completed=completed,
        run_id=run_id,
    )
    assert stats.total == 3
    assert stats.completed_preexisting == 2
    assert cutoff_rows == []
    assert len(todo) == 1
    assert todo[0].sample_idx == 2  # the network-failed row comes back to the queue


def test_resume_scoped_to_run_id(tmp_path: Path) -> None:
    conn = dbmod.connect(tmp_path / "r.db")
    dbmod.init_schema(conn)
    conn.execute(
        "INSERT INTO questions (id, choice_type, question_type, event, options, answer, end_time, imported_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        ("q1", "single", "yes_no", "ev", json.dumps(["Yes", "No"]), "A", "2026-05-01", dbmod.utcnow_iso()),
    )
    old_run = "20260424-110000-a000"
    new_run = "20260424-120000-a001"
    _seed_run(conn, old_run, [_row(old_run, "q1", "m/a", i, None) for i in range(3)])
    dbmod.register_run(
        conn,
        run_id=new_run,
        filters_snapshot={},
        config_snapshot={},
        source_db_hash="a" * 64,
        metadata_hash="b" * 64,
        prompt_templates_hash="c" * 64,
    )

    completed_new = dbmod.load_completed(conn, new_run)
    assert completed_new == set(), "a fresh run_id must not inherit completion state"


def test_register_run_preserves_finished_at_on_resume(tmp_path: Path) -> None:
    conn = dbmod.connect(tmp_path / "r.db")
    dbmod.init_schema(conn)
    run_id = "20260424-120000-a000"
    dbmod.register_run(
        conn,
        run_id=run_id,
        filters_snapshot={},
        config_snapshot={},
        source_db_hash="a" * 64,
        metadata_hash="b" * 64,
        prompt_templates_hash="c" * 64,
    )
    dbmod.finish_run(conn, run_id, "2026-04-24T12:00:00+00:00")

    dbmod.register_run(
        conn,
        run_id=run_id,
        filters_snapshot={"new": 1},
        config_snapshot={"new_key": "v"},
        source_db_hash="a" * 64,
        metadata_hash="b" * 64,
        prompt_templates_hash="c" * 64,
    )
    row = conn.execute(
        "SELECT finished_at, filters_snapshot FROM runs WHERE run_id=?",
        (run_id,),
    ).fetchone()
    assert row["finished_at"] == "2026-04-24T12:00:00+00:00"
    # config/filter updates should still be applied
    assert json.loads(row["filters_snapshot"]) == {"new": 1}
