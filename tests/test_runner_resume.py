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


def _seed_question(conn, qid: str = "q1") -> None:
    conn.execute(
        "INSERT INTO questions (id, choice_type, question_type, event, options, answer, end_time, imported_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (qid, "single", "yes_no", "ev", json.dumps(["Yes", "No"]), "A", "2026-05-01", dbmod.utcnow_iso()),
    )


def _sample_payload(sample_idx: int, error: str | None) -> dict:
    return {
        "question_id": "q1",
        "sample_idx": sample_idx,
        "user_prompt": "RENDERED",
        "final_answer_letters": None,
        "final_answer_raw": None,
        "correct": None if error else 1,
        "parse_ok": 0 if error else 1,
        "tool_calls_count": 0,
        "react_steps": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "latency_ms": 0,
        "messages_trace": None,
        "search_calls": None,
        "error": error,
        "created_at": dbmod.utcnow_iso(),
        # v3 observability columns. error rows mirror runner._error_row:
        # finish_reason / response_id / system_fingerprint / service_tier
        # are None because the LLM never returned a usable envelope.
        "finish_reason": None if error else "stop",
        "nudges_used": 0,
        "step_metrics": None if error else json.dumps([]),
        "response_id": None,
        "system_fingerprint": None,
        "service_tier": None,
    }


def test_load_completed_samples_keeps_cutoff_out_of_retry(tmp_path: Path) -> None:
    conn = dbmod.connect(tmp_path / "r.db")
    dbmod.init_schema(conn, sampling_n=3)
    _seed_question(conn)

    # s0 = normal completion, s1 = cutoff, s2 = network (retryable)
    dbmod.upsert_sample_sync(conn, 3, _sample_payload(0, None))
    dbmod.upsert_sample_sync(conn, 3, _sample_payload(1, "skipped_training_cutoff"))
    dbmod.upsert_sample_sync(conn, 3, _sample_payload(2, "network"))

    completed = dbmod.load_completed_samples(conn, sampling_n=3)
    assert completed == {("q1", 0), ("q1", 1)}


def test_build_task_plan_drops_resumed_and_keeps_retries() -> None:
    settings = _StubSettings()
    questions = [_q("q1")]
    # Already-done set for model m/a: s0 normal completion, s1 cutoff
    completed = {"m/a": {("q1", 0), ("q1", 1)}}
    todo, cutoff_rows, stats = build_task_plan(
        questions=questions,
        settings=settings,
        completed=completed,
        run_id="run-1",
    )
    assert stats.total == 3
    assert stats.completed_preexisting == 2
    assert cutoff_rows == {"m/a": []}
    assert len(todo) == 1
    assert todo[0].sample_idx == 2  # the network-failed row comes back to the queue


def test_completed_set_is_per_model() -> None:
    settings = _StubSettings(MODELS=["m/a", "m/b"])
    questions = [_q("q1")]
    completed = {
        "m/a": {("q1", 0), ("q1", 1), ("q1", 2)},  # fully done on m/a
        "m/b": set(),                              # nothing done on m/b
    }
    todo, cutoff_rows, stats = build_task_plan(
        questions=questions,
        settings=settings,
        completed=completed,
        run_id="run-1",
    )
    # m/a contributes nothing; m/b contributes 3 tasks
    assert {t.model for t in todo} == {"m/b"}
    assert len(todo) == 3
    assert cutoff_rows == {"m/a": [], "m/b": []}


def test_missing_model_in_completed_falls_back_to_empty() -> None:
    settings = _StubSettings(MODELS=["m/a", "m/b"])
    questions = [_q("q1")]
    # completed only has m/a; m/b key absent — must default to empty set
    completed = {"m/a": {("q1", 0), ("q1", 1), ("q1", 2)}}
    todo, _, stats = build_task_plan(
        questions=questions,
        settings=settings,
        completed=completed,
        run_id="run-1",
    )
    assert len(todo) == 3
    assert all(t.model == "m/b" for t in todo)
