from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from forecast_eval import db as dbmod
from forecast_eval.config import Settings


def _make_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("LLM_API_KEY", "sk-or-v1-ABCDEFGHIJKLMNOP0123")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-ABCDEFGH01234567")
    monkeypatch.setenv("MODELS", "openai/gpt-5,anthropic/claude-sonnet-4.5")
    monkeypatch.setenv(
        "MODEL_TRAINING_CUTOFFS",
        "openai/gpt-5=2024-10-01,anthropic/claude-sonnet-4.5=2025-03-01",
    )
    monkeypatch.setenv("RESULTS_DB", str(tmp_path / "results.db"))
    monkeypatch.setenv("SOURCE_DB", str(tmp_path / "forecast_eval_set.db"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    return Settings(_env_file=None)


def test_schema_and_pragmas(tmp_path: Path) -> None:
    conn = dbmod.connect(tmp_path / "r.db")
    dbmod.init_schema(conn)

    journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
    fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    sync = conn.execute("PRAGMA synchronous").fetchone()[0]
    assert journal.lower() == "wal"
    assert fk == 1
    assert sync in (1, 2)

    tables = {
        r["name"]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    }
    assert {"questions", "prompt_templates", "runs", "schema_version", "run_results"}.issubset(tables)

    version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
    assert version == dbmod.SCHEMA_VERSION

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO questions (id, choice_type, question_type, event, options, answer, end_time, imported_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("qid1", "triple", "yes_no", "e", "[]", "A", "2026-01-01", dbmod.utcnow_iso()),
        )


def test_hashes_are_stable(tmp_path: Path) -> None:
    src = tmp_path / "source.db"
    src.write_bytes(b"\x00" * 128)
    h1 = dbmod.compute_source_db_hash(src)
    h2 = dbmod.compute_source_db_hash(src)
    assert h1 == h2

    features = {"a": 1, "b": {"c": [2, 3]}}
    assert dbmod.compute_metadata_hash(json.dumps(features)) == dbmod.compute_metadata_hash(features)

    tmpl = {"agent_role": "You are an agent.", "guidance": "Answer clearly."}
    assert dbmod.compute_prompt_templates_hash(tmpl) == dbmod.compute_prompt_templates_hash(dict(tmpl))


def test_hash_changes_when_content_changes(tmp_path: Path) -> None:
    tmpl1 = {"agent_role": "You are an agent."}
    tmpl2 = {"agent_role": "You are a different agent."}
    assert dbmod.compute_prompt_templates_hash(tmpl1) != dbmod.compute_prompt_templates_hash(tmpl2)


def test_redact_api_key_hides_plaintext() -> None:
    raw = "sk-or-v1-ABCDEFGHIJKL"
    red = dbmod.redact_api_key(raw, "llm")
    assert red["provider"] == "llm"
    assert red["prefix"] == "sk-o"
    assert red["length"] == len(raw)
    assert len(red["sha256_12"]) == 12
    serialised = json.dumps(red)
    assert raw not in serialised
    assert "ABCDEFG" not in serialised


def test_snapshot_settings_redacts_all_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    s = _make_settings(tmp_path, monkeypatch)
    snap = dbmod.snapshot_settings(s)
    blob = json.dumps(snap)
    assert s.LLM_API_KEY not in blob
    assert s.TAVILY_API_KEY not in blob
    assert snap["LLM_API_KEY"]["provider"] == "llm"
    assert snap["TAVILY_API_KEY"]["provider"] == "tavily"
    assert snap["MODEL_TRAINING_CUTOFFS"]["openai/gpt-5"] == "2024-10-01"
    # repr must also be safe for logs
    assert s.LLM_API_KEY not in repr(s)
    assert s.TAVILY_API_KEY not in repr(s)


def test_register_run_and_load_completed(tmp_path: Path) -> None:
    conn = dbmod.connect(tmp_path / "r.db")
    dbmod.init_schema(conn)

    dbmod.register_run(
        conn,
        run_id="20260424-120000-abcd",
        filters_snapshot={"question_types": None, "choice_types": None, "question_count": 2},
        config_snapshot={"MODELS": ["m1"]},
        source_db_hash="a" * 64,
        metadata_hash="b" * 64,
        prompt_templates_hash="c" * 64,
    )
    conn.execute(
        "INSERT INTO questions (id, choice_type, question_type, event, options, answer, end_time, imported_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        ("q1", "single", "yes_no", "ev", "[\"Yes\",\"No\"]", "A", "2026-01-01", dbmod.utcnow_iso()),
    )

    def _row(run_id: str, sample_idx: int, error: str | None) -> dict:
        return {
            "run_id": run_id,
            "question_id": "q1",
            "model": "m1",
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

    run_id = "20260424-120000-abcd"
    rows = [
        _row(run_id, 0, None),
        _row(run_id, 1, "skipped_training_cutoff"),
        _row(run_id, 2, "network"),
    ]
    dbmod._insert_rows_sync(conn, rows)

    completed = dbmod.load_completed(conn, run_id)
    assert completed == {("q1", "m1", 0), ("q1", "m1", 1)}
    # the other run_id is independent
    assert dbmod.load_completed(conn, "other-run") == set()


def test_insert_or_replace_overrides_existing(tmp_path: Path) -> None:
    conn = dbmod.connect(tmp_path / "r.db")
    dbmod.init_schema(conn)
    dbmod.register_run(
        conn,
        run_id="20260424-120000-abcd",
        filters_snapshot={},
        config_snapshot={},
        source_db_hash="a" * 64,
        metadata_hash="b" * 64,
        prompt_templates_hash="c" * 64,
    )
    conn.execute(
        "INSERT INTO questions (id, choice_type, question_type, event, options, answer, end_time, imported_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        ("q1", "single", "yes_no", "ev", "[\"Yes\",\"No\"]", "A", "2026-01-01", dbmod.utcnow_iso()),
    )

    base = {
        "run_id": "20260424-120000-abcd",
        "question_id": "q1",
        "model": "m1",
        "sample_idx": 0,
        "final_answer_letters": None,
        "final_answer_raw": "first",
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
        "error": "content_policy",
        "created_at": dbmod.utcnow_iso(),
    }
    dbmod._insert_rows_sync(conn, [base])
    updated = {**base, "final_answer_raw": "second", "error": None, "parse_ok": 1, "correct": 1}
    dbmod._insert_rows_sync(conn, [updated])

    row = conn.execute(
        "SELECT final_answer_raw, error, parse_ok, correct FROM run_results WHERE run_id=? AND question_id=? AND model=? AND sample_idx=?",
        ("20260424-120000-abcd", "q1", "m1", 0),
    ).fetchone()
    assert row["final_answer_raw"] == "second"
    assert row["error"] is None
    assert row["parse_ok"] == 1
    assert row["correct"] == 1


async def test_async_writer_batches_commits(tmp_path: Path) -> None:
    conn = dbmod.connect(tmp_path / "r.db")
    dbmod.init_schema(conn)
    dbmod.register_run(
        conn,
        run_id="20260424-120000-abcd",
        filters_snapshot={},
        config_snapshot={},
        source_db_hash="a" * 64,
        metadata_hash="b" * 64,
        prompt_templates_hash="c" * 64,
    )
    conn.execute(
        "INSERT INTO questions (id, choice_type, question_type, event, options, answer, end_time, imported_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        ("q1", "single", "yes_no", "ev", "[\"Yes\",\"No\"]", "A", "2026-01-01", dbmod.utcnow_iso()),
    )

    writer = dbmod.AsyncWriter(conn, batch=5)
    await writer.start()
    try:
        for i in range(7):
            await writer.enqueue_result(
                {
                    "run_id": "20260424-120000-abcd",
                    "question_id": "q1",
                    "model": "m1",
                    "sample_idx": i,
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
                    "error": None,
                    "created_at": dbmod.utcnow_iso(),
                }
            )
        await writer.drain()
    finally:
        await writer.close()

    count = conn.execute("SELECT COUNT(*) FROM run_results").fetchone()[0]
    assert count == 7


def test_finish_run_updates_timestamp(tmp_path: Path) -> None:
    conn = dbmod.connect(tmp_path / "r.db")
    dbmod.init_schema(conn)
    dbmod.register_run(
        conn,
        run_id="20260424-120000-abcd",
        filters_snapshot={},
        config_snapshot={},
        source_db_hash="a" * 64,
        metadata_hash="b" * 64,
        prompt_templates_hash="c" * 64,
    )
    dbmod.finish_run(conn, "20260424-120000-abcd")
    row = conn.execute("SELECT finished_at FROM runs WHERE run_id=?", ("20260424-120000-abcd",)).fetchone()
    assert row["finished_at"] is not None
