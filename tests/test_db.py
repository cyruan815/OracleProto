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
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("SOURCE_DB", str(tmp_path / "forecast_eval_set.db"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    return Settings(_env_file=None)


def _sample_payload(
    *,
    question_id: str = "q1",
    sample_idx: int = 0,
    user_prompt: str | None = "RENDERED",
    correct: int | None = 1,
    parse_ok: int = 1,
    error: str | None = None,
    final_answer_raw: str | None = "answer",
) -> dict:
    return {
        "question_id": question_id,
        "sample_idx": sample_idx,
        "user_prompt": user_prompt,
        "final_answer_letters": json.dumps(["A"]) if correct is not None else None,
        "final_answer_raw": final_answer_raw,
        "correct": correct,
        "parse_ok": parse_ok,
        "tool_calls_count": 3,
        "react_steps": 2,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "reasoning_tokens": 0,
        "latency_ms": 1234,
        "messages_trace": None,
        "search_calls": json.dumps([]),
        "error": error,
        "created_at": dbmod.utcnow_iso(),
    }


def _seed_question(conn: sqlite3.Connection, qid: str = "q1") -> None:
    conn.execute(
        "INSERT INTO questions (id, choice_type, question_type, event, options, answer, end_time, imported_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (qid, "single", "yes_no", "ev", json.dumps(["Yes", "No"]), "A", "2026-01-01", dbmod.utcnow_iso()),
    )


def test_schema_and_pragmas(tmp_path: Path) -> None:
    conn = dbmod.connect(tmp_path / "r.db")
    dbmod.init_schema(conn, sampling_n=3)

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
    assert {"questions", "prompt_templates", "run_meta", "schema_version", "run_results"}.issubset(tables)
    assert "runs" not in tables, "old 'runs' table is gone — run_meta replaces it"

    version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
    assert version == dbmod.SCHEMA_VERSION

    # Dynamic per-sample columns exist (s0_, s1_, s2_)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(run_results)").fetchall()}
    for i in range(3):
        assert f"s{i}_correct" in cols
        assert f"s{i}_final_answer_letters" in cols
        assert f"s{i}_created_at" in cols
    assert "s3_correct" not in cols

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO questions (id, choice_type, question_type, event, options, answer, end_time, imported_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("qid1", "triple", "yes_no", "e", "[]", "A", "2026-01-01", dbmod.utcnow_iso()),
        )


def test_init_schema_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "r.db"
    conn = dbmod.connect(db_path)
    dbmod.init_schema(conn, sampling_n=2)
    dbmod.init_schema(conn, sampling_n=2)  # second call must not raise
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(run_results)").fetchall()}
    assert "s0_correct" in cols and "s1_correct" in cols


def test_init_schema_rejects_n_mismatch(tmp_path: Path) -> None:
    """If a DB was created with N=2, re-opening with N=3 must fail fast."""
    db_path = tmp_path / "r.db"
    conn = dbmod.connect(db_path)
    dbmod.init_schema(conn, sampling_n=2)
    with pytest.raises(ValueError, match="missing columns"):
        dbmod.init_schema(conn, sampling_n=3)


def test_model_slug_safe() -> None:
    assert dbmod.model_slug_safe("openai/gpt-4o-mini") == "openai__gpt-4o-mini"
    assert dbmod.model_slug_safe("anthropic/claude-sonnet-4.5") == "anthropic__claude-sonnet-4.5"
    assert dbmod.model_slug_safe("qwen3.6-plus-2026-04-02") == "qwen3.6-plus-2026-04-02"
    # Exotic punctuation is replaced with underscore
    assert dbmod.model_slug_safe("foo:bar/baz qux") == "foo_bar__baz_qux"
    with pytest.raises(ValueError):
        dbmod.model_slug_safe("")


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


def test_hash_changes_when_content_changes() -> None:
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
    assert s.LLM_API_KEY not in repr(s)
    assert s.TAVILY_API_KEY not in repr(s)


def test_register_run_meta_and_load_completed(tmp_path: Path) -> None:
    conn = dbmod.connect(tmp_path / "r.db")
    dbmod.init_schema(conn, sampling_n=3)
    dbmod.register_run_meta(
        conn,
        run_id="20260424-120000-abcd",
        model="openai/gpt-5",
        sampling_n=3,
        filters_snapshot={"question_types": None, "choice_types": None, "question_count": 1},
        config_snapshot={"MODELS": ["openai/gpt-5"]},
        source_db_hash="a" * 64,
        metadata_hash="b" * 64,
        prompt_templates_hash="c" * 64,
        training_cutoff="2024-10-01",
    )
    _seed_question(conn)

    # Sample 0: normal completion; sample 1: cutoff; sample 2: network (retryable)
    dbmod.upsert_sample_sync(conn, 3, _sample_payload(sample_idx=0))
    dbmod.upsert_sample_sync(conn, 3, _sample_payload(
        sample_idx=1, correct=None, parse_ok=0, error="skipped_training_cutoff",
    ))
    dbmod.upsert_sample_sync(conn, 3, _sample_payload(
        sample_idx=2, correct=None, parse_ok=0, error="network",
    ))

    done = dbmod.load_completed_samples(conn, sampling_n=3)
    assert done == {("q1", 0), ("q1", 1)}


def test_upsert_overrides_existing_sample(tmp_path: Path) -> None:
    conn = dbmod.connect(tmp_path / "r.db")
    dbmod.init_schema(conn, sampling_n=2)
    _seed_question(conn)
    dbmod.upsert_sample_sync(conn, 2, _sample_payload(
        sample_idx=0, correct=None, parse_ok=0, error="content_policy", final_answer_raw="first",
    ))
    dbmod.upsert_sample_sync(conn, 2, _sample_payload(
        sample_idx=0, correct=1, parse_ok=1, error=None, final_answer_raw="second",
    ))
    row = conn.execute("SELECT * FROM run_results WHERE question_id='q1'").fetchone()
    assert row["s0_final_answer_raw"] == "second"
    assert row["s0_error"] is None
    assert row["s0_parse_ok"] == 1
    assert row["s0_correct"] == 1


def test_upsert_writes_sibling_samples_independently(tmp_path: Path) -> None:
    conn = dbmod.connect(tmp_path / "r.db")
    dbmod.init_schema(conn, sampling_n=3)
    _seed_question(conn)

    # Write sample 0 then sample 2; sample 1 should remain NULL.
    dbmod.upsert_sample_sync(conn, 3, _sample_payload(sample_idx=0, correct=1))
    dbmod.upsert_sample_sync(conn, 3, _sample_payload(sample_idx=2, correct=0, final_answer_raw="diff"))
    row = conn.execute("SELECT * FROM run_results WHERE question_id='q1'").fetchone()
    assert row["s0_correct"] == 1
    assert row["s1_correct"] is None
    assert row["s1_created_at"] is None
    assert row["s2_correct"] == 0
    # user_prompt written on first sample — COALESCE preserved on second
    assert row["user_prompt"] == "RENDERED"


async def test_async_writer_groups_by_sample_idx(tmp_path: Path) -> None:
    conn = dbmod.connect(tmp_path / "r.db")
    dbmod.init_schema(conn, sampling_n=3)
    dbmod.register_run_meta(
        conn,
        run_id="20260424-120000-abcd",
        model="m1",
        sampling_n=3,
        filters_snapshot={},
        config_snapshot={},
        source_db_hash="a" * 64,
        metadata_hash="b" * 64,
        prompt_templates_hash="c" * 64,
    )
    # Two questions so we exercise multi-row writes.
    _seed_question(conn, "q1")
    _seed_question(conn, "q2")

    writer = dbmod.AsyncWriter(conn, sampling_n=3, batch=4)
    await writer.start()
    try:
        # 2 questions × 3 samples = 6 writes, order shuffled
        for qid in ("q1", "q2"):
            for s in (1, 0, 2):
                await writer.enqueue_result(
                    _sample_payload(question_id=qid, sample_idx=s, correct=s % 2)
                )
        await writer.drain()
    finally:
        await writer.close()

    rows = conn.execute(
        "SELECT question_id, s0_correct, s1_correct, s2_correct FROM run_results ORDER BY question_id"
    ).fetchall()
    assert [r["question_id"] for r in rows] == ["q1", "q2"]
    for r in rows:
        assert r["s0_correct"] == 0
        assert r["s1_correct"] == 1
        assert r["s2_correct"] == 0


async def test_async_writer_rejects_missing_keys(tmp_path: Path) -> None:
    conn = dbmod.connect(tmp_path / "r.db")
    dbmod.init_schema(conn, sampling_n=1)
    writer = dbmod.AsyncWriter(conn, sampling_n=1, batch=1)
    with pytest.raises(ValueError):
        await writer.enqueue_result({"question_id": "q1"})  # sample_idx missing


def test_finish_run_meta_updates_timestamp(tmp_path: Path) -> None:
    conn = dbmod.connect(tmp_path / "r.db")
    dbmod.init_schema(conn, sampling_n=1)
    dbmod.register_run_meta(
        conn,
        run_id="20260424-120000-abcd",
        model="m1",
        sampling_n=1,
        filters_snapshot={},
        config_snapshot={},
        source_db_hash="a" * 64,
        metadata_hash="b" * 64,
        prompt_templates_hash="c" * 64,
    )
    dbmod.finish_run_meta(conn, "20260424-120000-abcd")
    row = conn.execute("SELECT finished_at FROM run_meta WHERE run_id=?", ("20260424-120000-abcd",)).fetchone()
    assert row["finished_at"] is not None


def test_register_run_meta_preserves_started_at_on_resume(tmp_path: Path) -> None:
    conn = dbmod.connect(tmp_path / "r.db")
    dbmod.init_schema(conn, sampling_n=1)
    run_id = "20260424-120000-abcd"
    dbmod.register_run_meta(
        conn,
        run_id=run_id,
        model="m1",
        sampling_n=1,
        filters_snapshot={},
        config_snapshot={},
        source_db_hash="a" * 64,
        metadata_hash="b" * 64,
        prompt_templates_hash="c" * 64,
        started_at="2026-04-24T12:00:00+00:00",
    )
    dbmod.finish_run_meta(conn, run_id, finished_at="2026-04-24T13:00:00+00:00")
    # Second register (= resume) should preserve original started_at and finished_at
    dbmod.register_run_meta(
        conn,
        run_id=run_id,
        model="m1",
        sampling_n=1,
        filters_snapshot={"new": 1},
        config_snapshot={"new_key": "v"},
        source_db_hash="a" * 64,
        metadata_hash="b" * 64,
        prompt_templates_hash="c" * 64,
    )
    row = conn.execute(
        "SELECT started_at, finished_at, filters_snapshot FROM run_meta WHERE run_id=?",
        (run_id,),
    ).fetchone()
    assert row["started_at"] == "2026-04-24T12:00:00+00:00"
    assert row["finished_at"] == "2026-04-24T13:00:00+00:00"
    assert json.loads(row["filters_snapshot"]) == {"new": 1}
