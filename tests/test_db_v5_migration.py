"""v5 harness-resilience schema migration tests.

Covers:
- v4 → v5 ALTER TABLE chain on a synthetic v4-shaped DB
- v5 DB idempotency (no re-INSERT, no re-ALTER on second open)
- v3 → v5 chained migration (both v3→v4 and v4→v5 stamps land)

The v4 fixture is hand-written here (not via `dbmod` helpers) so the test
stays frozen against the historical v4 column suite even as
`PER_SAMPLE_COLUMNS` keeps evolving.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from forecast_eval import db as dbmod


# Mirrors the historical v4 column suite — keep this immutable here even as
# `PER_SAMPLE_COLUMNS` evolves further. The v4→v5 migration code is responsible
# for adding the v5 column on top of this snapshot.
_V4_PER_SAMPLE_COLUMNS = (
    ("final_answer_letters", "TEXT"),
    ("final_answer_raw", "TEXT"),
    ("correct", "INTEGER"),
    ("parse_ok", "INTEGER"),
    ("tool_calls_count", "INTEGER"),
    ("react_steps", "INTEGER"),
    ("prompt_tokens", "INTEGER"),
    ("completion_tokens", "INTEGER"),
    ("reasoning_tokens", "INTEGER"),
    ("latency_ms", "INTEGER"),
    ("messages_trace", "TEXT"),
    ("search_calls", "TEXT"),
    ("error", "TEXT"),
    ("created_at", "TEXT"),
    ("finish_reason", "TEXT"),
    ("nudges_used", "INTEGER"),
    ("step_metrics", "TEXT"),
    ("response_id", "TEXT"),
    ("system_fingerprint", "TEXT"),
    ("service_tier", "TEXT"),
    ("belief_final", "TEXT"),
    ("belief_trace", "TEXT"),
    ("belief_parse_ok", "INTEGER"),
)


def _build_v4_db(db_path: Path, sampling_n: int) -> sqlite3.Connection:
    """Construct a v4-schema DB by hand: `run_results` lacks the v5
    `final_answer_retry_used` column; `schema_version` is stamped with `4`."""
    conn = dbmod.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
        CREATE TABLE questions (
            id TEXT PRIMARY KEY,
            choice_type TEXT NOT NULL CHECK (choice_type IN ('single','multi')),
            question_type TEXT NOT NULL CHECK (question_type IN ('yes_no','binary_named','multiple_choice')),
            event TEXT NOT NULL,
            options TEXT NOT NULL,
            answer TEXT NOT NULL,
            end_time TEXT NOT NULL,
            imported_at TEXT NOT NULL
        );
        CREATE TABLE prompt_templates (key TEXT PRIMARY KEY, value TEXT NOT NULL, imported_at TEXT NOT NULL);
        CREATE TABLE run_meta (
            run_id                    TEXT PRIMARY KEY,
            model                     TEXT NOT NULL,
            sampling_n                INTEGER NOT NULL,
            config_snapshot           TEXT NOT NULL,
            filters_snapshot          TEXT NOT NULL,
            source_db_hash            TEXT NOT NULL,
            metadata_hash             TEXT NOT NULL,
            prompt_templates_hash     TEXT NOT NULL,
            training_cutoff           TEXT,
            started_at                TEXT NOT NULL,
            finished_at               TEXT,
            reflection_protocol_text  TEXT,
            reflection_protocol_hash  TEXT,
            belief_protocol_text      TEXT,
            belief_protocol_hash      TEXT
        );
        """
    )
    cols = ["question_id TEXT PRIMARY KEY", "user_prompt TEXT"]
    for i in range(sampling_n):
        for name, sql_type in _V4_PER_SAMPLE_COLUMNS:
            cols.append(f"s{i}_{name} {sql_type}")
    cols.append("FOREIGN KEY (question_id) REFERENCES questions(id)")
    conn.execute(f"CREATE TABLE run_results ({', '.join(cols)})")
    conn.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
        (4, dbmod.utcnow_iso()),
    )
    return conn


def _seed_question(conn: sqlite3.Connection, qid: str = "q42") -> None:
    import json

    conn.execute(
        "INSERT INTO questions (id, choice_type, question_type, event, options, answer, end_time, imported_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (qid, "single", "yes_no", "ev", json.dumps(["Yes", "No"]), "A", "2026-01-01", dbmod.utcnow_iso()),
    )


def test_v4_to_v5_adds_column(tmp_path: Path) -> None:
    """An existing v4 DB MUST be ALTERed in place: one new
    `s{i}_final_answer_retry_used` column per sample. Pre-existing data
    survives; new column starts NULL."""
    db_path = tmp_path / "v4.db"
    conn = _build_v4_db(db_path, sampling_n=3)
    _seed_question(conn)
    legacy_created_at = dbmod.utcnow_iso()
    conn.execute(
        "INSERT INTO run_results (question_id, user_prompt, s0_correct, s0_created_at, s0_belief_parse_ok) "
        "VALUES (?, ?, ?, ?, ?)",
        ("q42", "PROMPT", 1, legacy_created_at, 1),
    )

    dbmod.init_schema(conn, sampling_n=3)

    versions = {
        int(r["version"])
        for r in conn.execute("SELECT version FROM schema_version").fetchall()
    }
    assert versions == {4, 5}, "v4 → v5 migration must keep v4 row and stamp v5"

    cols = {r["name"] for r in conn.execute("PRAGMA table_info(run_results)").fetchall()}
    for i in range(3):
        assert f"s{i}_final_answer_retry_used" in cols, (
            f"missing s{i}_final_answer_retry_used after v4→v5 migration"
        )

    # Pre-existing v4 data survives; new column defaults to NULL.
    row = conn.execute("SELECT * FROM run_results WHERE question_id='q42'").fetchone()
    assert row["s0_correct"] == 1
    assert row["s0_created_at"] == legacy_created_at
    assert row["s0_belief_parse_ok"] == 1
    assert row["s0_final_answer_retry_used"] is None


def test_v5_idempotent(tmp_path: Path) -> None:
    """A brand-new v5 DB stamped with `version=5` MUST NOT re-INSERT or re-ALTER
    on subsequent `init_schema` calls."""
    conn = dbmod.connect(tmp_path / "v5.db")
    dbmod.init_schema(conn, sampling_n=2)

    versions_first = {
        int(r["version"])
        for r in conn.execute("SELECT version FROM schema_version").fetchall()
    }
    assert versions_first == {5}

    cols_first = [
        r["name"] for r in conn.execute("PRAGMA table_info(run_results)").fetchall()
    ]

    dbmod.init_schema(conn, sampling_n=2)
    versions_second = {
        int(r["version"])
        for r in conn.execute("SELECT version FROM schema_version").fetchall()
    }
    assert versions_second == {5}, "second init_schema must not re-INSERT version=5"
    cols_second = [
        r["name"] for r in conn.execute("PRAGMA table_info(run_results)").fetchall()
    ]
    assert cols_first == cols_second, "second init_schema must not re-ALTER"


def test_v3_to_v5_chained(tmp_path: Path) -> None:
    """A v3 DB jumped through v4 to land at v5 — both stamps must be recorded.
    We use the existing v3 builder from test_db_v4_migration to keep the
    historical schema definition in one place."""
    from tests.test_db_v4_migration import _build_v3_db, _seed_question as _seed_q3

    db_path = tmp_path / "v3.db"
    conn = _build_v3_db(db_path, sampling_n=2)
    _seed_q3(conn)

    dbmod.init_schema(conn, sampling_n=2)

    versions = {
        int(r["version"])
        for r in conn.execute("SELECT version FROM schema_version").fetchall()
    }
    assert versions == {3, 4, 5}, (
        "v3→v5 chained migration must stamp 4 (transitive) and 5"
    )

    cols = {r["name"] for r in conn.execute("PRAGMA table_info(run_results)").fetchall()}
    # Both v4 belief columns and v5 retry column should be present.
    for i in range(2):
        assert f"s{i}_belief_final" in cols
        assert f"s{i}_final_answer_retry_used" in cols


