"""v4 belief schema migration tests.

Covers:
- v3 → v4 ALTER TABLE chain on a synthetic v3-shaped DB
- v4 DB idempotency (no re-INSERT, no re-ALTER on second open)
- `register_run_meta` writes `belief_protocol_text` / `belief_protocol_hash`
  on both nullable paths (enabled / disabled)
- `_assert_run_results_matches` rejects sampling_n mismatch on a v4 DB

The v3 fixture is hand-written here (not via `dbmod` helpers) so the test
stays frozen against the historical v3 column suite even if the canonical
`PER_SAMPLE_COLUMNS` keeps evolving.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from forecast_eval import db as dbmod


# Mirrors the historical v3 column suite — keep this immutable here even as
# `PER_SAMPLE_COLUMNS` evolves further. The migration code is responsible for
# adding the v4 columns on top of this snapshot.
_V3_PER_SAMPLE_COLUMNS = (
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
)


def _build_v3_db(db_path: Path, sampling_n: int) -> sqlite3.Connection:
    """Construct a v3-schema DB by hand: `run_meta` lacks the two belief
    columns, `run_results` lacks the three v4 belief columns. `schema_version`
    is stamped with `version=3`."""
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
            reflection_protocol_hash  TEXT
        );
        """
    )
    cols = ["question_id TEXT PRIMARY KEY", "user_prompt TEXT"]
    for i in range(sampling_n):
        for name, sql_type in _V3_PER_SAMPLE_COLUMNS:
            cols.append(f"s{i}_{name} {sql_type}")
    cols.append("FOREIGN KEY (question_id) REFERENCES questions(id)")
    conn.execute(f"CREATE TABLE run_results ({', '.join(cols)})")
    conn.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
        (3, dbmod.utcnow_iso()),
    )
    return conn


def _seed_question(conn: sqlite3.Connection, qid: str = "q42") -> None:
    import json

    conn.execute(
        "INSERT INTO questions (id, choice_type, question_type, event, options, answer, end_time, imported_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (qid, "single", "yes_no", "ev", json.dumps(["Yes", "No"]), "A", "2026-01-01", dbmod.utcnow_iso()),
    )


def test_init_schema_migrates_v3_to_v4(tmp_path: Path) -> None:
    """An existing v3 DB must be ALTERed in place to v4 on re-open: three new
    `s{i}_belief_*` columns per sample plus two `belief_protocol_*` columns
    on `run_meta`. The historical `version=3` row is preserved alongside `4`."""
    db_path = tmp_path / "v3.db"
    conn = _build_v3_db(db_path, sampling_n=3)
    _seed_question(conn)
    # Hand-write a v3-shaped row to confirm the migration preserves data.
    legacy_created_at = dbmod.utcnow_iso()
    conn.execute(
        "INSERT INTO run_results (question_id, user_prompt, s0_correct, s0_created_at, s0_finish_reason) "
        "VALUES (?, ?, ?, ?, ?)",
        ("q42", "PROMPT", 1, legacy_created_at, "stop"),
    )

    dbmod.init_schema(conn, sampling_n=3)

    versions = {
        int(r["version"])
        for r in conn.execute("SELECT version FROM schema_version").fetchall()
    }
    assert versions == {3, 4}, "v3 → v4 migration must keep v3 row and stamp v4"

    cols = {r["name"] for r in conn.execute("PRAGMA table_info(run_results)").fetchall()}
    for i in range(3):
        for name in ("belief_final", "belief_trace", "belief_parse_ok"):
            assert f"s{i}_{name}" in cols, f"missing s{i}_{name} after v3→v4 migration"

    meta_cols = {r["name"] for r in conn.execute("PRAGMA table_info(run_meta)").fetchall()}
    assert "belief_protocol_text" in meta_cols
    assert "belief_protocol_hash" in meta_cols

    # Pre-existing v3 data survives; new belief columns default to NULL.
    row = conn.execute("SELECT * FROM run_results WHERE question_id='q42'").fetchone()
    assert row["s0_correct"] == 1
    assert row["s0_created_at"] == legacy_created_at
    assert row["s0_finish_reason"] == "stop"
    assert row["s0_belief_final"] is None
    assert row["s0_belief_trace"] is None
    assert row["s0_belief_parse_ok"] is None

    # Idempotency: a second init_schema must not re-stamp v4.
    dbmod.init_schema(conn, sampling_n=3)
    versions_after = {
        int(r["version"])
        for r in conn.execute("SELECT version FROM schema_version").fetchall()
    }
    assert versions_after == {3, 4}


def test_init_schema_idempotent_on_fresh_v4(tmp_path: Path) -> None:
    """A brand-new v4 DB stamped with `version=4` must not re-INSERT or
    re-ALTER on subsequent `init_schema` calls."""
    conn = dbmod.connect(tmp_path / "v4.db")
    dbmod.init_schema(conn, sampling_n=2)

    versions_first = {
        int(r["version"])
        for r in conn.execute("SELECT version FROM schema_version").fetchall()
    }
    assert versions_first == {4}

    dbmod.init_schema(conn, sampling_n=2)
    versions_second = {
        int(r["version"])
        for r in conn.execute("SELECT version FROM schema_version").fetchall()
    }
    assert versions_second == {4}, "second init_schema must not re-INSERT version=4"


def test_init_schema_rejects_n_mismatch_v4(tmp_path: Path) -> None:
    """Schema-of-N guardrail still triggers on v4 DBs: opening a `sampling_n=3`
    DB with `sampling_n=5` raises ValueError listing missing columns (e.g. the
    new s3_belief_* / s4_belief_* slots)."""
    conn = dbmod.connect(tmp_path / "v4.db")
    dbmod.init_schema(conn, sampling_n=3)
    with pytest.raises(ValueError, match="missing columns"):
        dbmod.init_schema(conn, sampling_n=5)


def test_register_run_meta_writes_belief_fields(tmp_path: Path) -> None:
    """Both nullable paths: explicit belief values land in run_meta, and
    omitting the kwargs writes NULLs. Independent of reflection fields."""
    conn = dbmod.connect(tmp_path / "r.db")
    dbmod.init_schema(conn, sampling_n=1)

    dbmod.register_run_meta(
        conn,
        run_id="run-with-belief",
        model="m1",
        sampling_n=1,
        filters_snapshot={},
        config_snapshot={},
        source_db_hash="a" * 64,
        metadata_hash="b" * 64,
        prompt_templates_hash="c" * 64,
        belief_protocol_text="emit <belief>{...}</belief> please",
        belief_protocol_hash="cafef00ddeadbeef",
    )
    dbmod.register_run_meta(
        conn,
        run_id="run-without-belief",
        model="m1",
        sampling_n=1,
        filters_snapshot={},
        config_snapshot={},
        source_db_hash="a" * 64,
        metadata_hash="b" * 64,
        prompt_templates_hash="c" * 64,
    )

    rows = {
        r["run_id"]: r
        for r in conn.execute(
            "SELECT run_id, belief_protocol_text, belief_protocol_hash, "
            "reflection_protocol_text, reflection_protocol_hash FROM run_meta"
        ).fetchall()
    }
    assert rows["run-with-belief"]["belief_protocol_text"] == "emit <belief>{...}</belief> please"
    assert rows["run-with-belief"]["belief_protocol_hash"] == "cafef00ddeadbeef"
    # belief flag enabled does NOT auto-fill reflection: protocols are independent.
    assert rows["run-with-belief"]["reflection_protocol_text"] is None
    assert rows["run-with-belief"]["reflection_protocol_hash"] is None
    assert rows["run-without-belief"]["belief_protocol_text"] is None
    assert rows["run-without-belief"]["belief_protocol_hash"] is None


def test_register_run_meta_belief_independent_of_reflection(tmp_path: Path) -> None:
    """Reflection-enabled run with belief disabled: reflection columns set,
    belief columns NULL — and vice versa. Verifies the two protocol
    fingerprints never bleed into each other."""
    conn = dbmod.connect(tmp_path / "r.db")
    dbmod.init_schema(conn, sampling_n=1)

    dbmod.register_run_meta(
        conn,
        run_id="A",
        model="m1",
        sampling_n=1,
        filters_snapshot={},
        config_snapshot={},
        source_db_hash="a" * 64,
        metadata_hash="b" * 64,
        prompt_templates_hash="c" * 64,
        reflection_protocol_text="reflect",
        reflection_protocol_hash="reflect_hash",
    )
    dbmod.register_run_meta(
        conn,
        run_id="B",
        model="m1",
        sampling_n=1,
        filters_snapshot={},
        config_snapshot={},
        source_db_hash="a" * 64,
        metadata_hash="b" * 64,
        prompt_templates_hash="c" * 64,
        belief_protocol_text="belief",
        belief_protocol_hash="belief_hash",
    )

    a = conn.execute(
        "SELECT reflection_protocol_hash, belief_protocol_hash FROM run_meta WHERE run_id='A'"
    ).fetchone()
    b = conn.execute(
        "SELECT reflection_protocol_hash, belief_protocol_hash FROM run_meta WHERE run_id='B'"
    ).fetchone()
    assert a["reflection_protocol_hash"] == "reflect_hash"
    assert a["belief_protocol_hash"] is None
    assert b["reflection_protocol_hash"] is None
    assert b["belief_protocol_hash"] == "belief_hash"
