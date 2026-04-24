from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .db import utcnow_iso, connect as db_connect
from .types import QFilter, Question


_REQUIRED_TEMPLATE_KEYS = (
    "agent_role",
    "guidance",
    "prompt_template",
    "outcomes_block_rule",
    "yes_no_output_format",
    "binary_named_output_format",
    "multiple_choice_output_format",
)


def _read_features_json(src_conn: sqlite3.Connection) -> dict[str, Any]:
    row = src_conn.execute("SELECT features_json FROM dataset_metadata").fetchone()
    if row is None:
        raise ValueError("dataset_metadata is empty; source DB is not a valid forecast_eval_set")
    return json.loads(row["features_json"])


def sync_prompt_templates(
    source_db: str | Path,
    results_conn: sqlite3.Connection,
) -> dict[str, str]:
    """Flatten `dataset_metadata.features_json.prompt_reconstruction` into
    `results.db.prompt_templates` and return it as a dict for in-memory use.

    String fields are stored verbatim; nested (dict/list) values are JSON-serialised
    into the same key so downstream readers can `json.loads` on demand.
    """
    src = db_connect(source_db)
    try:
        features = _read_features_json(src)
    finally:
        src.close()

    reconstruction = features.get("prompt_reconstruction")
    if not isinstance(reconstruction, dict):
        raise ValueError("features_json.prompt_reconstruction missing or malformed")

    flat: dict[str, str] = {}
    for key, value in reconstruction.items():
        if isinstance(value, str):
            flat[key] = value
        else:
            flat[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)

    missing = [k for k in _REQUIRED_TEMPLATE_KEYS if k not in flat]
    if missing:
        raise ValueError(f"prompt_reconstruction missing required keys: {missing}")

    now = utcnow_iso()
    results_conn.executemany(
        "INSERT INTO prompt_templates (key, value, imported_at) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value, imported_at = excluded.imported_at",
        [(k, v, now) for k, v in flat.items()],
    )
    return flat


def sync_questions(
    source_db: str | Path,
    results_conn: sqlite3.Connection,
    filters: QFilter,
) -> list[Question]:
    """Copy the filtered `forecast_eval_set` rows into `results.db.questions`."""
    src = db_connect(source_db)
    try:
        where, params = filters.apply_sql()
        rows = src.execute(
            f"SELECT id, choice_type, question_type, event, options, answer, end_time "
            f"FROM forecast_eval_set WHERE {where} "
            f"ORDER BY end_time, id",
            params,
        ).fetchall()
    finally:
        src.close()

    now = utcnow_iso()
    results_conn.executemany(
        "INSERT INTO questions (id, choice_type, question_type, event, options, answer, end_time, imported_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET "
        "choice_type=excluded.choice_type, question_type=excluded.question_type, "
        "event=excluded.event, options=excluded.options, answer=excluded.answer, "
        "end_time=excluded.end_time, imported_at=excluded.imported_at",
        [
            (
                r["id"],
                r["choice_type"],
                r["question_type"],
                r["event"],
                r["options"],
                r["answer"],
                r["end_time"],
                now,
            )
            for r in rows
        ],
    )

    return [
        Question(
            id=r["id"],
            choice_type=r["choice_type"],
            question_type=r["question_type"],
            event=r["event"],
            options=r["options"],
            answer=r["answer"],
            end_time=r["end_time"],
        )
        for r in rows
    ]


def load_raw_features_json(source_db: str | Path) -> str:
    """Return the raw `features_json` string for metadata hashing."""
    src = db_connect(source_db)
    try:
        row = src.execute("SELECT features_json FROM dataset_metadata").fetchone()
    finally:
        src.close()
    if row is None:
        raise ValueError("dataset_metadata is empty")
    return row["features_json"]
