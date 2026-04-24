from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import Settings


SCHEMA_VERSION = 1


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS questions (
    id            TEXT PRIMARY KEY,
    choice_type   TEXT NOT NULL CHECK (choice_type IN ('single','multi')),
    question_type TEXT NOT NULL CHECK (question_type IN ('yes_no','binary_named','multiple_choice')),
    event         TEXT NOT NULL,
    options       TEXT NOT NULL,
    answer        TEXT NOT NULL,
    end_time      TEXT NOT NULL,
    imported_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_questions_choice_type   ON questions(choice_type);
CREATE INDEX IF NOT EXISTS idx_questions_question_type ON questions(question_type);

CREATE TABLE IF NOT EXISTS prompt_templates (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id                TEXT PRIMARY KEY,
    config_snapshot       TEXT NOT NULL,
    filters_snapshot      TEXT NOT NULL,
    source_db_hash        TEXT NOT NULL,
    metadata_hash         TEXT NOT NULL,
    prompt_templates_hash TEXT NOT NULL,
    started_at            TEXT NOT NULL,
    finished_at           TEXT
);

CREATE TABLE IF NOT EXISTS run_results (
    run_id      TEXT NOT NULL,
    question_id TEXT NOT NULL,
    model       TEXT NOT NULL,
    sample_idx  INTEGER NOT NULL,

    final_answer_letters TEXT,
    final_answer_raw     TEXT,
    correct              INTEGER,
    parse_ok             INTEGER NOT NULL,

    tool_calls_count   INTEGER NOT NULL,
    react_steps        INTEGER NOT NULL,
    prompt_tokens      INTEGER,
    completion_tokens  INTEGER,
    reasoning_tokens   INTEGER,
    latency_ms         INTEGER NOT NULL,

    user_prompt    TEXT,
    messages_trace TEXT,
    search_calls   TEXT,
    error          TEXT,
    created_at     TEXT NOT NULL,

    PRIMARY KEY (run_id, question_id, model, sample_idx),
    FOREIGN KEY (question_id) REFERENCES questions(id),
    FOREIGN KEY (run_id)      REFERENCES runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_run_results_lookup ON run_results(run_id, model, question_id);
"""


RUN_RESULT_COLUMNS = (
    "run_id",
    "question_id",
    "model",
    "sample_idx",
    "final_answer_letters",
    "final_answer_raw",
    "correct",
    "parse_ok",
    "tool_calls_count",
    "react_steps",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "latency_ms",
    "user_prompt",
    "messages_trace",
    "search_calls",
    "error",
    "created_at",
)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")


def apply_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a sqlite3 connection with standard PRAGMAs applied."""
    conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    apply_pragmas(conn)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create all five tables if missing and stamp schema_version=1."""
    conn.executescript(SCHEMA_SQL)
    row = conn.execute("SELECT version FROM schema_version WHERE version = ?", (SCHEMA_VERSION,)).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, utcnow_iso()),
        )


# ---------- Hashing ----------

def compute_source_db_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_kv_string(mapping: dict[str, Any]) -> str:
    """Stable canonicalization: sorted by key, lines 'key=value' joined by '\\n'.

    Values are JSON-encoded (sort_keys, ensure_ascii=False) so nested dicts are stable.
    """
    parts = []
    for k in sorted(mapping):
        v = mapping[k]
        if isinstance(v, str):
            enc = v
        else:
            enc = json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        parts.append(f"{k}={enc}")
    return "\n".join(parts)


def compute_metadata_hash(features_json: str | dict[str, Any]) -> str:
    """Hash dataset_metadata.features_json in a normalized form."""
    if isinstance(features_json, str):
        try:
            data = json.loads(features_json)
        except json.JSONDecodeError:
            return hashlib.sha256(features_json.encode("utf-8")).hexdigest()
    else:
        data = features_json
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_prompt_templates_hash(templates: dict[str, str]) -> str:
    canonical = _canonical_kv_string(templates)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------- Redaction ----------

def redact_api_key(raw: str | None, provider: str) -> dict[str, Any]:
    if not raw:
        return {"provider": provider, "prefix": "", "length": 0, "sha256_12": ""}
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return {
        "provider": provider,
        "prefix": raw[:4],
        "length": len(raw),
        "sha256_12": digest,
    }


_API_KEY_FIELDS = {
    "LLM_API_KEY": "llm",
    "TAVILY_API_KEY": "tavily",
}


def snapshot_settings(settings: Settings) -> dict[str, Any]:
    """Return a JSON-ready dict with API keys redacted. Dates are isoformatted."""
    raw = settings.model_dump()
    redacted: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _API_KEY_FIELDS:
            redacted[key] = redact_api_key(getattr(settings, key), _API_KEY_FIELDS[key])
        elif key == "MODEL_TRAINING_CUTOFFS":
            redacted[key] = {m: d.isoformat() for m, d in settings.MODEL_TRAINING_CUTOFFS.items()}
        else:
            redacted[key] = value
    return redacted


# ---------- Run management ----------

def register_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    filters_snapshot: dict[str, Any],
    config_snapshot: dict[str, Any],
    source_db_hash: str,
    metadata_hash: str,
    prompt_templates_hash: str,
    started_at: str | None = None,
) -> None:
    """Insert a new run, or refresh config/filters/hash on an existing run.

    `finished_at` is preserved on UPSERT so a successfully-finished run stays
    finished; `started_at` is preserved so resume keeps the original start time.
    """
    conn.execute(
        """
        INSERT INTO runs (
            run_id, config_snapshot, filters_snapshot,
            source_db_hash, metadata_hash, prompt_templates_hash,
            started_at, finished_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
        ON CONFLICT(run_id) DO UPDATE SET
            config_snapshot       = excluded.config_snapshot,
            filters_snapshot      = excluded.filters_snapshot,
            source_db_hash        = excluded.source_db_hash,
            metadata_hash         = excluded.metadata_hash,
            prompt_templates_hash = excluded.prompt_templates_hash
        """,
        (
            run_id,
            json.dumps(config_snapshot, ensure_ascii=False, sort_keys=True),
            json.dumps(filters_snapshot, ensure_ascii=False, sort_keys=True),
            source_db_hash,
            metadata_hash,
            prompt_templates_hash,
            started_at or utcnow_iso(),
        ),
    )


def finish_run(conn: sqlite3.Connection, run_id: str, finished_at: str | None = None) -> None:
    conn.execute(
        "UPDATE runs SET finished_at = ? WHERE run_id = ?",
        (finished_at or utcnow_iso(), run_id),
    )


def load_completed(conn: sqlite3.Connection, run_id: str) -> set[tuple[str, str, int]]:
    """Return the set of (question_id, model, sample_idx) already accounted for.

    "Accounted for" = normal completion OR actively skipped training cutoff.
    Any other error (network/server_5xx/bad_request/content_policy) is retried.
    """
    rows = conn.execute(
        """
        SELECT question_id, model, sample_idx
        FROM run_results
        WHERE run_id = ?
          AND (error IS NULL OR error = 'skipped_training_cutoff')
        """,
        (run_id,),
    ).fetchall()
    return {(r["question_id"], r["model"], r["sample_idx"]) for r in rows}


# ---------- Async writer ----------

_SENTINEL = object()


class AsyncWriter:
    """Single-consumer async writer that batches commits.

    All producers `await enqueue_result(row_dict)`. The writer task dequeues,
    accumulates up to DB_COMMIT_BATCH rows (or until 1 second elapses since the
    first pending row), then issues one INSERT OR REPLACE transaction via
    `asyncio.to_thread` so sqlite stays off the event loop.

    Must live in a single event loop — do NOT share across threads (asyncio.Queue
    is not thread-safe).
    """

    FLUSH_INTERVAL_S = 1.0

    def __init__(self, conn: sqlite3.Connection, batch: int) -> None:
        self._conn = conn
        self._batch = max(1, int(batch))
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._task = asyncio.create_task(self._run(), name="db-writer")

    async def enqueue_result(self, row: dict[str, Any]) -> None:
        await self._queue.put(row)

    async def drain(self) -> None:
        """Block until the queue is empty and any in-flight batch has committed."""
        await self._queue.join()

    async def close(self) -> None:
        if not self._started or self._task is None:
            return
        await self._queue.put(_SENTINEL)
        await self._task
        self._task = None
        self._started = False

    async def _run(self) -> None:
        pending: list[dict[str, Any]] = []
        first_enqueued_at: float | None = None
        while True:
            timeout: float | None
            if pending and first_enqueued_at is not None:
                timeout = max(0.0, self.FLUSH_INTERVAL_S - (time.monotonic() - first_enqueued_at))
            else:
                timeout = None
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                await self._flush(pending)
                for _ in range(len(pending)):
                    self._queue.task_done()
                pending.clear()
                first_enqueued_at = None
                continue

            if item is _SENTINEL:
                self._queue.task_done()
                if pending:
                    await self._flush(pending)
                    for _ in range(len(pending)):
                        self._queue.task_done()
                    pending.clear()
                return

            pending.append(item)
            if first_enqueued_at is None:
                first_enqueued_at = time.monotonic()

            if len(pending) >= self._batch:
                await self._flush(pending)
                for _ in range(len(pending)):
                    self._queue.task_done()
                pending.clear()
                first_enqueued_at = None

    async def _flush(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        await asyncio.to_thread(_insert_rows_sync, self._conn, rows)


def _insert_rows_sync(conn: sqlite3.Connection, rows: Iterable[dict[str, Any]]) -> None:
    sql = (
        "INSERT OR REPLACE INTO run_results ("
        + ",".join(RUN_RESULT_COLUMNS)
        + ") VALUES ("
        + ",".join("?" for _ in RUN_RESULT_COLUMNS)
        + ")"
    )
    payload = [
        tuple(row.get(col) for col in RUN_RESULT_COLUMNS) for row in rows
    ]
    # isolation_level=None means autocommit; wrap explicitly for batch.
    conn.execute("BEGIN")
    try:
        conn.executemany(sql, payload)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
