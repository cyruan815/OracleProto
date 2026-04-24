"""Per-model SQLite persistence layer.

Each evaluation run owns a directory `RUNS_ROOT/{run_id}/`; inside `db/` we
write ONE SQLite file per model. Every model DB is self-contained: it has a
copy of the source questions, the prompt templates, a single `run_meta` row,
and a WIDE `run_results` table with one row per question and one column group
`s{i}_*` per sample index `0..SAMPLING_N-1`.

The DB stores raw observations only. Statistics (pass@1, pass_any@N, majority
vote, etc.) are computed post-hoc by `forecast_eval.analysis` and written to
`RUNS_ROOT/{run_id}/analysis/`.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import Settings


SCHEMA_VERSION = 2


# ---------- Per-sample column definitions ----------

# Each sample_idx contributes this column suite (prefixed with "s{i}_").
PER_SAMPLE_COLUMNS: tuple[tuple[str, str], ...] = (
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
)
PER_SAMPLE_FIELD_NAMES: tuple[str, ...] = tuple(name for name, _ in PER_SAMPLE_COLUMNS)


def sample_col(sample_idx: int, field: str) -> str:
    return f"s{sample_idx}_{field}"


# ---------- Schema (static parts + dynamic run_results) ----------

_STATIC_SCHEMA_SQL = """
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

CREATE TABLE IF NOT EXISTS run_meta (
    run_id                TEXT PRIMARY KEY,
    model                 TEXT NOT NULL,
    sampling_n            INTEGER NOT NULL,
    config_snapshot       TEXT NOT NULL,
    filters_snapshot      TEXT NOT NULL,
    source_db_hash        TEXT NOT NULL,
    metadata_hash         TEXT NOT NULL,
    prompt_templates_hash TEXT NOT NULL,
    training_cutoff       TEXT,
    started_at            TEXT NOT NULL,
    finished_at           TEXT
);
"""


def _run_results_ddl(sampling_n: int) -> str:
    if sampling_n < 1:
        raise ValueError(f"sampling_n must be >= 1, got {sampling_n}")
    cols: list[str] = [
        "question_id TEXT PRIMARY KEY",
        "user_prompt TEXT",
    ]
    for i in range(sampling_n):
        for name, sql_type in PER_SAMPLE_COLUMNS:
            cols.append(f"{sample_col(i, name)} {sql_type}")
    cols.append("FOREIGN KEY (question_id) REFERENCES questions(id)")
    joined = ",\n    ".join(cols)
    return (
        "CREATE TABLE IF NOT EXISTS run_results (\n    "
        + joined
        + "\n);\n"
        + "CREATE INDEX IF NOT EXISTS idx_run_results_question ON run_results(question_id);"
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


def init_schema(conn: sqlite3.Connection, sampling_n: int) -> None:
    """Create all tables for one model's DB with a run_results shape matching
    `sampling_n`. Idempotent as long as `sampling_n` stays the same across runs
    that reuse the same DB.
    """
    conn.executescript(_STATIC_SCHEMA_SQL)
    conn.executescript(_run_results_ddl(sampling_n))
    row = conn.execute(
        "SELECT version FROM schema_version WHERE version = ?",
        (SCHEMA_VERSION,),
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, utcnow_iso()),
        )
    _assert_run_results_matches(conn, sampling_n)


def _assert_run_results_matches(conn: sqlite3.Connection, sampling_n: int) -> None:
    """Guardrail: if a DB already exists with a different SAMPLING_N, fail fast."""
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(run_results)").fetchall()}
    expected = {"question_id", "user_prompt"}
    for i in range(sampling_n):
        for name, _ in PER_SAMPLE_COLUMNS:
            expected.add(sample_col(i, name))
    missing = expected - existing
    if missing:
        raise ValueError(
            "run_results table is missing columns for the configured SAMPLING_N; "
            "the DB was created with a different N. "
            f"missing={sorted(missing)}"
        )


# ---------- Hashing ----------

def compute_source_db_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_kv_string(mapping: dict[str, Any]) -> str:
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


# ---------- Model slug safety ----------

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def model_slug_safe(model: str) -> str:
    """Produce a filesystem-safe filename stem for a model slug.

    `/` is converted to `__` (to keep the provider/model visual split) and any
    other character outside `[A-Za-z0-9._-]` becomes `_`. Collisions are not
    expected in practice because the original slug space is small.
    """
    if not model:
        raise ValueError("model slug must not be empty")
    safe = model.replace("/", "__")
    safe = _UNSAFE_CHARS.sub("_", safe)
    return safe


# ---------- Run meta (per-model DB) ----------

def register_run_meta(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    model: str,
    sampling_n: int,
    filters_snapshot: dict[str, Any],
    config_snapshot: dict[str, Any],
    source_db_hash: str,
    metadata_hash: str,
    prompt_templates_hash: str,
    training_cutoff: str | None = None,
    started_at: str | None = None,
) -> None:
    """Insert or refresh the single `run_meta` row for this model's DB.

    `finished_at` and `started_at` are preserved on resume so the original
    start timestamp survives.
    """
    conn.execute(
        """
        INSERT INTO run_meta (
            run_id, model, sampling_n,
            config_snapshot, filters_snapshot,
            source_db_hash, metadata_hash, prompt_templates_hash,
            training_cutoff, started_at, finished_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
        ON CONFLICT(run_id) DO UPDATE SET
            model                 = excluded.model,
            sampling_n            = excluded.sampling_n,
            config_snapshot       = excluded.config_snapshot,
            filters_snapshot      = excluded.filters_snapshot,
            source_db_hash        = excluded.source_db_hash,
            metadata_hash         = excluded.metadata_hash,
            prompt_templates_hash = excluded.prompt_templates_hash,
            training_cutoff       = excluded.training_cutoff
        """,
        (
            run_id,
            model,
            sampling_n,
            json.dumps(config_snapshot, ensure_ascii=False, sort_keys=True),
            json.dumps(filters_snapshot, ensure_ascii=False, sort_keys=True),
            source_db_hash,
            metadata_hash,
            prompt_templates_hash,
            training_cutoff,
            started_at or utcnow_iso(),
        ),
    )


def finish_run_meta(
    conn: sqlite3.Connection,
    run_id: str,
    finished_at: str | None = None,
) -> None:
    conn.execute(
        "UPDATE run_meta SET finished_at = ? WHERE run_id = ?",
        (finished_at or utcnow_iso(), run_id),
    )


def load_completed_samples(
    conn: sqlite3.Connection,
    sampling_n: int,
) -> set[tuple[str, int]]:
    """Return {(question_id, sample_idx)} already accounted for.

    A sample counts as accounted-for if its `s{i}_created_at` is set AND its
    `s{i}_error` is either NULL (normal completion) or 'skipped_training_cutoff'
    (actively filtered). Any other `s{i}_error` value will be retried.
    """
    done: set[tuple[str, int]] = set()
    for i in range(sampling_n):
        created = sample_col(i, "created_at")
        err = sample_col(i, "error")
        rows = conn.execute(
            f"""
            SELECT question_id FROM run_results
            WHERE {created} IS NOT NULL
              AND ({err} IS NULL OR {err} = 'skipped_training_cutoff')
            """
        ).fetchall()
        for r in rows:
            done.add((r["question_id"], i))
    return done


# ---------- Async writer ----------

_SENTINEL = object()


class AsyncWriter:
    """Per-DB single-consumer async writer that batches UPSERTs.

    Each producer hands in a dict shaped like:
        {
            "question_id": str,
            "sample_idx": int,
            "user_prompt": str | None,
            "final_answer_letters": str | None,
            "final_answer_raw": str | None,
            "correct": int | None,
            "parse_ok": int,
            "tool_calls_count": int,
            "react_steps": int,
            "prompt_tokens": int,
            "completion_tokens": int,
            "reasoning_tokens": int,
            "latency_ms": int,
            "messages_trace": str | None,
            "search_calls": str | None,
            "error": str | None,
            "created_at": str,
        }

    The writer looks up `sample_idx` and UPSERTs the matching `s{i}_*` columns
    on the `question_id` row. `user_prompt` is written once (COALESCE) so the
    first sample that lands for a question persists the canonical rendered
    prompt; later samples don't overwrite it.
    """

    FLUSH_INTERVAL_S = 1.0

    def __init__(self, conn: sqlite3.Connection, sampling_n: int, batch: int) -> None:
        self._conn = conn
        self._sampling_n = sampling_n
        self._batch = max(1, int(batch))
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._started = False
        self._sql_cache: dict[int, str] = {}

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._task = asyncio.create_task(self._run(), name="db-writer")

    async def enqueue_result(self, row: dict[str, Any]) -> None:
        if "question_id" not in row or "sample_idx" not in row:
            raise ValueError("writer row requires question_id and sample_idx")
        await self._queue.put(row)

    async def drain(self) -> None:
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

    def _sql_for(self, sample_idx: int) -> str:
        cached = self._sql_cache.get(sample_idx)
        if cached is not None:
            return cached
        if not (0 <= sample_idx < self._sampling_n):
            raise ValueError(
                f"sample_idx {sample_idx} out of range for SAMPLING_N={self._sampling_n}"
            )
        sample_cols = [sample_col(sample_idx, name) for name in PER_SAMPLE_FIELD_NAMES]
        insert_cols = ["question_id", "user_prompt", *sample_cols]
        placeholders = ",".join("?" * len(insert_cols))
        update_parts = [
            "user_prompt = COALESCE(run_results.user_prompt, excluded.user_prompt)"
        ]
        update_parts.extend(f"{col} = excluded.{col}" for col in sample_cols)
        sql = (
            "INSERT INTO run_results ("
            + ",".join(insert_cols)
            + ") VALUES ("
            + placeholders
            + ") ON CONFLICT(question_id) DO UPDATE SET "
            + ", ".join(update_parts)
        )
        self._sql_cache[sample_idx] = sql
        return sql

    async def _flush(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        await asyncio.to_thread(self._insert_rows_sync, rows)

    def _insert_rows_sync(self, rows: Iterable[dict[str, Any]]) -> None:
        # Group rows by sample_idx so each SQL stays stable within executemany.
        buckets: dict[int, list[tuple[Any, ...]]] = {}
        for row in rows:
            sample_idx = int(row["sample_idx"])
            qid = row["question_id"]
            payload = [qid, row.get("user_prompt")]
            payload.extend(row.get(name) for name in PER_SAMPLE_FIELD_NAMES)
            buckets.setdefault(sample_idx, []).append(tuple(payload))

        self._conn.execute("BEGIN")
        try:
            for sample_idx, batch in buckets.items():
                self._conn.executemany(self._sql_for(sample_idx), batch)
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise


# Standalone helper: synchronous upsert used by tests and by the runner's
# best-effort cutoff pre-seeding before the async writer starts.
def upsert_sample_sync(
    conn: sqlite3.Connection,
    sampling_n: int,
    row: dict[str, Any],
) -> None:
    writer_like = AsyncWriter(conn, sampling_n=sampling_n, batch=1)
    writer_like._insert_rows_sync([row])
