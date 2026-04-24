"""Top-level orchestration for one evaluation run.

Contracts:
- `evaluation.py` prepares per-model SQLite connections under
  `RUNS_ROOT/{run_id}/db/` and hands them to `run()`.
- This module owns the task queue, the async writers (one per model), the
  cutoff-filter + resume logic, and the progress log.
- The DB layer stores raw observations only; statistics are computed post-hoc
  by `forecast_eval.analysis`.
"""
from __future__ import annotations

import asyncio
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from loguru import logger

from . import db as dbmod
from .config import Settings
from .db import AsyncWriter, utcnow_iso
from .errors import AuthError, ErrorKind, classify
from .llm import AuthError as _LLMAuthError  # noqa: F401 — re-exported for callers
from .react import run_react
from .types import QFilter, Question, SampleResult


@dataclass
class Task:
    question: Question
    model: str
    sample_idx: int


@dataclass
class RunStats:
    total: int = 0
    completed_preexisting: int = 0
    skipped_cutoff: int = 0
    planned: int = 0
    done: int = 0
    errors: dict[str, int] = field(default_factory=dict)


def generate_run_id(now: datetime | None = None) -> str:
    ts = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{uuid.uuid4().hex[:4]}"


def _skipped_cutoff_row(q: Question, sample_idx: int) -> dict[str, Any]:
    return SampleResult(
        run_id="",
        question_id=q.id,
        model="",
        sample_idx=sample_idx,
        final_answer_letters=None,
        final_answer_raw=None,
        correct=None,
        parse_ok=0,
        tool_calls_count=0,
        react_steps=0,
        prompt_tokens=0,
        completion_tokens=0,
        reasoning_tokens=0,
        latency_ms=0,
        user_prompt=None,
        messages_trace=None,
        search_calls=None,
        error="skipped_training_cutoff",
        created_at=utcnow_iso(),
    ).to_row()


def _error_row(q: Question, sample_idx: int, error: str) -> dict[str, Any]:
    return SampleResult(
        run_id="",
        question_id=q.id,
        model="",
        sample_idx=sample_idx,
        final_answer_letters=None,
        final_answer_raw=None,
        correct=None,
        parse_ok=0,
        tool_calls_count=0,
        react_steps=0,
        prompt_tokens=0,
        completion_tokens=0,
        reasoning_tokens=0,
        latency_ms=0,
        user_prompt=None,
        messages_trace=None,
        search_calls=None,
        error=error,
        created_at=utcnow_iso(),
    ).to_row()


def build_task_plan(
    *,
    questions: list[Question],
    settings: Settings,
    completed: dict[str, set[tuple[str, int]]],
    run_id: str,
) -> tuple[list[Task], dict[str, list[dict[str, Any]]], RunStats]:
    """Expand (questions × models × samples), drop resumed cells, then split:
       - `todo`: LLM work to dispatch (per-model writers consume this)
       - `cutoff_rows`: model -> list of pre-seeded rows marked
         `error="skipped_training_cutoff"` (not counted as LLM work)
       - `stats`: counters for progress logging

    Resume takes precedence over cutoff filtering — a cell already completed
    must never be re-emitted as skipped_training_cutoff.
    """
    stats = RunStats()
    todo: list[Task] = []
    cutoff_rows: dict[str, list[dict[str, Any]]] = {m: [] for m in settings.MODELS}

    for q in questions:
        q_end = date.fromisoformat(q.end_time)
        for model in settings.MODELS:
            cutoff = settings.MODEL_TRAINING_CUTOFFS.get(model)
            is_cutoff_hit = cutoff is not None and q_end <= cutoff
            done_for_model = completed.get(model, set())
            for s in range(settings.SAMPLING_N):
                stats.total += 1
                key = (q.id, s)
                if key in done_for_model:
                    stats.completed_preexisting += 1
                    continue
                if is_cutoff_hit:
                    cutoff_rows[model].append(_skipped_cutoff_row(q, s))
                    stats.skipped_cutoff += 1
                    continue
                todo.append(Task(question=q, model=model, sample_idx=s))
                stats.planned += 1

    return todo, cutoff_rows, stats


async def _run_task_with_retry(
    task: Task,
    *,
    settings: Settings,
    templates: dict[str, str],
    run_id: str,
    llm_semaphore: asyncio.Semaphore,
    search_semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Execute one sample; classify any terminal exception into an error row.

    AUTH errors re-raise so the caller aborts the whole run; everything else
    produces a row with `error=...` the writer can still land.
    """
    try:
        async with llm_semaphore:
            result = await run_react(
                task.question,
                model=task.model,
                sample_idx=task.sample_idx,
                settings=settings,
                templates=templates,
                run_id=run_id,
                search_semaphore=search_semaphore,
            )
        return result.to_row()
    except AuthError:
        raise
    except _LLMAuthError:
        raise
    except BaseException as exc:  # noqa: BLE001 — map to row-level error
        kind = classify(exc)
        if kind is ErrorKind.AUTH:
            raise AuthError(str(exc)) from exc
        error_str = str(kind) if kind is not ErrorKind.UNKNOWN else "unknown"
        logger.exception(
            "sample failed q={} model={} sample={} kind={}",
            task.question.id,
            task.model,
            task.sample_idx,
            kind,
        )
        return _error_row(task.question, task.sample_idx, error_str)


def _log_progress(
    *,
    run_id: str,
    done: int,
    total: int,
    sampling_n: int,
    task: Task,
    row: dict[str, Any],
) -> None:
    error = row.get("error")
    q = task.question
    if error:
        logger.error(
            "[run={}] [{}/{}] q={} qt={} ct={} model={} sample={}/{} error={} retry_exhausted",
            run_id,
            done,
            total,
            q.id,
            q.question_type,
            q.choice_type,
            task.model,
            task.sample_idx + 1,
            sampling_n,
            error,
        )
    else:
        logger.info(
            "[run={}] [{}/{}] q={} qt={} ct={} model={} sample={}/{} correct={} parse_ok={} steps={} tool_calls={} latency={}ms",
            run_id,
            done,
            total,
            q.id,
            q.question_type,
            q.choice_type,
            task.model,
            task.sample_idx + 1,
            sampling_n,
            row.get("correct"),
            row.get("parse_ok"),
            row.get("react_steps"),
            row.get("tool_calls_count"),
            row.get("latency_ms"),
        )


async def run(
    *,
    settings: Settings,
    filters: QFilter,
    questions: list[Question],
    templates: dict[str, str],
    run_id: str,
    conns: dict[str, sqlite3.Connection],
) -> RunStats:
    """Orchestrate one run across all configured models.

    Caller responsibility (see `evaluation.py`):
      * create `RUNS_ROOT/{run_id}/{db,analysis,logs}/`
      * open one sqlite connection per model under db/
      * run `db.init_schema(conn, SAMPLING_N)`
      * run `loader.sync_questions(...)` + `loader.sync_prompt_templates(...)` per DB
      * run `db.register_run_meta(...)` per DB

    This function then:
      * loads per-model resume sets
      * plans the task list + cutoff rows (one list per model)
      * spawns one `AsyncWriter` per model
      * drives the ReAct loop across `LLM_MAX_CONCURRENCY` workers
      * finishes each model's `run_meta` row on clean exit (or leaves it open
        if aborted by AUTH)
    """
    sampling_n = settings.SAMPLING_N
    models = list(conns.keys())

    # Per-model resume set
    completed: dict[str, set[tuple[str, int]]] = {
        m: dbmod.load_completed_samples(conns[m], sampling_n) for m in models
    }
    todo, cutoff_rows, stats = build_task_plan(
        questions=questions,
        settings=settings,
        completed=completed,
        run_id=run_id,
    )

    logger.info(
        "[run={}] plan: total={} already_done={} skipped_cutoff={} to_run={}",
        run_id,
        stats.total,
        stats.completed_preexisting,
        stats.skipped_cutoff,
        stats.planned,
    )

    writers: dict[str, AsyncWriter] = {
        m: AsyncWriter(conns[m], sampling_n=sampling_n, batch=settings.DB_COMMIT_BATCH)
        for m in models
    }
    for w in writers.values():
        await w.start()

    llm_sem = asyncio.Semaphore(settings.LLM_MAX_CONCURRENCY)
    search_sem = asyncio.Semaphore(settings.SEARCH_MAX_CONCURRENCY)

    # Cutoff rows are enqueued first. They flow through each model's writer and
    # inflate the [done/total] denominator in a predictable way.
    for model, rows in cutoff_rows.items():
        w = writers[model]
        for row in rows:
            await w.enqueue_result(row)

    done_counter = stats.completed_preexisting + stats.skipped_cutoff
    aborted = False

    async def _worker(task: Task) -> None:
        nonlocal done_counter
        try:
            row = await _run_task_with_retry(
                task,
                settings=settings,
                templates=templates,
                run_id=run_id,
                llm_semaphore=llm_sem,
                search_semaphore=search_sem,
            )
        except AuthError:
            raise
        await writers[task.model].enqueue_result(row)
        done_counter += 1
        kind = row.get("error")
        if kind:
            stats.errors[kind] = stats.errors.get(kind, 0) + 1
        _log_progress(
            run_id=run_id,
            done=done_counter,
            total=stats.total,
            sampling_n=sampling_n,
            task=task,
            row=row,
        )

    worker_tasks: list[asyncio.Task] = []
    try:
        for t in todo:
            worker_tasks.append(asyncio.create_task(_worker(t)))
        for fut in asyncio.as_completed(worker_tasks):
            try:
                await fut
            except AuthError:
                logger.error("[run={}] AUTH error; aborting run", run_id)
                aborted = True
                for t in worker_tasks:
                    t.cancel()
                break
    finally:
        for w in writers.values():
            await w.drain()
        for w in writers.values():
            await w.close()
        if not aborted:
            for m, conn in conns.items():
                dbmod.finish_run_meta(conn, run_id)

    stats.done = done_counter
    return stats
