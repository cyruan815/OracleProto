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
from .llm import AuthError as _LLMAuthError  # re-export alias for callers
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


def _skipped_cutoff_row(
    run_id: str,
    q: Question,
    model: str,
    sample_idx: int,
) -> dict[str, Any]:
    return SampleResult(
        run_id=run_id,
        question_id=q.id,
        model=model,
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


def _error_row(
    run_id: str,
    q: Question,
    model: str,
    sample_idx: int,
    error: str,
) -> dict[str, Any]:
    return SampleResult(
        run_id=run_id,
        question_id=q.id,
        model=model,
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
    completed: set[tuple[str, str, int]],
    run_id: str,
) -> tuple[list[Task], list[dict[str, Any]], RunStats]:
    """Expand the cartesian product, remove resumed rows, then split into:
       - `todo`: actual LLM work to dispatch
       - `cutoff_rows`: `skipped_training_cutoff` rows to enqueue directly
       - `stats`: counters for progress logging

    Resume takes precedence over cutoff filtering — a question already completed
    in a previous run must NOT be re-written as skipped_training_cutoff.
    """
    stats = RunStats()
    todo: list[Task] = []
    cutoff_rows: list[dict[str, Any]] = []

    for q in questions:
        q_end = date.fromisoformat(q.end_time)
        for model in settings.MODELS:
            cutoff = settings.MODEL_TRAINING_CUTOFFS.get(model)
            is_cutoff_hit = cutoff is not None and q_end <= cutoff
            for s in range(settings.SAMPLING_N):
                stats.total += 1
                key = (q.id, model, s)
                if key in completed:
                    stats.completed_preexisting += 1
                    continue
                if is_cutoff_hit:
                    cutoff_rows.append(_skipped_cutoff_row(run_id, q, model, s))
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
    """Execute one sample, classify any terminal exception into an error row.

    AUTH errors are re-raised so the caller can abort the whole run; everything
    else produces a row with `error=...` we can keep running past.
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
        return _error_row(run_id, task.question, task.model, task.sample_idx, error_str)


def _log_progress(
    *,
    run_id: str,
    done: int,
    total: int,
    task: Task,
    row: dict[str, Any],
) -> None:
    error = row.get("error")
    correct = row.get("correct")
    parse_ok = row.get("parse_ok")
    steps = row.get("react_steps")
    tool_calls = row.get("tool_calls_count")
    latency = row.get("latency_ms")
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
            "N",  # SAMPLING_N injected via settings at the runner call site below
            error,
        )
    else:
        logger.info(
            "[run={}] [{}/{}] q={} qt={} ct={} model={} sample={} correct={} parse_ok={} steps={} tool_calls={} latency={}ms",
            run_id,
            done,
            total,
            q.id,
            q.question_type,
            q.choice_type,
            task.model,
            task.sample_idx,
            correct,
            parse_ok,
            steps,
            tool_calls,
            latency,
        )


async def run(
    *,
    settings: Settings,
    filters: QFilter,
    questions: list[Question],
    templates: dict[str, str],
    run_id: str,
    conn: sqlite3.Connection,
) -> RunStats:
    """Top-level orchestration. Assumes loader.sync_* + register_run are done."""
    completed = dbmod.load_completed(conn, run_id)
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

    writer = AsyncWriter(conn, batch=settings.DB_COMMIT_BATCH)
    await writer.start()

    llm_sem = asyncio.Semaphore(settings.LLM_MAX_CONCURRENCY)
    search_sem = asyncio.Semaphore(settings.SEARCH_MAX_CONCURRENCY)

    # Enqueue cutoff rows first — they are effectively instant and they inflate
    # the denominator of [done/total] progress log in a predictable way.
    for row in cutoff_rows:
        await writer.enqueue_result(row)

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
        await writer.enqueue_result(row)
        done_counter += 1
        kind = row.get("error")
        if kind:
            stats.errors[kind] = stats.errors.get(kind, 0) + 1
        _log_progress(
            run_id=run_id,
            done=done_counter,
            total=stats.total,
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
        await writer.drain()
        await writer.close()
        if not aborted:
            dbmod.finish_run(conn, run_id)

    stats.done = done_counter
    return stats
