#!/usr/bin/env python3
"""Forecast evaluation entry point.

Usage:
    python evaluation.py
    python evaluation.py --question-type yes_no
    python evaluation.py --question-type multiple_choice --choice-type multi

All runtime knobs live in `.env`; the CLI only filters questions.
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from forecast_eval import db as dbmod
from forecast_eval import loader, runner
from forecast_eval.config import Settings
from forecast_eval.errors import AuthError
from forecast_eval.llm import AuthError as LLMAuthError
from forecast_eval.types import QFilter


QUESTION_TYPE_CHOICES = ("yes_no", "binary_named", "multiple_choice")
CHOICE_TYPE_CHOICES = ("single", "multi")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluation",
        description="Run LLM forecast evaluation against forecast_eval_set.db.",
    )
    parser.add_argument(
        "--question-type",
        action="append",
        choices=QUESTION_TYPE_CHOICES,
        help="Filter by question_type; pass multiple times to allow several.",
    )
    parser.add_argument(
        "--choice-type",
        action="append",
        choices=CHOICE_TYPE_CHOICES,
        help="Filter by choice_type; pass multiple times to allow several.",
    )
    return parser.parse_args(argv)


def _configure_logging(log_dir: Path, run_id: str, level: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / f"{run_id}.log",
        level="DEBUG",
        rotation="100 MB",
        retention=5,
    )


async def _run_async(settings: Settings, filters: QFilter, run_id: str) -> int:
    source_path = settings.source_db_path()
    results_path = settings.results_db_path()

    if not source_path.exists():
        logger.error("source DB not found at {}", source_path)
        return 2

    conn = dbmod.connect(results_path)
    dbmod.init_schema(conn)

    templates = loader.sync_prompt_templates(source_path, conn)
    questions = loader.sync_questions(source_path, conn, filters)
    if not questions:
        logger.error("no questions matched the filter; aborting")
        return 3

    raw_features = loader.load_raw_features_json(source_path)
    source_hash = dbmod.compute_source_db_hash(source_path)
    metadata_hash = dbmod.compute_metadata_hash(raw_features)
    templates_hash = dbmod.compute_prompt_templates_hash(templates)

    filters_snapshot: dict[str, Any] = {
        **filters.snapshot(),
        "question_count": len(questions),
        "question_ids": [q.id for q in questions],
    }

    dbmod.register_run(
        conn,
        run_id=run_id,
        filters_snapshot=filters_snapshot,
        config_snapshot=dbmod.snapshot_settings(settings),
        source_db_hash=source_hash,
        metadata_hash=metadata_hash,
        prompt_templates_hash=templates_hash,
    )

    try:
        stats = await runner.run(
            settings=settings,
            filters=filters,
            questions=questions,
            templates=templates,
            run_id=run_id,
            conn=conn,
        )
    except (AuthError, LLMAuthError) as e:
        logger.error("AUTH error, aborting run: {}", e)
        return 4

    logger.info(
        "[run={}] finished. total={} done={} skipped_cutoff={} errors={}",
        run_id,
        stats.total,
        stats.done,
        stats.skipped_cutoff,
        stats.errors,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        settings = Settings()
    except Exception as exc:
        print(f"[evaluation] failed to load .env: {exc}", file=sys.stderr)
        return 2

    filters = QFilter(
        question_types=frozenset(args.question_type) if args.question_type else None,
        choice_types=frozenset(args.choice_type) if args.choice_type else None,
    )

    run_id = settings.RUN_ID or runner.generate_run_id()
    _configure_logging(settings.log_dir_path(), run_id, settings.LOG_LEVEL)
    logger.info("run_id={} filters={}", run_id, filters.snapshot())

    def _handle_sigint(signum: int, frame: Any) -> None:  # noqa: ANN401
        logger.warning("SIGINT received; attempting graceful shutdown...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        return asyncio.run(_run_async(settings, filters, run_id))
    except KeyboardInterrupt:
        logger.warning("interrupted; some rows may not have been persisted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
