#!/usr/bin/env python3
"""Forecast evaluation entry point.

Usage:
    python evaluation.py
    python evaluation.py --question-type yes_no
    python evaluation.py --question-type multiple_choice --choice-type multi
    python evaluation.py --skip-analysis                  # run only, no post-hoc stats

All runtime knobs live in `.env`; the CLI only filters questions.

Layout produced per run:
    {RUNS_ROOT}/{run_id}/
        manifest.json          # run-level metadata + per-model file map
        db/
            <model_slug>.db    # one SQLite per model (wide run_results table)
        analysis/              # CSV/MD/JSON written after the run finishes
        logs/
            {run_id}.log
"""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sqlite3
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from forecast_eval import analysis
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
        description="Run LLM forecast evaluation against forecast_eval_set_example.db.",
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
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip the post-run statistics pass; raw DBs still land in db/.",
    )
    return parser.parse_args(argv)


def _configure_logging(log_file: Path, level: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        level="DEBUG",
        rotation="100 MB",
        retention=5,
    )


def _write_manifest(
    manifest_path: Path,
    *,
    run_id: str,
    settings: Settings,
    filters: QFilter,
    question_count: int,
    question_ids: list[str],
    model_files: dict[str, str],
    source_db_hash: str,
    metadata_hash: str,
    prompt_templates_hash: str,
    started_at: str,
) -> None:
    payload: dict[str, Any] = {
        "run_id": run_id,
        "schema_version": dbmod.SCHEMA_VERSION,
        "sampling_n": settings.SAMPLING_N,
        "models": list(settings.MODELS),
        "model_files": model_files,
        "model_training_cutoffs": {
            m: d.isoformat() for m, d in settings.MODEL_TRAINING_CUTOFFS.items()
        },
        "filters": {
            **filters.snapshot(),
            "question_count": question_count,
            "question_ids": question_ids,
        },
        "hashes": {
            "source_db": source_db_hash,
            "metadata": metadata_hash,
            "prompt_templates": prompt_templates_hash,
        },
        "started_at": started_at,
        "finished_at": None,
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _finalise_manifest(manifest_path: Path, finished_at: str) -> None:
    if not manifest_path.exists():
        return
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    data["finished_at"] = finished_at
    manifest_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _init_model_db(
    *,
    db_path: Path,
    settings: Settings,
    run_id: str,
    model: str,
    source_path: Path,
    filters: QFilter,
    config_snapshot: dict[str, Any],
    filters_snapshot: dict[str, Any],
    source_db_hash: str,
    metadata_hash: str,
    prompt_templates_hash: str,
) -> tuple[sqlite3.Connection, dict[str, str], list]:
    conn = dbmod.connect(db_path)
    dbmod.init_schema(conn, settings.SAMPLING_N)
    templates = loader.sync_prompt_templates(source_path, conn)
    questions = loader.sync_questions(source_path, conn, filters, table=settings.SOURCE_TABLE)
    cutoff = settings.MODEL_TRAINING_CUTOFFS.get(model)
    dbmod.register_run_meta(
        conn,
        run_id=run_id,
        model=model,
        sampling_n=settings.SAMPLING_N,
        filters_snapshot=filters_snapshot,
        config_snapshot=config_snapshot,
        source_db_hash=source_db_hash,
        metadata_hash=metadata_hash,
        prompt_templates_hash=prompt_templates_hash,
        training_cutoff=cutoff.isoformat() if cutoff else None,
    )
    return conn, templates, questions


async def _run_async(
    settings: Settings,
    filters: QFilter,
    run_id: str,
    run_dir: Path,
    skip_analysis: bool,
) -> int:
    source_path = settings.source_db_path()
    if not source_path.exists():
        logger.error("source DB not found at {}", source_path)
        return 2

    db_dir = run_dir / "db"
    analysis_dir = run_dir / "analysis"
    logs_dir = run_dir / "logs"
    for d in (db_dir, analysis_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Peek at questions once to compute filters_snapshot + question_count. We
    # use a scratch in-memory DB so the loader logic stays unchanged; each
    # per-model DB below will re-sync the questions into its own file.
    scratch = dbmod.connect(":memory:")
    dbmod.init_schema(scratch, settings.SAMPLING_N)
    templates_preview = loader.sync_prompt_templates(source_path, scratch)
    questions_preview = loader.sync_questions(source_path, scratch, filters, table=settings.SOURCE_TABLE)
    scratch.close()

    if not questions_preview:
        logger.error("no questions matched the filter; aborting")
        return 3

    raw_features = loader.load_raw_features_json(source_path)
    source_hash = dbmod.compute_source_db_hash(source_path)
    metadata_hash = dbmod.compute_metadata_hash(raw_features)
    templates_hash = dbmod.compute_prompt_templates_hash(templates_preview)

    filters_snapshot: dict[str, Any] = {
        **filters.snapshot(),
        "question_count": len(questions_preview),
        "question_ids": [q.id for q in questions_preview],
    }
    config_snapshot = dbmod.snapshot_settings(settings)
    started_at = dbmod.utcnow_iso()

    model_files: dict[str, str] = {}
    conns: dict[str, sqlite3.Connection] = {}
    templates_by_model: dict[str, dict[str, str]] = {}
    questions_by_model: dict[str, list] = {}

    try:
        for model in settings.MODELS:
            slug = dbmod.model_slug_safe(model)
            db_path = db_dir / f"{slug}.db"
            model_files[model] = db_path.name
            conn, templates, questions = _init_model_db(
                db_path=db_path,
                settings=settings,
                run_id=run_id,
                model=model,
                source_path=source_path,
                filters=filters,
                config_snapshot=config_snapshot,
                filters_snapshot=filters_snapshot,
                source_db_hash=source_hash,
                metadata_hash=metadata_hash,
                prompt_templates_hash=templates_hash,
            )
            conns[model] = conn
            templates_by_model[model] = templates
            questions_by_model[model] = questions

        manifest_path = run_dir / "manifest.json"
        _write_manifest(
            manifest_path,
            run_id=run_id,
            settings=settings,
            filters=filters,
            question_count=len(questions_preview),
            question_ids=[q.id for q in questions_preview],
            model_files=model_files,
            source_db_hash=source_hash,
            metadata_hash=metadata_hash,
            prompt_templates_hash=templates_hash,
            started_at=started_at,
        )

        primary_model = settings.MODELS[0]
        templates = templates_by_model[primary_model]
        questions = questions_by_model[primary_model]

        try:
            stats = await runner.run(
                settings=settings,
                filters=filters,
                questions=questions,
                templates=templates,
                run_id=run_id,
                conns=conns,
            )
        except (AuthError, LLMAuthError) as e:
            logger.error("AUTH error, aborting run: {}", e)
            return 4

        finished_at = dbmod.utcnow_iso()
        _finalise_manifest(manifest_path, finished_at)

        logger.info(
            "[run={}] finished. total={} done={} skipped_cutoff={} errors={}",
            run_id,
            stats.total,
            stats.done,
            stats.skipped_cutoff,
            stats.errors,
        )
    finally:
        for conn in conns.values():
            try:
                conn.close()
            except Exception:  # noqa: BLE001 — best-effort close
                logger.exception("failed to close a model DB connection")

    if skip_analysis:
        logger.info("[run={}] --skip-analysis: analysis/ not generated", run_id)
    else:
        try:
            written = analysis.run_analysis(run_dir)
            logger.info(
                "[run={}] analysis written: {}",
                run_id,
                ", ".join(p.name for p in written),
            )
        except Exception:  # noqa: BLE001 — analysis must never fail the run
            logger.exception("analysis pass failed; raw DBs are intact")

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
    run_dir = settings.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(run_dir / "logs" / f"{run_id}.log", settings.LOG_LEVEL)
    logger.info("run_id={} dir={} filters={}", run_id, run_dir, filters.snapshot())

    def _handle_sigint(signum: int, frame: Any) -> None:  # noqa: ANN401
        logger.warning("SIGINT received; attempting graceful shutdown...")
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        return asyncio.run(
            _run_async(settings, filters, run_id, run_dir, args.skip_analysis)
        )
    except KeyboardInterrupt:
        logger.warning("interrupted; some rows may not have been persisted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
