#!/usr/bin/env python3
"""One-shot migration: split `multiple_choice_output_format` into
`multiple_choice_single_output_format` and `multiple_choice_multi_output_format`
inside `dataset_metadata.features_json.prompt_reconstruction`.

Why: the old single template was identical for `choice_type=single` and
`choice_type=multi` and explicitly told the LLM "list all correct option(s)
... \boxed{B, C} for multiple correct options", which produced two failure
modes seen in run 20260426-062100-8570:
    - single questions: model emitted multiple letters, judged wrong
    - multi questions:  model emitted only one letter, judged wrong

The render_user_prompt logic now dispatches on `q.choice_type` and reads one
of the two new keys. This script rewrites the source DBs in place. It's
idempotent: if the new keys are already present and the old key is gone, it's
a no-op.

Run once, then commit the updated .db files. Note this changes
`prompt_templates_hash`, so new runs are not directly comparable to runs
made before the migration.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


SINGLE_TEMPLATE = """\
This is a SINGLE-ANSWER question: exactly ONE of the listed options is correct.
Your prediction will be scored on strict equality with the unique correct letter; choosing the wrong letter, or selecting more than one letter, scores zero.
Your final answer MUST end with this exact format:
the single correct letter inside the box, e.g. \\boxed{A}.
Do NOT list more than one letter, even if you believe two outcomes are tied — pick the one you find most likely."""


MULTI_TEMPLATE = """\
This is a MULTI-SELECT question: ONE OR MORE of the listed options can be correct.
Your prediction will be scored on strict equality with the FULL set of correct letters: any extra letter, any missing letter, or any wrong letter scores zero. You must include ALL correct options and NO incorrect options.
Your final answer MUST end with this exact format:
listing all correct option(s) you have identified, separated by commas, within the box.
For example: \\boxed{A} for a single correct option, or \\boxed{B, C} for multiple correct options."""


def migrate_one(db_path: Path, *, dry_run: bool) -> str:
    if not db_path.exists():
        return f"SKIP (missing): {db_path}"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT rowid, features_json FROM dataset_metadata").fetchall()
        if not rows:
            return f"SKIP (empty dataset_metadata): {db_path}"
        if len(rows) != 1:
            return f"SKIP (unexpected {len(rows)} rows): {db_path}"

        row = rows[0]
        features = json.loads(row["features_json"])
        recon = features.get("prompt_reconstruction")
        if not isinstance(recon, dict):
            return f"SKIP (no prompt_reconstruction): {db_path}"

        already = (
            "multiple_choice_single_output_format" in recon
            and "multiple_choice_multi_output_format" in recon
            and "multiple_choice_output_format" not in recon
        )
        if already:
            return f"NOOP (already migrated): {db_path}"

        recon.pop("multiple_choice_output_format", None)
        recon["multiple_choice_single_output_format"] = SINGLE_TEMPLATE
        recon["multiple_choice_multi_output_format"] = MULTI_TEMPLATE
        features["prompt_reconstruction"] = recon

        new_json = json.dumps(features, ensure_ascii=False)
        if dry_run:
            return f"DRY-RUN would update: {db_path}"
        conn.execute(
            "UPDATE dataset_metadata SET features_json = ? WHERE rowid = ?",
            (new_json, row["rowid"]),
        )
        conn.commit()
        return f"UPDATED: {db_path}"
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended action; do not write.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[
            Path("forecast_eval_set.db"),
            Path("forecast_eval_set_example.db"),
            Path("forecast_eval_set_smoke.db"),
        ],
        help="Source DBs to migrate (defaults to the three repo DBs).",
    )
    args = parser.parse_args(argv)

    rc = 0
    for p in args.paths:
        try:
            print(migrate_one(p, dry_run=args.dry_run))
        except Exception as exc:  # noqa: BLE001 — surface per-file failure
            print(f"ERROR ({p}): {exc!r}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
