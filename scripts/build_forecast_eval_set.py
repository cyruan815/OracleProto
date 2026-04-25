"""Build a curated `forecast_eval_set.db` from `forecast_eval_set_example.db`.

Run from the repo root:

    python scripts/build_forecast_eval_set.py

Refuses to overwrite an existing destination DB; delete it manually first
if you want to regenerate.

Sampling design (target N=90):

* Time floor `end_time >= 2026-03-01` — keeps the recent half of the
  3-month example span, biasing the curated set toward events that fall
  after typical model training cutoffs.
* Stratified by `(question_type, choice_type)` per `QUOTA` below.  The
  ratios approximate the example DB's overall distribution.
* Topic diversification for `mc/single` and `yes_no`: cap `TOPIC_CAP=3`
  per event-prefix (first `PREFIX_LEN` chars, lower-cased).  Prevents
  one trending topic (Oscars / elections / playoffs) from monopolising
  the set.  `mc/multi` and `binary_named` keep tiny candidate pools, so
  they skip the cap.
* `random.Random(SEED)` makes the output deterministic; bump `SEED` if
  you want a fresh sample.
* Output table is `forecast_eval_set` (no `_example` suffix), with the
  same 7-column schema + 3 indexes + a `dataset_metadata` row.  `.env`
  consumers switch datasets via `SOURCE_DB` + `SOURCE_TABLE`.
"""
from __future__ import annotations

import hashlib
import random
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC_DB = REPO / "forecast_eval_set_example.db"
SRC_TABLE = "forecast_eval_set_example"
DST_DB = REPO / "forecast_eval_set.db"
DST_TABLE = "forecast_eval_set"

TIME_FLOOR = "2026-03-01"
QUOTA = {
    ("multiple_choice", "single"): 50,
    ("yes_no", "single"): 25,
    ("multiple_choice", "multi"): 10,
    ("binary_named", "single"): 5,
}
TOPIC_CAP = 3
PREFIX_LEN = 50
TOPIC_CAP_STRATA = {("multiple_choice", "single"), ("yes_no", "single")}
SEED = 42


def main() -> None:
    if not SRC_DB.exists():
        print(f"source DB missing: {SRC_DB}", file=sys.stderr)
        sys.exit(1)
    if DST_DB.exists():
        print(f"refusing to overwrite existing {DST_DB}", file=sys.stderr)
        sys.exit(1)

    src = sqlite3.connect(SRC_DB)
    src.row_factory = sqlite3.Row

    rng = random.Random(SEED)
    selected: list[dict] = []
    print("=== sampling ===")
    for (qt, ct), n in QUOTA.items():
        rows = src.execute(
            f"SELECT id, choice_type, question_type, event, options, answer, end_time "
            f"FROM {SRC_TABLE} "
            f"WHERE question_type=? AND choice_type=? AND end_time >= ? "
            f"ORDER BY end_time DESC, id",
            (qt, ct, TIME_FLOOR),
        ).fetchall()
        candidates = [dict(r) for r in rows]

        if len(candidates) <= n:
            picked = candidates
        elif (qt, ct) in TOPIC_CAP_STRATA:
            shuffled = list(candidates)
            rng.shuffle(shuffled)
            topic_count: Counter = Counter()
            picked = []
            for q in shuffled:
                prefix = q["event"][:PREFIX_LEN].lower().strip()
                if topic_count[prefix] >= TOPIC_CAP:
                    continue
                picked.append(q)
                topic_count[prefix] += 1
                if len(picked) >= n:
                    break
            if len(picked) < n:
                picked_ids = {q["id"] for q in picked}
                for q in shuffled:
                    if q["id"] in picked_ids:
                        continue
                    picked.append(q)
                    if len(picked) >= n:
                        break
        else:
            shuffled = list(candidates)
            rng.shuffle(shuffled)
            picked = shuffled[:n]

        print(f"  {qt}/{ct}: candidates={len(candidates)}, picked={len(picked)}")
        selected.extend(picked)

    target_total = sum(QUOTA.values())
    assert len(selected) == target_total, f"got {len(selected)} != target {target_total}"
    selected.sort(key=lambda q: (q["end_time"], q["id"]))

    src_meta = dict(src.execute("SELECT * FROM dataset_metadata").fetchone())
    features_json = src_meta["features_json"]

    dst = sqlite3.connect(DST_DB)
    dst.row_factory = sqlite3.Row
    dst.executescript(
        f"""
        CREATE TABLE {DST_TABLE} (
            id TEXT PRIMARY KEY,
            choice_type TEXT NOT NULL,
            question_type TEXT NOT NULL,
            event TEXT NOT NULL,
            options TEXT NOT NULL,
            answer TEXT NOT NULL,
            end_time TEXT NOT NULL
        );
        CREATE INDEX idx_{DST_TABLE}_choice_type ON {DST_TABLE}(choice_type);
        CREATE INDEX idx_{DST_TABLE}_question_type ON {DST_TABLE}(question_type);
        CREATE INDEX idx_{DST_TABLE}_end_time ON {DST_TABLE}(end_time);
        CREATE TABLE dataset_metadata (
            dataset_name TEXT NOT NULL,
            split_name TEXT NOT NULL,
            table_name TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            imported_at_utc TEXT NOT NULL,
            features_json TEXT NOT NULL
        );
        """
    )

    dst.executemany(
        f"INSERT INTO {DST_TABLE} "
        f"(id, choice_type, question_type, event, options, answer, end_time) "
        f"VALUES (?,?,?,?,?,?,?)",
        [
            (
                q["id"],
                q["choice_type"],
                q["question_type"],
                q["event"],
                q["options"],
                q["answer"],
                q["end_time"],
            )
            for q in selected
        ],
    )

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    dst.execute(
        "INSERT INTO dataset_metadata "
        "(dataset_name, split_name, table_name, row_count, imported_at_utc, features_json) "
        "VALUES (?,?,?,?,?,?)",
        (
            "forecast_eval_set",
            "train",
            DST_TABLE,
            len(selected),
            now_iso,
            features_json,
        ),
    )
    dst.commit()

    print("\n=== verification ===")
    total = dst.execute(f"SELECT COUNT(*) FROM {DST_TABLE}").fetchone()[0]
    print(f"row_count: {total}")
    for r in dst.execute(
        f"SELECT question_type, choice_type, COUNT(*) AS n FROM {DST_TABLE} "
        f"GROUP BY question_type, choice_type ORDER BY question_type, choice_type"
    ):
        print(f"  {dict(r)}")
    rng_row = dict(
        dst.execute(
            f"SELECT MIN(end_time) AS mn, MAX(end_time) AS mx, "
            f"COUNT(DISTINCT end_time) AS days FROM {DST_TABLE}"
        ).fetchone()
    )
    print(f"end_time: min={rng_row['mn']}, max={rng_row['mx']}, distinct_days={rng_row['days']}")

    by_month = dict(
        dst.execute(
            f"SELECT substr(end_time,1,7) AS ym, COUNT(*) AS n FROM {DST_TABLE} "
            f"GROUP BY ym ORDER BY ym"
        ).fetchall()
    )
    print(f"by_month: {dict(by_month)}")

    meta = dict(dst.execute("SELECT * FROM dataset_metadata").fetchone())
    print(f"metadata.dataset_name={meta['dataset_name']}, table_name={meta['table_name']}, "
          f"row_count={meta['row_count']}, imported_at_utc={meta['imported_at_utc']}")

    dst.close()
    src.close()

    size = DST_DB.stat().st_size
    sha = hashlib.sha256(DST_DB.read_bytes()).hexdigest()
    print(f"\nfile: {DST_DB.name}, size={size}B, sha256={sha}")


if __name__ == "__main__":
    main()
