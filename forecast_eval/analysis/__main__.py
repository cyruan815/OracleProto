"""CLI entry: `python -m forecast_eval.analysis RUNS_ROOT/{run_id}`.

Kept as a thin wrapper so `run_analysis` stays importable without argparse
overhead. Behaviour matches the v3 single-file `analysis.py` CLI byte-for-byte.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import run_analysis


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="forecast_eval.analysis",
        description="Compute statistics for one completed evaluation run.",
    )
    parser.add_argument("run_dir", help="Path to a RUNS_ROOT/{run_id} directory.")
    args = parser.parse_args(argv)
    paths = run_analysis(Path(args.run_dir))
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
