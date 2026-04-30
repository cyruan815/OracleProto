"""CLI entry: `python -m forecast_eval.analysis RUNS_ROOT/{run_id}`.

Kept as a thin wrapper so `run_analysis` stays importable without argparse
overhead. composite-score-by-subtype: when ``.env`` is readable, pass the
subtype weights through to ``run_analysis``; otherwise fall back to default
weights (synonymous with ``Settings`` defaults).
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
    kwargs: dict[str, object] = {}
    try:
        from ..config import load_settings

        cfg = load_settings()
    except Exception:  # noqa: BLE001 — covers missing .env or unset LLM_API_KEY
        cfg = None
    if cfg is not None:
        kwargs.update(
            composite_weights_qtype=cfg.COMPOSITE_WEIGHTS_QTYPE,
            composite_weights_ctype=cfg.COMPOSITE_WEIGHTS_CTYPE,
            composite_overrides_qtype=cfg.COMPOSITE_WEIGHT_OVERRIDES_QTYPE,
            composite_overrides_ctype=cfg.COMPOSITE_WEIGHT_OVERRIDES_CTYPE,
        )
    paths = run_analysis(Path(args.run_dir), **kwargs)
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
