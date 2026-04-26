"""Tests for grid-search plots in `scripts/plot_analysis.py`.

Phase 2 of the `react-tavily-grid-search` change. Skipped when matplotlib
or Pillow aren't installed (neither is in the core conda environment), so
CI stays green; locally these tests double as paper-figure regression
guards.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

# Matplotlib + Pillow are optional plot-time deps; if either is missing the
# whole module is skipped (matches the lazy-import contract in
# `scripts/plot_analysis.py`).
pytest.importorskip("matplotlib")
pytest.importorskip("PIL")

# scripts/ is not a package — drop it on `sys.path` once for the file.
_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from PIL import Image  # noqa: E402

import plot_analysis  # noqa: E402


# `_GRID_SUMMARY_HEADER` from `forecast_eval.analysis.grid` — duplicated
# here so the test file is independent of the producer module's import
# path. Out-of-sync columns will surface immediately as `KeyError` from
# the plot loaders.
_GRID_SUMMARY_HEADER = (
    "real_model",
    "R",
    "C",
    "n_eligible",
    "n_total",
    "acc_mean",
    "acc_ci_lo",
    "acc_ci_hi",
    "bi_mean",
    "bi_ci_lo",
    "bi_ci_hi",
    "nll_mean",
    "ece",
    "mean_search_calls",
    "mean_latency_ms",
    "parse_ok_rate",
    "belief_parse_ok_rate",
)


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def _make_grid_cell(
    *,
    real_model: str,
    R: int,
    C: int,
    bi: float,
    bi_lo: float | None = None,
    bi_hi: float | None = None,
    mean_search: float = 1.0,
    mean_lat: float = 100.0,
    acc: float = 0.5,
    n_eligible: int = 10,
    n_total: int = 10,
) -> list:
    if bi_lo is None:
        bi_lo = max(0.0, bi - 5.0)
    if bi_hi is None:
        bi_hi = min(100.0, bi + 5.0)
    return [
        real_model, R, C,
        n_eligible, n_total,
        round(acc, 4),
        round(max(0.0, acc - 0.05), 4),
        round(min(1.0, acc + 0.05), 4),
        round(bi, 4),
        round(bi_lo, 4),
        round(bi_hi, 4),
        0.5,  # nll
        None,  # ece
        round(mean_search, 2),
        round(mean_lat, 1),
        0.95,  # parse_ok_rate
        0.93,  # belief_parse_ok_rate
    ]


def _build_multi_cell_run(tmp_path: Path) -> Path:
    """2 real_models × 2 R × 2 C grid with a manifest carrying `grid` block."""
    run_dir = tmp_path / "runs" / "multi-cell"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True)
    cells = []
    real_models = ["model_a", "model_b"]
    r_list = [5, 10]
    c_list = [1, 3]
    for rm_idx, rm in enumerate(real_models):
        for R in r_list:
            for C in c_list:
                base_bi = 30.0 + rm_idx * 5.0 + (R - 5) * 0.5 + C * 1.5
                cells.append(_make_grid_cell(
                    real_model=rm, R=R, C=C, bi=base_bi,
                    mean_search=float(C),
                    mean_lat=100.0 + C * 50,
                ))
    _write_csv(
        analysis_dir / "grid_summary.csv",
        list(_GRID_SUMMARY_HEADER), cells,
    )

    pareto_header = [
        "real_model", "R", "C", "mean_search_calls", "bi_mean", "dominated_by",
    ]
    pareto_rows = []
    for c in cells:
        rm, R, C = c[0], c[1], c[2]
        msc, bi = c[13], c[8]
        # Synthetic Pareto: every cell at C == max(c_list) is on the frontier;
        # cells at C == min(c_list) are dominated by their model's C-max cell.
        is_pareto = C == max(c_list)
        dominator = "" if is_pareto else f"{rm}::r{R}::c{max(c_list)}"
        pareto_rows.append([rm, R, C, msc, bi, dominator])
    _write_csv(
        analysis_dir / "grid_pareto.csv",
        pareto_header, pareto_rows,
    )

    winrate_header = [
        "model_a", "model_b",
        "total_cells", "wins_a", "wins_b", "ties",
        "sig_cells_a", "sig_cells_b",
    ]
    _write_csv(
        analysis_dir / "grid_winrate.csv",
        winrate_header,
        [["model_a", "model_b", 4, 1, 3, 0, 0, 2]],
    )

    manifest = {
        "run_id": "multi-cell",
        "schema_version": 4,
        "grid": {
            "r_list": r_list,
            "c_list": c_list,
            "default_r": r_list[0],
            "default_c": c_list[0],
            "real_models": real_models,
            "n_cells": len(real_models) * len(r_list) * len(c_list),
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    return run_dir


def _build_single_cell_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "runs" / "single-cell"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True)
    cells = [_make_grid_cell(real_model="model_a", R=5, C=3, bi=25.0)]
    _write_csv(
        analysis_dir / "grid_summary.csv",
        list(_GRID_SUMMARY_HEADER), cells,
    )
    _write_csv(
        analysis_dir / "grid_pareto.csv",
        ["real_model", "R", "C", "mean_search_calls", "bi_mean", "dominated_by"],
        [["model_a", 5, 3, 3.0, 25.0, ""]],
    )
    # No winrate.csv: winrate matrix needs >=2 real_models.
    manifest = {
        "run_id": "single-cell",
        "schema_version": 4,
        "grid": {
            "r_list": [5],
            "c_list": [3],
            "default_r": 5,
            "default_c": 3,
            "real_models": ["model_a"],
            "n_cells": 1,
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    return run_dir


def _build_legacy_run(tmp_path: Path) -> Path:
    """v4 run without a `grid` block in manifest — emulates pre-change runs."""
    run_dir = tmp_path / "runs" / "legacy-v4"
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True)
    manifest = {
        "run_id": "legacy-v4",
        "schema_version": 4,
        # No "grid" block.
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    return run_dir


def _png_is_decodable(path: Path) -> bool:
    with Image.open(path) as img:
        img.verify()
    return True


# --------------------------------------------------------------------------- #
# Direct plot function tests (13.1)
# --------------------------------------------------------------------------- #


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def test_plot_pareto_frontier_writes_png(tmp_path: Path):
    plt = _plt()
    rows = [
        _make_grid_cell(real_model="model_a", R=5, C=1, bi=20.0, mean_search=1.0)
        + [],  # already complete
    ]
    rows = [
        dict(zip(_GRID_SUMMARY_HEADER, [str(v) if v is not None else "" for v in r]))
        for r in [
            _make_grid_cell(real_model="model_a", R=5, C=1, bi=20.0, mean_search=1.0),
            _make_grid_cell(real_model="model_a", R=5, C=3, bi=28.0, mean_search=3.0),
            _make_grid_cell(real_model="model_b", R=5, C=1, bi=22.0, mean_search=1.0),
            _make_grid_cell(real_model="model_b", R=5, C=3, bi=25.0, mean_search=3.0),
        ]
    ]
    pareto_keys = {("model_a", 5, 3), ("model_b", 5, 3)}
    out = tmp_path / "pareto.png"
    plot_analysis.plot_pareto_frontier(
        plt, rows, pareto_keys, fix_R=5, out_path=out, default_r=5,
    )
    assert out.exists()
    assert out.stat().st_size > 1024
    assert _png_is_decodable(out)


def test_plot_grid_heatmap_writes_png(tmp_path: Path):
    plt = _plt()
    rows = [
        dict(zip(_GRID_SUMMARY_HEADER, [str(v) if v is not None else "" for v in r]))
        for r in [
            _make_grid_cell(real_model="model_a", R=5, C=1, bi=20.0),
            _make_grid_cell(real_model="model_a", R=5, C=3, bi=28.0),
            _make_grid_cell(real_model="model_a", R=10, C=1, bi=22.0),
            _make_grid_cell(real_model="model_a", R=10, C=3, bi=30.0),
        ]
    ]
    out = tmp_path / "heatmap.png"
    plot_analysis.plot_grid_heatmap(
        plt, rows, real_model="model_a", out_path=out,
        bi_vmin=0.0, bi_vmax=50.0,
    )
    assert out.exists()
    assert out.stat().st_size > 1024
    assert _png_is_decodable(out)


def test_plot_marginal_curves_writes_png(tmp_path: Path):
    plt = _plt()
    rows = [
        dict(zip(_GRID_SUMMARY_HEADER, [str(v) if v is not None else "" for v in r]))
        for r in [
            _make_grid_cell(real_model="model_a", R=5, C=1, bi=20.0, mean_search=1.0),
            _make_grid_cell(real_model="model_a", R=5, C=3, bi=28.0, mean_search=3.0),
            _make_grid_cell(real_model="model_b", R=5, C=1, bi=22.0, mean_search=1.0),
            _make_grid_cell(real_model="model_b", R=5, C=3, bi=25.0, mean_search=3.0),
        ]
    ]
    out = tmp_path / "curves.png"
    plot_analysis.plot_marginal_curves(
        plt, rows,
        axis="C",
        real_models=["model_a", "model_b"],
        fixed_other={"R": 5},
        out_path=out,
    )
    assert out.exists()
    assert out.stat().st_size > 1024
    assert _png_is_decodable(out)


def test_plot_winrate_matrix_writes_png(tmp_path: Path):
    plt = _plt()
    rows = [
        {
            "model_a": "model_a", "model_b": "model_b",
            "total_cells": "4", "wins_a": "1", "wins_b": "3", "ties": "0",
            "sig_cells_a": "0", "sig_cells_b": "2",
        },
    ]
    out = tmp_path / "winrate.png"
    plot_analysis.plot_winrate_matrix(plt, rows, out_path=out)
    assert out.exists()
    assert out.stat().st_size > 1024
    assert _png_is_decodable(out)


# --------------------------------------------------------------------------- #
# Full render_all integration (13.2 / 13.3 / 13.4)
# --------------------------------------------------------------------------- #


def test_render_all_multi_cell_emits_complete_grid_family(tmp_path: Path):
    run_dir = _build_multi_cell_run(tmp_path)
    written = plot_analysis.render_all(run_dir)
    figs_dir = run_dir / "analysis" / "figs"
    grid_pngs = sorted(figs_dir.glob("grid_*.png"))
    names = {p.name for p in grid_pngs}
    # 1 main pareto + |R|-1 secondary pareto (|R|=2 → 1 secondary file: R=10
    # since default_r=5 is the main figure) + |M|=2 heatmaps + 2 curve
    # families + 1 winrate = 1 + 1 + 2 + 2 + 1 = 7 PNGs.
    assert "grid_pareto_C.png" in names
    assert "grid_pareto_C_R10.png" in names
    assert "grid_heatmap_RC_model_a.png" in names
    assert "grid_heatmap_RC_model_b.png" in names
    assert "grid_curve_C.png" in names
    assert "grid_curve_R.png" in names
    assert "grid_winrate_matrix.png" in names
    assert len(grid_pngs) == 7
    for png in grid_pngs:
        assert png.stat().st_size > 1024
        assert _png_is_decodable(png)
    assert all(p in written for p in grid_pngs)


def test_render_all_single_cell_skips_extras(tmp_path: Path):
    run_dir = _build_single_cell_run(tmp_path)
    plot_analysis.render_all(run_dir)
    figs_dir = run_dir / "analysis" / "figs"
    grid_pngs = {p.name for p in figs_dir.glob("grid_*.png")}
    # Single-cell main pareto still renders (one data point, no CI band);
    # secondary pareto is skipped because default_r is the only R; one
    # heatmap is fine (1×1 grid is degenerate but `imshow` accepts it);
    # curve panels still render (single x-point per panel) but their main
    # value is in multi-cell runs. Winrate matrix is skipped because
    # there's only one real_model.
    assert "grid_pareto_C.png" in grid_pngs
    assert "grid_winrate_matrix.png" not in grid_pngs
    assert not any(name.startswith("grid_pareto_C_R") for name in grid_pngs)


def test_render_all_legacy_run_emits_no_grid_pngs(tmp_path: Path):
    run_dir = _build_legacy_run(tmp_path)
    plot_analysis.render_all(run_dir)
    figs_dir = run_dir / "analysis" / "figs"
    grid_pngs = list(figs_dir.glob("grid_*.png"))
    # Legacy v4 manifest has no `grid` key — Phase 2 grid family must stay
    # silent; existing v3/v4 plots may still render (or skip — they're
    # gated by their own input CSVs).
    assert grid_pngs == []
