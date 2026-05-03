"""On-demand visualisation for one analysis run.

CLI:
    python scripts/plot_analysis.py runs/<run_id>

Reads `runs/<run_id>/analysis/*.{csv,json}` produced by `run_analysis` and
emits PNGs into `runs/<run_id>/analysis/figs/`. Matplotlib is loaded lazily so the
core analysis path stays free of plot-time dependencies; missing matplotlib
prints a helpful install hint and exits 1.

Each figure is best-effort: a missing source file (e.g. no `reflection_ab.csv`
because no paired runs were found) skips the corresponding plot rather than
failing the whole pipeline. This matches the design.md "on-demand plotting" decision.
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Lazy matplotlib import + small CSV helpers
# --------------------------------------------------------------------------- #


def _import_plt():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no GUI required
        import matplotlib.pyplot as plt
        return plt
    except ImportError:  # pragma: no cover — exercised manually
        print(
            "matplotlib is not installed. Install it with:\n"
            "    pip install matplotlib\n"
            "and rerun this script.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_int(s: str | None) -> int | None:
    if s is None or s == "":
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return None


# --------------------------------------------------------------------------- #
# Grid-search loaders
# --------------------------------------------------------------------------- #


def _read_grid_artifacts(analysis_dir: Path) -> dict[str, list[dict[str, str]]]:
    """Bundle every grid_*.csv into one dict for the render loop.

    Missing CSVs degrade to empty lists — multi-cell runs always have
    `grid_summary.csv`; single-cell runs only have `grid_summary.csv` (no
    pareto / winrate when there's nothing to dominate or compare).
    """
    return {
        "summary": _read_csv(analysis_dir / "grid_summary.csv"),
        "pareto": _read_csv(analysis_dir / "grid_pareto.csv"),
        "winrate": _read_csv(analysis_dir / "grid_winrate.csv"),
        "marginal_C": _read_csv(analysis_dir / "grid_marginal_C.csv"),
        "marginal_R": _read_csv(analysis_dir / "grid_marginal_R.csv"),
    }


def _grid_pareto_keys(pareto_rows: list[dict[str, str]]) -> set[tuple[str, int, int]]:
    """Return the `(real_model, R, C)` triplets on the Pareto frontier.

    `_write_grid_pareto_csv` puts every cell in the file and uses the empty
    `dominated_by` column to mark frontier membership."""
    keys: set[tuple[str, int, int]] = set()
    for row in pareto_rows:
        if (row.get("dominated_by") or "").strip():
            continue
        rm = row.get("real_model")
        R = _to_int(row.get("R"))
        C = _to_int(row.get("C"))
        if rm is None or R is None or C is None:
            continue
        keys.add((rm, R, C))
    return keys


def _real_models_in_grid(grid_rows: list[dict[str, str]]) -> list[str]:
    """Sorted, de-duplicated real_model list straight from grid_summary.csv."""
    return sorted({r["real_model"] for r in grid_rows if r.get("real_model")})


def _r_values_in_grid(grid_rows: list[dict[str, str]]) -> list[int]:
    """Sorted unique R values; skips rows with malformed R."""
    rs: set[int] = set()
    for r in grid_rows:
        v = _to_int(r.get("R"))
        if v is not None:
            rs.add(v)
    return sorted(rs)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def plot_bi_bar_with_ci(
    plt, summary_rows: list[dict], paired_rows: list[dict], out_path: Path
) -> None:
    """Horizontal bar of `bi` per model with a half-CI from paired bootstrap.

    The half-CI is averaged over all pairs the model appears in — enough to
    give a sense of "this BI carries about ±X uncertainty from question
    sampling" without committing to a specific opponent.
    """
    if not summary_rows:
        return
    bi_by_model: dict[str, float] = {}
    for r in summary_rows:
        bi = _to_float(r.get("bi") or r.get("bi_uncal"))
        if bi is not None:
            bi_by_model[r["model"]] = bi
    half_ci_by_model: dict[str, float] = {}
    for r in paired_rows:
        for col in ("model_a", "model_b"):
            m = r.get(col)
            if not m:
                continue
            ci_low = _to_float(r.get("ci_low"))
            ci_high = _to_float(r.get("ci_high"))
            if ci_low is None or ci_high is None:
                continue
            half = (ci_high - ci_low) / 2.0
            cur = half_ci_by_model.get(m, [])
            if isinstance(cur, list):
                cur.append(half)
                half_ci_by_model[m] = cur
    half_ci_by_model = {
        m: (sum(v) / len(v) if isinstance(v, list) and v else 0.0)
        for m, v in half_ci_by_model.items()
    }
    models = sorted(bi_by_model.keys(), key=lambda m: bi_by_model[m])
    if not models:
        return
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(models) + 2)))
    ys = list(range(len(models)))
    bars = [bi_by_model[m] for m in models]
    errs = [half_ci_by_model.get(m, 0.0) * 100 for m in models]  # CI is on BS scale; BI is 100*sqrt(BS)
    ax.barh(ys, bars, xerr=errs, color="#4C72B0", alpha=0.85)
    ax.set_yticks(ys)
    ax.set_yticklabels(models)
    ax.set_xlabel("Brier Index (higher better)")
    ax.set_title("BI per model (error bar = mean half-CI from paired bootstrap)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_delta_bi_forest(
    plt, paired_rows: list[dict], out_path: Path
) -> None:
    """Forest plot of pairwise ΔBI with 95% CI."""
    if not paired_rows:
        return
    # Reverse sort so largest absolute ΔBI is on top.
    rows = []
    for r in paired_rows:
        delta = _to_float(r.get("delta_bi"))
        if delta is None:
            continue
        ci_low = _to_float(r.get("ci_low_bi"))
        ci_high = _to_float(r.get("ci_high_bi"))
        rows.append({
            "label": f"{r.get('model_a', '?')} vs {r.get('model_b', '?')}",
            "delta": delta,
            "ci_low": ci_low if ci_low is not None else delta,
            "ci_high": ci_high if ci_high is not None else delta,
            "p_holm": _to_float(r.get("p_holm")) or 1.0,
        })
    if not rows:
        return
    rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
    rows = rows[:30]  # cap rows so the figure stays legible
    rows.reverse()  # plot largest at top
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(rows) + 2)))
    ys = list(range(len(rows)))
    deltas = [r["delta"] for r in rows]
    err_lo = [d - r["ci_low"] for d, r in zip(deltas, rows)]
    err_hi = [r["ci_high"] - d for d, r in zip(deltas, rows)]
    colors = ["#C44E52" if r["p_holm"] < 0.05 else "#4C72B0" for r in rows]
    ax.errorbar(deltas, ys, xerr=[err_lo, err_hi], fmt="o", color="black", ecolor=colors, capsize=3)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8)
    ax.set_xlabel("ΔBI = BI_A − BI_B (red = Holm-adj p < 0.05)")
    ax.set_title("Pairwise ΔBI forest plot")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_fss_bar_with_ci(
    plt, summary_rows: list[dict], pairwise_rows: list[dict], out_path: Path
) -> None:
    """Main figure: horizontal bar of `fss` per model with half-CI from
    the multi-metric paired bootstrap (rows where `metric == 'fss'`).
    Mirrors `plot_bi_bar_with_ci` for FSS."""
    if not summary_rows:
        return
    fss_by_model: dict[str, float] = {}
    for r in summary_rows:
        v = _to_float(r.get("fss"))
        if v is not None:
            fss_by_model[r["model"]] = v
    if not fss_by_model:
        return
    half_ci_by_model: dict[str, list[float]] = {}
    for r in pairwise_rows:
        if r.get("metric") != "fss":
            continue
        for col in ("model_a", "model_b"):
            m = r.get(col)
            if not m:
                continue
            ci_low = _to_float(r.get("ci_low"))
            ci_high = _to_float(r.get("ci_high"))
            if ci_low is None or ci_high is None:
                continue
            half = (ci_high - ci_low) / 2.0
            half_ci_by_model.setdefault(m, []).append(half)
    half_ci_avg = {
        m: (sum(v) / len(v) if v else 0.0) for m, v in half_ci_by_model.items()
    }
    models = sorted(fss_by_model.keys(), key=lambda m: fss_by_model[m])
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(models) + 2)))
    ys = list(range(len(models)))
    bars = [fss_by_model[m] for m in models]
    errs = [half_ci_avg.get(m, 0.0) for m in models]
    ax.barh(ys, bars, xerr=errs, color="#55A868", alpha=0.85)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(models)
    ax.set_xlabel("FSS (Tversky α=2 β=0.5, chance-corrected) — higher better")
    ax.set_title("FSS per model (error bar = mean half-CI from paired bootstrap)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_delta_fss_forest(
    plt, pairwise_rows: list[dict], out_path: Path
) -> None:
    """Forest plot of pairwise ΔFSS with 95% CI; rows where
    `metric == 'fss'` from `pairwise_bootstrap.csv`."""
    rows = []
    for r in pairwise_rows:
        if r.get("metric") != "fss":
            continue
        delta = _to_float(r.get("delta_mean"))
        if delta is None:
            continue
        ci_low = _to_float(r.get("ci_low"))
        ci_high = _to_float(r.get("ci_high"))
        rows.append({
            "label": f"{r.get('model_a', '?')} vs {r.get('model_b', '?')}",
            "delta": delta,
            "ci_low": ci_low if ci_low is not None else delta,
            "ci_high": ci_high if ci_high is not None else delta,
            "p_value": _to_float(r.get("p_value")) or 1.0,
        })
    if not rows:
        return
    rows.sort(key=lambda r: abs(r["delta"]), reverse=True)
    rows = rows[:30]
    rows.reverse()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(rows) + 2)))
    ys = list(range(len(rows)))
    deltas = [r["delta"] for r in rows]
    err_lo = [d - r["ci_low"] for d, r in zip(deltas, rows)]
    err_hi = [r["ci_high"] - d for d, r in zip(deltas, rows)]
    colors = ["#C44E52" if r["p_value"] < 0.05 else "#4C72B0" for r in rows]
    ax.errorbar(deltas, ys, xerr=[err_lo, err_hi], fmt="o", color="black", ecolor=colors, capsize=3)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8)
    ax.set_xlabel("ΔFSS = FSS_A − FSS_B (red = p < 0.05)")
    ax.set_title("Pairwise ΔFSS forest plot")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_entropy_accuracy_grid(
    plt, bins_rows: list[dict], out_dir: Path,
) -> list[Path]:
    """Entropy-Acc joint grid — one PNG per model with 3 buckets × 3 metrics
    (Acc / MV Acc / Fleiss κ). Per-model bucket boundaries differ."""
    if not bins_rows:
        return []
    by_model: dict[str, list[dict]] = {}
    for r in bins_rows:
        m = r.get("model")
        if not m:
            continue
        by_model.setdefault(m, []).append(r)
    written: list[Path] = []
    bucket_order = {"low": 0, "mid": 1, "high": 2}
    for model in sorted(by_model.keys()):
        rows = sorted(
            by_model[model],
            key=lambda b: bucket_order.get(b.get("bucket", ""), 99),
        )
        if not rows:
            continue
        labels = [r.get("bucket", "?") for r in rows]
        acc_vals = [_to_float(r.get("acc")) or 0.0 for r in rows]
        mv_vals = [_to_float(r.get("mv_acc")) or 0.0 for r in rows]
        fk_vals = [_to_float(r.get("fleiss_kappa")) or 0.0 for r in rows]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
        axes[0].bar(labels, acc_vals, color="#4C72B0")
        axes[0].set_title("Pass@1 Acc")
        axes[0].set_ylim(0, 1)
        axes[1].bar(labels, mv_vals, color="#55A868")
        axes[1].set_title("Majority Vote Acc")
        axes[1].set_ylim(0, 1)
        axes[2].bar(labels, fk_vals, color="#C44E52")
        axes[2].set_title("Fleiss κ")
        axes[2].set_ylim(-0.2, 1)
        for ax in axes:
            ax.set_xlabel("entropy bucket (per-model)")
        fig.suptitle(f"{model} — entropy-accuracy grid (per-model tertiles)")
        fig.tight_layout()
        safe = model.replace("/", "_").replace(":", "_")
        out_path = out_dir / f"entropy_accuracy_grid_{safe}.png"
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        if out_path.exists():
            written.append(out_path)
    return written


def plot_belief_trajectories(
    plt, run_dir: Path, out_dir: Path, n_questions: int = 5
) -> None:
    """Pick `n_questions` from the first model's DB and plot per-step max p."""
    db_dir = run_dir / "db"
    if not db_dir.exists():
        return
    db_files = sorted(db_dir.glob("*.db"))
    if not db_files:
        return
    db_path = db_files[0]
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        meta = conn.execute(
            "SELECT sampling_n, model FROM run_meta ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        if meta is None:
            return
        sampling_n = int(meta["sampling_n"])
        model = meta["model"]
        # Pick first n_questions IDs that have at least one parsed belief in s0.
        rows = conn.execute(
            "SELECT q.id FROM questions q JOIN run_results r ON q.id = r.question_id "
            "WHERE r.s0_belief_trace IS NOT NULL LIMIT ?",
            (n_questions,),
        ).fetchall()
        for q_row in rows:
            qid = q_row["id"]
            opts = conn.execute(
                "SELECT options FROM questions WHERE id = ?", (qid,)
            ).fetchone()
            try:
                options = json.loads(opts["options"]) if opts and opts["options"] else []
            except (TypeError, ValueError):
                options = []
            fig, ax = plt.subplots(figsize=(7, 4))
            for i in range(sampling_n):
                trace_raw = conn.execute(
                    f"SELECT s{i}_belief_trace FROM run_results WHERE question_id = ?",
                    (qid,),
                ).fetchone()
                if not trace_raw or not trace_raw[0]:
                    continue
                try:
                    trace = json.loads(trace_raw[0])
                except (TypeError, ValueError):
                    continue
                if not isinstance(trace, list):
                    continue
                xs: list[int] = []
                ys: list[float] = []
                for step in trace:
                    if not isinstance(step, dict):
                        continue
                    p_dict = step.get("p")
                    if not isinstance(p_dict, dict):
                        continue
                    if not p_dict:
                        continue
                    xs.append(int(step.get("step", len(xs))))
                    ys.append(max(p_dict.values()))
                if not xs:
                    continue
                ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"trial {i}")
            ax.set_xlabel("ReAct step")
            ax.set_ylabel("max p (top-1 confidence)")
            ax.set_ylim(0, 1)
            ax.set_title(f"belief trajectory: {qid} ({model})")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            safe_qid = qid.replace("/", "_").replace(" ", "_")
            fig.savefig(out_dir / f"belief_trajectory_{safe_qid}.png", dpi=140)
            plt.close(fig)
    finally:
        conn.close()


def plot_tool_pdp(plt, pdp_rows: list[dict], out_dir: Path) -> None:
    """One PNG per feature: PDP_correct (left axis) + PDP_NLL (right axis)."""
    if not pdp_rows:
        return
    by_feature: dict[str, list[dict]] = {}
    for r in pdp_rows:
        by_feature.setdefault(r["feature"], []).append(r)
    for feat, rows in by_feature.items():
        rows = sorted(rows, key=lambda r: _to_float(r["feature_value"]) or 0.0)
        models = sorted({r["model"] for r in rows})
        fig, ax_left = plt.subplots(figsize=(7, 4))
        ax_right = ax_left.twinx()
        for model in models:
            sub = [r for r in rows if r["model"] == model]
            xs = [_to_float(r["feature_value"]) for r in sub]
            ys_correct = [_to_float(r["pdp_correct"]) for r in sub]
            ys_nll = [_to_float(r["pdp_nll"]) for r in sub]
            xs_clean = [x for x, y in zip(xs, ys_correct) if x is not None and y is not None]
            ys_correct_clean = [y for y in ys_correct if y is not None]
            xs_nll_clean = [x for x, y in zip(xs, ys_nll) if x is not None and y is not None]
            ys_nll_clean = [y for y in ys_nll if y is not None]
            if xs_clean and ys_correct_clean:
                ax_left.plot(xs_clean, ys_correct_clean, marker="o",
                             label=f"{model} (P correct)")
            if xs_nll_clean and ys_nll_clean:
                ax_right.plot(xs_nll_clean, ys_nll_clean, marker="x", linestyle="--",
                              alpha=0.6, label=f"{model} (E[NLL])")
        ax_left.set_xlabel(feat)
        ax_left.set_ylabel("Pr(correct)")
        ax_right.set_ylabel("E[NLL]")
        ax_left.set_title(f"Tool-usage PDP: {feat}")
        ax_left.legend(loc="upper left", fontsize=7)
        ax_right.legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"tool_pdp_{feat}.png", dpi=140)
        plt.close(fig)


def plot_difficulty_grid(
    plt, by_diff_rows: list[dict], paired_diff_rows: list[dict], out_path: Path
) -> None:
    """Heatmap-style grid: difficulty tier × model with BI as color."""
    if not by_diff_rows:
        return
    tiers = ("low", "mid", "high")
    by_model_tier: dict[str, dict[str, float]] = {}
    for r in by_diff_rows:
        m = r.get("model")
        t = r.get("difficulty_tier") or r.get("tier") or r.get("difficulty")
        bi = _to_float(r.get("bi"))
        if not m or t not in tiers or bi is None:
            continue
        by_model_tier.setdefault(m, {})[t] = bi
    if not by_model_tier:
        return
    models = sorted(by_model_tier.keys())
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(models)), 4))
    matrix = []
    for tier in tiers:
        matrix.append([by_model_tier[m].get(tier, float("nan")) for m in models])
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(list(range(len(models))))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(list(range(len(tiers))))
    ax.set_yticklabels(tiers)
    ax.set_xlabel("model")
    ax.set_ylabel("difficulty tier (γ-tertile)")
    ax.set_title("BI by difficulty tier")
    for i in range(len(tiers)):
        for j in range(len(models)):
            v = matrix[i][j]
            if v == v:  # not NaN
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="BI")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Grid-search plots
# --------------------------------------------------------------------------- #


def plot_pareto_frontier(
    plt,
    grid_rows: list[dict[str, str]],
    pareto_keys: set[tuple[str, int, int]],
    *,
    fix_R: int,
    out_path: Path,
    default_r: int | None = None,
) -> None:
    """Cost-quality Pareto figure for one fixed R.

    For each `real_model` we draw a polyline of `(mean_search_calls, bi_mean)`
    sorted by `mean_search_calls`, plus a `fill_between` 95% CI band derived
    from `grid_summary.csv` (`bi_ci_lo` / `bi_ci_hi`). Cells whose
    `(real_model, R, C)` triplet is in `pareto_keys` get a star marker on
    top of the line so the frontier is visually distinct from the per-model
    curve. The legend is anchored outside the right edge so long real_model
    slugs (e.g. `anthropic/claude-sonnet-4.5::r5::c3`) don't squash the
    plotting area. The figure renders even when only one cell exists for
    a real_model — it just shows a single marker without a CI band.
    """
    cells = [r for r in grid_rows if _to_int(r.get("R")) == fix_R]
    if not cells:
        return
    real_models = sorted({r["real_model"] for r in cells if r.get("real_model")})
    if not real_models:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, rm in enumerate(real_models):
        sub = sorted(
            [r for r in cells if r["real_model"] == rm],
            key=lambda r: _to_float(r.get("mean_search_calls")) or 0.0,
        )
        xs: list[float] = []
        ys: list[float] = []
        ci_lo: list[float] = []
        ci_hi: list[float] = []
        keys: list[tuple[str, int, int]] = []
        for row in sub:
            x = _to_float(row.get("mean_search_calls"))
            y = _to_float(row.get("bi_mean"))
            if x is None or y is None:
                continue
            R_val = _to_int(row.get("R"))
            C_val = _to_int(row.get("C"))
            if R_val is None or C_val is None:
                continue
            xs.append(x)
            ys.append(y)
            lo = _to_float(row.get("bi_ci_lo"))
            hi = _to_float(row.get("bi_ci_hi"))
            ci_lo.append(lo if lo is not None else y)
            ci_hi.append(hi if hi is not None else y)
            keys.append((rm, R_val, C_val))
        if not xs:
            continue
        color = color_cycle[i % len(color_cycle)]
        ax.plot(xs, ys, marker="o", color=color, linewidth=1.5, label=rm)
        if len(xs) >= 2:
            ax.fill_between(xs, ci_lo, ci_hi, alpha=0.15, color=color, linewidth=0)
        for x, y, key in zip(xs, ys, keys):
            if key in pareto_keys:
                ax.plot(
                    x, y, marker="*", color=color, markersize=14,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=5,
                )
    ax.set_xlabel("average search calls per sample")
    ax.set_ylabel("Brier Index (higher is better)")
    title = f"Cost-quality Pareto (R={fix_R}"
    if default_r is not None and fix_R == default_r:
        title += ", default"
    title += ")"
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grid_heatmap(
    plt,
    grid_rows: list[dict[str, str]],
    *,
    real_model: str,
    out_path: Path,
    bi_vmin: float | None = None,
    bi_vmax: float | None = None,
) -> None:
    """(R × C) BI heatmap for a single real_model.

    Each cell shows `bi_mean` rounded to 3 decimals in the upper-right
    corner. Cells whose 95% BI CI overlaps the best cell's CI (a CI-overlap
    proxy for "paired bootstrap p > 0.05" — strict pairwise paired bs at
    plot time would re-open every .db, which violates the CSV-only
    contract of this script) are overlaid with a hatched `x` patch. The
    color range is shared across real_models via `bi_vmin` / `bi_vmax` so
    multi-model heatmaps stay comparable.
    """
    sub = [r for r in grid_rows if r.get("real_model") == real_model]
    if not sub:
        return
    rs = sorted({_to_int(r["R"]) for r in sub if _to_int(r.get("R")) is not None})
    cs = sorted({_to_int(r["C"]) for r in sub if _to_int(r.get("C")) is not None})
    if not rs or not cs:
        return
    bi_lookup: dict[tuple[int, int], dict[str, float | None]] = {}
    for row in sub:
        R = _to_int(row.get("R"))
        C = _to_int(row.get("C"))
        if R is None or C is None:
            continue
        bi_lookup[(R, C)] = {
            "bi": _to_float(row.get("bi_mean")),
            "ci_lo": _to_float(row.get("bi_ci_lo")),
            "ci_hi": _to_float(row.get("bi_ci_hi")),
        }
    matrix: list[list[float]] = []
    for R in rs:
        row_vals: list[float] = []
        for C in cs:
            cell = bi_lookup.get((R, C))
            row_vals.append(float("nan") if cell is None or cell["bi"] is None else cell["bi"])
        matrix.append(row_vals)
    flat = [v for row in matrix for v in row if v == v]
    if not flat:
        return
    vmin = bi_vmin if bi_vmin is not None else 0.0
    vmax = bi_vmax if bi_vmax is not None else max(50.0, max(flat) + 1.0)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(list(range(len(cs))))
    ax.set_xticklabels([str(c) for c in cs])
    ax.set_yticks(list(range(len(rs))))
    ax.set_yticklabels([str(r) for r in rs])
    ax.set_xlabel("C (REACT_MAX_SEARCH_CALLS)")
    ax.set_ylabel("R (TAVILY_MAX_RESULTS)")
    ax.set_title(f"BI heatmap: {real_model}")
    # Best cell = highest BI for this real_model (BI higher-is-better).
    best_key: tuple[int, int] | None = None
    best_bi = -float("inf")
    for (R, C), cell in bi_lookup.items():
        bi = cell["bi"]
        if bi is None:
            continue
        if bi > best_bi:
            best_bi = bi
            best_key = (R, C)
    best_cell = bi_lookup.get(best_key) if best_key is not None else None
    for i, R in enumerate(rs):
        for j, C in enumerate(cs):
            cell = bi_lookup.get((R, C))
            if cell is None or cell["bi"] is None:
                continue
            ax.text(
                j + 0.32, i - 0.32, f"{cell['bi']:.3f}",
                ha="left", va="top", fontsize=7,
                color="white" if cell["bi"] < (vmin + vmax) / 2 else "black",
            )
            if best_cell is None or (R, C) == best_key:
                continue
            cell_lo, cell_hi = cell["ci_lo"], cell["ci_hi"]
            best_lo, best_hi = best_cell["ci_lo"], best_cell["ci_hi"]
            if cell_lo is None or cell_hi is None or best_lo is None or best_hi is None:
                continue
            overlaps = max(cell_lo, best_lo) <= min(cell_hi, best_hi)
            if overlaps:
                from matplotlib.patches import Rectangle

                rect = Rectangle(
                    (j - 0.5, i - 0.5), 1.0, 1.0,
                    fill=False, hatch="xx", edgecolor="white",
                    linewidth=0.0,
                )
                ax.add_patch(rect)
    fig.colorbar(im, ax=ax, label="BI")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_marginal_curves(
    plt,
    grid_rows: list[dict[str, str]],
    *,
    axis: str,
    real_models: list[str],
    fixed_other: dict[str, int],
    out_path: Path,
) -> None:
    """3-row (BI / NLL / Acc) × |real_models|-col panel of marginal curves.

    `axis` ∈ {"C", "R"} — the varying axis on each subplot's x-axis.
    `fixed_other` is the orthogonal pin (`{"R": 5}` when axis="C",
    `{"C": 3}` when axis="R"). Each curve gets a 95% CI shading where
    available; the saturation marker is a vertical dashed line at the
    smallest x where consecutive y values differ by < 0.01 in the BI
    domain.
    """
    if axis not in {"C", "R"}:
        return
    if not real_models:
        return
    fix_axis = "R" if axis == "C" else "C"
    fix_value = fixed_other.get(fix_axis)
    if fix_value is None:
        return
    cells = [r for r in grid_rows if _to_int(r.get(fix_axis)) == fix_value]
    if not cells:
        return
    n_cols = len(real_models)
    fig, axes = plt.subplots(
        3, n_cols, figsize=(4 * n_cols, 9), squeeze=False, sharex="col",
    )
    metrics = (
        ("bi_mean", "BI", "bi_ci_lo", "bi_ci_hi"),
        ("nll_mean", "NLL", None, None),
        ("acc_mean", "Accuracy", "acc_ci_lo", "acc_ci_hi"),
    )
    for col_idx, rm in enumerate(real_models):
        sub = sorted(
            [r for r in cells if r.get("real_model") == rm],
            key=lambda r: _to_int(r.get(axis)) or 0,
        )
        if not sub:
            continue
        xs = [_to_int(r.get(axis)) for r in sub]
        for row_idx, (key, label, lo_key, hi_key) in enumerate(metrics):
            ax = axes[row_idx][col_idx]
            ys = [_to_float(r.get(key)) for r in sub]
            xs_clean = [x for x, y in zip(xs, ys) if x is not None and y is not None]
            ys_clean = [y for y in ys if y is not None]
            if not xs_clean:
                ax.set_visible(False)
                continue
            ax.plot(xs_clean, ys_clean, marker="o", linewidth=1.5)
            if lo_key and hi_key:
                lo_vals = [_to_float(r.get(lo_key)) for r in sub]
                hi_vals = [_to_float(r.get(hi_key)) for r in sub]
                xs_ci, lo_ci, hi_ci = [], [], []
                for x, lo, hi, y in zip(xs, lo_vals, hi_vals, ys):
                    if x is None or y is None:
                        continue
                    xs_ci.append(x)
                    lo_ci.append(lo if lo is not None else y)
                    hi_ci.append(hi if hi is not None else y)
                if len(xs_ci) >= 2:
                    ax.fill_between(xs_ci, lo_ci, hi_ci, alpha=0.18, linewidth=0)
            if row_idx == 0 and len(ys_clean) >= 2:
                for x_prev, x_now, y_prev, y_now in zip(
                    xs_clean[:-1], xs_clean[1:], ys_clean[:-1], ys_clean[1:],
                ):
                    if abs(y_now - y_prev) < 0.01:
                        ax.axvline(x_now, color="gray", linestyle="--", linewidth=1)
                        break
            if row_idx == 0:
                ax.set_title(rm, fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(label)
            if row_idx == 2:
                ax.set_xlabel(axis)
    fig.suptitle(
        f"Marginal curves along {axis} (fixed {fix_axis}={fix_value})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_winrate_matrix(
    plt,
    winrate_rows: list[dict[str, str]],
    *,
    out_path: Path,
    sig_threshold: int = 1,
) -> None:
    """`M × M` heatmap of "row beats column" cell-win share.

    `grid_winrate.csv` contains one ordered pair (`a < b` lex) per row, so
    we mirror each row to populate `matrix[a][b]` (= wins_a / total_cells)
    and `matrix[b][a]` (= wins_b / total_cells). The diagonal is NaN
    (self-comparison is meaningless). Cells whose `sig_cells_*` count is
    `>= sig_threshold` get a `*` annotation appended to the proportion
    text.
    """
    if not winrate_rows:
        return
    models: set[str] = set()
    for r in winrate_rows:
        ma = r.get("model_a")
        mb = r.get("model_b")
        if ma:
            models.add(ma)
        if mb:
            models.add(mb)
    sorted_models = sorted(models)
    if len(sorted_models) < 2:
        return
    n = len(sorted_models)
    idx = {m: i for i, m in enumerate(sorted_models)}
    matrix: list[list[float]] = [
        [float("nan")] * n for _ in range(n)
    ]
    sig_mask: list[list[bool]] = [
        [False] * n for _ in range(n)
    ]
    for r in winrate_rows:
        ma = r.get("model_a") or ""
        mb = r.get("model_b") or ""
        if ma not in idx or mb not in idx:
            continue
        total = _to_int(r.get("total_cells")) or 0
        if total <= 0:
            continue
        wins_a = _to_int(r.get("wins_a")) or 0
        wins_b = _to_int(r.get("wins_b")) or 0
        sig_a = _to_int(r.get("sig_cells_a")) or 0
        sig_b = _to_int(r.get("sig_cells_b")) or 0
        i, j = idx[ma], idx[mb]
        matrix[i][j] = wins_a / total
        matrix[j][i] = wins_b / total
        sig_mask[i][j] = sig_a >= sig_threshold
        sig_mask[j][i] = sig_b >= sig_threshold
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(list(range(n)))
    ax.set_xticklabels(sorted_models, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(list(range(n)))
    ax.set_yticklabels(sorted_models, fontsize=8)
    ax.set_title("Pairwise winrate (row beats column across (R, C) cells)")
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = matrix[i][j]
            if v != v:  # NaN
                continue
            label = f"{v:.2f}"
            if sig_mask[i][j]:
                label += "*"
            ax.text(
                j, i, label, ha="center", va="center",
                fontsize=8,
                color="white" if v < 0.4 or v > 0.6 else "black",
            )
    fig.colorbar(im, ax=ax, label="row-wins share")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _render_grid_figures(
    plt,
    *,
    run_dir: Path,
    analysis_dir: Path,
    figs_dir: Path,
) -> list[Path]:
    """Grid-figure entry: emit `figs/grid_*.png` when manifest carries a grid block.

    Best-effort like the rest of `render_all` — a missing CSV or manifest
    block silently skips its plot. Returns the list of PNG paths actually
    written.
    """
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    grid_meta = manifest.get("grid") if isinstance(manifest, dict) else None
    if not grid_meta:
        return []
    artifacts = _read_grid_artifacts(analysis_dir)
    summary = artifacts["summary"]
    if not summary:
        return []
    written: list[Path] = []
    pareto_keys = _grid_pareto_keys(artifacts["pareto"])
    real_models_meta = list(grid_meta.get("real_models") or [])
    real_models = real_models_meta or _real_models_in_grid(summary)
    r_list = [int(r) for r in (grid_meta.get("r_list") or _r_values_in_grid(summary))]
    default_r = grid_meta.get("default_r")
    if default_r is None and r_list:
        default_r = r_list[0]
    default_c = grid_meta.get("default_c")
    c_list_meta = list(grid_meta.get("c_list") or [])
    if default_c is None and c_list_meta:
        default_c = int(c_list_meta[0])

    if default_r is not None:
        out = figs_dir / "grid_pareto_C.png"
        plot_pareto_frontier(
            plt, summary, pareto_keys,
            fix_R=int(default_r), out_path=out, default_r=int(default_r),
        )
        if out.exists():
            written.append(out)
    for r_value in r_list:
        if default_r is not None and r_value == int(default_r):
            continue
        out = figs_dir / f"grid_pareto_C_R{r_value}.png"
        plot_pareto_frontier(
            plt, summary, pareto_keys,
            fix_R=r_value, out_path=out,
            default_r=int(default_r) if default_r is not None else None,
        )
        if out.exists():
            written.append(out)

    bi_values = [v for r in summary if (v := _to_float(r.get("bi_mean"))) is not None]
    bi_vmax: float | None = None
    if bi_values:
        bi_vmax = max(50.0, max(bi_values) + 1.0)
    for rm in real_models:
        safe_rm = rm.replace("/", "__")
        out = figs_dir / f"grid_heatmap_RC_{safe_rm}.png"
        plot_grid_heatmap(
            plt, summary, real_model=rm, out_path=out,
            bi_vmin=0.0, bi_vmax=bi_vmax,
        )
        if out.exists():
            written.append(out)

    if default_r is not None:
        out = figs_dir / "grid_curve_C.png"
        plot_marginal_curves(
            plt, summary,
            axis="C",
            real_models=real_models,
            fixed_other={"R": int(default_r)},
            out_path=out,
        )
        if out.exists():
            written.append(out)
    if default_c is not None:
        out = figs_dir / "grid_curve_R.png"
        plot_marginal_curves(
            plt, summary,
            axis="R",
            real_models=real_models,
            fixed_other={"C": int(default_c)},
            out_path=out,
        )
        if out.exists():
            written.append(out)

    if artifacts["winrate"]:
        out = figs_dir / "grid_winrate_matrix.png"
        plot_winrate_matrix(plt, artifacts["winrate"], out_path=out)
        if out.exists():
            written.append(out)
    return written


def render_all(run_dir: Path) -> list[Path]:
    plt = _import_plt()
    analysis_dir = run_dir / "analysis"
    figs_dir = analysis_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    summary_rows = _read_csv(analysis_dir / "per_model_summary.csv")
    paired_rows = _read_csv(analysis_dir / "paired_delta_bi.csv")
    pairwise_v5_rows = _read_csv(analysis_dir / "pairwise_bootstrap.csv")
    by_diff_rows = _read_csv(analysis_dir / "per_model_by_difficulty.csv")
    paired_diff_rows = _read_csv(analysis_dir / "paired_delta_bi_by_difficulty.csv")
    pdp_rows = _read_csv(analysis_dir / "tool_usage_pdp.csv")
    entropy_acc_rows = _read_csv(analysis_dir / "entropy_accuracy_bins.csv")

    # Main figure: FSS bar with CI.
    if summary_rows:
        out = figs_dir / "fss_bar_with_ci.png"
        plot_fss_bar_with_ci(plt, summary_rows, pairwise_v5_rows, out)
        if out.exists():
            written.append(out)

    # Main figure: ΔFSS forest.
    if pairwise_v5_rows:
        out = figs_dir / "delta_fss_forest.png"
        plot_delta_fss_forest(plt, pairwise_v5_rows, out)
        if out.exists():
            written.append(out)

    # Main figure: per-model entropy-Acc grid.
    if entropy_acc_rows:
        new_pngs = plot_entropy_accuracy_grid(plt, entropy_acc_rows, figs_dir)
        written.extend(p for p in new_pngs if p not in written)

    # Appendix: BI bar with CI (companion to FSS, for BLF anchoring).
    if summary_rows:
        out = figs_dir / "bi_bar_with_ci.png"
        plot_bi_bar_with_ci(plt, summary_rows, paired_rows, out)
        if out.exists():
            written.append(out)

    # Appendix: ΔBI forest plot.
    if paired_rows:
        out = figs_dir / "delta_bi_forest.png"
        plot_delta_bi_forest(plt, paired_rows, out)
        if out.exists():
            written.append(out)

    if by_diff_rows:
        out = figs_dir / "difficulty_grid.png"
        plot_difficulty_grid(plt, by_diff_rows, paired_diff_rows, out)
        if out.exists():
            written.append(out)

    if pdp_rows:
        plot_tool_pdp(plt, pdp_rows, figs_dir)
        for png in figs_dir.glob("tool_pdp_*.png"):
            written.append(png)

    plot_belief_trajectories(plt, run_dir, figs_dir, n_questions=5)
    for png in figs_dir.glob("belief_trajectory_*.png"):
        if png not in written:
            written.append(png)

    written.extend(_render_grid_figures(
        plt, run_dir=run_dir, analysis_dir=analysis_dir, figs_dir=figs_dir,
    ))

    return sorted(set(written))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="path to runs/<run_id>")
    args = parser.parse_args()
    if not args.run_dir.exists():
        print(f"run_dir does not exist: {args.run_dir}", file=sys.stderr)
        return 2
    written = render_all(args.run_dir)
    print(f"wrote {len(written)} figure(s) to {args.run_dir / 'analysis' / 'figs'}")
    for p in written:
        print(f"  - {p.relative_to(args.run_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
