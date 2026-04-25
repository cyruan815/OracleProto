"""On-demand visualisation for one analysis run.

CLI:
    python scripts/plot_analysis.py runs/<run_id>

Reads `runs/<run_id>/analysis/*.{csv,json}` produced by Phase 1-3 and emits
PNGs into `runs/<run_id>/analysis/figs/`. Matplotlib is loaded lazily so the
core analysis path stays free of plot-time dependencies; missing matplotlib
prints a helpful install hint and exits 1.

Each figure is best-effort: a missing source file (e.g. no `reflection_ab.csv`
because no paired runs were found) skips the corresponding plot rather than
failing the whole pipeline. This matches the design.md "按需出图" decision.
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
# Plots
# --------------------------------------------------------------------------- #


def plot_reliability(
    plt, data: dict, out_path: Path, *, title: str
) -> None:
    """Per-model reliability diagram from `reliability_data*.json`.

    Each model gets one polyline of (mean_p, mean_o) per non-empty bin. Light
    gray y=x reference. Bin counts are NOT shown — it would clutter the line
    plot; users who care can read the JSON directly.
    """
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="ideal")
    for model, by_qtype in sorted(data.items()):
        # Aggregate bins across question_types — match `_bins_for_model_qtype`'s
        # output schema.
        all_pts: list[tuple[float, float]] = []
        if isinstance(by_qtype, dict):
            for qt, bins in sorted(by_qtype.items()):
                if not isinstance(bins, list):
                    continue
                for b in bins:
                    if not isinstance(b, dict):
                        continue
                    n = b.get("n") or 0
                    if n <= 0:
                        continue
                    mp = b.get("mean_p")
                    mo = b.get("mean_o")
                    if mp is None or mo is None:
                        continue
                    all_pts.append((float(mp), float(mo)))
        elif isinstance(by_qtype, list):
            for b in by_qtype:
                n = b.get("n") or 0
                if n <= 0:
                    continue
                mp = b.get("mean_p")
                mo = b.get("mean_o")
                if mp is None or mo is None:
                    continue
                all_pts.append((float(mp), float(mo)))
        all_pts.sort()
        if not all_pts:
            continue
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=model)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


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


def plot_brier_decomposition(
    plt, decomp_rows: list[dict], out_path: Path
) -> None:
    """Stacked bar of Murphy three-decomposition (rel - res + unc) per model."""
    if not decomp_rows:
        return
    # Use uncalibrated columns; calibrated also exists but uncal is the
    # "honest" signal pre-correction.
    by_model: dict[str, dict[str, float]] = {}
    for r in decomp_rows:
        m = r.get("model")
        if not m:
            continue
        by_model[m] = {
            "rel": _to_float(r.get("rel_uncal")) or 0.0,
            "res": _to_float(r.get("res_uncal")) or 0.0,
            "unc": _to_float(r.get("unc_uncal")) or 0.0,
        }
    if not by_model:
        return
    models = sorted(by_model.keys())
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(models)), 5))
    rel = [by_model[m]["rel"] for m in models]
    res_neg = [-by_model[m]["res"] for m in models]
    unc = [by_model[m]["unc"] for m in models]
    ax.bar(models, rel, color="#C44E52", label="rel (calibration ↓ better)")
    ax.bar(models, res_neg, bottom=rel, color="#55A868", label="−res (resolution ↑ better)")
    ax.bar(
        models,
        unc,
        bottom=[r + n for r, n in zip(rel, res_neg)],
        color="#8172B2",
        label="unc (irreducible)",
    )
    ax.set_ylabel("Brier components")
    ax.set_title("Murphy three-decomposition (uncalibrated)")
    ax.legend(loc="best", fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


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
# Entry point
# --------------------------------------------------------------------------- #


def render_all(run_dir: Path) -> list[Path]:
    plt = _import_plt()
    analysis_dir = run_dir / "analysis"
    figs_dir = analysis_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Reliability diagrams.
    rel_uncal = _read_json(analysis_dir / "reliability_data.json")
    if rel_uncal:
        out = figs_dir / "reliability_diagram_per_model.png"
        plot_reliability(plt, rel_uncal, out, title="Reliability (uncalibrated)")
        if out.exists():
            written.append(out)
    rel_cal = _read_json(analysis_dir / "reliability_data_calibrated.json")
    if rel_cal:
        out = figs_dir / "reliability_diagram_calibrated.png"
        plot_reliability(plt, rel_cal, out, title="Reliability (calibrated)")
        if out.exists():
            written.append(out)

    summary_rows = _read_csv(analysis_dir / "per_model_summary.csv")
    cal_summary_rows = _read_csv(analysis_dir / "per_model_summary_calibrated.csv")
    paired_rows = _read_csv(analysis_dir / "paired_delta_bi.csv")
    decomp_rows = _read_csv(analysis_dir / "brier_decomposition.csv")
    by_diff_rows = _read_csv(analysis_dir / "per_model_by_difficulty.csv")
    paired_diff_rows = _read_csv(analysis_dir / "paired_delta_bi_by_difficulty.csv")
    pdp_rows = _read_csv(analysis_dir / "tool_usage_pdp.csv")

    if summary_rows or cal_summary_rows:
        out = figs_dir / "bi_bar_with_ci.png"
        plot_bi_bar_with_ci(
            plt,
            summary_rows or cal_summary_rows,
            paired_rows,
            out,
        )
        if out.exists():
            written.append(out)

    if paired_rows:
        out = figs_dir / "delta_bi_forest.png"
        plot_delta_bi_forest(plt, paired_rows, out)
        if out.exists():
            written.append(out)

    if decomp_rows:
        out = figs_dir / "brier_decomp_stacked.png"
        plot_brier_decomposition(plt, decomp_rows, out)
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
