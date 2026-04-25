"""Hierarchical calibration for the v4 probabilistic family (Phase 2).

Implements the calibration stack described in `ANALYSIS_DESIGN_v4.md §3.3`
and `specs/probabilistic-analysis/spec.md`:

* **Per-question_type Platt scaling** — label-wise expansion $(q, l)$ as
  independent samples; $L_2$-regularized binary logistic regression solved
  via IRLS / Newton's method (pure Python, ~50 lines).
* **Multi-class temperature scaling** — for $k \\ge 3$ single questions,
  $\\hat{p}_l = \\mathrm{softmax}(\\log p / T)$; $T > 0$ via golden-section
  search on LOO NLL.
* **Leave-one-out evaluation** — every question's calibrated prediction is
  produced by params trained on $\\mathcal{Q}_t \\setminus \\{q\\}$. With
  $\\sim 300$ questions per cell, LOO Platt is sub-second.
* **ECE** — equal-width bins, default $M = 15$.
* **Murphy three-decomposition** — $\\overline{BS} = BS_{\\text{rel}} -
  BS_{\\text{res}} + BS_{\\text{unc}}$ in the same bin partition.

All math is pure Python — same convention as `proper_score.py` and
`aggregation.py`. The IRLS loop converges in ~10 iterations for typical
calibration data; we cap at 50 to keep worst-case bounded. The Brent /
golden-section search uses the standard $\\varphi$-ratio for guaranteed
convergence on a unimodal function (NLL in $T$ has a single minimum).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .aggregation import _clip, _logit, _sigmoid, _softmax
from .proper_score import (
    NLL_EPS,
    PerQuestionScore,
    ModelProbabilisticAggregate,
    aggregate_probabilistic,
    brier_score_lab,
    nll as nll_metric,
    per_question_scores_for,
)
from .probabilistic import _QuestionProbabilityRow


# --------------------------------------------------------------------------- #
# Platt (Newton-IRLS)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PlattParams:
    """$\\sigma(a \\cdot x + b)$ where $x = \\mathrm{logit}\\,p$."""

    a: float
    b: float


def fit_platt_l2(
    x: list[float],
    y: list[int],
    *,
    l2: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-7,
) -> PlattParams:
    """Fit Platt's $(a, b)$ via Newton-IRLS with $L_2$ regularization.

    Loss: $-\\sum_i [y_i \\log p_i + (1-y_i)\\log(1-p_i)] + \\frac{\\lambda}{2}(a^2 + b^2)$
    where $p_i = \\sigma(a x_i + b)$.

    Newton update: $\\beta \\leftarrow \\beta + (X^T W X + \\lambda I)^{-1}
    (X^T (y - p) - \\lambda \\beta)$. The $2 \\times 2$ system is solved
    directly via Cramer's rule.

    Returns the identity Platt $(a=1, b=0)$ when given empty input — caller
    can rely on that as a safe degenerate.
    """
    n = len(x)
    if n == 0 or n != len(y):
        return PlattParams(a=1.0, b=0.0)
    # Start at the identity (a=1, b=0) — already-calibrated input is the
    # most common path in our setting.
    a, b = 1.0, 0.0
    # Step cap protects against IRLS divergence when the working sigmoids
    # saturate (extreme `xi` values produce near-singular Hessians; without a
    # cap the Newton step can launch to ±1e6 in one iteration). With a unit
    # cap, IRLS still converges in ~20 iterations even on overconfident data.
    max_step = 1.0
    for _ in range(max_iter):
        s_aa = 0.0  # Σ x² · w
        s_ab = 0.0  # Σ x  · w
        s_bb = 0.0  # Σ      w
        g_a = 0.0   # Σ x · (y - p)
        g_b = 0.0   # Σ     (y - p)
        for xi, yi in zip(x, y):
            z = a * xi + b
            p = _sigmoid(z)
            w = p * (1.0 - p)
            s_aa += xi * xi * w
            s_ab += xi * w
            s_bb += w
            g_a += xi * (yi - p)
            g_b += yi - p
        # Regularize both params for numerical stability; λ = 1 is mild.
        g_a -= l2 * a
        g_b -= l2 * b
        s_aa += l2
        s_bb += l2
        det = s_aa * s_bb - s_ab * s_ab
        if abs(det) < 1e-15:
            break
        da = (s_bb * g_a - s_ab * g_b) / det
        db = (-s_ab * g_a + s_aa * g_b) / det
        step_norm = math.sqrt(da * da + db * db)
        if step_norm > max_step:
            scale = max_step / step_norm
            da *= scale
            db *= scale
        a += da
        b += db
        if abs(da) < tol and abs(db) < tol:
            break
    return PlattParams(a=a, b=b)


def apply_platt(
    probs: list[float], params: PlattParams, choice_type: str
) -> list[float]:
    """Apply $\\sigma(a \\cdot \\mathrm{logit}\\,p_l + b)$ per label.

    For `single` choice_type, renormalize so the simplex constraint
    $\\sum_l p_l = 1$ still holds. For `multi`, leave the per-label
    Bernoulli probabilities independent (no normalization).
    """
    cal: list[float] = []
    for p in probs:
        z = _logit(p)
        cal.append(_sigmoid(params.a * z + params.b))
    if choice_type == "single":
        s = sum(cal)
        if s > 0:
            cal = [c / s for c in cal]
        else:
            # Degenerate: all zeros after sigmoid (should not happen with
            # finite a/b, but guard against it). Fall back to uniform.
            k = len(cal)
            cal = [1.0 / k] * k
    return cal


# --------------------------------------------------------------------------- #
# Temperature scaling (golden-section search)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TemperatureParams:
    """Calibrated by softmax($\\log p / T$); $T > 0$."""

    T: float


def _avg_nll_at_T(
    log_probs_per_q: list[list[float]],
    obs_per_q: list[list[int]],
    T: float,
    eps: float = NLL_EPS,
) -> float:
    """Mean NLL after applying temperature $T$ to per-question log-probs."""
    if T <= 0:
        return float("inf")
    total = 0.0
    n = 0
    for log_p, obs in zip(log_probs_per_q, obs_per_q):
        # Apply softmax(log_p / T)
        scaled = [v / T for v in log_p]
        mx = max(scaled)
        exps = [math.exp(v - mx) for v in scaled]
        s = sum(exps)
        if s <= 0:
            continue
        # Find the positive observation index (single-choice convention).
        try:
            l_star = next(i for i, oi in enumerate(obs) if oi == 1)
        except StopIteration:
            continue
        p_lstar = exps[l_star] / s
        p_clipped = max(min(p_lstar, 1.0 - eps), eps)
        total += -math.log(p_clipped)
        n += 1
    return total / n if n > 0 else float("inf")


def fit_temperature(
    probs_per_q: list[list[float]],
    obs_per_q: list[list[int]],
    *,
    T_min: float = 0.05,
    T_max: float = 20.0,
    tol: float = 1e-4,
    max_iter: int = 60,
) -> TemperatureParams:
    """Golden-section search for $T \\in [T_{\\min}, T_{\\max}]$ minimizing avg NLL.

    Inputs: per-question probability vectors and per-question Bernoulli obs
    vectors (single-choice; positive label is the unique $l^*$ with $o = 1$).
    Identity ($T = 1$) is the no-op default; values $T > 1$ flatten an
    over-confident model, values $T < 1$ sharpen an under-confident one.

    Returns $T = 1$ when given empty input — same identity convention as
    `fit_platt_l2`.
    """
    if not probs_per_q or len(probs_per_q) != len(obs_per_q):
        return TemperatureParams(T=1.0)
    if T_min <= 0 or T_max <= T_min:
        raise ValueError(f"invalid temperature search interval [{T_min}, {T_max}]")

    log_probs_per_q = [
        [math.log(_clip(p, eps=NLL_EPS)) for p in probs] for probs in probs_per_q
    ]

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    inv_phi2 = 1.0 / (phi * phi)
    a, b = T_min, T_max
    h = b - a
    if h <= tol:
        return TemperatureParams(T=(a + b) / 2.0)

    c = a + inv_phi2 * h
    d = a + inv_phi * h
    yc = _avg_nll_at_T(log_probs_per_q, obs_per_q, c)
    yd = _avg_nll_at_T(log_probs_per_q, obs_per_q, d)

    for _ in range(max_iter):
        if yc < yd:
            b = d
            d, yd = c, yc
            h = inv_phi * h
            c = a + inv_phi2 * h
            yc = _avg_nll_at_T(log_probs_per_q, obs_per_q, c)
        else:
            a = c
            c, yc = d, yd
            h = inv_phi * h
            d = a + inv_phi * h
            yd = _avg_nll_at_T(log_probs_per_q, obs_per_q, d)
        if h < tol:
            break

    T_star = (a + b) / 2.0
    return TemperatureParams(T=T_star)


def apply_temperature(probs: list[float], params: TemperatureParams) -> list[float]:
    """Apply softmax($\\log p / T$). Preserves the simplex by construction."""
    if params.T <= 0:
        return probs
    log_p = [math.log(_clip(p)) for p in probs]
    scaled = [v / params.T for v in log_p]
    return _softmax(scaled)


# --------------------------------------------------------------------------- #
# ECE and Murphy decomposition
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CalibrationBin:
    """One bin in a reliability diagram."""

    bin_lo: float
    bin_hi: float
    n: int
    mean_p: float
    mean_o: float


def _bin_for(p: float, n_bins: int) -> int:
    """Map probability to bin index $0 \\le i < n_{\\text{bins}}$."""
    idx = int(p * n_bins)
    if idx >= n_bins:
        idx = n_bins - 1
    if idx < 0:
        idx = 0
    return idx


def reliability_bins(
    probs: list[float], obs: list[int], n_bins: int = 15
) -> list[CalibrationBin]:
    """Equal-width reliability bins on $[0, 1]$. Empty bins are skipped.

    Each `(p, o)` in the input is one sample. Caller decides what to feed:
    * top-1 confidence + top-1 hit (single-choice ECE),
    * per-(q, l) probability + observation (label-wise / multi-choice ECE).
    """
    if not probs or len(probs) != len(obs):
        return []
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, o in zip(probs, obs):
        bins[_bin_for(p, n_bins)].append((float(p), int(o)))
    out: list[CalibrationBin] = []
    for i, bucket in enumerate(bins):
        if not bucket:
            continue
        n_b = len(bucket)
        mean_p = sum(b[0] for b in bucket) / n_b
        mean_o = sum(b[1] for b in bucket) / n_b
        out.append(
            CalibrationBin(
                bin_lo=i / n_bins,
                bin_hi=(i + 1) / n_bins,
                n=n_b,
                mean_p=mean_p,
                mean_o=mean_o,
            )
        )
    return out


def compute_ece(
    probs: list[float], obs: list[int], n_bins: int = 15
) -> float | None:
    """Expected Calibration Error: $\\sum_m \\frac{|B_m|}{N}|\\overline{p}_m - \\overline{a}_m|$.

    Returns None if input is empty.
    """
    if not probs:
        return None
    n = len(probs)
    bins = reliability_bins(probs, obs, n_bins)
    if not bins:
        return 0.0
    return sum((b.n / n) * abs(b.mean_p - b.mean_o) for b in bins)


@dataclass(frozen=True)
class MurphyDecomposition:
    """$\\overline{BS} = BS_{\\text{rel}} - BS_{\\text{res}} + BS_{\\text{unc}}$.

    Reliability ($BS_{\\text{rel}}$) is the calibration error term — the
    bin-weighted squared distance between predicted and observed frequency.
    Resolution ($BS_{\\text{res}}$) measures how well the model spreads
    probabilities across distinct outcome classes. Uncertainty ($BS_{\\text{unc}}$)
    is the data's intrinsic randomness lower bound (binary case).
    """

    rel: float
    res: float
    unc: float
    total: float


def murphy_decomposition(
    probs: list[float], obs: list[int], n_bins: int = 15
) -> MurphyDecomposition | None:
    """Murphy three-decomposition with equal-width bins."""
    if not probs:
        return None
    n = len(probs)
    bins = reliability_bins(probs, obs, n_bins)
    o_global = sum(obs) / n if n > 0 else 0.0
    rel = 0.0
    res = 0.0
    for b in bins:
        rel += (b.n / n) * (b.mean_p - b.mean_o) ** 2
        res += (b.n / n) * (b.mean_o - o_global) ** 2
    unc = o_global * (1.0 - o_global)
    return MurphyDecomposition(rel=rel, res=res, unc=unc, total=rel - res + unc)


# --------------------------------------------------------------------------- #
# LOO orchestration: per-(model, qtype) calibration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CellKey:
    """Calibration cell = (question_type, choice_type) combination.

    Our dataset has at most 4 cells (yes_no/single, binary_named/single,
    multiple_choice/single, multiple_choice/multi). Per-cell calibration
    matches the spec's "per-question_type Platt" intent.
    """

    question_type: str
    choice_type: str

    @property
    def name(self) -> str:
        return f"{self.question_type}__{self.choice_type}"


def _determine_method(cell: CellKey, k: int) -> str:
    """Pick `platt` or `temperature` based on the cell's properties.

    * `multi` → `platt` (label-wise, no simplex constraint).
    * `single` with $k = 2$ → `platt` (binary, with simplex renormalization).
    * `single` with $k \\ge 3$ → `temperature` (preserves simplex by softmax).
    """
    if cell.choice_type != "single":
        return "platt"
    if k <= 2:
        return "platt"
    return "temperature"


def _flatten_for_platt(
    rows: list[_QuestionProbabilityRow],
) -> tuple[list[float], list[int], list[tuple[str, int]]]:
    """Per-(q, l) flatten: returns (logit_p, obs, [(qid, label_idx)])."""
    xs: list[float] = []
    ys: list[int] = []
    keys: list[tuple[str, int]] = []
    for r in rows:
        for li, (p, o) in enumerate(zip(r.probs, r.obs)):
            xs.append(_logit(p))
            ys.append(int(o))
            keys.append((r.question_id, li))
    return xs, ys, keys


@dataclass(frozen=True)
class CellCalibration:
    """Calibration result for one (model, cell) pair.

    `params` is the all-data fit (used for `calibration_params.json`); the
    LOO-fitted per-question params live in `loo_params_by_qid`.
    """

    cell: CellKey
    method: str
    params: PlattParams | TemperatureParams
    loo_params_by_qid: dict[str, PlattParams | TemperatureParams]
    n_questions: int


def _fit_platt_for_cell(
    rows: list[_QuestionProbabilityRow], *, l2: float
) -> tuple[PlattParams, dict[str, PlattParams]]:
    """All-data fit + per-question LOO fit (Platt)."""
    xs, ys, keys = _flatten_for_platt(rows)
    if not xs:
        identity = PlattParams(a=1.0, b=0.0)
        return identity, {r.question_id: identity for r in rows}
    full_params = fit_platt_l2(xs, ys, l2=l2)
    # LOO: for each question, refit excluding that question's (q, l) samples.
    qid_to_indices: dict[str, list[int]] = {}
    for i, (qid, _) in enumerate(keys):
        qid_to_indices.setdefault(qid, []).append(i)
    loo_params: dict[str, PlattParams] = {}
    for qid, exclude in qid_to_indices.items():
        excl = set(exclude)
        x_train = [v for i, v in enumerate(xs) if i not in excl]
        y_train = [v for i, v in enumerate(ys) if i not in excl]
        if not x_train:
            loo_params[qid] = full_params
        else:
            loo_params[qid] = fit_platt_l2(x_train, y_train, l2=l2)
    return full_params, loo_params


def _fit_temperature_for_cell(
    rows: list[_QuestionProbabilityRow],
) -> tuple[TemperatureParams, dict[str, TemperatureParams]]:
    """All-data fit + per-question LOO fit (Temperature)."""
    if not rows:
        identity = TemperatureParams(T=1.0)
        return identity, {}
    probs = [r.probs for r in rows]
    obs = [r.obs for r in rows]
    full_params = fit_temperature(probs, obs)
    loo_params: dict[str, TemperatureParams] = {}
    for i, r in enumerate(rows):
        train_probs = probs[:i] + probs[i + 1:]
        train_obs = obs[:i] + obs[i + 1:]
        if not train_probs:
            loo_params[r.question_id] = full_params
        else:
            loo_params[r.question_id] = fit_temperature(train_probs, train_obs)
    return full_params, loo_params


def calibrate_cell(
    rows: list[_QuestionProbabilityRow], *, l2: float = 1.0
) -> CellCalibration | None:
    """Fit per-cell calibration with all-data + LOO. Returns None for empty cell."""
    if not rows:
        return None
    cell = CellKey(
        question_type=rows[0].question_type, choice_type=rows[0].choice_type
    )
    k = len(rows[0].probs)
    method = _determine_method(cell, k)
    if method == "platt":
        full, loo = _fit_platt_for_cell(rows, l2=l2)
    else:
        full, loo = _fit_temperature_for_cell(rows)
    return CellCalibration(
        cell=cell,
        method=method,
        params=full,
        loo_params_by_qid=loo,
        n_questions=len(rows),
    )


def apply_cell_calibration(
    row: _QuestionProbabilityRow,
    calibration: CellCalibration,
    *,
    use_loo: bool = True,
) -> list[float]:
    """Apply LOO (default) or all-data params to a single question's probs."""
    if use_loo:
        params = calibration.loo_params_by_qid.get(row.question_id, calibration.params)
    else:
        params = calibration.params
    if calibration.method == "platt":
        assert isinstance(params, PlattParams)
        return apply_platt(row.probs, params, row.choice_type)
    assert isinstance(params, TemperatureParams)
    return apply_temperature(row.probs, params)


# --------------------------------------------------------------------------- #
# Top-level: per-model calibrated predictions + aggregates
# --------------------------------------------------------------------------- #


@dataclass
class CalibratedRow:
    """A `_QuestionProbabilityRow` with both uncal and cal probabilities."""

    row: _QuestionProbabilityRow
    cal_probs: list[float]
    cell_method: str


@dataclass
class ModelCalibrationReport:
    """Per-model calibration outputs.

    `cells` lists every cell with its `CellCalibration` (params + LOO params).
    `calibrated_rows` is the per-question post-calibration probs. Aggregates
    cover both uncalibrated (for sanity) and calibrated metrics.
    """

    model: str
    cells: dict[str, CellCalibration]   # keyed by CellKey.name
    calibrated_rows: list[CalibratedRow]
    uncal_aggregate: ModelProbabilisticAggregate
    cal_aggregate: ModelProbabilisticAggregate
    ece_uncal: float | None
    ece_cal: float | None
    murphy_uncal: MurphyDecomposition | None
    murphy_cal: MurphyDecomposition | None

    @property
    def overfit_warning(self) -> bool:
        """Spec哨兵: True if cal BI > uncal BI + 5 (calibration likely overfit)."""
        u = self.uncal_aggregate.bi
        c = self.cal_aggregate.bi
        if u is None or c is None:
            return False
        # u and c are scaled to [0, 100]; cal is "worse" when its BI is
        # smaller (closer to 0) — but spec uses the wording "cal BI 比 uncal
        # 高 5+" meaning cal BI exceeds uncal BI by 5 (cal is BETTER by 5).
        # Interpretation: too-large positive jump suggests overfit on LOO
        # train.
        return (c - u) > 5.0


def _flat_pairs_for_ece(
    rows: list[CalibratedRow], *, calibrated: bool
) -> tuple[list[float], list[int]]:
    """Flatten rows to (top-1 prob, top-1 hit) for single-choice ECE.

    For multi rows we use per-(q, l) pairs since "top-1" doesn't carry the
    same semantics. Same convention paper §C.11 applies.
    """
    probs: list[float] = []
    obs: list[int] = []
    for cr in rows:
        r = cr.row
        p_vec = cr.cal_probs if calibrated else r.probs
        if r.choice_type == "single":
            best_i = 0
            best_p = p_vec[0]
            for i, p in enumerate(p_vec):
                if p > best_p:
                    best_p = p
                    best_i = i
            probs.append(best_p)
            obs.append(int(r.obs[best_i]))
        else:
            for p, o in zip(p_vec, r.obs):
                probs.append(p)
                obs.append(int(o))
    return probs, obs


def calibrate_model(
    rows: list[_QuestionProbabilityRow],
    *,
    crowd_gammas: dict[str, float | None] | None = None,
    uniform_gammas: dict[str, float] | None = None,
    l2: float = 1.0,
    n_bins: int = 15,
) -> ModelCalibrationReport:
    """End-to-end per-model calibration.

    1. Group `rows` by `CellKey`.
    2. Fit per-cell calibration (all-data + LOO).
    3. Apply LOO calibration to each row.
    4. Re-aggregate proper scores on calibrated probs.
    5. Compute ECE / Murphy on top-1 confidence (single) or per-label (multi).
    """
    if not rows:
        empty = ModelProbabilisticAggregate(
            n_questions=0, n_fallback=0, fallback_share=None,
            bi=None, bi_dec=None, nll=None, mbs=None,
            abi_crowd=None, abi_uniform=None,
        )
        return ModelCalibrationReport(
            model="",
            cells={},
            calibrated_rows=[],
            uncal_aggregate=empty,
            cal_aggregate=empty,
            ece_uncal=None,
            ece_cal=None,
            murphy_uncal=None,
            murphy_cal=None,
        )

    model = rows[0].model
    by_cell: dict[CellKey, list[_QuestionProbabilityRow]] = {}
    for r in rows:
        cell = CellKey(question_type=r.question_type, choice_type=r.choice_type)
        by_cell.setdefault(cell, []).append(r)

    cells: dict[str, CellCalibration] = {}
    calibrated_rows: list[CalibratedRow] = []
    for cell, cell_rows in by_cell.items():
        cal = calibrate_cell(cell_rows, l2=l2)
        if cal is None:
            continue
        cells[cell.name] = cal
        for r in cell_rows:
            cal_probs = apply_cell_calibration(r, cal, use_loo=True)
            calibrated_rows.append(
                CalibratedRow(row=r, cal_probs=cal_probs, cell_method=cal.method)
            )

    # Re-aggregate proper scores on uncal vs cal probabilities.
    uncal_per_q = [
        per_question_scores_for(
            question_id=cr.row.question_id,
            choice_type=cr.row.choice_type,
            probs=cr.row.probs,
            obs=cr.row.obs,
            is_fallback=cr.row.is_fallback,
        )
        for cr in calibrated_rows
    ]
    cal_per_q = [
        per_question_scores_for(
            question_id=cr.row.question_id,
            choice_type=cr.row.choice_type,
            probs=cr.cal_probs,
            obs=cr.row.obs,
            is_fallback=cr.row.is_fallback,
        )
        for cr in calibrated_rows
    ]
    uncal_aggregate = aggregate_probabilistic(
        uncal_per_q, crowd_gammas=crowd_gammas, uniform_gammas=uniform_gammas,
    )
    cal_aggregate = aggregate_probabilistic(
        cal_per_q, crowd_gammas=crowd_gammas, uniform_gammas=uniform_gammas,
    )

    # ECE + Murphy on top-1 confidence (single) or per-label (multi).
    uncal_probs, uncal_obs = _flat_pairs_for_ece(calibrated_rows, calibrated=False)
    cal_probs, cal_obs = _flat_pairs_for_ece(calibrated_rows, calibrated=True)
    ece_uncal = compute_ece(uncal_probs, uncal_obs, n_bins)
    ece_cal = compute_ece(cal_probs, cal_obs, n_bins)
    murphy_uncal = murphy_decomposition(uncal_probs, uncal_obs, n_bins)
    murphy_cal = murphy_decomposition(cal_probs, cal_obs, n_bins)

    return ModelCalibrationReport(
        model=model,
        cells=cells,
        calibrated_rows=calibrated_rows,
        uncal_aggregate=uncal_aggregate,
        cal_aggregate=cal_aggregate,
        ece_uncal=ece_uncal,
        ece_cal=ece_cal,
        murphy_uncal=murphy_uncal,
        murphy_cal=murphy_cal,
    )


def calibrate_run(
    rows_by_model: dict[str, list[_QuestionProbabilityRow]],
    *,
    crowd_gammas_by_model: dict[str, dict[str, float | None]] | None = None,
    uniform_gammas: dict[str, float] | None = None,
    l2: float = 1.0,
    n_bins: int = 15,
) -> dict[str, ModelCalibrationReport]:
    """Top-level: run `calibrate_model` for every model in the run."""
    out: dict[str, ModelCalibrationReport] = {}
    for model, rows in rows_by_model.items():
        crowd_gammas = (
            crowd_gammas_by_model.get(model) if crowd_gammas_by_model else None
        )
        out[model] = calibrate_model(
            rows,
            crowd_gammas=crowd_gammas,
            uniform_gammas=uniform_gammas,
            l2=l2,
            n_bins=n_bins,
        )
    return out


__all__ = [
    "PlattParams",
    "TemperatureParams",
    "CalibrationBin",
    "MurphyDecomposition",
    "CellKey",
    "CellCalibration",
    "CalibratedRow",
    "ModelCalibrationReport",
    "fit_platt_l2",
    "apply_platt",
    "fit_temperature",
    "apply_temperature",
    "compute_ece",
    "murphy_decomposition",
    "reliability_bins",
    "calibrate_cell",
    "apply_cell_calibration",
    "calibrate_model",
    "calibrate_run",
]
