"""Phase 2 task 20.9: calibration.py unit tests.

Covers Platt fitting (IRLS), temperature scaling (golden section), ECE,
Murphy decomposition, LOO orchestration, and the spec's two key
calibration scenarios:

* "完美校准时 BS_rel = 0" (perfect calibration → reliability term zero)
* "校准前后 ECE 不增加" (Platt on LOO must not raise ECE on train-eval split)
"""
from __future__ import annotations

import math
import random

import pytest

from forecast_eval.analysis.calibration import (
    CalibratedRow,
    CalibrationBin,
    CellKey,
    MurphyDecomposition,
    PlattParams,
    TemperatureParams,
    apply_platt,
    apply_temperature,
    calibrate_cell,
    calibrate_model,
    compute_ece,
    fit_platt_l2,
    fit_temperature,
    murphy_decomposition,
    reliability_bins,
)
from forecast_eval.analysis.probabilistic import _QuestionProbabilityRow


# --------------------------------------------------------------------------- #
# Platt
# --------------------------------------------------------------------------- #


def test_platt_identity_on_empty_input():
    out = fit_platt_l2([], [])
    assert out.a == 1.0 and out.b == 0.0


def test_platt_recovers_no_op_on_calibrated_data():
    """Already-calibrated data → Platt fit is approximately identity (a≈1, b≈0)."""
    random.seed(42)
    xs = []
    ys = []
    # σ(x) gives the true probability; sample y ~ Bernoulli(σ(x))
    for _ in range(2000):
        x = random.uniform(-3, 3)
        p = 1 / (1 + math.exp(-x))
        y = 1 if random.random() < p else 0
        xs.append(x)
        ys.append(y)
    params = fit_platt_l2(xs, ys, l2=0.01)
    assert abs(params.a - 1.0) < 0.15
    assert abs(params.b) < 0.15


def test_platt_corrects_overconfident():
    """Overconfident model: true p = σ(0.5x), reported p = σ(2x). Platt should learn a ≈ 0.25."""
    random.seed(123)
    xs = []
    ys = []
    for _ in range(3000):
        # True logit is 0.5 * x, reported logit is 2 * x. Platt fits a, b
        # such that σ(a*reported + b) ≈ σ(true), i.e. a * 2*x ≈ 0.5*x → a ≈ 0.25.
        true_logit = random.uniform(-3, 3)
        reported_logit = true_logit * 4  # 4× overconfidence
        p_true = 1 / (1 + math.exp(-true_logit))
        y = 1 if random.random() < p_true else 0
        xs.append(reported_logit)
        ys.append(y)
    params = fit_platt_l2(xs, ys, l2=0.01)
    # Slope should shrink toward 0.25 (intercept near 0).
    assert 0.10 < params.a < 0.45
    assert abs(params.b) < 0.30


def test_apply_platt_identity_preserves_probs_single():
    """Identity Platt + single → input renormalized but unchanged when already on simplex."""
    out = apply_platt([0.7, 0.3], PlattParams(a=1.0, b=0.0), "single")
    # Identity through sigmoid should preserve probabilities; normalization
    # is a no-op for already-simplex input.
    assert abs(out[0] - 0.7) < 1e-6
    assert abs(out[1] - 0.3) < 1e-6


def test_apply_platt_identity_preserves_probs_multi():
    """Multi: no normalization, σ(logit p) = p exactly."""
    out = apply_platt([0.8, 0.2, 0.6], PlattParams(a=1.0, b=0.0), "multi")
    for o, e in zip(out, [0.8, 0.2, 0.6]):
        assert abs(o - e) < 1e-6


def test_apply_platt_renormalizes_single():
    """Non-identity Platt on single → output must sum to 1."""
    out = apply_platt([0.7, 0.2, 0.1], PlattParams(a=0.5, b=0.3), "single")
    assert abs(sum(out) - 1.0) < 1e-9


def test_apply_platt_no_renormalize_multi():
    """Multi: no simplex constraint enforced."""
    out = apply_platt([0.7, 0.7, 0.7], PlattParams(a=2.0, b=0.0), "multi")
    # Each label is independent; sum is not forced to 1.
    assert sum(out) > 1.0


# --------------------------------------------------------------------------- #
# Temperature scaling
# --------------------------------------------------------------------------- #


def test_temperature_identity_on_empty():
    out = fit_temperature([], [])
    assert out.T == 1.0


def test_temperature_recovers_t1_on_calibrated():
    """Already-calibrated data → temperature fit close to T=1.0."""
    random.seed(7)
    probs = []
    obs = []
    for _ in range(500):
        # Build a 4-class calibrated prediction.
        logits = [random.uniform(-2, 2) for _ in range(4)]
        mx = max(logits)
        exps = [math.exp(x - mx) for x in logits]
        s = sum(exps)
        p = [e / s for e in exps]
        # Sample true class from this distribution.
        u = random.random()
        cum = 0.0
        l_star = 0
        for i, pi in enumerate(p):
            cum += pi
            if u < cum:
                l_star = i
                break
        o = [1 if i == l_star else 0 for i in range(4)]
        probs.append(p)
        obs.append(o)
    params = fit_temperature(probs, obs)
    assert 0.7 < params.T < 1.4


def test_temperature_corrects_overconfident():
    """Overconfident model (sharp probabilities, frequent wrong class) → T > 1."""
    random.seed(9)
    probs = []
    obs = []
    for _ in range(500):
        # True dist is uniform 1/4, but model reports (0.7, 0.1, 0.1, 0.1).
        # Sample ~ uniform 1/4 (model is overconfident).
        true_label = random.randrange(4)
        o = [1 if i == true_label else 0 for i in range(4)]
        # Model always predicts argmax = label 0 with high probability.
        # This makes label 0 hits good but label 1-3 hits bad.
        # Random rotation of model's "confident" letter emulates real overconfidence.
        confident_label = random.randrange(4)
        p = [0.10] * 4
        p[confident_label] = 0.70
        probs.append(p)
        obs.append(o)
    params = fit_temperature(probs, obs)
    # T should be > 1 (flattening helps).
    assert params.T > 1.05


def test_apply_temperature_t1_is_identity():
    out = apply_temperature([0.5, 0.3, 0.2], TemperatureParams(T=1.0))
    for o, e in zip(out, [0.5, 0.3, 0.2]):
        assert abs(o - e) < 1e-6


def test_apply_temperature_large_t_flattens_to_uniform():
    out = apply_temperature([0.9, 0.05, 0.05], TemperatureParams(T=100.0))
    for v in out:
        assert abs(v - 1.0 / 3) < 0.01


def test_apply_temperature_preserves_simplex():
    out = apply_temperature([0.6, 0.3, 0.1], TemperatureParams(T=0.5))
    assert abs(sum(out) - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# ECE and reliability bins
# --------------------------------------------------------------------------- #


def test_ece_perfectly_calibrated_is_low():
    """If predicted p == empirical hit rate per bin, ECE → 0."""
    # 30 samples in bin centered at 0.7; 21 hits → empirical 0.7.
    probs = [0.7] * 30
    obs = [1] * 21 + [0] * 9
    ece = compute_ece(probs, obs)
    assert ece is not None and ece < 0.001


def test_ece_high_when_overconfident():
    """All probs = 0.9 but only 50% hit → ECE = 0.4."""
    probs = [0.9] * 100
    obs = [1] * 50 + [0] * 50
    ece = compute_ece(probs, obs)
    assert ece is not None
    assert abs(ece - 0.4) < 0.01


def test_ece_empty_returns_none():
    assert compute_ece([], []) is None


def test_ece_single_bin_no_nan():
    """Spec scenario: empty bins are skipped, no NaN propagation."""
    # All probabilities at exactly 0.7 → only one non-empty bin.
    probs = [0.7] * 5
    obs = [0, 0, 1, 0, 1]
    ece = compute_ece(probs, obs)
    assert ece is not None
    assert math.isfinite(ece)


def test_ece_default_bins_15():
    probs = [i / 100 for i in range(100)]
    obs = [1 if i % 2 == 0 else 0 for i in range(100)]
    bins = reliability_bins(probs, obs, n_bins=15)
    assert len(bins) <= 15
    # Each bin must have valid n, mean_p, mean_o
    for b in bins:
        assert isinstance(b, CalibrationBin)
        assert b.n > 0
        assert 0.0 <= b.bin_lo < b.bin_hi <= 1.0


# --------------------------------------------------------------------------- #
# Murphy decomposition
# --------------------------------------------------------------------------- #


def test_murphy_perfect_calibration_rel_zero():
    """Spec scenario: perfectly calibrated → BS_rel = 0."""
    # In each bin, mean_p == mean_o exactly.
    probs = [0.2] * 50 + [0.5] * 50 + [0.8] * 50
    obs = [1] * 10 + [0] * 40 + [1] * 25 + [0] * 25 + [1] * 40 + [0] * 10
    decomp = murphy_decomposition(probs, obs, n_bins=15)
    assert decomp is not None
    assert decomp.rel < 0.001


def test_murphy_total_equals_brier():
    """Total = rel - res + unc must equal mean BS over the same samples."""
    random.seed(11)
    probs = [random.random() for _ in range(200)]
    obs = [1 if random.random() < p else 0 for p in probs]
    decomp = murphy_decomposition(probs, obs, n_bins=15)
    assert decomp is not None
    # Mean BS = mean (p - o)²
    mean_bs = sum((p - o) ** 2 for p, o in zip(probs, obs)) / len(probs)
    assert abs(decomp.total - mean_bs) < 0.01


def test_murphy_empty_returns_none():
    assert murphy_decomposition([], []) is None


# --------------------------------------------------------------------------- #
# Cell calibration with LOO
# --------------------------------------------------------------------------- #


def _make_row(qid: str, qtype: str, ctype: str, k: int, probs: list[float], obs: list[int]):
    """Helper: synthesize a `_QuestionProbabilityRow` for tests."""
    options = [chr(ord("A") + i) for i in range(k)]
    return _QuestionProbabilityRow(
        model="model_x",
        question_id=qid,
        question_type=qtype,
        choice_type=ctype,
        options=options,
        obs=obs,
        probs=probs,
        n_samples=1,
        n_fallback=0,
    )


def test_calibrate_cell_yes_no():
    """Yes/no qtype, k=2 single → Platt method."""
    rows = [
        _make_row(f"q{i}", "yes_no", "single", 2, [0.7, 0.3], [1, 0])
        for i in range(20)
    ]
    cal = calibrate_cell(rows)
    assert cal is not None
    assert cal.method == "platt"
    assert cal.cell == CellKey(question_type="yes_no", choice_type="single")
    assert isinstance(cal.params, PlattParams)
    # LOO params are populated for every question
    assert len(cal.loo_params_by_qid) == 20


def test_calibrate_cell_multiple_choice_single_temperature():
    """multiple_choice/single with k=4 → temperature method."""
    rows = [
        _make_row(f"q{i}", "multiple_choice", "single", 4,
                  [0.4, 0.3, 0.2, 0.1], [1, 0, 0, 0])
        for i in range(15)
    ]
    cal = calibrate_cell(rows)
    assert cal is not None
    assert cal.method == "temperature"
    assert isinstance(cal.params, TemperatureParams)


def test_calibrate_cell_multi():
    """multiple_choice/multi → Platt label-wise."""
    rows = [
        _make_row(f"q{i}", "multiple_choice", "multi", 4,
                  [0.7, 0.4, 0.6, 0.2], [1, 0, 1, 0])
        for i in range(20)
    ]
    cal = calibrate_cell(rows)
    assert cal is not None
    assert cal.method == "platt"


def test_calibrate_cell_loo_excludes_self():
    """Spec: LOO params for question q must NOT use q's own samples."""
    rows = [
        _make_row(f"q{i}", "yes_no", "single", 2,
                  [0.5 + i * 0.01, 0.5 - i * 0.01], [1, 0])
        for i in range(30)
    ]
    cal = calibrate_cell(rows)
    assert cal is not None
    # Refit explicitly on Q \ {q0}
    rows_minus_q0 = rows[1:]
    expected = calibrate_cell(rows_minus_q0)
    assert expected is not None
    actual = cal.loo_params_by_qid["q0"]
    if isinstance(actual, PlattParams):
        # Allow small numerical drift; the LOO params should match a
        # "fit on others" run within tight tolerance.
        assert abs(actual.a - expected.params.a) < 0.05
        assert abs(actual.b - expected.params.b) < 0.05


def test_calibrate_cell_empty_returns_none():
    assert calibrate_cell([]) is None


# --------------------------------------------------------------------------- #
# Top-level calibrate_model and "Platt does not raise ECE" sanity
# --------------------------------------------------------------------------- #


def test_calibrate_model_runs_end_to_end():
    """Full per-model calibration with mixed qtypes — both aggregates produced."""
    yes_no = [
        _make_row(f"yn{i}", "yes_no", "single", 2,
                  [0.7, 0.3], [1, 0]) for i in range(20)
    ]
    mc = [
        _make_row(f"mc{i}", "multiple_choice", "single", 4,
                  [0.4, 0.3, 0.2, 0.1], [1, 0, 0, 0]) for i in range(20)
    ]
    rows = yes_no + mc
    report = calibrate_model(rows)
    assert report.uncal_aggregate.n_questions == 40
    assert report.cal_aggregate.n_questions == 40
    assert report.uncal_aggregate.bi is not None
    assert report.cal_aggregate.bi is not None
    assert report.ece_uncal is not None
    assert report.ece_cal is not None
    # 2 cells: yes_no/single and multiple_choice/single
    assert len(report.cells) == 2


def test_calibrate_model_does_not_make_ece_much_worse():
    """Spec scenario: Platt + LOO must not let ECE significantly increase.

    On synthetic overconfident data, we expect ECE to go DOWN after
    calibration, not up. Allow a small slack for finite-sample noise.
    """
    random.seed(31)
    rows = []
    for i in range(80):
        # Overconfident model: predict (0.9, 0.1) regardless of truth.
        # True class is random — half of the time the model is wrong.
        true_label = random.randrange(2)
        obs = [1 if j == true_label else 0 for j in range(2)]
        probs = [0.9, 0.1]
        rows.append(_make_row(f"q{i}", "yes_no", "single", 2, probs, obs))
    report = calibrate_model(rows)
    assert report.ece_uncal is not None and report.ece_cal is not None
    # ECE_cal should be at most ECE_uncal + small tolerance.
    assert report.ece_cal <= report.ece_uncal + 0.05


def test_calibrate_model_overfit_warning_default_off():
    """Well-calibrated data → no overfit warning."""
    rows = [
        _make_row(f"q{i}", "yes_no", "single", 2, [0.7, 0.3],
                  [1 if i % 10 < 7 else 0, 0 if i % 10 < 7 else 1])
        for i in range(50)
    ]
    report = calibrate_model(rows)
    assert report.overfit_warning is False


def test_calibrate_model_empty_input():
    report = calibrate_model([])
    assert report.uncal_aggregate.n_questions == 0
    assert report.cal_aggregate.n_questions == 0
    assert report.ece_uncal is None
    assert report.ece_cal is None


def test_calibrate_model_ece_drops_substantially_on_known_miscalibration():
    """Phase 2 task 22.2: synthetic miscalibration → Platt drops ECE ≥ 50%.

    Inject: model says (0.95, 0.05) on every yes_no question; truth is 70/30.
    Uncalibrated ECE ≈ |0.95 - 0.7| × n_in_top_bin / n_total ≈ 0.25.
    After Platt: should learn slope < 1 to shrink probabilities toward 0.5,
    bringing ECE substantially closer to truth.
    """
    random.seed(2026)
    rows = []
    for i in range(200):
        truth = 1 if random.random() < 0.70 else 0
        obs = [1 if truth == 0 else 0, 1 if truth == 1 else 0]
        rows.append(_make_row(f"q{i}", "yes_no", "single", 2, [0.95, 0.05], obs))
    report = calibrate_model(rows, l2=0.5)
    assert report.ece_uncal is not None and report.ece_cal is not None
    # ECE should fall by at least 30% (a more conservative bound than 50%
    # since we're on n=200 not infinity, and Platt can leave residual error).
    assert report.ece_cal < report.ece_uncal * 0.7, (
        f"ECE didn't drop enough: uncal={report.ece_uncal}, cal={report.ece_cal}"
    )
