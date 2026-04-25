"""Phase 2 task 21.7: inference.py unit tests.

Covers `paired_bootstrap`, `holm_bonferroni`, `difficulty_tertile`,
`paired_bootstrap_by_difficulty`, `posterior_a_better_than_b`, and the
`pairwise_paired_bootstrap` orchestrator.

Spec scenarios verified:
* Paired bootstrap is paired, not independent (same indices for A and B).
* Holm correction direction (multiply by remaining ranks descending).
* B = 5000 default; same seed → reproducible.
* Posterior probability is a valid [0, 1] number, same direction as p-value.
"""
from __future__ import annotations

import math
import random

import pytest

from forecast_eval.analysis.inference import (
    DifficultyTertile,
    ModelPairResult,
    PairedBootstrapResult,
    _normal_cdf,
    difficulty_tertile,
    holm_bonferroni,
    paired_bootstrap,
    paired_bootstrap_by_difficulty,
    pairwise_paired_bootstrap,
    posterior_a_better_than_b,
    posterior_normal_fit,
)


# --------------------------------------------------------------------------- #
# Paired bootstrap
# --------------------------------------------------------------------------- #


def test_paired_bootstrap_default_n_bootstrap_is_5000():
    """Spec scenario: B = 5000 by default."""
    bs_a = [0.1] * 50
    bs_b = [0.2] * 50
    res = paired_bootstrap(bs_a, bs_b)
    assert res.n_bootstrap == 5000


def test_paired_bootstrap_seed_is_reproducible():
    """Same seed → same bootstrap distribution → CIs differ by < 0.01."""
    bs_a = [0.1 + i * 0.01 for i in range(100)]
    bs_b = [0.2 - i * 0.01 for i in range(100)]
    res1 = paired_bootstrap(bs_a, bs_b, seed=999)
    res2 = paired_bootstrap(bs_a, bs_b, seed=999)
    assert abs(res1.ci_low - res2.ci_low) < 1e-9
    assert abs(res1.ci_high - res2.ci_high) < 1e-9
    assert abs(res1.p_two_sided - res2.p_two_sided) < 1e-9


def test_paired_bootstrap_zero_delta_p_value_high():
    """Identical models → delta = 0, p-value should be high (no signal)."""
    random.seed(7)
    bs_a = [random.random() for _ in range(100)]
    bs_b = list(bs_a)  # exactly the same
    res = paired_bootstrap(bs_a, bs_b)
    assert abs(res.delta_mean) < 1e-12
    # All bootstrap deltas = 0; p_two = 2 * min(1.0, 1.0) = 2.0 → clipped to 1
    assert res.p_two_sided >= 0.99


def test_paired_bootstrap_strong_difference_p_value_low():
    """Model A always 0.05 worse → strong signal, p < 0.01."""
    bs_a = [0.10] * 100
    bs_b = [0.05] * 100
    res = paired_bootstrap(bs_a, bs_b)
    assert res.delta_mean == pytest.approx(0.05, abs=1e-9)
    assert res.ci_low > 0  # CI excludes 0
    assert res.ci_high > 0
    assert res.p_two_sided < 0.01


def test_paired_bootstrap_validates_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap([0.1, 0.2], [0.1, 0.2, 0.3])


def test_paired_bootstrap_empty_input():
    res = paired_bootstrap([], [])
    assert res.n_questions == 0
    assert res.delta_mean == 0.0


def test_paired_bootstrap_pairing_property():
    """Spec scenario: the same bootstrap index hits BOTH arrays.

    With paired bootstrap, the variance of the delta is the variance of the
    per-question deltas. With independent bootstrap, it would be the sum
    of A's and B's variances. We verify by constructing data where paired
    deltas are constant (zero variance) — paired CI must be tight, while
    independent would be wide.
    """
    n = 50
    bs_a = [0.1 + i * 0.01 for i in range(n)]
    bs_b = [v - 0.05 for v in bs_a]   # delta is always exactly 0.05
    res = paired_bootstrap(bs_a, bs_b)
    # Paired delta has zero variance → CI collapses to ~0.05.
    assert abs(res.delta_mean - 0.05) < 1e-9
    assert abs(res.ci_low - 0.05) < 1e-9
    assert abs(res.ci_high - 0.05) < 1e-9


# --------------------------------------------------------------------------- #
# Holm-Bonferroni
# --------------------------------------------------------------------------- #


def test_holm_empty_input():
    assert holm_bonferroni([]) == []


def test_holm_single_pvalue_unchanged():
    """n=1: no correction needed."""
    assert holm_bonferroni([0.03]) == [0.03]


def test_holm_correction_direction_simple():
    """5 raw p-values → adjusted increase monotonically along sorted rank."""
    p_raw = [0.01, 0.02, 0.03, 0.04, 0.05]
    adj = holm_bonferroni(p_raw)
    # Smallest p multiplied by 5 = 0.05
    assert abs(adj[0] - 0.05) < 1e-9
    # 2nd smallest: max(0.05, 0.02 * 4) = max(0.05, 0.08) = 0.08
    assert abs(adj[1] - 0.08) < 1e-9
    # 3rd: max(0.08, 0.03 * 3) = max(0.08, 0.09) = 0.09
    assert abs(adj[2] - 0.09) < 1e-9
    # 4th: max(0.09, 0.04 * 2) = max(0.09, 0.08) = 0.09
    assert abs(adj[3] - 0.09) < 1e-9
    # 5th: max(0.09, 0.05 * 1) = max(0.09, 0.05) = 0.09
    assert abs(adj[4] - 0.09) < 1e-9


def test_holm_correction_with_unsorted_input():
    """Adjusted p-values returned in INPUT order, not sorted order."""
    p_raw = [0.04, 0.01, 0.03, 0.02]
    adj = holm_bonferroni(p_raw)
    assert len(adj) == 4
    # smallest is index 1 (p=0.01) → adjusted to 0.04
    assert abs(adj[1] - 0.04) < 1e-9


def test_holm_clips_to_one():
    """Adjusted p > 1 → clipped to 1.0."""
    adj = holm_bonferroni([0.5, 0.6])
    # 0.5 * 2 = 1.0; 0.6 * 1 = 0.6 < 1.0 → after cum max, 0.6 → 1.0
    assert all(v <= 1.0 for v in adj)


# --------------------------------------------------------------------------- #
# Difficulty tertile
# --------------------------------------------------------------------------- #


def test_difficulty_tertile_balanced_split():
    """N divisible by 3 → tertiles split as N/3 each."""
    gammas = {f"q{i}": i * 0.01 for i in range(30)}
    tert = difficulty_tertile(gammas)
    assert tert.n_low == 10
    assert tert.n_mid == 10
    assert tert.n_high == 10
    # Lowest gamma → "low"
    assert tert.by_question["q0"] == "low"
    assert tert.by_question["q15"] == "mid"
    assert tert.by_question["q29"] == "high"


def test_difficulty_tertile_uneven_split():
    """N=10 → tertiles 3/3/4 (or similar)."""
    gammas = {f"q{i}": i * 0.01 for i in range(10)}
    tert = difficulty_tertile(gammas)
    assert tert.n_low + tert.n_mid + tert.n_high == 10
    assert tert.n_high >= tert.n_mid >= tert.n_low - 1


def test_difficulty_tertile_empty_input():
    tert = difficulty_tertile({})
    assert tert.n_low == 0 and tert.n_mid == 0 and tert.n_high == 0
    assert tert.by_question == {}


def test_paired_bootstrap_by_difficulty_three_tiers():
    """Spec scenario: synthetic data where only 'high' tier has a difference.

    Construct: low tier all delta=0, mid tier all delta=0, high tier all
    delta=0.05. Per-tier paired bootstrap should give CI excluding 0 only
    on high.
    """
    bs_a_by_q: dict[str, float] = {}
    bs_b_by_q: dict[str, float] = {}
    gammas: dict[str, float] = {}
    for i in range(30):
        # Low tertile: gamma 0.0-0.1, no model difference.
        bs_a_by_q[f"l{i}"] = 0.10
        bs_b_by_q[f"l{i}"] = 0.10
        gammas[f"l{i}"] = 0.0 + i * 0.001
    for i in range(30):
        bs_a_by_q[f"m{i}"] = 0.20
        bs_b_by_q[f"m{i}"] = 0.20
        gammas[f"m{i}"] = 0.20 + i * 0.001
    for i in range(30):
        # High tertile: model A is 0.05 better.
        bs_a_by_q[f"h{i}"] = 0.30
        bs_b_by_q[f"h{i}"] = 0.35
        gammas[f"h{i}"] = 0.40 + i * 0.001
    tert = difficulty_tertile(gammas)
    out = paired_bootstrap_by_difficulty(bs_a_by_q, bs_b_by_q, tert, n_bootstrap=2000)
    # All three tiers populated
    assert "low" in out and "mid" in out and "high" in out
    # Low and mid: CI brackets 0.
    assert out["low"].ci_low <= 0 <= out["low"].ci_high
    assert out["mid"].ci_low <= 0 <= out["mid"].ci_high
    # High: CI strictly negative (A is better than B).
    assert out["high"].ci_high < 0
    assert abs(out["high"].delta_mean - (-0.05)) < 1e-9


# --------------------------------------------------------------------------- #
# Posterior over BI
# --------------------------------------------------------------------------- #


def test_normal_cdf_basic():
    """erf-based CDF: Φ(0) = 0.5, Φ(2) ≈ 0.977, Φ(-2) ≈ 0.023."""
    assert abs(_normal_cdf(0.0) - 0.5) < 1e-9
    assert abs(_normal_cdf(2.0) - 0.9772) < 0.01
    assert abs(_normal_cdf(-2.0) - 0.0228) < 0.01


def test_posterior_a_better_when_a_consistently_smaller():
    """A's BS uniformly smaller → posterior should be ~1.0."""
    bs_a = [0.05] * 50
    bs_b = [0.10] * 50
    p = posterior_a_better_than_b(bs_a, bs_b, n_bootstrap=1000)
    assert p > 0.99


def test_posterior_a_better_when_a_consistently_larger():
    """A's BS uniformly larger → posterior should be ~0.0."""
    bs_a = [0.30] * 50
    bs_b = [0.05] * 50
    p = posterior_a_better_than_b(bs_a, bs_b, n_bootstrap=1000)
    assert p < 0.01


def test_posterior_a_better_tied():
    """Identical BS arrays → posterior at 0.5 (each bootstrap is a tie, half-credit)."""
    bs_a = [0.1] * 30
    bs_b = [0.1] * 30
    p = posterior_a_better_than_b(bs_a, bs_b, n_bootstrap=2000)
    assert 0.49 <= p <= 0.51


def test_posterior_normal_fit_matches_bootstrap_on_large_sample():
    """For N=200 with Gaussian-ish deltas, both methods agree to ±0.05."""
    random.seed(101)
    bs_a = [0.10 + random.gauss(0, 0.05) for _ in range(200)]
    bs_b = [0.15 + random.gauss(0, 0.05) for _ in range(200)]
    p_normal = posterior_normal_fit(bs_a, bs_b)
    p_boot = posterior_a_better_than_b(bs_a, bs_b, n_bootstrap=2000)
    assert abs(p_normal - p_boot) < 0.05


def test_posterior_validates_lengths():
    with pytest.raises(ValueError):
        posterior_a_better_than_b([0.1], [0.1, 0.2])
    with pytest.raises(ValueError):
        posterior_normal_fit([0.1], [0.1, 0.2])


# --------------------------------------------------------------------------- #
# Pairwise orchestrator
# --------------------------------------------------------------------------- #


def test_pairwise_paired_bootstrap_basic():
    """3 models → C(3, 2) = 3 ordered pairs."""
    bs = {
        "model_a": {"q1": 0.10, "q2": 0.20, "q3": 0.30},
        "model_b": {"q1": 0.05, "q2": 0.15, "q3": 0.25},
        "model_c": {"q1": 0.30, "q2": 0.30, "q3": 0.30},
    }
    pairs = pairwise_paired_bootstrap(bs, n_bootstrap=500)
    assert len(pairs) == 3
    # All ordered (a<b<c lexicographically)
    assert pairs[0].model_a == "model_a" and pairs[0].model_b == "model_b"
    assert pairs[1].model_a == "model_a" and pairs[1].model_b == "model_c"
    assert pairs[2].model_a == "model_b" and pairs[2].model_b == "model_c"
    # Holm-adjusted p-values populated
    for p in pairs:
        assert p.p_holm is not None
        assert 0 <= p.p_holm <= 1


def test_pairwise_paired_bootstrap_intersection_only():
    """Only questions that both models answered enter the comparison."""
    bs = {
        "model_a": {"q1": 0.10, "q2": 0.20},
        "model_b": {"q2": 0.15, "q3": 0.25},
    }
    pairs = pairwise_paired_bootstrap(bs, n_bootstrap=500)
    assert len(pairs) == 1
    # Common questions: only {q2}
    assert pairs[0].n_questions == 1


def test_pairwise_paired_bootstrap_empty_input():
    assert pairwise_paired_bootstrap({}) == []
