"""Statistical inference for model comparisons (Phase 2 task 21).

Implements the inference stack from `ANALYSIS_DESIGN_v4.md §3.4` and
`specs/probabilistic-analysis/spec.md`:

* `paired_bootstrap` — $B = 5000$ paired resamples on per-question Brier
  scores; returns 95% CI and two-sided p-value.
* `holm_bonferroni` — FWER control across multi-comparison families;
  returns adjusted p-values in original order.
* `difficulty_tertile` — tertile split on per-question $\\gamma_q$ (the
  baseline difficulty proxy used by ABI).
* `paired_bootstrap_by_difficulty` — runs `paired_bootstrap` once per
  tertile so a model pair can be compared on equal-difficulty subsets.
* `posterior_a_better_than_b` — direct Monte-Carlo posterior of
  $\\Pr(\\mathrm{BI}_A > \\mathrm{BI}_B)$ on bootstrap samples.
* `posterior_normal_fit` — closed-form normal approximation of the same
  quantity; reported as a sanity-check side channel.

Pure Python with `random` for the bootstrap RNG and `math.erf` for the
normal CDF — no numpy / scipy dependency.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# Paired bootstrap
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PairedBootstrapResult:
    """Output of `paired_bootstrap`. `bootstrap_means` is the resampled mean
    delta distribution — kept around so callers can derive posterior
    probabilities without repeating the bootstrap."""

    delta_mean: float
    ci_low: float
    ci_high: float
    p_two_sided: float
    n_bootstrap: int
    n_questions: int
    bootstrap_means: list[float]


def paired_bootstrap(
    bs_a: list[float],
    bs_b: list[float],
    *,
    n_bootstrap: int = 5000,
    seed: int = 42,
    ci_alpha: float = 0.05,
) -> PairedBootstrapResult:
    """Paired bootstrap on per-question Brier-score differences.

    The two arrays MUST be aligned by question (so `bs_a[i]` and `bs_b[i]`
    refer to the same question). Each bootstrap iteration samples $N$
    indices with replacement and uses them to subset BOTH arrays — this is
    the "paired" property that controls for question-level variance, which
    paper §G.2 quantifies at 62% of total variance.

    The two-sided p-value is the standard $2 \\min(\\Pr(\\Delta \\le 0),
    \\Pr(\\Delta \\ge 0))$ on the bootstrap distribution; without recentring
    it tests "is the observed delta consistent with zero?"
    """
    if len(bs_a) != len(bs_b):
        raise ValueError(
            f"paired arrays must be the same length: {len(bs_a)} vs {len(bs_b)}"
        )
    n = len(bs_a)
    if n == 0:
        return PairedBootstrapResult(
            delta_mean=0.0, ci_low=0.0, ci_high=0.0, p_two_sided=1.0,
            n_bootstrap=0, n_questions=0, bootstrap_means=[],
        )
    deltas = [float(a) - float(b) for a, b in zip(bs_a, bs_b)]
    delta_mean = sum(deltas) / n

    rng = random.Random(seed)
    bootstrap_means: list[float] = []
    for _ in range(n_bootstrap):
        s = 0.0
        for _ in range(n):
            s += deltas[rng.randrange(n)]
        bootstrap_means.append(s / n)

    sorted_means = sorted(bootstrap_means)
    lo_idx = max(0, int(ci_alpha / 2 * n_bootstrap))
    hi_idx = min(n_bootstrap - 1, int((1 - ci_alpha / 2) * n_bootstrap) - 1)
    ci_low = sorted_means[lo_idx]
    ci_high = sorted_means[hi_idx]

    n_le_0 = sum(1 for m in bootstrap_means if m <= 0)
    n_ge_0 = sum(1 for m in bootstrap_means if m >= 0)
    p_low = n_le_0 / n_bootstrap
    p_high = n_ge_0 / n_bootstrap
    p_two = min(1.0, 2.0 * min(p_low, p_high))

    return PairedBootstrapResult(
        delta_mean=delta_mean,
        ci_low=ci_low,
        ci_high=ci_high,
        p_two_sided=p_two,
        n_bootstrap=n_bootstrap,
        n_questions=n,
        bootstrap_means=bootstrap_means,
    )


# --------------------------------------------------------------------------- #
# Holm-Bonferroni
# --------------------------------------------------------------------------- #


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni FWER control. Returns adjusted p-values aligned with input order.

    Algorithm:
    1. Sort p-values ascending; let $i = 0, 1, \\dots, n-1$ be sorted ranks.
    2. For each rank $i$: $p^*_{(i)} = (n - i) \\cdot p_{(i)}$ (clipped to 1).
    3. Apply the cumulative max from left so adjusted p-values are monotone.
    4. Map back to the original input positions.
    """
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    cur_max = 0.0
    for rank, i in enumerate(indexed):
        adj = float(p_values[i]) * (n - rank)
        if adj > 1.0:
            adj = 1.0
        if adj > cur_max:
            cur_max = adj
        adjusted[i] = cur_max
    return adjusted


# --------------------------------------------------------------------------- #
# Difficulty tertile
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DifficultyTertile:
    """A question's difficulty bucket plus the boundary $\\gamma$ thresholds."""

    by_question: dict[str, str]   # question_id → "low" / "mid" / "high"
    threshold_low: float          # γ at the low/mid boundary
    threshold_high: float         # γ at the mid/high boundary
    n_low: int
    n_mid: int
    n_high: int


def difficulty_tertile(gammas: dict[str, float]) -> DifficultyTertile:
    """Split questions into low / mid / high tertiles by per-question $\\gamma$.

    Sorts questions ascending by $\\gamma$ (lower = easier — small baseline
    BS = baseline already gets it right). Splits into thirds; ties at the
    boundary go into the lower tertile, which is a deterministic choice
    that keeps results reproducible across runs.
    """
    if not gammas:
        return DifficultyTertile(
            by_question={}, threshold_low=0.0, threshold_high=0.0,
            n_low=0, n_mid=0, n_high=0,
        )
    items = sorted(gammas.items(), key=lambda kv: kv[1])
    n = len(items)
    one_third = n // 3
    two_third = (2 * n) // 3
    by_q: dict[str, str] = {}
    for i, (q, _) in enumerate(items):
        if i < one_third:
            by_q[q] = "low"
        elif i < two_third:
            by_q[q] = "mid"
        else:
            by_q[q] = "high"
    threshold_low = items[one_third][1] if one_third < n else items[-1][1]
    threshold_high = items[two_third][1] if two_third < n else items[-1][1]
    n_low = sum(1 for v in by_q.values() if v == "low")
    n_mid = sum(1 for v in by_q.values() if v == "mid")
    n_high = sum(1 for v in by_q.values() if v == "high")
    return DifficultyTertile(
        by_question=by_q,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
        n_low=n_low,
        n_mid=n_mid,
        n_high=n_high,
    )


def paired_bootstrap_by_difficulty(
    bs_a_by_q: dict[str, float],
    bs_b_by_q: dict[str, float],
    tertile: DifficultyTertile,
    *,
    n_bootstrap: int = 5000,
    seed: int = 42,
    ci_alpha: float = 0.05,
) -> dict[str, PairedBootstrapResult]:
    """Per-tertile paired bootstrap. Returns `{tier: result}` for low/mid/high."""
    out: dict[str, PairedBootstrapResult] = {}
    for tier in ("low", "mid", "high"):
        qs = [q for q, t in tertile.by_question.items() if t == tier]
        a_arr = [bs_a_by_q[q] for q in qs if q in bs_a_by_q and q in bs_b_by_q]
        b_arr = [bs_b_by_q[q] for q in qs if q in bs_a_by_q and q in bs_b_by_q]
        out[tier] = paired_bootstrap(
            a_arr, b_arr,
            n_bootstrap=n_bootstrap,
            seed=seed,
            ci_alpha=ci_alpha,
        )
    return out


# --------------------------------------------------------------------------- #
# Posterior over BI
# --------------------------------------------------------------------------- #


def _normal_cdf(z: float) -> float:
    """Standard normal CDF using `math.erf`."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def posterior_a_better_than_b(
    bs_a: list[float],
    bs_b: list[float],
    *,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> float:
    """$\\Pr(\\mathrm{BI}_A > \\mathrm{BI}_B)$ via paired bootstrap.

    BI is a monotone-decreasing function of mean BS, so the event is
    equivalent to $\\Pr(\\overline{BS}_A < \\overline{BS}_B)$. We compute it
    directly by counting bootstrap iterations where A's mean BS is smaller.
    """
    if len(bs_a) != len(bs_b):
        raise ValueError("paired arrays must be equal length")
    n = len(bs_a)
    if n == 0:
        return 0.5
    rng = random.Random(seed)
    n_a_better = 0
    for _ in range(n_bootstrap):
        sa = 0.0
        sb = 0.0
        for _ in range(n):
            j = rng.randrange(n)
            sa += bs_a[j]
            sb += bs_b[j]
        if sa < sb:
            n_a_better += 1
        elif sa == sb:
            n_a_better += 0.5  # half-credit on ties
    return n_a_better / n_bootstrap


def posterior_normal_fit(bs_a: list[float], bs_b: list[float]) -> float:
    """Normal-fit closed form: $\\Pr(\\Delta < 0)$ under $\\mathcal{N}(\\bar\\Delta, s^2/n)$.

    Reported alongside the bootstrap-based estimate as a sanity check —
    the two should agree closely on $N \\ge 100$ when the per-question
    delta distribution is roughly symmetric.
    """
    if len(bs_a) != len(bs_b):
        raise ValueError("paired arrays must be equal length")
    n = len(bs_a)
    if n == 0:
        return 0.5
    deltas = [float(a) - float(b) for a, b in zip(bs_a, bs_b)]
    mean = sum(deltas) / n
    if n < 2:
        return 1.0 if mean < 0 else (0.0 if mean > 0 else 0.5)
    var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    if var <= 0:
        return 1.0 if mean < 0 else (0.0 if mean > 0 else 0.5)
    se = math.sqrt(var / n)
    z = -mean / se
    return _normal_cdf(z)


# --------------------------------------------------------------------------- #
# Convenience: pairwise comparison across many models
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelPairResult:
    """Bundled per-pair statistics. Fields beyond `pair` are all in BS units
    (the writer converts to BI when serialising)."""

    model_a: str
    model_b: str
    delta_bs_mean: float
    ci_low: float
    ci_high: float
    p_raw: float
    p_holm: float | None
    posterior_a_better: float
    n_questions: int


def pairwise_paired_bootstrap(
    bs_by_model_qid: dict[str, dict[str, float]],
    *,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> list[ModelPairResult]:
    """Run paired bootstrap on every ordered pair of models, return Holm-adjusted p-values.

    Pairs are ordered (model_a, model_b) for `model_a < model_b` lexicographically;
    a positive `delta_bs_mean` means A is WORSE than B (higher BS), so positive
    `posterior_a_better` reflects "P(A's mean BS smaller than B's)".

    Pairs missing a question in either model are silently dropped — the
    intersection is the only fair comparison surface.
    """
    models = sorted(bs_by_model_qid.keys())
    pairs: list[ModelPairResult] = []
    raw_p: list[float] = []
    for i, ma in enumerate(models):
        for mb in models[i + 1:]:
            common = sorted(set(bs_by_model_qid[ma]) & set(bs_by_model_qid[mb]))
            bs_a = [bs_by_model_qid[ma][q] for q in common]
            bs_b = [bs_by_model_qid[mb][q] for q in common]
            res = paired_bootstrap(
                bs_a, bs_b, n_bootstrap=n_bootstrap, seed=seed,
            )
            posterior = posterior_a_better_than_b(
                bs_a, bs_b, n_bootstrap=n_bootstrap, seed=seed,
            )
            pairs.append(
                ModelPairResult(
                    model_a=ma,
                    model_b=mb,
                    delta_bs_mean=res.delta_mean,
                    ci_low=res.ci_low,
                    ci_high=res.ci_high,
                    p_raw=res.p_two_sided,
                    p_holm=None,
                    posterior_a_better=posterior,
                    n_questions=len(common),
                )
            )
            raw_p.append(res.p_two_sided)
    if pairs:
        adjusted = holm_bonferroni(raw_p)
        # Re-emit with p_holm filled in.
        pairs = [
            ModelPairResult(
                model_a=p.model_a,
                model_b=p.model_b,
                delta_bs_mean=p.delta_bs_mean,
                ci_low=p.ci_low,
                ci_high=p.ci_high,
                p_raw=p.p_raw,
                p_holm=adjusted[idx],
                posterior_a_better=p.posterior_a_better,
                n_questions=p.n_questions,
            )
            for idx, p in enumerate(pairs)
        ]
    return pairs


__all__ = [
    "PairedBootstrapResult",
    "DifficultyTertile",
    "ModelPairResult",
    "paired_bootstrap",
    "paired_bootstrap_by_difficulty",
    "holm_bonferroni",
    "difficulty_tertile",
    "posterior_a_better_than_b",
    "posterior_normal_fit",
    "pairwise_paired_bootstrap",
]
