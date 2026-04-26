"""Inter-trial consistency metrics for K-trial parallel sampling (v5 Phase B).

These metrics are only meaningful when $K \\ge 2$ — they extract structure
from the *agreement* across the K parallel trials. K=1 runs degrade
gracefully: every aggregator returns None and ConsistencyReport carries
None on every field, leaving the writer to emit blank cells.

Family:

* `fleiss_kappa(samples_by_q, options_map)` — Fleiss' κ multi-rater
  agreement; single-choice votes are letter argmax, multi-label votes are
  per-label binary Fleiss averaged across labels. Mixed-question-type runs
  use per-question-type weighted averaging.
* `mean_entropy(samples_by_q, options_map)` — Shannon entropy per question
  on $\\hat{p}_l = n_{q,l}/K$ (single) or per-label binary entropy mean
  (multi); averaged across questions.
* `entropy_accuracy_bins(samples_by_q, gt_map, *, n_buckets=3)` — per-model
  tertile bucketing of questions by predictive entropy. For each bucket
  reports n_questions, $H$ range, Acc, MV Acc, and Fleiss κ. The
  per-model boundaries make buckets diagnostic for that model only;
  cross-model bucket boundaries differ.
* `mean_vci(samples_by_q, options_map)` / `vci_per_question` — Vote
  Concentration Index $\\max_l n_{q,l} / K$ per question.
* `mvg(samples, gt_map)` — Majority Vote Gain = MV_Acc - Pass@1_Acc.
* `build_consistency_report(...)` — top-level entry building a
  `ConsistencyReport` dataclass with all five aggregates + bucket list.

The Fleiss formula handles per-question $K_q$ variability (some trials may
have parse_ok=0 / parsed_letters=None and are excluded). Questions with
$K_q < 2$ are silently dropped — they cannot contribute consistency signal.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..prompts import index_to_letter
from .accuracy import _aggregate
from .flatten import SampleRow

# `log(0)` guard for entropy. Pure-numerical safety; small enough not to
# move the legit entropy values.
_LOG_EPS: float = 1e-10


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _parsed_letters_per_trial(samples: list[SampleRow]) -> list[frozenset[str]]:
    """Filter the K samples for a question down to those with a parsed letter set.

    A trial with `parse_ok=0` or `final_answer_letters=None` doesn't qualify
    as a "rater" — Fleiss / entropy / VCI all need the per-trial vote.
    """
    out: list[frozenset[str]] = []
    for s in samples:
        if not s.is_eligible:
            continue
        parsed = s.parsed_letters
        if parsed is None:
            continue
        out.append(parsed)
    return out


def _vote_counts_single(parsed_per_trial: list[frozenset[str]], k: int) -> list[int]:
    """For single-choice: $n_{q,l}$ over the K letters for one question.

    A trial that voted for a letter outside the option range is silently
    dropped (defensive — parser should have caught it).
    """
    counts = [0] * k
    valid_letters = [index_to_letter(i) for i in range(k)]
    letter_to_idx = {l: i for i, l in enumerate(valid_letters)}
    for parsed in parsed_per_trial:
        if not parsed:
            continue
        # single-choice should have exactly one letter; if multiple, take
        # the first (sorted for determinism).
        letter = sorted(parsed)[0]
        idx = letter_to_idx.get(letter)
        if idx is not None:
            counts[idx] += 1
    return counts


def _per_label_select_counts_multi(
    parsed_per_trial: list[frozenset[str]], k: int
) -> list[int]:
    """For multi-label: per-label "selected" count over K trials."""
    counts = [0] * k
    valid_letters = [index_to_letter(i) for i in range(k)]
    for parsed in parsed_per_trial:
        for i, letter in enumerate(valid_letters):
            if letter in parsed:
                counts[i] += 1
    return counts


# --------------------------------------------------------------------------- #
# Fleiss' κ
# --------------------------------------------------------------------------- #


def _fleiss_kappa_from_counts(
    n_matrix: list[list[int]],
    k_per_q: list[int],
) -> float | None:
    """Standard Fleiss' κ on a per-question $n_{i,j}$ count matrix.

    Allows per-question $K_i$ to differ (Fleiss 1971 §3 generalised form).
    Skips questions with $K_i < 2$ since their per-question agreement is
    undefined. Returns None if no question has $K_i \\ge 2$ or all
    categories are degenerate.

    `n_matrix[i][j]` = count of category j votes on question i.
    `k_per_q[i]` = total raters on question i.

    All rows MUST share the same number of categories — Fleiss κ's marginal
    category proportions $p_j$ only have meaning inside a fixed category
    space. Callers with mixed-k items must stratify by k first (see
    `fleiss_kappa_single`).
    """
    valid = [
        (n_row, k) for n_row, k in zip(n_matrix, k_per_q) if k >= 2
    ]
    if not valid:
        return None
    n_q = len(valid)
    n_categories = len(valid[0][0])
    if n_categories == 0:
        return None
    if any(len(n_row) != n_categories for n_row, _ in valid):
        # Defensive — the older single-pool implementation passed mixed-k
        # rows here and silently produced an IndexError downstream. Surface
        # the misuse so future regressions fail loudly at the call site.
        raise ValueError(
            "_fleiss_kappa_from_counts requires all rows to share the same "
            "number of categories; stratify by k before calling."
        )

    # Per-question observed agreement: P_i = (sum_j n_ij^2 - K_i) / (K_i (K_i - 1))
    p_i_values: list[float] = []
    for n_row, k in valid:
        sum_sq = sum(n * n for n in n_row)
        denom = k * (k - 1)
        if denom == 0:
            continue
        p_i_values.append((sum_sq - k) / denom)
    if not p_i_values:
        return None
    p_bar = sum(p_i_values) / len(p_i_values)

    # Marginal category proportions: p_j = sum_i n_ij / sum_i K_i
    total_votes = sum(k for _, k in valid)
    if total_votes == 0:
        return None
    cat_totals = [0] * n_categories
    for n_row, _ in valid:
        for j, n in enumerate(n_row):
            cat_totals[j] += n
    p_j = [t / total_votes for t in cat_totals]
    p_e = sum(p ** 2 for p in p_j)

    if abs(1.0 - p_e) < 1e-12:
        # All raters agreed on one category every question — perfect agreement
        # AND zero expected variance; conventional Fleiss treats this as 1.0.
        return 1.0
    return (p_bar - p_e) / (1.0 - p_e)


def fleiss_kappa_single(
    samples_by_q: dict[str, list[SampleRow]],
    k_per_q: dict[str, int],
) -> float | None:
    """Fleiss κ on single-choice questions; vote = letter argmax per trial.

    `k_per_q` carries the question's option count (number of categories).
    Returns None when no question has $K_q \\ge 2$ effective trials.

    Stratified by k: standard Fleiss κ requires a shared category space, so
    questions are bucketed by their option count, κ is computed per stratum,
    then averaged across strata weighted by question count. Single-stratum
    runs collapse to the textbook formula.
    """
    by_k: dict[int, tuple[list[list[int]], list[int]]] = {}
    for qid, samples in samples_by_q.items():
        k = k_per_q.get(qid)
        if k is None or k <= 0:
            continue
        parsed = _parsed_letters_per_trial(samples)
        if len(parsed) < 2:
            continue
        counts = _vote_counts_single(parsed, k)
        n_rows, keffs = by_k.setdefault(k, ([], []))
        n_rows.append(counts)
        keffs.append(sum(counts))
    if not by_k:
        return None

    parts: list[tuple[float, int]] = []  # (kappa, n_questions_in_stratum)
    for _k, (n_rows, keffs) in by_k.items():
        kappa = _fleiss_kappa_from_counts(n_rows, keffs)
        if kappa is not None:
            parts.append((kappa, len(n_rows)))
    if not parts:
        return None
    total_w = sum(w for _, w in parts)
    if total_w == 0:
        return None
    return sum(kappa * w for kappa, w in parts) / total_w


def fleiss_kappa_multi_per_label(
    samples_by_q: dict[str, list[SampleRow]],
    k_per_q: dict[str, int],
) -> float | None:
    """Fleiss κ on multi-label questions; per-label binary κ then mean.

    Each label gets an independent binary Fleiss κ (categories: selected /
    not selected), then we average across labels and across questions.
    Per-question structure: each question contributes $k_q$ binary κ
    estimates (one per option), all averaged into the question's κ.
    """
    per_question_kappas: list[float] = []
    for qid, samples in samples_by_q.items():
        k = k_per_q.get(qid)
        if k is None or k <= 0:
            continue
        parsed = _parsed_letters_per_trial(samples)
        K_eff = len(parsed)
        if K_eff < 2:
            continue
        select_counts = _per_label_select_counts_multi(parsed, k)
        # Per-label binary Fleiss κ for this question.
        label_kappas: list[float] = []
        for sel in select_counts:
            n_select = sel
            n_not = K_eff - sel
            # Single-question binary Fleiss with one row.
            kappa = _fleiss_kappa_from_counts(
                [[n_select, n_not]], [K_eff]
            )
            if kappa is not None:
                label_kappas.append(kappa)
        if label_kappas:
            per_question_kappas.append(sum(label_kappas) / len(label_kappas))
    if not per_question_kappas:
        return None
    return sum(per_question_kappas) / len(per_question_kappas)


def fleiss_kappa(
    samples_by_q: dict[str, list[SampleRow]],
    options_map: dict[str, list[str]],
) -> float | None:
    """Mixed-question-type Fleiss κ; weighted-average across single/multi pools.

    Splits questions by `choice_type` of the first eligible sample, runs
    `fleiss_kappa_single` or `fleiss_kappa_multi_per_label` on each pool,
    then weights by question count.
    """
    single_qids: dict[str, list[SampleRow]] = {}
    multi_qids: dict[str, list[SampleRow]] = {}
    k_per_q: dict[str, int] = {}
    for qid, samples in samples_by_q.items():
        opts = options_map.get(qid)
        if opts is None or not opts:
            continue
        if not samples:
            continue
        ctype = samples[0].choice_type
        k_per_q[qid] = len(opts)
        if ctype == "single":
            single_qids[qid] = samples
        else:
            multi_qids[qid] = samples

    parts: list[tuple[float, int]] = []  # (kappa, n_questions_in_pool)
    if single_qids:
        kappa_s = fleiss_kappa_single(single_qids, k_per_q)
        if kappa_s is not None:
            parts.append((kappa_s, len(single_qids)))
    if multi_qids:
        kappa_m = fleiss_kappa_multi_per_label(multi_qids, k_per_q)
        if kappa_m is not None:
            parts.append((kappa_m, len(multi_qids)))
    if not parts:
        return None
    total_w = sum(w for _, w in parts)
    if total_w == 0:
        return None
    return sum(k * w for k, w in parts) / total_w


# --------------------------------------------------------------------------- #
# Predictive entropy
# --------------------------------------------------------------------------- #


def prediction_entropy_single(
    samples_for_q: list[SampleRow], k: int
) -> float | None:
    """Shannon entropy on $\\hat{p}_l = n_{q,l}/K$. K < 2 returns None."""
    if k <= 0:
        return None
    parsed = _parsed_letters_per_trial(samples_for_q)
    K_eff = len(parsed)
    if K_eff < 2:
        return None
    counts = _vote_counts_single(parsed, k)
    h = 0.0
    for n in counts:
        p = n / K_eff
        if p <= 0:
            continue
        h -= p * math.log2(p + _LOG_EPS)
    return h


def prediction_entropy_multi(
    samples_for_q: list[SampleRow], k: int
) -> float | None:
    """Per-label binary entropy mean: $\\frac{1}{k}\\sum_l h(\\hat{p}_l)$."""
    if k <= 0:
        return None
    parsed = _parsed_letters_per_trial(samples_for_q)
    K_eff = len(parsed)
    if K_eff < 2:
        return None
    select_counts = _per_label_select_counts_multi(parsed, k)
    total = 0.0
    for n in select_counts:
        p = n / K_eff
        h = 0.0
        if 0 < p < 1:
            h = -(p * math.log2(p + _LOG_EPS) + (1 - p) * math.log2(1 - p + _LOG_EPS))
        total += h
    return total / k


def _entropy_for_question(
    samples_for_q: list[SampleRow], options: list[str]
) -> float | None:
    """Dispatch single vs multi entropy by the first sample's choice_type."""
    if not samples_for_q or not options:
        return None
    ctype = samples_for_q[0].choice_type
    k = len(options)
    if ctype == "single":
        return prediction_entropy_single(samples_for_q, k)
    return prediction_entropy_multi(samples_for_q, k)


def mean_entropy(
    samples_by_q: dict[str, list[SampleRow]],
    options_map: dict[str, list[str]],
) -> float | None:
    """Across-question mean of per-question entropy. K_q < 2 questions skipped."""
    values: list[float] = []
    for qid, samples in samples_by_q.items():
        opts = options_map.get(qid)
        if opts is None:
            continue
        h = _entropy_for_question(samples, opts)
        if h is not None:
            values.append(h)
    if not values:
        return None
    return sum(values) / len(values)


# --------------------------------------------------------------------------- #
# Vote Concentration Index (VCI)
# --------------------------------------------------------------------------- #


def vci_per_question(samples_for_q: list[SampleRow]) -> float | None:
    """$\\max_l n_{q,l} / K$ — fraction of trials voting for the modal letter set.

    Single-choice: modal letter. Multi-label: modal letter set (treats the
    full set as the "category" — same convention as majority_vote_accuracy).
    K < 2 returns None.
    """
    parsed = _parsed_letters_per_trial(samples_for_q)
    K_eff = len(parsed)
    if K_eff < 2:
        return None
    counts = Counter(parsed)
    max_n = max(counts.values())
    return max_n / K_eff


def mean_vci(
    samples_by_q: dict[str, list[SampleRow]],
    options_map: dict[str, list[str]],
) -> float | None:
    """Across-question mean of VCI."""
    values: list[float] = []
    for qid, samples in samples_by_q.items():
        if options_map.get(qid) is None:
            continue
        v = vci_per_question(samples)
        if v is not None:
            values.append(v)
    if not values:
        return None
    return sum(values) / len(values)


# --------------------------------------------------------------------------- #
# Majority Vote Gain (MVG)
# --------------------------------------------------------------------------- #


def mvg(
    samples: list[SampleRow],
    gt_map: dict[str, frozenset[str]],
) -> float | None:
    """MV_Acc - Pass@1_Acc, computed via the existing `_aggregate` helper.

    Returns None when sampling_n < 2 (no MV signal possible) or when
    `_aggregate` returns None for either metric.
    """
    if not samples:
        return None
    # Determine effective K from samples (per-question max sample_idx + 1).
    by_q: dict[str, set[int]] = {}
    for s in samples:
        by_q.setdefault(s.question_id, set()).add(s.sample_idx)
    if not by_q:
        return None
    sampling_n = max(len(idxs) for idxs in by_q.values())
    if sampling_n < 2:
        return None
    agg = _aggregate(samples, sampling_n=sampling_n, gt_map=gt_map)
    if agg.pass_at_1_avg is None or agg.majority_vote_accuracy is None:
        return None
    return agg.majority_vote_accuracy - agg.pass_at_1_avg


# --------------------------------------------------------------------------- #
# Entropy-accuracy joint analysis (per-model tertile bucketing)
# --------------------------------------------------------------------------- #


def _split_quantile_indices(n: int, n_buckets: int) -> list[tuple[int, int]]:
    """Slice $[0, n)$ into `n_buckets` contiguous halves, distributing the
    remainder to the **leftmost** buckets (matching spec scenario "low gets
    the extra question on N % n != 0").

    For n=32, n_buckets=3 → [(0, 11), (11, 22), (22, 32)] = 11/11/10."""
    base = n // n_buckets
    extras = n % n_buckets
    out: list[tuple[int, int]] = []
    pos = 0
    for i in range(n_buckets):
        size = base + (1 if i < extras else 0)
        out.append((pos, pos + size))
        pos += size
    return out


def _bucket_label(i: int, n_buckets: int) -> str:
    """Pretty bucket name for n_buckets ∈ {2, 3, 4}; numeric otherwise."""
    if n_buckets == 3 and i in (0, 1, 2):
        return ["low", "mid", "high"][i]
    if n_buckets == 2 and i in (0, 1):
        return ["low", "high"][i]
    return f"q{i}"


def entropy_accuracy_bins(
    samples_by_q: dict[str, list[SampleRow]],
    gt_map: dict[str, frozenset[str]],
    options_map: dict[str, list[str]],
    *,
    n_buckets: int = 3,
) -> list[dict[str, Any]]:
    """Per-model tertile bucketing on per-question predictive entropy.

    For each (question, model):
    1. Compute $H_q$ via `_entropy_for_question`. Skip questions with K_q < 2.
    2. Sort ascending by $H_q$.
    3. Split into `n_buckets` quantile buckets (extras go to leftmost / "low").
    4. Per bucket: compute Acc (pass@1 across all eligible samples in those
       questions), MV Acc, and Fleiss' κ.

    Returns a list of dicts with `bucket_label / n_questions / h_lo / h_hi /
    acc / mv_acc / fleiss_kappa`. Empty list when no question has K_q ≥ 2.

    Per-model boundaries make this diagnostic for the model only — boundaries
    differ across models, so cross-model comparison of bucket cells is
    intentionally unavailable (Decision 5).
    """
    # Step 1: compute H_q per question.
    h_per_q: list[tuple[str, float]] = []
    for qid, samples in samples_by_q.items():
        opts = options_map.get(qid)
        if opts is None:
            continue
        h = _entropy_for_question(samples, opts)
        if h is None:
            continue
        h_per_q.append((qid, h))

    if not h_per_q or n_buckets <= 0:
        return []

    # Step 2: sort by H ascending.
    h_per_q.sort(key=lambda kv: kv[1])
    n = len(h_per_q)
    boundaries = _split_quantile_indices(n, n_buckets)

    out: list[dict[str, Any]] = []
    for i, (lo_idx, hi_idx) in enumerate(boundaries):
        if lo_idx >= hi_idx:
            continue
        bucket_qids = [qid for qid, _ in h_per_q[lo_idx:hi_idx]]
        bucket_h = [h for _, h in h_per_q[lo_idx:hi_idx]]
        h_lo = bucket_h[0] if bucket_h else None
        h_hi = bucket_h[-1] if bucket_h else None

        # Per-bucket samples, gt subset, options subset.
        bucket_samples: list[SampleRow] = []
        bucket_samples_by_q: dict[str, list[SampleRow]] = {}
        bucket_gt = {}
        bucket_opts = {}
        for qid in bucket_qids:
            ss = samples_by_q.get(qid, [])
            bucket_samples.extend(ss)
            bucket_samples_by_q[qid] = ss
            if qid in gt_map:
                bucket_gt[qid] = gt_map[qid]
            if qid in options_map:
                bucket_opts[qid] = options_map[qid]

        # Per-bucket Acc / MV Acc via existing helper.
        sampling_n = max(
            (len({s.sample_idx for s in ss}) for ss in bucket_samples_by_q.values()),
            default=1,
        )
        agg = _aggregate(bucket_samples, sampling_n=sampling_n, gt_map=bucket_gt)
        kappa = fleiss_kappa(bucket_samples_by_q, bucket_opts)
        out.append({
            "bucket_label": _bucket_label(i, n_buckets),
            "n_questions": len(bucket_qids),
            "h_lo": h_lo,
            "h_hi": h_hi,
            "acc": agg.pass_at_1_avg,
            "mv_acc": agg.majority_vote_accuracy,
            "fleiss_kappa": kappa,
        })
    return out


# --------------------------------------------------------------------------- #
# Top-level report
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ConsistencyReport:
    """Bundled consistency aggregates for one model.

    Field None semantics:
    * `fleiss_kappa / mean_entropy / vci / mvg` are None when K < 2 across
      the run (no question has ≥ 2 effective trials);
    * `entropy_accuracy_bins` is `[]` (empty list, not None) on K=1 runs to
      keep the writer's iteration uniform (zero-row bucket table).
    """

    fleiss_kappa: float | None
    mean_entropy: float | None
    vci: float | None
    mvg: float | None
    entropy_accuracy_bins: list[dict[str, Any]]
    n_questions_used: int


def build_consistency_report(
    samples: list[SampleRow],
    gt_map: dict[str, frozenset[str]],
    options_map: dict[str, list[str]],
) -> ConsistencyReport:
    """End-to-end ConsistencyReport build for one model's samples.

    Groups samples by question, dispatches each metric, returns the bundle.
    K=1 runs (every question has only one trial) get all-None aggregates and
    an empty bucket list — graceful degradation per Decision 13.
    """
    by_q: dict[str, list[SampleRow]] = {}
    for s in samples:
        by_q.setdefault(s.question_id, []).append(s)

    # Count questions with K_q >= 2 effective parsed trials.
    n_used = 0
    for samples_q in by_q.values():
        if len(_parsed_letters_per_trial(samples_q)) >= 2:
            n_used += 1

    return ConsistencyReport(
        fleiss_kappa=fleiss_kappa(by_q, options_map),
        mean_entropy=mean_entropy(by_q, options_map),
        vci=mean_vci(by_q, options_map),
        mvg=mvg(samples, gt_map),
        entropy_accuracy_bins=entropy_accuracy_bins(by_q, gt_map, options_map),
        n_questions_used=n_used,
    )


__all__ = [
    "ConsistencyReport",
    "fleiss_kappa",
    "fleiss_kappa_single",
    "fleiss_kappa_multi_per_label",
    "prediction_entropy_single",
    "prediction_entropy_multi",
    "mean_entropy",
    "vci_per_question",
    "mean_vci",
    "mvg",
    "entropy_accuracy_bins",
    "build_consistency_report",
]
