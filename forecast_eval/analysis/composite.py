"""Composite score weighted by composite difficulty coefficient.

This module hosts the aggregation logic for every data column in
``per_model_summary.csv``: "compute per bucket -> aggregate by weights ->
drop missing buckets and renormalize".

## Buckets

The dataset is partitioned by a *composite difficulty coefficient* that
factorises along two orthogonal axes. A question-type prior assigns
weights $(0.15, 0.15, 0.70)$ to ``{yes_no, binary_named, multiple_choice}``,
and within the multiple_choice family an answering-mode split
$(0.40, 0.60)$ further partitions ``{single, multi}``. Multiplying the
two yields the four scoring buckets

* ``yes_no``        — weight 0.15
* ``binary_named``  — weight 0.15
* ``mc_single``     — weight 0.28
* ``mc_multi``      — weight 0.42

summing to one. ``yes_no`` and ``binary_named`` carry a degenerate
single-mode share of 1, so their composite coefficients coincide with
their question-type priors. Bucket key for a sample is derived from
``(question_type, choice_type)`` by :func:`bucket_of`.

## Formula

For each (model, metric):

$$
\\text{composite}_{m} = \\frac{\\sum_{b \\in B_{\\text{valid}}} w_{m,b} \\cdot v_{m,b}}{\\sum_{b \\in B_{\\text{valid}}} w_{m,b}}
$$

where :math:`B_{\\text{valid}}` is the set of buckets under that
(model, metric) whose slice measurement is not ``None`` and whose weight is
> 0. All-``None`` -> composite returns ``None``.

## Alignment with ``_SUMMARY_FIELDS``

Output columns correspond one-to-one with ``writers._SUMMARY_FIELDS``
(minus the metadata columns ``model`` / ``sampling_n``) — meaning that
reading ``per_model_composite.csv`` directly gives the difficulty-weighted
summary table, and downstream scripts only need to swap the file path.

## Per-bucket slicing for the discrete family

``fss`` / ``cohen_kappa`` / ``hamming_score`` / ``fleiss_kappa`` /
``mean_entropy`` / ``vci`` / ``mvg`` are computed globally per model at the
top of ``__init__.py`` (written into ``per_model_summary.csv``), but for
weighted bucket aggregation they must be sliced per bucket before being
computed — :func:`slice_v5_metrics_by_bucket` in this module takes care of
that.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .accuracy import Aggregate, cohen_kappa, fss, hamming_score
from .consistency import ConsistencyReport, build_consistency_report
from .flatten import SampleRow
from .proper_score import ModelProbabilisticAggregate


# Composite difficulty coefficients. The four-bucket nested weights expand
# the question-type prior (0.15/0.15/0.70) by the within-mc answering-mode
# split (0.40/0.60), so 0.70 * 0.40 = 0.28 and 0.70 * 0.60 = 0.42.
DEFAULT_WEIGHTS: dict[str, float] = {
    "yes_no": 0.15,
    "binary_named": 0.15,
    "mc_single": 0.28,
    "mc_multi": 0.42,
}

# Allowed bucket keys; mirrored in ``config.COMPOSITE_BUCKETS`` for
# Settings-side validation (so the config module does not have to import
# this module).
COMPOSITE_BUCKETS: frozenset[str] = frozenset(DEFAULT_WEIGHTS)


def bucket_of(question_type: str, choice_type: str) -> str:
    """Map a sample's ``(question_type, choice_type)`` to a composite bucket key.

    ``yes_no`` and ``binary_named`` collapse to themselves regardless of
    ``choice_type`` (structurally always single). ``multiple_choice`` splits
    into ``mc_single`` / ``mc_multi`` by the sample's ``choice_type``.
    """
    if question_type in ("yes_no", "binary_named"):
        return question_type
    if question_type == "multiple_choice":
        if choice_type in ("single", "multi"):
            return f"mc_{choice_type}"
        raise ValueError(
            f"bucket_of: choice_type must be 'single' or 'multi' for "
            f"multiple_choice; got {choice_type!r}"
        )
    raise ValueError(
        f"bucket_of: unexpected question_type {question_type!r}"
    )


# --------------------------------------------------------------------------- #
# Known-metric allowlist
# --------------------------------------------------------------------------- #


# Must match ``writers._SUMMARY_FIELDS`` (excluding ``model`` /
# ``sampling_n``); ``tests/test_composite_score.py`` has an alignment
# assertion to prevent the two sides from drifting. A misspelled metric name
# or one not on the allowlist -> ``compute_composite`` raises at the entry,
# because this must be a user typo in ``COMPOSITE_WEIGHT_OVERRIDES``.
KNOWN_METRICS: frozenset[str] = frozenset(
    {
        # accuracy.Aggregate.as_ordered_dict()
        "eligible_samples",
        "eligible_questions",
        "resolvable_samples",
        "cutoff_skip_samples",
        "cutoff_skip_rate",
        "pass_at_1_avg",
        "resolvable_rate",
        "pass_any_at_n",
        "at_least_majority_at_n",
        "at_least_all_at_n",
        "exam_score_at_n_avg",
        "majority_vote_accuracy",
        "majority_vote_resolvable_rate",
        "parse_failure_rate",
        "error_rate",
        "avg_tool_calls",
        "avg_react_steps",
        "avg_latency_ms",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "avg_reasoning_tokens",
        "avg_nudges_used",
        "final_answer_retry_rate",
        # Discrete (FSS family)
        "fss",
        "fss_pe_mean",
        "cohen_kappa",
        "hamming_score",
        # Consistency family
        "fleiss_kappa",
        "mean_entropy",
        "vci",
        "mvg",
        # Probabilistic family
        "bi",
        "bi_dec",
        "nll",
        "mbs",
        "abi_crowd",
        "abi_uniform",
        "fallback_share",
    }
)


# --------------------------------------------------------------------------- #
# Per-bucket slicing for the discrete family
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class V5SliceResult:
    """Discrete-family slice result for a single (model, bucket).

    ``fss`` carries the two columns ``fss / fss_pe_mean``, so the original
    ``fss(...)`` dict is preserved; ``cohen_kappa`` / ``hamming_score`` are
    each single values; ``consistency`` provides four columns
    ``fleiss_kappa`` / ``mean_entropy`` / ``vci`` / ``mvg``.
    """

    fss_result: dict[str, Any] | None
    cohen_kappa: float | None
    hamming_score: float | None
    consistency: ConsistencyReport | None


def slice_v5_metrics_by_bucket(
    samples: list[SampleRow],
    gt_map: dict[str, frozenset[str]],
    options_map: dict[str, list[str]],
    *,
    key_fn: Callable[[SampleRow], str],
) -> dict[str, V5SliceResult]:
    """Recompute the discrete-family metrics on each bucket grouped by ``key_fn(sample)``.

    Empty buckets (no samples) are silently skipped; per-bucket None slots
    are handled by the caller during aggregation via "drop and renormalize".
    """
    by_bucket: dict[str, list[SampleRow]] = {}
    for s in samples:
        by_bucket.setdefault(key_fn(s), []).append(s)

    out: dict[str, V5SliceResult] = {}
    for bucket, bucket_samples in by_bucket.items():
        if not bucket_samples:
            continue
        # cohen_kappa / hamming share by_q (looking only at questions in this bucket).
        by_q: dict[str, list[SampleRow]] = {}
        for s in bucket_samples:
            by_q.setdefault(s.question_id, []).append(s)
        # cohen_kappa is still meaningful when the bucket is all-multi
        # (per_e_q=0.5); when the bucket is all-single it is still computed
        # as 1/k. The function itself handles None degeneracy.
        kappa = cohen_kappa(by_q, gt_map)
        # hamming_score is only defined for multi (accuracy.hamming_score
        # internally skips single), so it must be None under the yes_no /
        # binary_named / mc_single buckets; this is the expected behavior —
        # that bucket is dropped during aggregation.
        hamming_v = hamming_score(bucket_samples, gt_map)
        fss_result = fss(bucket_samples, gt_map)
        consistency = build_consistency_report(
            bucket_samples, gt_map, options_map
        )
        out[bucket] = V5SliceResult(
            fss_result=fss_result,
            cohen_kappa=kappa,
            hamming_score=hamming_v,
            consistency=consistency,
        )
    return out


def difficulty_bucket_key(s: SampleRow) -> str:
    """``key_fn`` for :func:`slice_v5_metrics_by_bucket` and friends."""
    return bucket_of(s.question_type, s.choice_type)


# --------------------------------------------------------------------------- #
# slice -> column values
# --------------------------------------------------------------------------- #


def _aggregate_to_columns(agg: Aggregate | None) -> dict[str, float | None]:
    """``Aggregate`` -> column dict aligned with ``_SUMMARY_FIELDS_V3`` (not rounded)."""
    if agg is None:
        return {
            "eligible_samples": None,
            "eligible_questions": None,
            "resolvable_samples": None,
            "cutoff_skip_samples": None,
            "cutoff_skip_rate": None,
            "pass_at_1_avg": None,
            "resolvable_rate": None,
            "pass_any_at_n": None,
            "at_least_majority_at_n": None,
            "at_least_all_at_n": None,
            "exam_score_at_n_avg": None,
            "majority_vote_accuracy": None,
            "majority_vote_resolvable_rate": None,
            "parse_failure_rate": None,
            "error_rate": None,
            "avg_tool_calls": None,
            "avg_react_steps": None,
            "avg_latency_ms": None,
            "avg_prompt_tokens": None,
            "avg_completion_tokens": None,
            "avg_reasoning_tokens": None,
            "avg_nudges_used": None,
            "final_answer_retry_rate": None,
        }
    return {
        "eligible_samples": agg.eligible_samples,
        "eligible_questions": agg.eligible_questions,
        "resolvable_samples": agg.resolvable_samples,
        "cutoff_skip_samples": agg.cutoff_skip_samples,
        "cutoff_skip_rate": agg.cutoff_skip_rate,
        "pass_at_1_avg": agg.pass_at_1_avg,
        "resolvable_rate": agg.resolvable_rate,
        "pass_any_at_n": agg.pass_any_at_n,
        "at_least_majority_at_n": agg.at_least_majority_at_n,
        "at_least_all_at_n": agg.at_least_all_at_n,
        "exam_score_at_n_avg": agg.exam_score_at_n_avg,
        "majority_vote_accuracy": agg.majority_vote_accuracy,
        "majority_vote_resolvable_rate": agg.majority_vote_resolvable_rate,
        "parse_failure_rate": agg.parse_failure_rate,
        "error_rate": agg.error_rate,
        "avg_tool_calls": agg.avg_tool_calls,
        "avg_react_steps": agg.avg_react_steps,
        "avg_latency_ms": agg.avg_latency_ms,
        "avg_prompt_tokens": agg.avg_prompt_tokens,
        "avg_completion_tokens": agg.avg_completion_tokens,
        "avg_reasoning_tokens": agg.avg_reasoning_tokens,
        "avg_nudges_used": agg.avg_nudges_used,
        "final_answer_retry_rate": agg.final_answer_retry_rate,
    }


def _v5_slice_to_columns(res: V5SliceResult | None) -> dict[str, float | None]:
    """``V5SliceResult`` -> the eight columns fss / fss_pe_mean / cohen_kappa /
    hamming_score / fleiss_kappa / mean_entropy / vci / mvg."""
    if res is None:
        return {
            "fss": None,
            "fss_pe_mean": None,
            "cohen_kappa": None,
            "hamming_score": None,
            "fleiss_kappa": None,
            "mean_entropy": None,
            "vci": None,
            "mvg": None,
        }
    fss_value = (
        res.fss_result.get("fss") if res.fss_result is not None else None
    )
    fss_pe = (
        res.fss_result.get("mean_pe") if res.fss_result is not None else None
    )
    rep = res.consistency
    return {
        "fss": fss_value,
        "fss_pe_mean": fss_pe,
        "cohen_kappa": res.cohen_kappa,
        "hamming_score": res.hamming_score,
        "fleiss_kappa": rep.fleiss_kappa if rep is not None else None,
        "mean_entropy": rep.mean_entropy if rep is not None else None,
        "vci": rep.vci if rep is not None else None,
        "mvg": rep.mvg if rep is not None else None,
    }


def _prob_slice_to_columns(
    agg: ModelProbabilisticAggregate | None,
) -> dict[str, float | None]:
    """``ModelProbabilisticAggregate`` -> the seven columns bi / bi_dec /
    nll / mbs / abi_crowd / abi_uniform / fallback_share (not rounded; raw
    values for the aggregator)."""
    if agg is None:
        return {
            "bi": None,
            "bi_dec": None,
            "nll": None,
            "mbs": None,
            "abi_crowd": None,
            "abi_uniform": None,
            "fallback_share": None,
        }
    return {
        "bi": agg.bi,
        "bi_dec": agg.bi_dec,
        "nll": agg.nll,
        "mbs": agg.mbs,
        "abi_crowd": agg.abi_crowd,
        "abi_uniform": agg.abi_uniform,
        "fallback_share": agg.fallback_share,
    }


def collect_bucket_values(
    *,
    aggregate_slice: dict[str, dict[str, Aggregate]],
    v5_slice: dict[str, dict[str, V5SliceResult]],
    prob_slice: dict[str, dict[str, ModelProbabilisticAggregate]] | None,
) -> dict[str, dict[str, dict[str, float | None]]]:
    """Stitch multiple slice sources into ``{model: {metric: {bucket: value}}}``.

    A missing (model, bucket) is represented as "every metric in that bucket
    slot is None" rather than "the bucket does not exist" — so the aggregator
    can consistently apply "drop None buckets".
    """
    out: dict[str, dict[str, dict[str, float | None]]] = {}
    models = set(aggregate_slice) | set(v5_slice)
    if prob_slice:
        models |= set(prob_slice)
    for model in models:
        per_metric: dict[str, dict[str, float | None]] = {
            m: {} for m in KNOWN_METRICS
        }
        agg_buckets = aggregate_slice.get(model, {})
        v5_buckets = v5_slice.get(model, {})
        prob_buckets = (prob_slice.get(model, {}) if prob_slice else {}) or {}
        all_buckets = (
            set(agg_buckets) | set(v5_buckets) | set(prob_buckets)
        )
        for bucket in all_buckets:
            agg_cols = _aggregate_to_columns(agg_buckets.get(bucket))
            v5_cols = _v5_slice_to_columns(v5_buckets.get(bucket))
            prob_cols = _prob_slice_to_columns(prob_buckets.get(bucket))
            for metric, value in agg_cols.items():
                per_metric[metric][bucket] = value
            for metric, value in v5_cols.items():
                per_metric[metric][bucket] = value
            for metric, value in prob_cols.items():
                per_metric[metric][bucket] = value
        out[model] = per_metric
    return out


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CompositeMetricInfo:
    """Aggregation result and metadata for a single (model, metric)."""

    value: float | None
    buckets_used: tuple[str, ...]
    weights_used_normalized: dict[str, float]
    bucket_values: dict[str, float | None]
    weights_kind: str  # "default" | "overridden"


@dataclass(frozen=True)
class CompositeReport:
    """Aggregation result under the difficulty-weighted composite."""

    weights_default: dict[str, float]
    overrides: dict[str, dict[str, float]]
    per_model: dict[str, dict[str, CompositeMetricInfo]] = field(
        default_factory=dict
    )

    def is_overridden(self, model: str) -> bool:
        """Whether this (model) row used any override (determines the ``weights_kind`` column)."""
        if not self.overrides:
            return False
        per_metric = self.per_model.get(model, {})
        return any(
            info.weights_kind == "overridden" for info in per_metric.values()
        )


def _select_weights(
    metric: str,
    *,
    weights_default: dict[str, float],
    overrides: dict[str, dict[str, float]],
) -> tuple[dict[str, float], str]:
    """Pick the weight dict for the metric and the source tag (``"default"`` / ``"overridden"``)."""
    if metric in overrides and overrides[metric]:
        return overrides[metric], "overridden"
    return weights_default, "default"


def _weighted_average(
    bucket_values: dict[str, float | None],
    weights: dict[str, float],
) -> tuple[float | None, tuple[str, ...], dict[str, float]]:
    """Weighted average: a bucket is included in the denominator only when its value is not None and its weight is > 0."""
    contributors: list[tuple[str, float, float]] = []
    for bucket, value in bucket_values.items():
        if value is None:
            continue
        w = weights.get(bucket, 0.0)
        if w <= 0:
            continue
        contributors.append((bucket, w, float(value)))
    if not contributors:
        return None, (), {}
    sum_w = sum(w for _, w, _ in contributors)
    if sum_w <= 0:
        return None, (), {}
    composite = sum(w * v for _, w, v in contributors) / sum_w
    used = tuple(b for b, _, _ in contributors)
    used_normalized = {b: w / sum_w for b, w, _ in contributors}
    return composite, used, used_normalized


def compute_composite(
    *,
    bucket_values_per_model: dict[str, dict[str, dict[str, float | None]]],
    weights_default: dict[str, float],
    overrides: dict[str, dict[str, float]],
) -> CompositeReport:
    """Compute the difficulty-weighted composite for each (model, metric) and return a :class:`CompositeReport`.

    A misspelled metric name (not in :data:`KNOWN_METRICS`) raises here —
    this is the redemption point of the design's promise that "a misspelled
    metric name must be surfaced to the user".
    """
    for metric in overrides:
        if metric not in KNOWN_METRICS:
            raise ValueError(
                f"COMPOSITE_WEIGHT_OVERRIDES[{metric!r}] is not a known "
                f"metric; pick from {sorted(KNOWN_METRICS)}"
            )

    per_model: dict[str, dict[str, CompositeMetricInfo]] = {}
    for model, per_metric in bucket_values_per_model.items():
        per_metric_info: dict[str, CompositeMetricInfo] = {}
        for metric in KNOWN_METRICS:
            bucket_values = per_metric.get(metric, {})
            weights, kind = _select_weights(
                metric,
                weights_default=weights_default,
                overrides=overrides,
            )
            value, used, used_norm = _weighted_average(bucket_values, weights)
            per_metric_info[metric] = CompositeMetricInfo(
                value=value,
                buckets_used=used,
                weights_used_normalized=used_norm,
                bucket_values=dict(bucket_values),
                weights_kind=kind,
            )
        per_model[model] = per_metric_info

    return CompositeReport(
        weights_default=dict(weights_default),
        overrides={m: dict(sub) for m, sub in overrides.items()},
        per_model=per_model,
    )


__all__ = [
    "DEFAULT_WEIGHTS",
    "COMPOSITE_BUCKETS",
    "KNOWN_METRICS",
    "V5SliceResult",
    "CompositeMetricInfo",
    "CompositeReport",
    "bucket_of",
    "difficulty_bucket_key",
    "slice_v5_metrics_by_bucket",
    "collect_bucket_values",
    "compute_composite",
]
