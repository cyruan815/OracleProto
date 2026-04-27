"""综合得分按子题型加权（composite-score-by-subtype）。

此模块承载 ``per_model_summary.csv`` 全部数据列在两个维度上独立做的
"先按桶分别算 → 再按权重合成 → 缺失桶剔除并归一化"的合成逻辑。

## 维度

* ``question_type``: ``yes_no`` / ``binary_named`` / ``multiple_choice``
* ``choice_type``:   ``single`` / ``multi``

每个维度独立做一遍合成，互不影响（``multiple_choice`` 内部既有 single 也有
multi，两个维度并不正交，但这就是用户语义里的"两种切法"）。

## 公式

对每个 (model, dimension, metric)：

$$
\\text{composite}_{m} = \\frac{\\sum_{b \\in B_{\\text{valid}}} w_{m,b} \\cdot v_{m,b}}{\\sum_{b \\in B_{\\text{valid}}} w_{m,b}}
$$

其中 :math:`B_{\\text{valid}}` 是该 (model, metric) 下 slice 实测值非
``None`` 且权重 > 0 的桶集合。全 ``None`` → composite 返 ``None``。

## 与 ``_SUMMARY_FIELDS`` 的对齐

输出列与 ``writers._SUMMARY_FIELDS``（去掉元数据列 ``model`` /
``sampling_n``）一一对应——这意味着读
``per_model_composite_by_question_type.csv`` 即可直接得到"按子题型加权后
的总表"，下游脚本只需改文件路径。

## v5 离散家族的"按桶切"

``fss`` / ``cohen_kappa`` / ``hamming_score`` / ``fleiss_kappa`` /
``mean_entropy`` / ``vci`` / ``mvg`` 在 ``__init__.py`` 顶层是按 model 全局
计算的（写入 ``per_model_summary.csv``），但要做按桶加权合成，必须先按桶
切再算——本模块的 :func:`slice_v5_metrics_by_bucket` 接管这件事。其结果
**同时**回流给 ``writers._write_slice_csv``，让 ``per_model_by_*.csv`` 这
两张明细表也带上这些列（不破坏既有"v3 + prob"的列序，只在尾部追加）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .accuracy import Aggregate, cohen_kappa, fss, hamming_score
from .consistency import ConsistencyReport, build_consistency_report
from .flatten import SampleRow
from .proper_score import ModelProbabilisticAggregate


# 默认权重 (与 ``config.Settings`` 默认值同值, 在 ``run_analysis`` 未拿到
# Settings 时作为零配置回退使用)。校验由 Settings 那一侧负责。
DEFAULT_WEIGHTS_QTYPE: dict[str, float] = {
    "yes_no": 0.15,
    "binary_named": 0.15,
    "multiple_choice": 0.70,
}
DEFAULT_WEIGHTS_CTYPE: dict[str, float] = {"single": 0.40, "multi": 0.60}


# --------------------------------------------------------------------------- #
# 已知指标白名单
# --------------------------------------------------------------------------- #


# 必须与 ``writers._SUMMARY_FIELDS``（除去 ``model`` / ``sampling_n``）一致；
# 在 ``tests/test_composite_score.py`` 里有对齐断言，防止两边漂移。指标名拼
# 错或者用了未在白名单上的名字 → ``compute_composite`` 在入口 raise，因为
# 这一定是用户在 ``COMPOSITE_WEIGHT_OVERRIDES_*`` 里写错了。
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
        # v5 discrete (FSS family)
        "fss",
        "fss_pe_mean",
        "cohen_kappa",
        "hamming_score",
        # v5 consistency family
        "fleiss_kappa",
        "mean_entropy",
        "vci",
        "mvg",
        # v4 probabilistic family
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
# v5 离散家族的"按桶切"
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class V5SliceResult:
    """一个 (model, bucket) 下的 v5 离散家族 slice 结果。

    ``fss`` 携带 ``fss / fss_pe_mean`` 两列，所以保留原 ``fss(...)`` 字典；
    ``cohen_kappa`` / ``hamming_score`` 各是单值；``consistency`` 给出
    ``fleiss_kappa`` / ``mean_entropy`` / ``vci`` / ``mvg`` 四列。
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
    """按 ``key_fn(sample)`` 把 v5 离散家族指标分别在每个桶上重新计算。

    ``__init__.py`` 顶层会用 ``key_fn=lambda s: s.question_type`` 与
    ``key_fn=lambda s: s.choice_type`` 各调一次。空桶（无 sample）静默
    跳过，每桶 None 桶位由调用方在合成时按"剔除并归一化"处理。
    """
    by_bucket: dict[str, list[SampleRow]] = {}
    for s in samples:
        by_bucket.setdefault(key_fn(s), []).append(s)

    out: dict[str, V5SliceResult] = {}
    for bucket, bucket_samples in by_bucket.items():
        if not bucket_samples:
            continue
        # cohen_kappa / hamming 共用 by_q（只看本桶内的题）。
        by_q: dict[str, list[SampleRow]] = {}
        for s in bucket_samples:
            by_q.setdefault(s.question_id, []).append(s)
        # cohen_kappa 在桶内全是 multi 时仍然有意义（per_e_q=0.5）；
        # 在桶内全是 single 时仍然按 1/k 计算。函数自身处理 None 退化。
        kappa = cohen_kappa(by_q, gt_map)
        # hamming_score 仅对 multi 有定义（accuracy.hamming_score 内部
        # 跳过 single），所以 yes_no / binary_named 桶下必为 None；这是
        # 预期行为，合成时该桶被剔除。
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


# --------------------------------------------------------------------------- #
# slice → 列值
# --------------------------------------------------------------------------- #


def _aggregate_to_columns(agg: Aggregate | None) -> dict[str, float | None]:
    """``Aggregate`` → 与 ``_SUMMARY_FIELDS_V3`` 对齐的列字典（不取整）。"""
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
    """``V5SliceResult`` → fss / fss_pe_mean / cohen_kappa / hamming_score /
    fleiss_kappa / mean_entropy / vci / mvg 八列。"""
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
    """``ModelProbabilisticAggregate`` → bi / bi_dec / nll / mbs /
    abi_crowd / abi_uniform / fallback_share 七列（不取整，给合成器原始值）。"""
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
    """把多个 slice 来源拼成 ``{model: {metric: {bucket: value}}}``。

    缺失的 (model, bucket) 被表示为 "桶位上每个指标都是 None" 而不是
    "桶不存在"——这样合成函数可以一致地按"None 桶剔除"处理。
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
# 合成
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CompositeMetricInfo:
    """单个 (model, metric) 的合成结果及元信息。"""

    value: float | None
    buckets_used: tuple[str, ...]
    weights_used_normalized: dict[str, float]
    bucket_values: dict[str, float | None]
    weights_kind: str  # "default" | "overridden"


@dataclass(frozen=True)
class CompositeReport:
    """一个维度（``question_type`` 或 ``choice_type``）下的合成结果。"""

    dimension: str
    weights_default: dict[str, float]
    overrides: dict[str, dict[str, float]]
    per_model: dict[str, dict[str, CompositeMetricInfo]] = field(
        default_factory=dict
    )

    def is_overridden(self, model: str) -> bool:
        """该 (model) 行是否使用了任意 override（决定 ``weights_kind`` 列）。"""
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
    """挑选指标对应的权重 dict 与来源标记 (``"default"`` / ``"overridden"``)。"""
    if metric in overrides and overrides[metric]:
        return overrides[metric], "overridden"
    return weights_default, "default"


def _weighted_average(
    bucket_values: dict[str, float | None],
    weights: dict[str, float],
) -> tuple[float | None, tuple[str, ...], dict[str, float]]:
    """加权平均: 仅在桶值非 None 且权重 > 0 时纳入分母。"""
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
    dimension: str,
    bucket_values_per_model: dict[str, dict[str, dict[str, float | None]]],
    weights_default: dict[str, float],
    overrides: dict[str, dict[str, float]],
) -> CompositeReport:
    """对每个 (model, metric) 计算加权综合值并返回 :class:`CompositeReport`。

    指标名拼错（不在 :data:`KNOWN_METRICS` 内）会在这里 raise——这是设计
    稿里"拼错指标名一定要让用户知道"的兑现点。
    """
    if dimension not in ("question_type", "choice_type"):
        raise ValueError(
            f"compute_composite: dimension must be 'question_type' or "
            f"'choice_type'; got {dimension!r}"
        )
    for metric in overrides:
        if metric not in KNOWN_METRICS:
            raise ValueError(
                f"COMPOSITE_WEIGHT_OVERRIDES_{dimension.upper()}[{metric!r}] "
                f"is not a known metric; pick from {sorted(KNOWN_METRICS)}"
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
        dimension=dimension,
        weights_default=dict(weights_default),
        overrides={m: dict(sub) for m, sub in overrides.items()},
        per_model=per_model,
    )


__all__ = [
    "DEFAULT_WEIGHTS_QTYPE",
    "DEFAULT_WEIGHTS_CTYPE",
    "KNOWN_METRICS",
    "V5SliceResult",
    "CompositeMetricInfo",
    "CompositeReport",
    "slice_v5_metrics_by_bucket",
    "collect_bucket_values",
    "compute_composite",
]
