"""Tests for `forecast_eval.analysis.exam_score`.

## 摘除验证清单（manual checklist — review 时执行一次）

本指标允诺"摘除等价性"：删除以下文件 / 段落后，仓库的所有现有测试 SHALL 全部
pass，且 `python -m forecast_eval.analysis <run_dir>` 生成的 CSV / Markdown
**除 `exam_score_at_n_avg` 列外** SHALL 与未引入本改动的输出字节级一致。

具体步骤（review 时复制到本地 worktree 执行）：

1. 在临时 worktree 上 `git checkout` 本 PR 的父 commit（即未引入 exam_score 之
   前），跑 `python -m pytest tests/ -x -q` 记录 pass 数；
2. 跑 `python -m forecast_eval.analysis runs/<某个 v5 run_dir>` 备份生成的
   `per_model_summary.csv` / `per_model_summary.md`；
3. 切回 PR 分支，删除以下文件 / 段落（marker 字面值统一为 `exam-score-metric:`）：
   - `forecast_eval/analysis/exam_score.py`
   - `tests/test_exam_score.py`（即本文件）
   - `forecast_eval/analysis/accuracy.py` 中 grep 该 marker 列出的 4 处挂接
     （import / Aggregate 字段 / as_ordered_dict / _aggregate 注入）
   - `forecast_eval/analysis/writers.py` 中 grep 该 marker 列出的 2 处挂接
     （CSV header / markdown header）
   - `README.md` / `DESIGN.md` / `FRAME.md` 中 HTML 注释包裹的段落
   - `.env.example` 中标记的 SAMPLING_N "独立测验次数"语义注释扩展
4. 跑 `python -m pytest tests/ -x -q` → MUST 与步骤 1 的 pass 数完全一致（不含
   已删除的本文件自身）；
5. 跑 `python -m forecast_eval.analysis runs/<同一 run_dir>` → 生成的 CSV 与
   步骤 2 的备份用 `cmp` 字节级比对 MUST 完全一致；
6. 跑 `grep -rn` 该 marker → 主仓库 MUST 返回 0 处（除
   `openspec/changes/add-exam-score-metric/` 内部 self-reference，archive 后
   随 change 整体迁移）。

任何一步失败即破坏"摘除等价性"约束，PR 不能合入。
"""
from __future__ import annotations

import json

import pytest

from forecast_eval.analysis.exam_score import exam_score, exam_score_at_n_avg
from forecast_eval.analysis.flatten import CUTOFF, SampleRow
from forecast_eval.parser import is_correct


def _make_sample(
    *,
    question_id: str = "q1",
    sample_idx: int = 0,
    choice_type: str = "multi",
    options: list[str] | None = None,
    parse_ok: int | None = 1,
    parsed: frozenset[str] | None = None,
    error: str | None = None,
) -> SampleRow:
    """Minimal SampleRow factory for exam_score tests.

    Defaults to a multi-choice eligible sample. Pass `parsed=None` to simulate
    `final_answer_letters` missing; pass `parse_ok=0` to simulate parse failure.
    `error="skipped_training_cutoff"` triggers the cutoff path.
    """
    if options is None:
        options = ["A", "B", "C", "D"]
    final_letters_json = None
    if parsed is not None:
        final_letters_json = json.dumps(sorted(parsed))
    correct = 1 if (parse_ok == 1 and parsed is not None) else None
    return SampleRow(
        model="model_x",
        question_id=question_id,
        question_type="forecast",
        choice_type=choice_type,
        options=options,
        sample_idx=sample_idx,
        correct=correct,
        parse_ok=parse_ok,
        tool_calls_count=0,
        react_steps=0,
        prompt_tokens=0,
        completion_tokens=0,
        reasoning_tokens=0,
        latency_ms=0,
        final_answer_letters=final_letters_json,
        error=error,
        created_at="2026-04-27T00:00:00Z",
        finish_reason="stop",
        nudges_used=0,
        belief_final=None,
        belief_trace=None,
        belief_parse_ok=0,
        probabilities=None,
        is_fallback=False,
    )


# --------------------------------------------------------------------------- #
# Section §3.2 — 公式（含用户原始两个例子）
# --------------------------------------------------------------------------- #


def test_user_multi_example_per_question() -> None:
    """用户原始多选题：GT={A,B,C}，三次回答 AB / ABC / AD → e_q ≈ 0.5556。"""
    gt = frozenset({"A", "B", "C"})
    samples = [
        _make_sample(sample_idx=0, parsed=frozenset({"A", "B"})),
        _make_sample(sample_idx=1, parsed=frozenset({"A", "B", "C"})),
        _make_sample(sample_idx=2, parsed=frozenset({"A", "D"})),
    ]
    scores = [exam_score(s, gt) for s in samples]
    assert scores[0] == pytest.approx(2 / 3, abs=1e-4)
    assert scores[1] == pytest.approx(1.0)
    assert scores[2] == pytest.approx(0.0)
    e_q = sum(scores) / len(scores)  # type: ignore[arg-type]
    assert e_q == pytest.approx(5 / 9, abs=1e-4)


def test_user_single_example_per_question() -> None:
    """用户原始单选题：GT={B}，三次回答 A / A / B → e_q ≈ 0.3333。"""
    gt = frozenset({"B"})
    samples = [
        _make_sample(sample_idx=0, choice_type="single",
                     options=["A", "B"], parsed=frozenset({"A"})),
        _make_sample(sample_idx=1, choice_type="single",
                     options=["A", "B"], parsed=frozenset({"A"})),
        _make_sample(sample_idx=2, choice_type="single",
                     options=["A", "B"], parsed=frozenset({"B"})),
    ]
    scores = [exam_score(s, gt) for s in samples]
    assert scores == [0.0, 0.0, 1.0]
    e_q = sum(scores) / len(scores)  # type: ignore[arg-type]
    assert e_q == pytest.approx(1 / 3, abs=1e-4)


def test_perfect_hit_returns_1() -> None:
    s = _make_sample(parsed=frozenset({"A", "B", "C"}))
    assert exam_score(s, frozenset({"A", "B", "C"})) == pytest.approx(1.0)


def test_only_missing_partial_credit() -> None:
    """仅漏选（FN > 0、FP = 0）→ TP/|G|。"""
    s = _make_sample(parsed=frozenset({"A"}))
    assert exam_score(s, frozenset({"A", "B", "C"})) == pytest.approx(1 / 3, abs=1e-4)


def test_any_wrong_choice_returns_0() -> None:
    """含错选 → 0 分（即使有部分正确也一票否决）。"""
    s = _make_sample(parsed=frozenset({"A", "D"}))  # 漏 B/C 且选了 D
    assert exam_score(s, frozenset({"A", "B", "C"})) == pytest.approx(0.0)


def test_superset_pred_still_zero() -> None:
    """全对 + 多选一个错选 → 仍 0 分（FP > 0 触发硬门）。"""
    s = _make_sample(parsed=frozenset({"A", "B", "C", "D"}))
    assert exam_score(s, frozenset({"A", "B", "C"})) == pytest.approx(0.0)


def test_single_correct() -> None:
    s = _make_sample(choice_type="single", options=["A", "B"],
                     parsed=frozenset({"B"}))
    assert exam_score(s, frozenset({"B"})) == pytest.approx(1.0)


def test_single_wrong() -> None:
    s = _make_sample(choice_type="single", options=["A", "B"],
                     parsed=frozenset({"A"}))
    assert exam_score(s, frozenset({"B"})) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Section §3.3 — 基数规则（cutoff / error / parse_ok / 防御）
# --------------------------------------------------------------------------- #


def test_cutoff_returns_none() -> None:
    """is_cutoff（error == CUTOFF）→ None，剔除。"""
    s = _make_sample(parsed=None, parse_ok=None, error=CUTOFF)
    assert exam_score(s, frozenset({"A"})) is None


def test_other_error_returns_none() -> None:
    """非 cutoff 的 error（如敏感词、API timeout）→ None，剔除。"""
    for err in ("content_policy", "api_timeout", "bad_request", "network"):
        s = _make_sample(parsed=None, parse_ok=None, error=err)
        assert exam_score(s, frozenset({"A"})) is None, f"error={err}"


def test_parse_ok_zero_returns_zero() -> None:
    """error=None 且 parse_ok=0 → 0.0，进基数（"完成但答错"）。"""
    s = _make_sample(parsed=None, parse_ok=0)
    assert exam_score(s, frozenset({"A"})) == 0.0


def test_parse_ok_none_returns_zero() -> None:
    """parse_ok 为 None 也走 0.0 路径（合 `parse_ok != 1` 分支）。"""
    s = _make_sample(parsed=None, parse_ok=None)
    assert exam_score(s, frozenset({"A"})) == 0.0


def test_parse_ok_one_but_parsed_none_returns_zero() -> None:
    """parse_ok=1 但 parsed_letters 解析后为 None（防御性）→ 0.0。"""
    # final_answer_letters 是空字符串 → parsed_letters returns None
    s = SampleRow(
        model="model_x", question_id="q1", question_type="forecast",
        choice_type="multi", options=["A", "B", "C", "D"], sample_idx=0,
        correct=None, parse_ok=1,
        tool_calls_count=0, react_steps=0,
        prompt_tokens=0, completion_tokens=0, reasoning_tokens=0,
        latency_ms=0,
        final_answer_letters="",  # → parsed_letters = None
        error=None, created_at="2026-04-27T00:00:00Z",
        finish_reason="stop", nudges_used=0,
        belief_final=None, belief_trace=None, belief_parse_ok=0,
        probabilities=None, is_fallback=False,
    )
    assert s.parsed_letters is None
    assert exam_score(s, frozenset({"A"})) == 0.0


def test_normal_path_does_not_hit_defensive_branches() -> None:
    """正常路径：error=None, parse_ok=1, parsed 有效 → 走公式。"""
    s = _make_sample(parsed=frozenset({"A", "B"}))
    assert exam_score(s, frozenset({"A", "B", "C"})) == pytest.approx(2 / 3, abs=1e-4)


# --------------------------------------------------------------------------- #
# Section §3.4 — 聚合（题内分母 = 实际进基数；空基数题剔除；按题等权）
# --------------------------------------------------------------------------- #


def test_aggregation_eligible_denominator_is_actual_in_basis() -> None:
    """用户原话："剩下 2 个做平均即可"。
    SAMPLING_N=3 中一个 cutoff，剩 2 个进基数 → 题内分母 = 2。
    """
    gt = frozenset({"A"})
    samples = [
        _make_sample(question_id="q1", sample_idx=0,
                     parsed=None, parse_ok=None, error=CUTOFF),  # 剔除
        _make_sample(question_id="q1", sample_idx=1,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"A"})),
        _make_sample(question_id="q1", sample_idx=2,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"B"})),
    ]
    # 单题 e_q = (1.0 + 0.0) / 2 = 0.5（cutoff 不进分母）
    result = exam_score_at_n_avg(samples, {"q1": gt})
    assert result == pytest.approx(0.5)


def test_aggregation_error_excluded_from_denominator() -> None:
    """敏感词失败的 sample 不进基数；剩下的 sample 做平均。"""
    gt = frozenset({"A", "B", "C"})
    samples = [
        _make_sample(question_id="q1", sample_idx=0,
                     parsed=None, parse_ok=None, error="content_policy"),  # 剔除
        _make_sample(question_id="q1", sample_idx=1,
                     parsed=frozenset({"A", "B", "C"})),  # 1.0
        _make_sample(question_id="q1", sample_idx=2,
                     parsed=frozenset({"A"})),  # 1/3
    ]
    # e_q = (1.0 + 1/3) / 2 = 2/3
    assert exam_score_at_n_avg(samples, {"q1": gt}) == pytest.approx(2 / 3, abs=1e-4)


def test_aggregation_parse_failure_counts_as_zero() -> None:
    """parse_ok=0 进基数计 0.0，跟"剔除"的语义不同。"""
    gt = frozenset({"A", "B", "C"})
    samples = [
        _make_sample(question_id="q1", sample_idx=0,
                     parsed=None, parse_ok=0),  # 0.0 进基数
        _make_sample(question_id="q1", sample_idx=1,
                     parsed=frozenset({"A", "B", "C"})),  # 1.0
        _make_sample(question_id="q1", sample_idx=2,
                     parsed=frozenset({"A", "B", "C"})),  # 1.0
    ]
    # e_q = (0.0 + 1.0 + 1.0) / 3 = 2/3
    assert exam_score_at_n_avg(samples, {"q1": gt}) == pytest.approx(2 / 3, abs=1e-4)


def test_aggregation_question_with_all_excluded_skipped_globally() -> None:
    """整题全部 sample 剔除 → e_q = None，不参与全局题间分母。"""
    gt_a = frozenset({"A"})
    gt_b = frozenset({"B"})
    samples = [
        # q1：3 个 sample 全部 cutoff，e_q = None
        _make_sample(question_id="q1", sample_idx=0,
                     parsed=None, parse_ok=None, error=CUTOFF),
        _make_sample(question_id="q1", sample_idx=1,
                     parsed=None, parse_ok=None, error=CUTOFF),
        # q2：2 个 sample 都 1.0，e_q = 1.0
        _make_sample(question_id="q2", sample_idx=0,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"B"})),
        _make_sample(question_id="q2", sample_idx=1,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"B"})),
    ]
    # 全局只统计 q2 → 1.0（q1 不进题间分母）
    result = exam_score_at_n_avg(samples, {"q1": gt_a, "q2": gt_b})
    assert result == pytest.approx(1.0)


def test_aggregation_empty_samples_returns_none() -> None:
    assert exam_score_at_n_avg([], {}) is None


def test_aggregation_all_excluded_returns_none() -> None:
    """所有 sample 都剔除（全 cutoff/error）→ 全局返 None。"""
    samples = [
        _make_sample(question_id="q1", sample_idx=0,
                     parsed=None, parse_ok=None, error=CUTOFF),
        _make_sample(question_id="q2", sample_idx=0,
                     parsed=None, parse_ok=None, error="content_policy"),
    ]
    assert exam_score_at_n_avg(
        samples, {"q1": frozenset({"A"}), "q2": frozenset({"B"})}
    ) is None


def test_aggregation_questions_equal_weighted_not_sample_weighted() -> None:
    """按题等权（不按 sample 数加权）：q1 有 3 个 sample，q2 有 1 个 sample，
    全局值 = (e_q1 + e_q2) / 2，而非 sample 加权。"""
    gt = frozenset({"A"})
    samples = [
        # q1：3 个 sample 都 0.0 → e_q1 = 0.0
        _make_sample(question_id="q1", sample_idx=0,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"B"})),
        _make_sample(question_id="q1", sample_idx=1,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"B"})),
        _make_sample(question_id="q1", sample_idx=2,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"B"})),
        # q2：1 个 sample 为 1.0 → e_q2 = 1.0
        _make_sample(question_id="q2", sample_idx=0,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"A"})),
    ]
    # 题等权：(0.0 + 1.0) / 2 = 0.5
    # 若按 sample 加权将得 (0+0+0+1)/4 = 0.25
    result = exam_score_at_n_avg(samples, {"q1": gt, "q2": gt})
    assert result == pytest.approx(0.5)


def test_aggregation_user_two_questions_global() -> None:
    """组合用户两个例子在全局聚合（题等权）：(5/9 + 1/3) / 2 ≈ 0.4444。"""
    gt_a = frozenset({"A", "B", "C"})
    gt_b = frozenset({"B"})
    samples = [
        _make_sample(question_id="qA", sample_idx=0, parsed=frozenset({"A", "B"})),
        _make_sample(question_id="qA", sample_idx=1, parsed=frozenset({"A", "B", "C"})),
        _make_sample(question_id="qA", sample_idx=2, parsed=frozenset({"A", "D"})),
        _make_sample(question_id="qB", sample_idx=0,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"A"})),
        _make_sample(question_id="qB", sample_idx=1,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"A"})),
        _make_sample(question_id="qB", sample_idx=2,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"B"})),
    ]
    result = exam_score_at_n_avg(samples, {"qA": gt_a, "qB": gt_b})
    expected = (5 / 9 + 1 / 3) / 2
    assert result == pytest.approx(expected, abs=1e-4)


# --------------------------------------------------------------------------- #
# Section §3.5 — 单选退化等价（与 parser.is_correct 字节级一致）
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "pred_letters,gt_letters",
    [
        ({"A"}, {"A"}),
        ({"A"}, {"B"}),
        ({"B"}, {"A"}),
        ({"B"}, {"B"}),
        ({"C"}, {"D"}),
    ],
)
def test_single_choice_degenerates_to_is_correct(pred_letters, gt_letters) -> None:
    pred = frozenset(pred_letters)
    gt = frozenset(gt_letters)
    s = _make_sample(choice_type="single", options=["A", "B", "C", "D"],
                     parsed=pred)
    expected = is_correct(pred, gt)
    assert expected is not None
    assert exam_score(s, gt) == float(expected)


# --------------------------------------------------------------------------- #
# Section §3.6 — 防御边界（gt 空集 / gt_map 缺 qid）
# --------------------------------------------------------------------------- #


def test_empty_gt_returns_zero() -> None:
    """gt 为空 frozenset → 0.0（数据集理论不会出现，防脏数据 NaN）。"""
    s = _make_sample(parsed=frozenset({"A"}))
    assert exam_score(s, frozenset()) == 0.0


def test_gt_map_missing_question_id_skipped() -> None:
    """gt_map 缺该 qid（理论不会出现）→ 该题被跳过，不参与全局。"""
    samples = [
        _make_sample(question_id="qA", sample_idx=0,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"A"})),
        _make_sample(question_id="qB_missing", sample_idx=0,
                     choice_type="single", options=["A", "B"],
                     parsed=frozenset({"B"})),
    ]
    # gt_map 只含 qA → 全局只统计 qA
    result = exam_score_at_n_avg(samples, {"qA": frozenset({"A"})})
    assert result == pytest.approx(1.0)


def test_pred_empty_set_with_nonempty_gt_returns_zero() -> None:
    """pred=空集（既无 FP 也无 TP）→ TP/|G| = 0/|G| = 0。"""
    # 这是 "FP=0 但 TP=0" 的边界 — 公式给 0/N = 0.0
    s = _make_sample(parsed=frozenset())
    assert exam_score(s, frozenset({"A", "B"})) == pytest.approx(0.0)
