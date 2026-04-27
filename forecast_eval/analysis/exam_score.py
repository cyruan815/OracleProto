"""考试式部分得分（Exam-style partial credit）—— 单文件、可整体摘除。

## 公式

对单个 sample，记题目正确答案集 $G$、模型解析出的选项集 $\\hat S$：

$$
\\text{exam\\_score}(\\hat S, G) = \\begin{cases}
|\\hat S \\cap G| / |G| & \\text{if } \\hat S \\setminus G = \\emptyset \\\\
0 & \\text{if } \\hat S \\setminus G \\ne \\emptyset
\\end{cases}
$$

聚合采用题内均值 → 题间均值的两步：先对 $q$ 题在基数内的样本算 $e_q$，再对所有
题等权求平均；任一步空基数返 `None`。

## 语义

把 SAMPLING_N 在该指标视角下重新解读为"独立测验次数"：每次测验得分独立 0~1，
最终取算术均值。形式上等价于 **"Recall under zero-FP gate"** —— 把 Recall
加一个"任何 FP 都一票否决"的硬门。

## 与既有评分的差异

| 指标 | 公式形态 | 错选惩罚 | 漏选惩罚 | 单选退化 |
| --- | --- | --- | --- | --- |
| `parser.is_correct`（strict） | $\\hat S = G$ 才得 1 | 0 分 | 0 分 | 0/1 |
| `tversky_score(α=2,β=0.5)`+chance correction → `fss` | 软惩罚 FP/FN | α 倍 | β 倍 | 含 chance |
| `hamming_score` | $1 - \\text{XOR位数}/k$ | 与漏选对称 | 与错选对称 | 0/1 |
| **`exam_score`（本文件）** | TP/|G|·𝟙(FP=0) | 一票否决 | 按比例扣分 | 0/1 |

`exam_score` 的卖点是"用一句话能解释"：含错选 0 分，否则按答对比例给分。

## 摘除等价性约束

本文件、`tests/test_exam_score.py`、`accuracy.py` / `writers.py` 中带 grep-able
注释标记的少量挂接点、`README.md` / `DESIGN.md` / `FRAME.md` / `.env.example`
中带 HTML 注释包裹的段落，共同构成可一次性整体移除的最小闭包。删除后仓库须
回到本次改动前的字节级一致状态（既有测试全 pass、CSV 既有列字节相同、文档段落
无残留）。Marker 字面值见 `openspec/changes/add-exam-score-metric/design.md` §D8。

允许依赖范围：标准库、`flatten.SampleRow`、`flatten._group_by_question`。
SHALL NOT 反向依赖 `accuracy.py` / `proper_score.py` / `consistency.py` /
`writers.py` / `inference.py` / `behavior.py` / `grid.py`，否则摘除时无法定位
最小闭包。
"""
from __future__ import annotations

from .flatten import SampleRow, _group_by_question


def exam_score(s: SampleRow, gt: frozenset[str]) -> float | None:
    """单 sample 考试式得分；进基数→浮点 [0,1]，剔除→`None`。

    判定顺序：
      1. `is_cutoff`（题目晚于训练截止）→ `None`，剔除（信息屏障）；
      2. `error is not None` 且非 cutoff → `None`，剔除（"未完成过程"）；
      3. `error is None` 且 `parse_ok != 1` → `0.0`，进基数（"完成但答错"）；
      4. `parsed_letters is None`（防御性）→ `0.0`，进基数；
      5. 含错选（FP > 0）→ `0.0`，进基数；
      6. 防御 `gt` 空集 → `0.0`；
      7. 否则 → $|\\hat S \\cap G| / |G|$，进基数。
    """
    if s.is_cutoff:
        return None
    if s.error is not None:
        return None
    if s.parse_ok != 1:
        return 0.0
    pred = s.parsed_letters
    if pred is None:
        return 0.0
    if pred - gt:
        return 0.0
    if not gt:
        return 0.0
    return len(pred & gt) / len(gt)


def exam_score_at_n_avg(
    samples: list[SampleRow],
    gt_map: dict[str, frozenset[str]],
) -> float | None:
    """题内均值 → 题间均值的两步聚合。

    Step 1（题内）：对每题 $q$ 取基数内 sample 的 `exam_score` 算术均值得 $e_q$；
    若该题全部 sample 均剔除（基数为 0），$e_q = \\text{None}$ 不参与全局。

    Step 2（题间）：对所有 $e_q \\ne \\text{None}$ 的题等权求均值；若全局基数为 0
    返 `None`（不返 0.0 / NaN / 抛异常）。

    `gt_map` 缺该 question_id 时该题被跳过（防御边界，理论数据不会出现）。
    """
    by_q = _group_by_question(samples)
    e_q_values: list[float] = []
    for qid, q_samples in by_q.items():
        gt = gt_map.get(qid)
        if gt is None:
            continue
        per_sample: list[float] = []
        for s in q_samples:
            score = exam_score(s, gt)
            if score is None:
                continue
            per_sample.append(score)
        if not per_sample:
            continue
        e_q_values.append(sum(per_sample) / len(per_sample))
    if not e_q_values:
        return None
    return sum(e_q_values) / len(e_q_values)


__all__ = [
    "exam_score",
    "exam_score_at_n_avg",
]
