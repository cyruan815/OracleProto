"""Glue layer between flattened samples and `proper_score` aggregates.

The multi-trial story is deliberately simple: per (model, question) we
arithmetic-mean the per-sample probability vectors. Logit-space mean and
LOO-tuned shrinkage live in `aggregation.py` as drop-in replacements for
`_aggregate_question_probs`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .flatten import SampleRow, gt_vector
from .proper_score import (
    ModelProbabilisticAggregate,
    PerQuestionScore,
    aggregate_probabilistic,
    crowd_gamma_for,
    per_question_scores_for,
    uniform_gamma_for,
)


@dataclass(frozen=True)
class _QuestionProbabilityRow:
    """One (model, question) row after aggregating across the K samples."""

    model: str
    question_id: str
    question_type: str
    choice_type: str
    options: list[str]
    obs: list[int]
    probs: list[float]
    n_samples: int
    n_fallback: int

    @property
    def is_fallback(self) -> bool:
        # A question is "fallback" iff every contributing sample was a
        # fallback. If even one sample produced a parsed belief, we trust
        # the arithmetic mean across them.
        return self.n_samples > 0 and self.n_fallback == self.n_samples


def _aggregate_question_probs(
    samples: list[SampleRow],
) -> list[float] | None:
    """Arithmetic-mean the per-sample probability vectors. Returns None if no
    sample produced a usable probability vector."""
    vecs = [s.probabilities for s in samples if s.probabilities is not None]
    if not vecs:
        return None
    k = len(vecs[0])
    sums = [0.0] * k
    for v in vecs:
        if len(v) != k:
            # Mismatched lengths shouldn't happen (per-question options are
            # constant), but skip gracefully.
            continue
        for i, val in enumerate(v):
            sums[i] += val
    return [s / len(vecs) for s in sums]


def _build_question_rows_for_model(
    samples: list[SampleRow],
    gt_map: dict[str, frozenset[str]],
) -> list[_QuestionProbabilityRow]:
    """Group eligible samples by question_id and aggregate."""
    grouped: dict[str, list[SampleRow]] = {}
    for s in samples:
        if not s.is_eligible:
            continue
        grouped.setdefault(s.question_id, []).append(s)

    rows: list[_QuestionProbabilityRow] = []
    for qid, ss in grouped.items():
        gt = gt_map.get(qid)
        if gt is None or not ss[0].options:
            continue
        k = len(ss[0].options)
        if k == 0:
            continue
        agg_probs = _aggregate_question_probs(ss)
        if agg_probs is None:
            continue  # No probability vector to score on this question.
        n_total = sum(1 for s in ss if s.probabilities is not None)
        n_fallback = sum(1 for s in ss if s.is_fallback)
        rows.append(
            _QuestionProbabilityRow(
                model=ss[0].model,
                question_id=qid,
                question_type=ss[0].question_type,
                choice_type=ss[0].choice_type,
                options=ss[0].options,
                obs=gt_vector(gt, k),
                probs=agg_probs,
                n_samples=n_total,
                n_fallback=n_fallback,
            )
        )
    return rows


def _per_question_scores_from_rows(
    rows: Iterable[_QuestionProbabilityRow],
) -> list[PerQuestionScore]:
    return [
        per_question_scores_for(
            question_id=r.question_id,
            choice_type=r.choice_type,
            probs=r.probs,
            obs=r.obs,
            is_fallback=r.is_fallback,
        )
        for r in rows
    ]


def _build_uniform_gammas(
    rows: Iterable[_QuestionProbabilityRow],
) -> dict[str, float]:
    """$\\gamma_q$ for the uniform $(1/k, \\dots, 1/k)$ baseline per question."""
    out: dict[str, float] = {}
    for r in rows:
        out[r.question_id] = uniform_gamma_for(r.obs)
    return out


def _build_crowd_gammas_per_model(
    rows_by_model: dict[str, list[_QuestionProbabilityRow]],
) -> dict[str, dict[str, float | None]]:
    """For each (model, question) → leave-one-out crowd γ (excluding self).

    The outer dict key is the model whose ABI we're computing; the inner dict
    is question_id → γ. Returns `None` for a question if no other model
    produced a probability vector on it (caller then falls back to uniform).
    """
    by_q: dict[str, list[tuple[str, list[float], list[int]]]] = {}
    for model, rows in rows_by_model.items():
        for r in rows:
            by_q.setdefault(r.question_id, []).append((model, r.probs, r.obs))

    out: dict[str, dict[str, float | None]] = {m: {} for m in rows_by_model}
    for qid, entries in by_q.items():
        # Validate observations agree across models for this qid; we trust
        # them to (questions table is shared) but defensive against fixture
        # bugs.
        obs = entries[0][2]
        for m_self in rows_by_model:
            other_probs = [probs for (m, probs, _) in entries if m != m_self]
            if not other_probs:
                out[m_self][qid] = None
            else:
                out[m_self][qid] = crowd_gamma_for(obs, other_probs)
    return out


def _aggregate_for_subset(
    rows: list[_QuestionProbabilityRow],
    *,
    crowd_gammas: dict[str, float | None] | None,
    uniform_gammas: dict[str, float],
) -> ModelProbabilisticAggregate:
    """Filter `crowd_gammas` keys to those in `rows` (slices truncate the set)."""
    per_q = _per_question_scores_from_rows(rows)
    if crowd_gammas is not None:
        crowd_subset = {
            qid: g
            for qid, g in crowd_gammas.items()
            if g is not None and qid in {r.question_id for r in rows}
        }
    else:
        crowd_subset = None
    uniform_subset = {
        qid: g
        for qid, g in uniform_gammas.items()
        if qid in {r.question_id for r in rows}
    }
    return aggregate_probabilistic(
        per_q, crowd_gammas=crowd_subset, uniform_gammas=uniform_subset,
    )


@dataclass
class ProbabilisticReport:
    """All probabilistic aggregates a single run produces, keyed by model.

    `per_model` is the headline aggregate; `per_model_by_difficulty` is the
    composite-difficulty-bucket slice aggregate for the wider CSV. Slice
    aggregates inherit the same crowd-γ map as the full model — slicing only
    restricts which questions enter the average, not which models contribute
    to the crowd baseline (consistent with paper §A.2's intent).
    """

    per_model: dict[str, ModelProbabilisticAggregate]
    per_model_by_difficulty: dict[str, dict[str, ModelProbabilisticAggregate]]
    rows_by_model: dict[str, list[_QuestionProbabilityRow]]


def build_probabilistic_report(
    samples_by_model: dict[str, list[SampleRow]],
    gt_map: dict[str, frozenset[str]],
) -> ProbabilisticReport:
    """End-to-end: per-model question rows → crowd γ → per-model aggregates."""
    from .composite import bucket_of

    rows_by_model: dict[str, list[_QuestionProbabilityRow]] = {
        m: _build_question_rows_for_model(samples, gt_map)
        for m, samples in samples_by_model.items()
    }
    crowd_gammas = _build_crowd_gammas_per_model(rows_by_model)
    # Uniform γ is a function of observations only — same across models.
    union_rows: list[_QuestionProbabilityRow] = []
    for rows in rows_by_model.values():
        union_rows.extend(rows)
    uniform_gammas = _build_uniform_gammas(union_rows)

    per_model: dict[str, ModelProbabilisticAggregate] = {}
    per_model_by_difficulty: dict[str, dict[str, ModelProbabilisticAggregate]] = {}

    for m, rows in rows_by_model.items():
        per_model[m] = _aggregate_for_subset(
            rows,
            crowd_gammas=crowd_gammas[m],
            uniform_gammas=uniform_gammas,
        )

        difficulty_buckets: dict[str, list[_QuestionProbabilityRow]] = {}
        for r in rows:
            difficulty_buckets.setdefault(
                bucket_of(r.question_type, r.choice_type), []
            ).append(r)

        per_model_by_difficulty[m] = {
            b: _aggregate_for_subset(
                rs, crowd_gammas=crowd_gammas[m], uniform_gammas=uniform_gammas,
            )
            for b, rs in sorted(difficulty_buckets.items())
        }

    return ProbabilisticReport(
        per_model=per_model,
        per_model_by_difficulty=per_model_by_difficulty,
        rows_by_model=rows_by_model,
    )


__all__ = [
    "ProbabilisticReport",
    "build_probabilistic_report",
]
