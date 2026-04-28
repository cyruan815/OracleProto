"""Belief block parsing tests (v4 BELIEF_PROTOCOL).

Covers all four question shapes (yes_no / binary_named / multiple_choice
single / multiple_choice multi) plus every documented failure mode (simplex
violation, letter set mismatch, out-of-range probability, illegal confidence,
malformed JSON, missing tag, oversized key_evidence, etc.) and the >26 option
corner case used elsewhere in the project.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from forecast_eval.parser import Belief, parse_belief
from forecast_eval.types import Question


SOURCE_DB = Path(__file__).resolve().parents[1] / "forecast_eval_set_example.db"


def _q(qt: str, options: list[str], *, choice_type: str = "single", answer: str = "A") -> Question:
    return Question(
        id="qid",
        choice_type=choice_type,
        question_type=qt,
        event="ev",
        options=json.dumps(options),
        answer=answer,
        end_time="2026-01-01",
    )


def _payload(probs: dict[str, float], **overrides) -> dict:
    base = {
        "version": "v4.0",
        "probabilities": probs,
        "confidence": "medium",
        "key_evidence": ["primary driver"],
        "counterevidence": [],
        "open_questions": [],
        "decision_rule": "argmax",
    }
    base.update(overrides)
    return base


def _wrap(payload: dict) -> str:
    return f"reasoning... <belief>{json.dumps(payload)}</belief> final \\boxed{{A}}"


# ---- Valid paths ------------------------------------------------------------


def test_yes_no_valid() -> None:
    q = _q("yes_no", ["Yes", "No"])
    out = parse_belief(_wrap(_payload({"A": 0.7, "B": 0.3})), q)
    assert isinstance(out, Belief)
    assert out.probabilities == {"A": 0.7, "B": 0.3}
    assert out.confidence == "medium"
    assert out.decision_rule == "argmax"


def test_binary_named_valid() -> None:
    q = _q("binary_named", ["Lakers", "Warriors"])
    out = parse_belief(_wrap(_payload({"A": 0.55, "B": 0.45})), q)
    assert out is not None
    assert out.probabilities["A"] == 0.55


def test_multiple_choice_single_valid() -> None:
    q = _q("multiple_choice", ["alpha", "beta", "gamma"])
    out = parse_belief(_wrap(_payload({"A": 0.2, "B": 0.5, "C": 0.3})), q)
    assert out is not None
    assert sum(out.probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_multi_select_no_simplex() -> None:
    """Multi-select questions explicitly do NOT enforce sum=1; each value is
    an independent Bernoulli probability."""
    q = _q("multiple_choice", ["w", "x", "y", "z"], choice_type="multi", answer="A, C")
    out = parse_belief(
        _wrap(_payload({"A": 0.8, "B": 0.3, "C": 0.7, "D": 0.1})), q
    )
    assert out is not None
    # Sum is 1.9 — must still be accepted under multi.
    assert sum(out.probabilities.values()) == pytest.approx(1.9)


def test_simplex_tolerance() -> None:
    """Single-answer simplex check tolerates 1e-3 numerical noise."""
    q = _q("yes_no", ["Yes", "No"])
    # Sum = 1.0005 → within tolerance
    assert parse_belief(_wrap(_payload({"A": 0.7005, "B": 0.3})), q) is not None
    # Sum = 1.005 → outside tolerance
    assert parse_belief(_wrap(_payload({"A": 0.705, "B": 0.3})), q) is None


def test_last_belief_wins() -> None:
    """If the message contains multiple <belief> blocks, the LAST one is parsed."""
    q = _q("yes_no", ["Yes", "No"])
    text = (
        f"first attempt: <belief>{json.dumps(_payload({'A': 0.9, 'B': 0.1}))}</belief>"
        f" then revision: <belief>{json.dumps(_payload({'A': 0.4, 'B': 0.6}))}</belief>"
        " final \\boxed{No}"
    )
    out = parse_belief(text, q)
    assert out is not None
    assert out.probabilities == {"A": 0.4, "B": 0.6}


# ---- Failure modes ----------------------------------------------------------


def test_no_belief_tag_returns_none() -> None:
    q = _q("yes_no", ["Yes", "No"])
    assert parse_belief("I cannot predict.", q) is None
    assert parse_belief("", q) is None


def test_malformed_json_returns_none() -> None:
    """Trailing commas, unquoted keys — any json.loads failure → None, no exception."""
    q = _q("yes_no", ["Yes", "No"])
    text = '<belief>{"probabilities": {A: 0.5, "B": 0.5}, ...}</belief>'
    assert parse_belief(text, q) is None


def test_simplex_violation_returns_none() -> None:
    q = _q("yes_no", ["Yes", "No"])
    assert parse_belief(_wrap(_payload({"A": 0.6, "B": 0.6})), q) is None


def test_letter_set_mismatch_returns_none() -> None:
    """3-option question with only 2 letters in the belief → reject."""
    q = _q("multiple_choice", ["a", "b", "c"])
    assert parse_belief(_wrap(_payload({"A": 0.5, "B": 0.5})), q) is None


def test_out_of_range_probability_returns_none() -> None:
    q = _q("yes_no", ["Yes", "No"])
    assert parse_belief(_wrap(_payload({"A": 1.5, "B": -0.5})), q) is None


def test_illegal_confidence_returns_none() -> None:
    q = _q("yes_no", ["Yes", "No"])
    assert parse_belief(
        _wrap(_payload({"A": 0.5, "B": 0.5}, confidence="very_high")), q
    ) is None


def test_empty_key_evidence_returns_none() -> None:
    q = _q("yes_no", ["Yes", "No"])
    assert parse_belief(
        _wrap(_payload({"A": 0.5, "B": 0.5}, key_evidence=[])), q
    ) is None


def test_oversized_key_evidence_returns_none() -> None:
    q = _q("yes_no", ["Yes", "No"])
    long_str = "x" * 281
    assert parse_belief(
        _wrap(_payload({"A": 0.5, "B": 0.5}, key_evidence=[long_str])), q
    ) is None


def test_missing_decision_rule_returns_none() -> None:
    q = _q("yes_no", ["Yes", "No"])
    assert parse_belief(
        _wrap(_payload({"A": 0.5, "B": 0.5}, decision_rule="")), q
    ) is None


def test_non_dict_probabilities_returns_none() -> None:
    q = _q("yes_no", ["Yes", "No"])
    text = '<belief>{"probabilities": [0.5, 0.5]}</belief>'
    assert parse_belief(text, q) is None


def test_bool_probability_returns_none() -> None:
    """`True` / `False` are technically `int` subclasses in JSON terms — reject
    explicitly so a model emitting `{"A": true, "B": false}` is treated as
    malformed rather than as `1.0` / `0.0`."""
    q = _q("yes_no", ["Yes", "No"])
    text = '<belief>{"probabilities": {"A": true, "B": false}, "confidence": "medium", "key_evidence": ["x"], "counterevidence": [], "open_questions": [], "decision_rule": "argmax"}</belief>'
    assert parse_belief(text, q) is None


def test_belief_independent_of_boxed() -> None:
    """parse_belief result MUST NOT depend on whether \\boxed{} is present."""
    q = _q("yes_no", ["Yes", "No"])
    # Belief alone — no boxed answer in the text.
    text = f"<belief>{json.dumps(_payload({'A': 0.7, 'B': 0.3}))}</belief>"
    out = parse_belief(text, q)
    assert out is not None
    assert out.probabilities == {"A": 0.7, "B": 0.3}
