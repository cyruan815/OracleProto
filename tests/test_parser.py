from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from forecast_eval.parser import is_correct, parse_answer, parse_gt
from forecast_eval.types import Question


SOURCE_DB = Path(__file__).resolve().parents[1] / "forecast_eval_set_example.db"


def _q(qt: str, options: list[str], answer: str = "A") -> Question:
    return Question(
        id="qid",
        choice_type="single",
        question_type=qt,
        event="ev",
        options=json.dumps(options),
        answer=answer,
        end_time="2026-01-01",
    )


def test_yes_no_case_insensitive() -> None:
    q = _q("yes_no", ["Yes", "No"])
    for s in ("yes", "YES", "Yes"):
        assert parse_answer(f"analysis... \\boxed{{{s}}}", q) == frozenset({"A"})
    for s in ("no", "NO", "No"):
        assert parse_answer(f"analysis... \\boxed{{{s}}}", q) == frozenset({"B"})
    assert parse_answer("analysis... \\boxed{Maybe}", q) is None


def test_last_boxed_wins() -> None:
    q = _q("yes_no", ["Yes", "No"])
    text = "I first wrote \\boxed{Yes} then changed to \\boxed{No}."
    assert parse_answer(text, q) == frozenset({"B"})


def test_binary_named_exact_match_only() -> None:
    q = _q("binary_named", ["Golden Knights", "Kings"], answer="A")
    assert parse_answer("final: \\boxed{ Kings }", q) == frozenset({"B"})
    assert parse_answer("final: \\boxed{kings}", q) == frozenset({"B"})
    # fuzzy match is explicitly rejected
    assert parse_answer("final: \\boxed{L.A. Kings}", q) is None


def test_multiple_choice_single_and_multi() -> None:
    q = _q("multiple_choice", ["x", "y", "z"])
    assert parse_answer("\\boxed{A}", q) == frozenset({"A"})
    assert parse_answer("\\boxed{A, C}", q) == frozenset({"A", "C"})
    assert parse_answer("\\boxed{A C}", q) == frozenset({"A", "C"})
    # out-of-range
    assert parse_answer("\\boxed{D}", q) is None
    # multi-char token
    assert parse_answer("\\boxed{AB}", q) is None
    # empty
    assert parse_answer("\\boxed{}", q) is None


def test_no_boxed_returns_none() -> None:
    q = _q("yes_no", ["Yes", "No"])
    assert parse_answer("I cannot predict the future.", q) is None
    assert parse_answer("", q) is None


def test_parse_gt() -> None:
    assert parse_gt("A") == frozenset({"A"})
    assert parse_gt("A, B") == frozenset({"A", "B"})
    assert parse_gt("A,B") == frozenset({"A", "B"})
    assert parse_gt("A, B, D, E") == frozenset({"A", "B", "D", "E"})
    with pytest.raises(ValueError):
        parse_gt("")
    with pytest.raises(ValueError):
        parse_gt("   ")


def test_is_correct() -> None:
    assert is_correct(frozenset({"A", "B"}), frozenset({"A", "B"})) is True
    assert is_correct(frozenset({"A"}), frozenset({"A", "B"})) is False
    assert is_correct(frozenset({"A", "B", "C"}), frozenset({"A", "B"})) is False
    assert is_correct(None, frozenset({"A"})) is None


