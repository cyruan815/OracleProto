from __future__ import annotations

import json
from pathlib import Path

import pytest

from forecast_eval import loader
from forecast_eval.prompts import (
    index_to_letter,
    letter_to_index,
    render_user_prompt,
)
from forecast_eval.types import Question


SOURCE_DB = Path(__file__).resolve().parents[1] / "forecast_eval_set_example.db"


@pytest.fixture(scope="module")
def templates() -> dict[str, str]:
    raw = loader.load_raw_features_json(SOURCE_DB)
    features = json.loads(raw)
    reconstruction = features["prompt_reconstruction"]
    return {
        k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, sort_keys=True)
        for k, v in reconstruction.items()
    }


def test_letter_index_roundtrip() -> None:
    for i in range(0, 35):
        letter = index_to_letter(i)
        assert letter_to_index(letter) == i
    # 26..34 must map onto the ASCII continuation set
    assert index_to_letter(26) == "["
    assert index_to_letter(31) == "`"
    assert index_to_letter(32) == "a"


def test_yes_no_render(templates: dict[str, str]) -> None:
    q = Question(
        id="q_yn",
        choice_type="single",
        question_type="yes_no",
        event="2026 a dream year for trump?",
        options=json.dumps(["Yes", "No"]),
        answer="B",
        end_time="2026-01-31",
    )
    rendered = render_user_prompt(q, templates)
    assert "2026 a dream year for trump?" in rendered
    assert "resolved around 2026-01-31 (GMT+8)" in rendered
    assert "\nA." not in rendered  # no outcomes block
    assert "\\boxed{Yes}" in rendered or "boxed{Yes}" in rendered


def test_binary_named_render_replaces_placeholders(templates: dict[str, str]) -> None:
    q = Question(
        id="q_bn",
        choice_type="single",
        question_type="binary_named",
        event="Golden Knights vs. Kings",
        options=json.dumps(["Golden Knights", "Kings"]),
        answer="A",
        end_time="2026-01-15",
    )
    rendered = render_user_prompt(q, templates)
    assert "<options[0]>" not in rendered
    assert "<options[1]>" not in rendered
    assert "Golden Knights" in rendered
    assert "Kings" in rendered


def test_multiple_choice_render_under_26(templates: dict[str, str]) -> None:
    q = Question(
        id="q_mc",
        choice_type="single",
        question_type="multiple_choice",
        event="Bank of Brazil decision",
        options=json.dumps(["No change", "Raise", "Lower"]),
        answer="A",
        end_time="2026-01-27",
    )
    rendered = render_user_prompt(q, templates)
    assert "\nA. No change" in rendered
    assert "\nB. Raise" in rendered
    assert "\nC. Lower" in rendered


def test_multiple_choice_over_26_protects_labels(templates: dict[str, str]) -> None:
    """Use real DB data so tests match what actually ships."""
    import sqlite3

    conn = sqlite3.connect(f"file:{SOURCE_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT id, choice_type, question_type, event, options, answer, end_time "
        "FROM forecast_eval_set_example WHERE json_array_length(options) > 26 ORDER BY json_array_length(options) DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row is not None

    q = Question(
        id=row["id"],
        choice_type=row["choice_type"],
        question_type=row["question_type"],
        event=row["event"],
        options=row["options"],
        answer=row["answer"],
        end_time=row["end_time"],
    )
    rendered = render_user_prompt(q, templates)

    options = json.loads(q.options)
    # Every option must appear in the rendered prompt, even those whose letter
    # falls outside A..Z (those lines are rendered with backtick protection).
    for i, label in enumerate(options):
        letter = index_to_letter(i)
        if i < 26:
            assert f"{letter}. {label}" in rendered
        else:
            assert f"`{letter}`. {label}" in rendered


def test_render_is_deterministic(templates: dict[str, str]) -> None:
    q = Question(
        id="q_mc",
        choice_type="single",
        question_type="multiple_choice",
        event="Coin flip",
        options=json.dumps(["A-label", "B-label", "C-label"]),
        answer="A",
        end_time="2026-02-14",
    )
    assert render_user_prompt(q, templates) == render_user_prompt(q, templates)


def test_unknown_question_type_raises(templates: dict[str, str]) -> None:
    q = Question(
        id="q_bad",
        choice_type="single",
        question_type="numeric",
        event="...",
        options="[]",
        answer="A",
        end_time="2026-01-01",
    )
    with pytest.raises(ValueError):
        render_user_prompt(q, templates)
