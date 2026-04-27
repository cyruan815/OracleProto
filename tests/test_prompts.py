from __future__ import annotations

import json
from pathlib import Path

import pytest

from forecast_eval import loader
from forecast_eval.prompts import (
    BELIEF_PROTOCOL,
    REFLECTION_PROTOCOL,
    _build_nudge_message,
    build_budget_awareness_protocol,
    build_last_step_force_finalisation,
    build_penultimate_step_warning,
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


def _yes_no_question() -> Question:
    return Question(
        id="q_yn_protocol",
        choice_type="single",
        question_type="yes_no",
        event="will the protocol fire?",
        options=json.dumps(["Yes", "No"]),
        answer="A",
        end_time="2026-03-01",
    )


def test_reflection_protocol_appended_when_provided(templates: dict[str, str]) -> None:
    q = _yes_no_question()
    plain = render_user_prompt(q, templates)
    with_protocol = render_user_prompt(q, templates, reflection_protocol=REFLECTION_PROTOCOL)

    # 协议必须以原 prompt 为前缀, 即只在末尾追加, 不修改原模板任何字符.
    assert with_protocol.startswith(plain)
    assert with_protocol != plain
    # 关键反思要素必须出现在协议里.
    for marker in (
        "Forecasting Protocol",
        "Decompose",
        "distinct angles",
        "Cross-validate",
        "OPPOSITE",
        "Calibrate",
    ):
        assert marker in with_protocol


def test_reflection_protocol_default_is_off(templates: dict[str, str]) -> None:
    """Calling render_user_prompt without the kw must keep historical behaviour."""
    q = _yes_no_question()
    rendered = render_user_prompt(q, templates)
    assert "Forecasting Protocol" not in rendered


def test_reflection_protocol_none_equivalent_to_off(templates: dict[str, str]) -> None:
    q = _yes_no_question()
    assert render_user_prompt(q, templates) == render_user_prompt(
        q, templates, reflection_protocol=None
    )


def test_nudge_message_mentions_counts_and_new_angle() -> None:
    msg = _build_nudge_message(searches_done=1, min_required=3)
    # 必须客观陈述事实, 不泄露 end_date 等内部信息, 同时要求 LLM 换角度而非重复.
    assert "1" in msg
    assert "3" in msg
    assert "NEW angle" in msg or "new angle" in msg.lower()
    assert "end_date" not in msg
    assert "training cutoff" not in msg.lower()


# ---- v4 belief protocol -----------------------------------------------------


def test_belief_protocol_contains_required_fields() -> None:
    """Static contract: the protocol body must mention the belief tag and
    every required JSON key so the model has unambiguous instructions."""
    for marker in (
        "<belief>",
        "</belief>",
        "probabilities",
        "confidence",
        "key_evidence",
        "counterevidence",
        "decision_rule",
    ):
        assert marker in BELIEF_PROTOCOL, f"BELIEF_PROTOCOL missing required marker {marker!r}"


def test_belief_protocol_token_budget_under_800() -> None:
    """Soft budget check. Goal is ~500 tokens (cl100k_base); ceiling is 800.

    Skips if `tiktoken` isn't installed — Phase 0 deliberately avoids adding
    new heavy deps, and a char-based proxy is sufficient signal here. The
    char proxy uses 3.5 chars/token (English text average), giving ~enough
    headroom that anyone running the test locally can spot regressions."""
    try:
        import tiktoken
    except ImportError:
        # Fallback: char-count proxy — 800 tokens * ~3.5 chars ≈ 2800 chars.
        assert len(BELIEF_PROTOCOL) < 3500, (
            f"BELIEF_PROTOCOL length {len(BELIEF_PROTOCOL)} chars exceeds soft "
            "budget; install tiktoken for an exact token count."
        )
        return
    enc = tiktoken.get_encoding("cl100k_base")
    n = len(enc.encode(BELIEF_PROTOCOL))
    assert n < 800, f"BELIEF_PROTOCOL is {n} tokens (cl100k_base); budget < 800"


def test_belief_protocol_appended_when_provided(templates: dict[str, str]) -> None:
    q = _yes_no_question()
    plain = render_user_prompt(q, templates)
    with_belief = render_user_prompt(q, templates, belief_protocol=BELIEF_PROTOCOL)
    # Belief is appended ONLY at the tail; original body is a strict prefix.
    assert with_belief.startswith(plain)
    assert with_belief != plain
    assert "<belief>" in with_belief
    assert "</belief>" in with_belief


def test_belief_protocol_default_is_off(templates: dict[str, str]) -> None:
    q = _yes_no_question()
    rendered = render_user_prompt(q, templates)
    assert "<belief>" not in rendered
    assert "Belief Protocol" not in rendered


def test_belief_protocol_none_equivalent_to_off(templates: dict[str, str]) -> None:
    q = _yes_no_question()
    assert render_user_prompt(q, templates) == render_user_prompt(
        q, templates, belief_protocol=None
    )


def test_reflection_then_belief_order(templates: dict[str, str]) -> None:
    """When BOTH protocols are enabled, reflection comes BEFORE belief."""
    q = _yes_no_question()
    rendered = render_user_prompt(
        q,
        templates,
        reflection_protocol=REFLECTION_PROTOCOL,
        belief_protocol=BELIEF_PROTOCOL,
    )
    refl_pos = rendered.find("Forecasting Protocol")
    belief_pos = rendered.find("Belief Protocol")
    assert refl_pos > 0 and belief_pos > 0
    assert refl_pos < belief_pos, "reflection MUST appear before belief in the rendered message"


# ---- force-final-answer-near-limit-v1 ---------------------------------------


def test_budget_awareness_protocol_contains_budget_numbers() -> None:
    """The rendered budget tail must surface the actual N / C the run uses, so
    the model's plan is grounded in the real harness limits (not a generic
    placeholder)."""
    text = build_budget_awareness_protocol(max_steps=12, max_search_calls=8)
    assert "Budget Awareness" in text
    assert "**12**" in text
    assert "**8**" in text
    # Plain prose tells the model to leave the last step for the boxed answer.
    assert "LAST step" in text or "final step" in text.lower()
    assert "\\boxed{...}" in text


def test_budget_awareness_protocol_handles_zero_searches() -> None:
    """When ENABLE_WEB_SEARCH=false the runtime passes max_search_calls=0;
    the protocol must NOT advertise a positive search budget in that case."""
    text = build_budget_awareness_protocol(max_steps=4, max_search_calls=0)
    assert "**4**" in text
    assert "web_search is disabled" in text
    # No "**N** web_search" claim when the tool is gone.
    assert "**0** `web_search`" not in text


def test_budget_awareness_protocol_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        build_budget_awareness_protocol(max_steps=0, max_search_calls=5)
    with pytest.raises(ValueError):
        build_budget_awareness_protocol(max_steps=4, max_search_calls=-1)


def test_budget_awareness_appended_when_provided(templates: dict[str, str]) -> None:
    q = _yes_no_question()
    plain = render_user_prompt(q, templates)
    tail = build_budget_awareness_protocol(max_steps=6, max_search_calls=4)
    with_budget = render_user_prompt(q, templates, budget_awareness_protocol=tail)
    assert with_budget.startswith(plain)
    assert "Budget Awareness" in with_budget
    assert "**6**" in with_budget
    assert "**4**" in with_budget


def test_budget_awareness_default_is_off(templates: dict[str, str]) -> None:
    q = _yes_no_question()
    rendered = render_user_prompt(q, templates)
    assert "Budget Awareness" not in rendered


def test_budget_awareness_before_reflection_and_belief(templates: dict[str, str]) -> None:
    """Order contract: budget → reflection → belief. Budget must come FIRST so
    the model has the global plan before reading methodology layers."""
    q = _yes_no_question()
    rendered = render_user_prompt(
        q,
        templates,
        budget_awareness_protocol=build_budget_awareness_protocol(
            max_steps=6, max_search_calls=4
        ),
        reflection_protocol=REFLECTION_PROTOCOL,
        belief_protocol=BELIEF_PROTOCOL,
    )
    budget_pos = rendered.find("Budget Awareness")
    refl_pos = rendered.find("Forecasting Protocol")
    belief_pos = rendered.find("Belief Protocol")
    assert 0 < budget_pos < refl_pos < belief_pos


def test_penultimate_warning_mentions_budget_state() -> None:
    """The soft warning must surface (current_step / max_steps) and the search
    budget snapshot so the model knows where it stands."""
    msg = build_penultimate_step_warning(
        current_step=2, max_steps=3, searches_done=1, max_search_calls=4
    )
    assert "2 of 3" in msg
    assert "1/4" in msg
    assert "second-to-last" in msg
    # The warning explicitly previews what happens next turn.
    assert "NEXT step" in msg


def test_last_step_force_finalisation_mentions_cutoff() -> None:
    msg = build_last_step_force_finalisation(current_step=3, max_steps=3)
    assert "Harness cutoff" in msg
    assert "3 of 3" in msg
    assert "\\boxed{...}" in msg
    # Tells the model not to default to an empty reply.
    assert "empty reply" in msg.lower() or "scores zero" in msg.lower()
