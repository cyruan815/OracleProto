"""Targeted tests for the ReAct reflection scaffold + soft search-floor nudge.

These do NOT spin the runner; they drive `run_react` directly with stubbed
LLM/Tavily so we can shape the message sequence and assert on the loop's
behaviour around protocol injection and the nudge mechanism.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from forecast_eval import loader, react
from forecast_eval.config import Settings
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


def _yes_no_question() -> Question:
    return Question(
        id="q_react_test",
        choice_type="single",
        question_type="yes_no",
        event="will the test pass?",
        options=json.dumps(["Yes", "No"]),
        answer="A",
        end_time="2026-03-01",
    )


def _make_settings(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    monkeypatch.setenv("LLM_API_KEY", "sk-or-v1-TEST_ABCDEFGH")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-TEST_ABCDEFGH")
    monkeypatch.setenv("MODELS", "openai/gpt-4o-mini")
    monkeypatch.setenv("MODEL_TRAINING_CUTOFFS", "")
    monkeypatch.setenv("REACT_MAX_STEPS", "6")
    monkeypatch.setenv("REACT_MAX_SEARCH_CALLS", "5")
    monkeypatch.setenv("REACT_REFLECTION_PROTOCOL", "true")
    monkeypatch.setenv("REACT_MIN_SEARCH_CALLS", "0")
    monkeypatch.setenv("REACT_MAX_NUDGES", "2")
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "true")
    for k, v in overrides.items():
        monkeypatch.setenv(k, v)
    return Settings(_env_file=None)


class _StubLLM:
    """Replays a scripted sequence of assistant responses."""

    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        # Snapshot the messages list at call time (deep enough for assertions).
        msgs = kwargs["messages"]
        self.calls.append([dict(m) for m in msgs])
        if not self.responses:
            raise AssertionError("LLM stub exhausted")
        nxt = self.responses.pop(0)
        from forecast_eval.llm import ChatResponse, Usage

        return ChatResponse(
            message=nxt,
            usage=Usage(prompt_tokens=10, completion_tokens=5, reasoning_tokens=0),
        )


def _final_msg(text: str = "After analysis: \\boxed{Yes}") -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def _tool_msg(tc_id: str, query: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc_id,
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": json.dumps({"query": query}),
                },
            }
        ],
    }


async def _stub_tavily(query: str, end_date: str, *args: Any, **kwargs: Any) -> Any:
    from forecast_eval.search import SearchResult, SearchResultItem

    return SearchResult(
        query=query,
        end_date=end_date,
        answer=None,
        results=[
            SearchResultItem(
                title="t",
                url="https://example.com/x",
                content="snippet",
                published_date="2026-02-01",
                raw_content=None,
            )
        ],
    )


async def test_protocol_appended_to_user_prompt_when_enabled(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _make_settings(monkeypatch, REACT_REFLECTION_PROTOCOL="true")
    stub = _StubLLM([_final_msg()])
    monkeypatch.setattr(react, "llm_chat", stub)

    result = await react.run_react(
        _yes_no_question(),
        model=settings.MODELS[0],
        sample_idx=0,
        settings=settings,
        templates=templates,
        run_id="test",
    )
    assert "Forecasting Protocol" in result.user_prompt
    assert result.tool_calls_count == 0
    assert result.react_steps == 1


async def test_protocol_absent_when_disabled(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _make_settings(monkeypatch, REACT_REFLECTION_PROTOCOL="false")
    stub = _StubLLM([_final_msg()])
    monkeypatch.setattr(react, "llm_chat", stub)

    result = await react.run_react(
        _yes_no_question(),
        model=settings.MODELS[0],
        sample_idx=0,
        settings=settings,
        templates=templates,
        run_id="test",
    )
    assert "Forecasting Protocol" not in result.user_prompt


async def test_nudge_fires_when_finalize_below_min_searches(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _make_settings(
        monkeypatch,
        REACT_MIN_SEARCH_CALLS="2",
        REACT_MAX_NUDGES="2",
    )
    # Sequence: try to finalize immediately (0 searches → nudge),
    #          then call web_search once (1 search; still < 2 → nudge again),
    #          then call web_search once more (now 2 searches),
    #          then finalize.
    stub = _StubLLM(
        [
            _final_msg("draft answer \\boxed{Yes}"),
            _tool_msg("call_1", "first angle"),
            _final_msg("another draft \\boxed{No}"),
            _tool_msg("call_2", "second angle"),
            _final_msg("final \\boxed{Yes}"),
        ]
    )
    monkeypatch.setattr(react, "llm_chat", stub)
    monkeypatch.setattr(react, "tavily_search", _stub_tavily)

    result = await react.run_react(
        _yes_no_question(),
        model=settings.MODELS[0],
        sample_idx=0,
        settings=settings,
        templates=templates,
        run_id="test",
    )
    assert result.tool_calls_count == 2
    assert result.final_answer_letters == json.dumps(["A"])
    # Two nudges should appear in the message trace as user messages with the
    # canonical phrasing.
    trace = json.loads(result.messages_trace)
    nudge_msgs = [
        m for m in trace
        if m.get("role") == "user"
        and "protocol requires consulting at least" in (m.get("content") or "")
    ]
    assert len(nudge_msgs) == 2


async def test_nudge_capped_by_max_nudges(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _make_settings(
        monkeypatch,
        REACT_MIN_SEARCH_CALLS="3",
        REACT_MAX_NUDGES="1",
        REACT_MAX_STEPS="6",
    )
    # LLM stubbornly tries to finalize without searching. After 1 nudge
    # (cap), the 2nd finalize attempt must be accepted as final, even
    # though search count is still 0.
    stub = _StubLLM(
        [
            _final_msg("first try \\boxed{Yes}"),
            _final_msg("second try \\boxed{No}"),
        ]
    )
    monkeypatch.setattr(react, "llm_chat", stub)

    result = await react.run_react(
        _yes_no_question(),
        model=settings.MODELS[0],
        sample_idx=0,
        settings=settings,
        templates=templates,
        run_id="test",
    )
    assert result.tool_calls_count == 0
    # The final answer is the SECOND attempt, after exactly one nudge.
    assert result.final_answer_letters == json.dumps(["B"])
    trace = json.loads(result.messages_trace)
    nudge_msgs = [
        m for m in trace
        if m.get("role") == "user"
        and "protocol requires consulting at least" in (m.get("content") or "")
    ]
    assert len(nudge_msgs) == 1


async def test_nudge_disabled_when_min_is_zero(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _make_settings(monkeypatch, REACT_MIN_SEARCH_CALLS="0")
    stub = _StubLLM([_final_msg()])
    monkeypatch.setattr(react, "llm_chat", stub)

    result = await react.run_react(
        _yes_no_question(),
        model=settings.MODELS[0],
        sample_idx=0,
        settings=settings,
        templates=templates,
        run_id="test",
    )
    # Single LLM call — no nudge attempted.
    assert len(stub.calls) == 1
    assert result.tool_calls_count == 0


async def test_nudge_skipped_when_web_search_disabled(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ENABLE_WEB_SEARCH=false, there's no search to nudge toward — the
    loop must NOT inject a nudge even with REACT_MIN_SEARCH_CALLS>0 set."""
    settings = _make_settings(
        monkeypatch,
        ENABLE_WEB_SEARCH="false",
        REACT_MIN_SEARCH_CALLS="3",
    )
    stub = _StubLLM([_final_msg()])
    monkeypatch.setattr(react, "llm_chat", stub)

    result = await react.run_react(
        _yes_no_question(),
        model=settings.MODELS[0],
        sample_idx=0,
        settings=settings,
        templates=templates,
        run_id="test",
    )
    # Exactly one LLM call, no nudge. Web-search-disabled mode is preserved.
    assert len(stub.calls) == 1
    assert result.tool_calls_count == 0


def test_settings_rejects_min_above_max(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-or-v1-TEST_ABCDEFGH")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-TEST_ABCDEFGH")
    monkeypatch.setenv("MODELS", "openai/gpt-4o-mini")
    monkeypatch.setenv("REACT_MAX_SEARCH_CALLS", "3")
    monkeypatch.setenv("REACT_MIN_SEARCH_CALLS", "5")
    with pytest.raises(ValueError, match="REACT_MIN_SEARCH_CALLS"):
        Settings(_env_file=None)


def test_settings_rejects_negative_min(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-or-v1-TEST_ABCDEFGH")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-TEST_ABCDEFGH")
    monkeypatch.setenv("MODELS", "openai/gpt-4o-mini")
    monkeypatch.setenv("REACT_MIN_SEARCH_CALLS", "-1")
    with pytest.raises(ValueError, match="REACT_MIN_SEARCH_CALLS"):
        Settings(_env_file=None)
