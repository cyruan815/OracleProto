"""Per-step observability tests for `react.run_react`.

Drives the loop directly with a stub LLM so we can shape `finish_reason` per
turn and assert how the loop assembles `step_metrics`, `nudges_used`, and the
last-response envelope (finish_reason / response_id / system_fingerprint /
service_tier).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from forecast_eval import loader, react
from forecast_eval.config import Settings
from forecast_eval.llm import ChatResponse, Usage
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
        id="q_step_metrics",
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
    monkeypatch.setenv("REACT_REFLECTION_PROTOCOL", "false")
    monkeypatch.setenv("REACT_MIN_SEARCH_CALLS", "0")
    monkeypatch.setenv("REACT_MAX_NUDGES", "0")
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "true")
    for k, v in overrides.items():
        monkeypatch.setenv(k, v)
    return Settings(_env_file=None)


class _ScriptedLLM:
    """Replays a list of (message, finish_reason, envelope) triples.

    Each entry is `(message_dict, finish_reason, envelope_kwargs)` — the
    envelope dict can override `response_id` / `system_fingerprint` / etc. We
    mint distinct usage values per step so step_metrics math is non-trivial.
    """

    def __init__(
        self,
        script: list[tuple[dict[str, Any], str | None, dict[str, Any]]],
        *,
        prompt_seq: tuple[int, ...] = (101, 202, 303),
        completion_seq: tuple[int, ...] = (11, 22, 33),
        reasoning_seq: tuple[int, ...] = (1, 2, 3),
    ) -> None:
        self.script = list(script)
        self.prompt_seq = prompt_seq
        self.completion_seq = completion_seq
        self.reasoning_seq = reasoning_seq
        self._step = 0
        self.calls: list[list[dict[str, Any]]] = []

    async def __call__(self, **kwargs: Any) -> ChatResponse:
        self.calls.append(list(kwargs["messages"]))
        if not self.script:
            raise AssertionError("scripted LLM exhausted")
        msg, finish, envelope = self.script.pop(0)
        i = self._step
        self._step += 1
        usage = Usage(
            prompt_tokens=self.prompt_seq[i % len(self.prompt_seq)],
            completion_tokens=self.completion_seq[i % len(self.completion_seq)],
            reasoning_tokens=self.reasoning_seq[i % len(self.reasoning_seq)],
        )
        return ChatResponse(
            message=msg,
            usage=usage,
            finish_reason=finish,
            response_id=envelope.get("response_id"),
            system_fingerprint=envelope.get("system_fingerprint"),
            service_tier=envelope.get("service_tier"),
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


async def test_step_metrics_and_nudges(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two-step trace: one tool_call (finish_reason=tool_calls), one final
    answer (finish_reason=stop). step_metrics should mirror the per-step
    snapshots; nudges_used should stay 0 because REACT_MIN_SEARCH_CALLS=0."""
    settings = _make_settings(monkeypatch)
    script = [
        (_tool_msg("call_1", "evidence please"), "tool_calls", {
            "response_id": "resp_step0",
            "system_fingerprint": "fp_a",
            "service_tier": "default",
        }),
        (_final_msg("\\boxed{Yes}"), "stop", {
            "response_id": "resp_step1",
            "system_fingerprint": "fp_b",
            "service_tier": "scale",
        }),
    ]
    llm = _ScriptedLLM(script)
    monkeypatch.setattr(react, "llm_chat", llm)
    monkeypatch.setattr(react, "tavily_search", _stub_tavily)

    result = await react.run_react(
        _yes_no_question(),
        model=settings.MODELS[0],
        sample_idx=0,
        settings=settings,
        templates=templates,
        run_id="test",
    )

    assert result.nudges_used == 0
    assert result.tool_calls_count == 1
    assert result.react_steps == 2

    # Last-response envelope must reflect the FINAL llm.chat call.
    assert result.finish_reason == "stop"
    assert result.response_id == "resp_step1"
    assert result.system_fingerprint == "fp_b"
    assert result.service_tier == "scale"

    metrics = json.loads(result.step_metrics)
    assert len(metrics) == 2
    expected_keys = {"step", "prompt", "completion", "reasoning", "latency_ms", "finish_reason", "n_tool_calls"}
    assert set(metrics[0]) == expected_keys
    # Step 0: tool_call assistant message — n_tool_calls should be 1.
    assert metrics[0]["step"] == 0
    assert metrics[0]["finish_reason"] == "tool_calls"
    assert metrics[0]["n_tool_calls"] == 1
    assert metrics[0]["prompt"] == 101
    assert metrics[0]["completion"] == 11
    # Step 1: final answer — no tool_calls on the assistant message.
    assert metrics[1]["step"] == 1
    assert metrics[1]["finish_reason"] == "stop"
    assert metrics[1]["n_tool_calls"] == 0
    assert metrics[1]["prompt"] == 202
    # latency_ms should be a non-negative int — wall clock for the single call.
    for m in metrics:
        assert isinstance(m["latency_ms"], int) and m["latency_ms"] >= 0

    # Token totals on the SampleResult are sums across the per-step snapshots.
    assert result.prompt_tokens == 101 + 202
    assert result.completion_tokens == 11 + 22
    assert result.reasoning_tokens == 1 + 2


async def test_step_metrics_counts_nudge_step(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the loop nudges and the LLM finalises afterward, step_metrics must
    include both the pre-nudge attempt AND the post-nudge final, and the
    `nudges_used` counter must reflect the actual nudge count."""
    settings = _make_settings(
        monkeypatch,
        REACT_MIN_SEARCH_CALLS="1",
        REACT_MAX_NUDGES="2",
    )
    script = [
        # Step 0: tries to finalise with 0 searches → triggers nudge.
        (_final_msg("draft \\boxed{Yes}"), "stop", {"response_id": "r0"}),
        # Step 1: actually searches → satisfies the floor.
        (_tool_msg("call_1", "evidence"), "tool_calls", {"response_id": "r1"}),
        # Step 2: final answer accepted (search count >= min).
        (_final_msg("\\boxed{Yes}"), "stop", {"response_id": "r2"}),
    ]
    llm = _ScriptedLLM(script)
    monkeypatch.setattr(react, "llm_chat", llm)
    monkeypatch.setattr(react, "tavily_search", _stub_tavily)

    result = await react.run_react(
        _yes_no_question(),
        model=settings.MODELS[0],
        sample_idx=0,
        settings=settings,
        templates=templates,
        run_id="test",
    )

    assert result.nudges_used == 1
    assert result.tool_calls_count == 1
    metrics = json.loads(result.step_metrics)
    assert len(metrics) == 3
    assert [m["step"] for m in metrics] == [0, 1, 2]
    # Final envelope reflects the LAST llm.chat (step 2).
    assert result.response_id == "r2"
    assert result.finish_reason == "stop"


async def test_finish_reason_length_path(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the last response was truncated by `length`, the SampleResult's
    `finish_reason` must reflect that — even when content still parses."""
    settings = _make_settings(monkeypatch)
    script = [
        # Single turn, no tool calls. content parses as "Yes" (=A) but we
        # advertise finish_reason=length to simulate output truncation.
        (_final_msg("\\boxed{Yes}"), "length", {
            "response_id": "resp_truncated",
            "system_fingerprint": "fp_trunc",
            "service_tier": "flex",
        }),
    ]
    llm = _ScriptedLLM(script)
    monkeypatch.setattr(react, "llm_chat", llm)
    monkeypatch.setattr(react, "tavily_search", _stub_tavily)

    result = await react.run_react(
        _yes_no_question(),
        model=settings.MODELS[0],
        sample_idx=0,
        settings=settings,
        templates=templates,
        run_id="test",
    )

    assert result.finish_reason == "length"
    assert result.response_id == "resp_truncated"
    assert result.system_fingerprint == "fp_trunc"
    assert result.service_tier == "flex"
    # Per-step snapshot for step 0 should also surface the length finish reason.
    metrics = json.loads(result.step_metrics)
    assert len(metrics) == 1
    assert metrics[0]["finish_reason"] == "length"
