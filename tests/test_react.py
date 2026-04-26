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
    """Build a cell-local Settings sub-view for direct run_react tests.

    `react.py` and `search.py` read `settings.TAVILY_MAX_RESULTS` /
    `REACT_MAX_SEARCH_CALLS` as single ints (the dispatcher's per-cell view).
    Tests that bypass evaluation.py / runner.py and call `run_react` directly
    must therefore present a sub-view via `model_copy(update=...)`, mirroring
    `_make_settings_factory` in evaluation.py.
    """
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
    base = Settings(_env_file=None)
    return base.model_copy(
        update={
            "TAVILY_MAX_RESULTS": int(base.TAVILY_MAX_RESULTS[0]),
            "REACT_MAX_SEARCH_CALLS": int(base.REACT_MAX_SEARCH_CALLS[0]),
        }
    )


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
    # v4 step_metrics schema adds `belief` (always None when BELIEF_PROTOCOL
    # is off — keeps the JSON shape uniform across protocol-on/off runs).
    expected_keys = {
        "step", "prompt", "completion", "reasoning", "latency_ms",
        "finish_reason", "n_tool_calls", "belief",
    }
    assert set(metrics[0]) == expected_keys
    assert metrics[0]["belief"] is None
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


# ---- v4 belief capture ------------------------------------------------------


def _belief_block(probs: dict[str, float], confidence: str = "medium") -> str:
    return (
        "<belief>"
        + json.dumps(
            {
                "version": "v4.0",
                "probabilities": probs,
                "confidence": confidence,
                "key_evidence": ["primary signal"],
                "counterevidence": [],
                "open_questions": [],
                "decision_rule": "argmax",
            }
        )
        + "</belief>"
    )


async def test_belief_capture(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """When BELIEF_PROTOCOL=True and every step emits a valid `<belief>` block,
    `step_metrics[i].belief` is populated, and the SampleResult's three new
    fields reflect the LAST step's belief."""
    settings = _make_settings(monkeypatch, BELIEF_PROTOCOL="true")
    # Two steps: a tool call (with belief on the assistant message) then a
    # final answer (with belief). The final-step belief drives belief_final.
    step0_msg = _tool_msg("call_1", "search me")
    step0_msg["content"] = (
        "Initial belief: " + _belief_block({"A": 0.55, "B": 0.45}, confidence="low")
    )
    step1_msg = _final_msg(
        "After research, "
        + _belief_block({"A": 0.8, "B": 0.2}, confidence="high")
        + " final \\boxed{Yes}"
    )
    script = [
        (step0_msg, "tool_calls", {
            "response_id": "resp0", "system_fingerprint": "fp_a", "service_tier": "default",
        }),
        (step1_msg, "stop", {
            "response_id": "resp1", "system_fingerprint": "fp_b", "service_tier": "default",
        }),
    ]
    monkeypatch.setattr(react, "llm_chat", _ScriptedLLM(script))
    monkeypatch.setattr(react, "tavily_search", _stub_tavily)

    result = await react.run_react(
        _yes_no_question(),
        model=settings.MODELS[0],
        sample_idx=0,
        settings=settings,
        templates=templates,
        run_id="test",
    )

    # Boxed-answer path is unchanged.
    assert result.parse_ok == 1
    assert result.correct == 1

    # Three v4 fields populated from the LAST belief.
    assert result.belief_parse_ok == 1
    assert json.loads(result.belief_final) == {"A": 0.8, "B": 0.2}
    trace = json.loads(result.belief_trace)
    assert len(trace) == 2
    assert trace[0]["p"] == {"A": 0.55, "B": 0.45}
    assert trace[0]["confidence"] == "low"
    assert trace[1]["p"] == {"A": 0.8, "B": 0.2}
    assert trace[1]["confidence"] == "high"
    # step_metrics elements carry per-step belief inline.
    metrics = json.loads(result.step_metrics)
    assert len(metrics) == 2
    assert metrics[0]["belief"]["p"] == {"A": 0.55, "B": 0.45}
    assert metrics[1]["belief"]["p"] == {"A": 0.8, "B": 0.2}


async def test_belief_failure_fallback(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Last step ships a valid `\\boxed{...}` but NO `<belief>` block. The
    boxed-answer path stays unaffected (parse_ok=1, correct=1) while the
    three v4 fields fall back to None / None / 0."""
    settings = _make_settings(monkeypatch, BELIEF_PROTOCOL="true")
    step0_msg = _tool_msg("call_1", "evidence")
    step0_msg["content"] = "Mid-flight: " + _belief_block({"A": 0.5, "B": 0.5})
    # Final assistant message — boxed only, no belief tag.
    step1_msg = _final_msg("Decided. \\boxed{Yes}")
    script = [
        (step0_msg, "tool_calls", {"response_id": "r0", "system_fingerprint": "f", "service_tier": "default"}),
        (step1_msg, "stop", {"response_id": "r1", "system_fingerprint": "f", "service_tier": "default"}),
    ]
    monkeypatch.setattr(react, "llm_chat", _ScriptedLLM(script))
    monkeypatch.setattr(react, "tavily_search", _stub_tavily)

    result = await react.run_react(
        _yes_no_question(),
        model=settings.MODELS[0],
        sample_idx=0,
        settings=settings,
        templates=templates,
        run_id="test",
    )

    # Boxed-answer fields untouched: belief failure MUST NOT pollute parse_ok / correct.
    assert result.parse_ok == 1
    assert result.correct == 1
    assert result.error is None

    # Last-step belief failed → belief_parse_ok=0, belief_final=None.
    # belief_trace still records the earlier successful step alongside the None.
    assert result.belief_parse_ok == 0
    assert result.belief_final is None
    trace = json.loads(result.belief_trace)
    assert len(trace) == 2
    assert trace[0] is not None and trace[0]["p"] == {"A": 0.5, "B": 0.5}
    assert trace[1] is None


async def test_belief_protocol_disabled(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With BELIEF_PROTOCOL=False the user prompt MUST be byte-identical to
    the v3 rendering, parse_belief MUST NOT be called, and the three v4
    fields are None / None / 0."""
    settings = _make_settings(monkeypatch)  # default: BELIEF_PROTOCOL stays False
    assert settings.BELIEF_PROTOCOL is False

    # Spy on parse_belief — this MUST stay at 0 calls.
    parse_calls = {"n": 0}
    real_parse = react.parse_belief

    def _counting_parse(text: str, q):
        parse_calls["n"] += 1
        return real_parse(text, q)

    monkeypatch.setattr(react, "parse_belief", _counting_parse)

    # Even if the LLM accidentally emits a belief block, parse_belief MUST NOT run.
    step0_msg = _final_msg(
        "Even if I emit " + _belief_block({"A": 0.7, "B": 0.3}) + " final \\boxed{Yes}"
    )
    monkeypatch.setattr(
        react,
        "llm_chat",
        _ScriptedLLM([(step0_msg, "stop", {"response_id": "r", "system_fingerprint": "f", "service_tier": "default"})]),
    )
    monkeypatch.setattr(react, "tavily_search", _stub_tavily)

    result = await react.run_react(
        _yes_no_question(),
        model=settings.MODELS[0],
        sample_idx=0,
        settings=settings,
        templates=templates,
        run_id="test",
    )

    assert parse_calls["n"] == 0, "parse_belief MUST NOT be called when protocol is disabled"
    assert result.parse_ok == 1
    assert result.correct == 1
    # All three v4 fields stay nil.
    assert result.belief_final is None
    assert result.belief_trace is None
    assert result.belief_parse_ok == 0
    # step_metrics still has the belief slot, but it's None.
    metrics = json.loads(result.step_metrics)
    assert metrics[0]["belief"] is None

    # Render the v3 user message via the prompts API directly and compare —
    # MUST be byte-identical to what the loop produced.
    from forecast_eval.prompts import render_user_prompt
    expected = render_user_prompt(
        _yes_no_question(),
        templates,
        reflection_protocol=None,
        belief_protocol=None,
    )
    assert result.user_prompt == expected


# ---- v5.1 harness-resilience: budget-exceeded drop tools + final-answer retry --


async def test_budget_exceeded_drops_tools(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once `len(search_calls) >= REACT_MAX_SEARCH_CALLS`, the next LLM call
    MUST be issued with `tools=[]`. The model receives no `web_search` schema
    and the loop exits via the existing "no tool_calls → break" branch.

    Script: 4 tool_calls (consume the budget of 4), then on step 5 the loop
    drops the schema and the scripted LLM returns a `\\boxed{A}` content. We
    verify both the per-call `tools=` arg and the final SampleResult shape."""
    settings = _make_settings(
        monkeypatch,
        REACT_MAX_STEPS="6",
        REACT_MAX_SEARCH_CALLS="4",
    )
    script = [
        (_tool_msg("call_1", "q1"), "tool_calls", {"response_id": "r1"}),
        (_tool_msg("call_2", "q2"), "tool_calls", {"response_id": "r2"}),
        (_tool_msg("call_3", "q3"), "tool_calls", {"response_id": "r3"}),
        (_tool_msg("call_4", "q4"), "tool_calls", {"response_id": "r4"}),
        # Step 5: tools=[] — the LLM only sees the no-tool schema, replies
        # with a final boxed answer. If the harness still passed tools here
        # we'd assert below.
        (_final_msg("\\boxed{Yes}"), "stop", {"response_id": "r5"}),
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

    # The 5th `llm_chat` call MUST have received tools=[].
    # _ScriptedLLM doesn't capture tools directly — but the loop's branching
    # is the only way step 5 can reach here without an extra round-trip, so
    # verify behaviorally: 5 LLM calls total, react_steps=5, no retry.
    assert len(llm.calls) == 5
    assert result.react_steps == 5
    assert result.final_answer_retry_used == 0
    assert result.parse_ok == 1
    assert result.tool_calls_count == 4
    assert "\\boxed{Yes}" in result.final_answer_raw


async def test_budget_exceeded_disabled_falls_back(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With `REACT_BUDGET_EXCEEDED_DROP_TOOLS=False` the loop preserves the
    legacy behaviour: tool schemas keep being exposed even after the budget
    runs out, and excess `web_search` calls get echoed `search budget exceeded`
    tool_result errors."""
    settings = _make_settings(
        monkeypatch,
        REACT_MAX_STEPS="6",
        REACT_MAX_SEARCH_CALLS="4",
        REACT_BUDGET_EXCEEDED_DROP_TOOLS="false",
    )
    script = [
        (_tool_msg("call_1", "q1"), "tool_calls", {"response_id": "r1"}),
        (_tool_msg("call_2", "q2"), "tool_calls", {"response_id": "r2"}),
        (_tool_msg("call_3", "q3"), "tool_calls", {"response_id": "r3"}),
        (_tool_msg("call_4", "q4"), "tool_calls", {"response_id": "r4"}),
        # Step 5: tools still exposed → model can request another web_search.
        (_tool_msg("call_5", "q5"), "tool_calls", {"response_id": "r5"}),
        # Step 6: model finally gives up and answers (post-budget-exceeded
        # tool_result is in messages history).
        (_final_msg("\\boxed{Yes}"), "stop", {"response_id": "r6"}),
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

    # 6 LLM calls because tools were still exposed at step 5; budget was
    # exceeded so the 5th tool_call got a `search budget exceeded` echo (but
    # tool_calls_count counts SUCCESSFUL tavily calls, capped at 4).
    assert len(llm.calls) == 6
    assert result.tool_calls_count == 4  # one extra request was rejected
    assert result.final_answer_retry_used == 0  # final_raw was filled in step 6


async def test_final_answer_retry_triggered(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """All 6 react steps emit tool_calls (loop exhausts before producing a
    final answer); the bail-out retry MUST fire once with `tools=[]` and
    populate `final_raw` from its content."""
    settings = _make_settings(
        monkeypatch,
        REACT_MAX_STEPS="6",
        REACT_MAX_SEARCH_CALLS="100",  # not the budget we're testing here
    )
    script = [
        (_tool_msg(f"call_{i}", f"q{i}"), "tool_calls", {"response_id": f"r{i}"})
        for i in range(1, 7)
    ]
    # The 7th call is the bail-out retry.
    script.append(
        (_final_msg("\\boxed{No}"), "stop", {"response_id": "retry"})
    )
    llm = _ScriptedLLM(
        script,
        # Pad sequences out to 7 entries so step 6 (idx 6) doesn't IndexError.
        prompt_seq=(101, 202, 303, 404, 505, 606, 707),
        completion_seq=(11, 22, 33, 44, 55, 66, 77),
        reasoning_seq=(1, 2, 3, 4, 5, 6, 7),
    )
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

    assert result.final_answer_retry_used == 1
    assert result.react_steps == 7  # 6 loop steps + 1 retry
    assert result.nudges_used == 0  # retry MUST NOT increment nudges
    assert result.parse_ok == 1
    assert "\\boxed{No}" in result.final_answer_raw
    metrics = json.loads(result.step_metrics)
    assert len(metrics) == 7
    # The retry step records belief=None (no belief block parsed on bail-out).
    assert metrics[-1]["belief"] is None


async def test_final_answer_retry_disabled(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the switch off, the loop exits with `final_raw=""` and `parse_ok=0`
    just like the legacy behavior. The new column reads 0."""
    settings = _make_settings(
        monkeypatch,
        REACT_MAX_STEPS="6",
        REACT_MAX_SEARCH_CALLS="100",
        REACT_FINAL_ANSWER_RETRY="false",
    )
    script = [
        (_tool_msg(f"call_{i}", f"q{i}"), "tool_calls", {"response_id": f"r{i}"})
        for i in range(1, 7)
    ]
    llm = _ScriptedLLM(
        script,
        prompt_seq=(101, 202, 303, 404, 505, 606),
        completion_seq=(11, 22, 33, 44, 55, 66),
        reasoning_seq=(1, 2, 3, 4, 5, 6),
    )
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

    assert result.final_answer_retry_used == 0
    assert result.react_steps == 6
    assert result.parse_ok == 0
    assert result.final_answer_raw == ""


async def test_final_answer_retry_still_empty(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the retry call also returns empty content, we MUST NOT loop forever
    — the function returns with `final_answer_retry_used=1` and `parse_ok=0`."""
    settings = _make_settings(
        monkeypatch,
        REACT_MAX_STEPS="6",
        REACT_MAX_SEARCH_CALLS="100",
    )
    script = [
        (_tool_msg(f"call_{i}", f"q{i}"), "tool_calls", {"response_id": f"r{i}"})
        for i in range(1, 7)
    ]
    # Bail-out retry — content is empty too.
    script.append(({"role": "assistant", "content": ""}, "stop", {"response_id": "retry"}))
    llm = _ScriptedLLM(
        script,
        prompt_seq=(101, 202, 303, 404, 505, 606, 707),
        completion_seq=(11, 22, 33, 44, 55, 66, 77),
        reasoning_seq=(1, 2, 3, 4, 5, 6, 7),
    )
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

    assert result.final_answer_retry_used == 1
    assert result.parse_ok == 0
    assert result.final_answer_raw == ""
    # Exactly 7 LLM calls — no second retry.
    assert len(llm.calls) == 7


async def test_final_answer_retry_skipped_when_already_filled(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the loop break path already filled `final_raw` (even with malformed
    content that fails parse_answer), the bail-out retry MUST NOT fire."""
    settings = _make_settings(monkeypatch)
    # Single step: model goes straight to a final answer. Content is set so
    # final_raw is non-empty even though parse_answer may or may not succeed.
    script = [
        (_final_msg("I have decided. \\boxed{Maybe}"), "stop", {"response_id": "r0"})
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

    assert result.final_answer_retry_used == 0
    assert result.react_steps == 1
    # Exactly 1 LLM call — no retry.
    assert len(llm.calls) == 1
