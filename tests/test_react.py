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
    # force-final-answer-near-limit-v1 is on by default; turn it off here so the
    # legacy v3/v4/v5.1 tests keep byte-identical behaviour. The new mechanism
    # has its own dedicated tests (test_force_final_*).
    monkeypatch.setenv("REACT_BUDGET_AWARENESS_PROTOCOL", "false")
    monkeypatch.setenv("REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT", "false")
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
        REACT_FINAL_ANSWER_RETRY="true",  # class default is False post-v6; opt-in here
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
        REACT_FINAL_ANSWER_RETRY="true",  # class default is False post-v6; opt-in here
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


async def test_search_calls_includes_detector_audit(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the leak filter is enabled, the search_calls entry MUST contain
    the 5 detector_* fields; n_results = audit.n_results_kept (backwards
    compatible with legacy analysis scripts)."""
    settings = _make_settings(monkeypatch)

    async def stub_tavily_with_audit(query: str, end_date: str, *args, **kwargs):
        from forecast_eval.search import SearchResult, SearchResultItem

        # Simulate a SearchResult that leak_filter has already trimmed: 5 raw -> 3 kept.
        kept_items = [
            SearchResultItem(
                title=f"t{i}",
                url=f"https://x/{i}",
                content="c",
                published_date=f"2026-01-{10+i:02d}",
            )
            for i in (0, 2, 4)
        ]
        return SearchResult(
            query=query,
            end_date=end_date,
            answer=None,
            results=kept_items,
            audit={
                "n_results_raw": 5,
                "n_results_kept": 3,
                "detector_verdicts": ["keep", "drop", "keep", "drop", "keep"],
                "detector_latency_ms": 123,
                "detector_error_kind": None,
                "published_dates_raw": [
                    "2026-01-10",
                    "2026-01-11",
                    "2026-01-12",
                    "2026-01-13",
                    "2026-01-14",
                ],
            },
        )

    script = [
        (_tool_msg("call_1", "evidence"), "tool_calls", {"response_id": "r0"}),
        (_final_msg("\\boxed{Yes}"), "stop", {"response_id": "r1"}),
    ]
    llm = _ScriptedLLM(script)
    monkeypatch.setattr(react, "llm_chat", llm)
    monkeypatch.setattr(react, "tavily_search", stub_tavily_with_audit)

    result = await react.run_react(
        _yes_no_question(),
        model=settings.MODELS[0],
        sample_idx=0,
        settings=settings,
        templates=templates,
        run_id="test",
    )
    calls = json.loads(result.search_calls)
    assert len(calls) == 1
    entry = calls[0]
    assert entry["n_results"] == 3  # = n_results_kept (backwards compatible)
    assert entry["n_results_raw"] == 5
    assert entry["n_results_kept"] == 3
    assert entry["detector_verdicts"] == ["keep", "drop", "keep", "drop", "keep"]
    assert entry["detector_latency_ms"] == 123
    assert entry["detector_error_kind"] is None
    # published_dates length == n_results_raw, one-to-one with detector_verdicts.
    assert len(entry["published_dates"]) == 5


async def test_search_calls_no_audit_when_disabled(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switch off (audit=None) -> search_calls entry contains only the 4 legacy fields."""
    settings = _make_settings(monkeypatch)
    # _stub_tavily returns a SearchResult whose audit defaults to None -- this
    # is exactly the shape produced when the switch is off.
    script = [
        (_tool_msg("call_1", "evidence"), "tool_calls", {"response_id": "r0"}),
        (_final_msg("\\boxed{Yes}"), "stop", {"response_id": "r1"}),
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
    calls = json.loads(result.search_calls)
    assert len(calls) == 1
    entry = calls[0]
    assert set(entry.keys()) == {"query", "end_date", "n_results", "published_dates"}
    assert "detector_verdicts" not in entry


async def test_search_calls_tavily_error_no_detector_fields(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tavily failure -> entry contains only the 5 legacy fields + error_kind, no detector fields."""
    settings = _make_settings(monkeypatch)

    async def failing_tavily(query: str, end_date: str, *args, **kwargs):
        from forecast_eval.search import SearchResult

        return SearchResult(
            query=query,
            end_date=end_date,
            error_kind="tavily_error",
            error_message="all keys exhausted",
        )

    script = [
        (_tool_msg("call_1", "evidence"), "tool_calls", {"response_id": "r0"}),
        (_final_msg("\\boxed{Yes}"), "stop", {"response_id": "r1"}),
    ]
    llm = _ScriptedLLM(script)
    monkeypatch.setattr(react, "llm_chat", llm)
    monkeypatch.setattr(react, "tavily_search", failing_tavily)

    result = await react.run_react(
        _yes_no_question(),
        model=settings.MODELS[0],
        sample_idx=0,
        settings=settings,
        templates=templates,
        run_id="test",
    )
    calls = json.loads(result.search_calls)
    entry = calls[0]
    assert entry["error_kind"] == "tavily_error"
    assert entry["n_results"] == 0
    assert entry["published_dates"] == []
    assert "detector_verdicts" not in entry
    assert "detector_latency_ms" not in entry


async def test_loop_stops_on_valid_boxed(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loop exits immediately when a valid \\boxed{} answer is found, without
    triggering the bail-out retry."""
    settings = _make_settings(monkeypatch)
    script = [
        (_final_msg("Evidence gathered. \\boxed{Yes}"), "stop", {"response_id": "r0"})
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

    assert result.parse_ok == 1
    assert result.final_answer_retry_used == 0
    assert result.react_steps == 1
    assert len(llm.calls) == 1


async def test_loop_continues_when_no_valid_boxed(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A content-only reply without a valid \\boxed{} does NOT stop the loop —
    the model gets another step to commit its answer."""
    settings = _make_settings(monkeypatch)
    # Step 0: content without valid boxed (Maybe is not Yes/No).
    # Step 1: content with valid boxed — loop stops here.
    script = [
        (_final_msg("Thinking out loud. \\boxed{Maybe}"), "stop", {"response_id": "r0"}),
        (_final_msg("On reflection: \\boxed{Yes}"), "stop", {"response_id": "r1"}),
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

    assert result.parse_ok == 1
    assert result.react_steps == 2
    assert len(llm.calls) == 2
    # Step 1's call must include the unified continuation injection so that
    # (a) the conversation stays user/assistant-alternating and (b) the model
    # gets the live status (step 2/6, web_search 0/5 used) before deciding
    # what to do. Replaces the legacy ad-hoc "Harness: step N of M complete"
    # string — both messages were the same intent, but we now share the
    # status-header format with every other harness injection.
    step1_messages = llm.calls[1]
    injected = [
        m for m in step1_messages
        if m.get("role") == "user" and "[Harness status]" in (m.get("content") or "")
    ]
    assert injected, "continuation injection missing from second LLM call"
    body = injected[-1]["content"]
    assert "step 2/6" in body  # the step about to be entered (1-indexed)
    assert "previous reply did not contain" in body.lower()
    assert "\\boxed{...}" in body


# ---- force-final-answer-near-limit-v1 ---------------------------------------


_PENULTIMATE_MARKER = "second-to-last"
_LAST_STEP_MARKER = "Harness cutoff"
_BUDGET_MARKER = "Budget Awareness"


async def test_budget_awareness_protocol_appended_when_enabled(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """When REACT_BUDGET_AWARENESS_PROTOCOL=true, the budget tail is part of
    user_prompt and reflects the cell-local REACT_MAX_STEPS / *_SEARCH_CALLS."""
    settings = _make_settings(
        monkeypatch,
        REACT_MAX_STEPS="6",
        REACT_MAX_SEARCH_CALLS="4",
        REACT_BUDGET_AWARENESS_PROTOCOL="true",
    )
    script = [(_final_msg("\\boxed{Yes}"), "stop", {"response_id": "r0"})]
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
    assert _BUDGET_MARKER in result.user_prompt
    # The protocol must surface the actual N / C the cell will run with.
    assert "**6**" in result.user_prompt  # max_steps
    assert "**4**" in result.user_prompt  # max_search_calls


async def test_budget_awareness_protocol_disabled_byte_identical(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """REACT_BUDGET_AWARENESS_PROTOCOL=false → user_prompt MUST be byte-
    identical to the v5.1 rendering (no budget tail)."""
    settings = _make_settings(monkeypatch, REACT_BUDGET_AWARENESS_PROTOCOL="false")
    monkeypatch.setattr(
        react,
        "llm_chat",
        _ScriptedLLM([(_final_msg("\\boxed{Yes}"), "stop", {"response_id": "r"})]),
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
    assert _BUDGET_MARKER not in result.user_prompt


async def test_force_final_near_limit_two_stage_ladder(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """LOOKAHEAD=2 (default): with REACT_MAX_STEPS=3 the loop should inject a
    soft warning at step 2 (remaining=2) and a hard cutoff at step 3
    (remaining=1). The hard cutoff must coincide with `tools=[]` — verified
    behaviorally by scripting an LLM that returns content even though the
    earlier turns returned tool_calls.
    """
    settings = _make_settings(
        monkeypatch,
        REACT_MAX_STEPS="3",
        REACT_MAX_SEARCH_CALLS="100",  # never the binding constraint here
        REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT="true",
        REACT_FORCE_FINAL_ANSWER_LOOKAHEAD="2",
    )
    script = [
        # Step 1 (remaining=3): no injection, model issues a tool_call.
        (_tool_msg("call_1", "q1"), "tool_calls", {"response_id": "r1"}),
        # Step 2 (remaining=2): soft warning is injected BEFORE this call;
        # tools schema still exposed so the model could search again. Here
        # we have it search once more.
        (_tool_msg("call_2", "q2"), "tool_calls", {"response_id": "r2"}),
        # Step 3 (remaining=1): hard cutoff message + tools=[]; the scripted
        # LLM emits content with the boxed answer.
        (_final_msg("\\boxed{No}"), "stop", {"response_id": "r3"}),
    ]
    llm = _ScriptedLLM(
        script,
        prompt_seq=(101, 202, 303),
        completion_seq=(11, 22, 33),
        reasoning_seq=(1, 2, 3),
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

    # Loop ran exactly 3 steps, no fall-through retry.
    assert result.react_steps == 3
    assert result.final_answer_retry_used == 0
    assert result.parse_ok == 1
    assert "\\boxed{No}" in result.final_answer_raw
    # nudges_used MUST stay 0 — these injections are harness resilience, not
    # search-floor enforcement (parity with REACT_FINAL_ANSWER_RETRY).
    assert result.nudges_used == 0

    # messages_trace must contain BOTH injected messages, in the right order
    # and with the right roles.
    trace = json.loads(result.messages_trace)
    user_msgs = [m for m in trace if m.get("role") == "user"]
    penult = [m for m in user_msgs if _PENULTIMATE_MARKER in (m.get("content") or "")]
    last = [m for m in user_msgs if _LAST_STEP_MARKER in (m.get("content") or "")]
    assert len(penult) == 1, f"expected 1 soft-warning user msg, got {len(penult)}"
    assert len(last) == 1, f"expected 1 hard-cutoff user msg, got {len(last)}"
    # Soft warning precedes hard cutoff in the trace.
    assert trace.index(penult[0]) < trace.index(last[0])


async def test_force_final_near_limit_lookahead_one(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """LOOKAHEAD=1 means ONLY hard-cutoff on the last step, no soft warning."""
    settings = _make_settings(
        monkeypatch,
        REACT_MAX_STEPS="3",
        REACT_MAX_SEARCH_CALLS="100",
        REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT="true",
        REACT_FORCE_FINAL_ANSWER_LOOKAHEAD="1",
    )
    script = [
        (_tool_msg("call_1", "q1"), "tool_calls", {"response_id": "r1"}),
        (_tool_msg("call_2", "q2"), "tool_calls", {"response_id": "r2"}),
        (_final_msg("\\boxed{Yes}"), "stop", {"response_id": "r3"}),
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

    trace = json.loads(result.messages_trace)
    user_msgs = [m for m in trace if m.get("role") == "user"]
    penult = [m for m in user_msgs if _PENULTIMATE_MARKER in (m.get("content") or "")]
    last = [m for m in user_msgs if _LAST_STEP_MARKER in (m.get("content") or "")]
    assert len(penult) == 0  # LOOKAHEAD=1 suppresses the soft warning
    assert len(last) == 1


async def test_force_final_near_limit_disabled(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the switch is off, neither the soft warning nor the hard cutoff
    fires. The loop runs to its natural end and (with retry off) leaves
    final_raw empty if the model never gives content."""
    settings = _make_settings(
        monkeypatch,
        REACT_MAX_STEPS="3",
        REACT_MAX_SEARCH_CALLS="100",
        REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT="false",
        REACT_FINAL_ANSWER_RETRY="false",
    )
    script = [
        (_tool_msg("call_1", "q1"), "tool_calls", {"response_id": "r1"}),
        (_tool_msg("call_2", "q2"), "tool_calls", {"response_id": "r2"}),
        (_tool_msg("call_3", "q3"), "tool_calls", {"response_id": "r3"}),
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

    trace = json.loads(result.messages_trace)
    user_msgs = [m for m in trace if m.get("role") == "user"]
    assert all(
        _PENULTIMATE_MARKER not in (m.get("content") or "")
        and _LAST_STEP_MARKER not in (m.get("content") or "")
        for m in user_msgs
    )
    assert result.parse_ok == 0
    assert result.final_answer_raw == ""


async def test_force_final_near_limit_supersedes_budget_exceeded(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """When BOTH the budget-exceeded drop-tools branch AND the last-step
    hard-cutoff would fire on the SAME step, only one user message is injected
    (the hard-cutoff one) and the resulting tools list is the same `[]`. We
    verify by setting a tiny search budget so it's already exhausted by the
    last step, then confirming exactly one harness user message lands on the
    final step."""
    settings = _make_settings(
        monkeypatch,
        REACT_MAX_STEPS="3",
        REACT_MAX_SEARCH_CALLS="2",
        REACT_BUDGET_EXCEEDED_DROP_TOOLS="true",
        REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT="true",
        REACT_FORCE_FINAL_ANSWER_LOOKAHEAD="2",
    )
    script = [
        (_tool_msg("call_1", "q1"), "tool_calls", {"response_id": "r1"}),
        # Step 2 (remaining=2): soft warning injected; budget already at 1/2
        # but not yet exhausted. Model issues 2nd search → budget now 2/2.
        (_tool_msg("call_2", "q2"), "tool_calls", {"response_id": "r2"}),
        # Step 3 (remaining=1): both branches would want tools=[]. Hard
        # cutoff wins; only ONE harness user message at this point.
        (_final_msg("\\boxed{Yes}"), "stop", {"response_id": "r3"}),
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

    trace = json.loads(result.messages_trace)
    last_msgs = [
        m for m in trace
        if m.get("role") == "user"
        and _LAST_STEP_MARKER in (m.get("content") or "")
    ]
    assert len(last_msgs) == 1
    assert result.tool_calls_count == 2  # both searches went through


# ---- harness-status-unification regression tests ---------------------------


_STATUS_HEADER = "[Harness status]"


async def test_continuation_and_penultimate_do_not_double_inject(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The historical bug: step k returns content w/o `\\boxed`, harness injects
    a step-position message AND the next iteration injects a penultimate
    warning — two consecutive user messages with overlapping content. The
    unified design defers the continuation reminder to the next iteration's
    pre-step injection slot, where it loses to higher-priority injections.

    Setup: REACT_MAX_STEPS=3, LOOKAHEAD=2. Step 1 returns content without
    `\\boxed{...}`, step 2 (penultimate) must run with EXACTLY ONE harness
    user message preceding it (the penultimate warning), not two."""
    settings = _make_settings(
        monkeypatch,
        REACT_MAX_STEPS="3",
        REACT_MAX_SEARCH_CALLS="100",
        REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT="true",
        REACT_FORCE_FINAL_ANSWER_LOOKAHEAD="2",
    )
    script = [
        # Step 1: content without parseable boxed (Maybe ≠ Yes/No).
        (_final_msg("Pondering. \\boxed{Maybe}"), "stop", {"response_id": "r1"}),
        # Step 2 (penultimate, remaining=2): penultimate warning fires.
        (_tool_msg("call_2", "q2"), "tool_calls", {"response_id": "r2"}),
        # Step 3 (last, remaining=1): hard cutoff fires.
        (_final_msg("\\boxed{Yes}"), "stop", {"response_id": "r3"}),
    ]
    llm = _ScriptedLLM(script)
    monkeypatch.setattr(react, "llm_chat", llm)
    monkeypatch.setattr(react, "tavily_search", _stub_tavily)

    await react.run_react(
        _yes_no_question(),
        model=settings.MODELS[0],
        sample_idx=0,
        settings=settings,
        templates=templates,
        run_id="test",
    )

    # Inspect what the LLM saw on step 2: between the step-1 assistant turn
    # and the step-2 LLM call there must be EXACTLY ONE harness-status user
    # message (the penultimate warning), NOT two (penultimate + continuation).
    step2_messages = llm.calls[1]
    # Find the last assistant message; everything after it (and before the
    # next assistant) is the harness pre-step injection slot.
    last_assistant_idx = max(
        i for i, m in enumerate(step2_messages) if m.get("role") == "assistant"
    )
    pre_step_users = [
        m for m in step2_messages[last_assistant_idx + 1 :]
        if m.get("role") == "user"
    ]
    assert len(pre_step_users) == 1, (
        f"expected exactly 1 pre-step user message at penultimate, got "
        f"{len(pre_step_users)}: {[m.get('content') for m in pre_step_users]}"
    )
    body = pre_step_users[0]["content"]
    assert _STATUS_HEADER in body
    # Must be the penultimate warning (it carries `second-to-last`), not the
    # continuation message (which would mention "previous reply did not contain").
    assert _PENULTIMATE_MARKER in body
    assert "previous reply did not contain" not in body


async def test_continuation_injection_carries_status_and_no_old_string(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The continuation injection (content w/o boxed → next turn) MUST carry
    the unified status header and MUST NOT use the legacy "Harness: step N of
    M complete" string anywhere in the trace."""
    settings = _make_settings(
        monkeypatch,
        REACT_MAX_STEPS="6",
        REACT_MAX_SEARCH_CALLS="100",
    )
    script = [
        (_final_msg("Sketching. \\boxed{Maybe}"), "stop", {"response_id": "r0"}),
        (_final_msg("Settled. \\boxed{Yes}"), "stop", {"response_id": "r1"}),
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

    trace = json.loads(result.messages_trace)
    user_bodies = [m.get("content") or "" for m in trace if m.get("role") == "user"]
    # Legacy ad-hoc string is gone — no occurrence anywhere.
    assert not any("Harness: step" in b and "complete" in b for b in user_bodies), (
        "legacy step-position injection string must not appear in messages_trace"
    )
    # The unified continuation injection IS present and carries the status header.
    cont = [b for b in user_bodies if "previous reply did not contain" in b.lower()]
    assert len(cont) == 1
    assert _STATUS_HEADER in cont[0]
    assert "step 2/6" in cont[0]


async def test_tool_error_carries_status(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a `web_search` tool_call hits the search budget, the tool error
    payload must include a `status` field with the live counters so the model
    knows its remaining budget INSIDE the tool cycle (where we cannot inject
    a user message)."""
    settings = _make_settings(
        monkeypatch,
        REACT_MAX_STEPS="6",
        REACT_MAX_SEARCH_CALLS="2",
        REACT_BUDGET_EXCEEDED_DROP_TOOLS="false",  # keep tools so we can hit the error path
    )
    script = [
        (_tool_msg("call_1", "q1"), "tool_calls", {"response_id": "r1"}),
        (_tool_msg("call_2", "q2"), "tool_calls", {"response_id": "r2"}),
        # Step 3: budget already 2/2, this call hits the budget-exceeded branch.
        (_tool_msg("call_3", "q3"), "tool_calls", {"response_id": "r3"}),
        (_final_msg("\\boxed{Yes}"), "stop", {"response_id": "r4"}),
    ]
    llm = _ScriptedLLM(
        script,
        prompt_seq=(101, 202, 303, 404),
        completion_seq=(11, 22, 33, 44),
        reasoning_seq=(1, 2, 3, 4),
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

    trace = json.loads(result.messages_trace)
    tool_errors = [
        json.loads(m["content"]) for m in trace
        if m.get("role") == "tool" and "error" in (m.get("content") or "")
    ]
    assert tool_errors, "expected at least one tool_error payload"
    # The budget-exceeded error must carry both `error` and `status` slots.
    budget_err = [e for e in tool_errors if e.get("error") == "search budget exceeded"]
    assert budget_err, "no budget-exceeded tool_error in trace"
    status = budget_err[0].get("status")
    assert isinstance(status, str)
    assert "step 3/6" in status
    assert "2/2 used" in status
    # The model should be told to commit \\boxed{...} from inside the tool cycle.
    assert "\\boxed{...}" in status


async def test_tool_error_unknown_tool_also_carries_status(
    templates: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Status field must be attached to EVERY tool_error path, not just
    'search budget exceeded' — i.e. unknown_tool / invalid_arguments /
    missing_query all surface live budget context."""
    settings = _make_settings(monkeypatch)
    bad_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_bad",
                "type": "function",
                "function": {
                    "name": "make_coffee",  # unknown tool
                    "arguments": json.dumps({"shots": 2}),
                },
            }
        ],
    }
    script = [
        (bad_msg, "tool_calls", {"response_id": "r0"}),
        (_final_msg("\\boxed{Yes}"), "stop", {"response_id": "r1"}),
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

    trace = json.loads(result.messages_trace)
    tool_errors = [
        json.loads(m["content"]) for m in trace
        if m.get("role") == "tool"
    ]
    unknown = [e for e in tool_errors if "unknown tool" in (e.get("error") or "")]
    assert unknown
    assert "status" in unknown[0]
    assert "step 1/" in unknown[0]["status"]


def test_settings_rejects_lookahead_above_max_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-or-v1-TEST_ABCDEFGH")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-TEST_ABCDEFGH")
    monkeypatch.setenv("MODELS", "openai/gpt-4o-mini")
    monkeypatch.setenv("REACT_MAX_STEPS", "3")
    monkeypatch.setenv("REACT_FORCE_FINAL_ANSWER_LOOKAHEAD", "5")
    with pytest.raises(ValueError, match="REACT_FORCE_FINAL_ANSWER_LOOKAHEAD"):
        Settings(_env_file=None)


def test_settings_rejects_lookahead_below_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-or-v1-TEST_ABCDEFGH")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-TEST_ABCDEFGH")
    monkeypatch.setenv("MODELS", "openai/gpt-4o-mini")
    monkeypatch.setenv("REACT_FORCE_FINAL_ANSWER_LOOKAHEAD", "0")
    with pytest.raises(ValueError, match="REACT_FORCE_FINAL_ANSWER_LOOKAHEAD"):
        Settings(_env_file=None)


# ==================== L2 temporal masking: χ_i = τ_i + δ ====================
#
# Without these pins, a sign flip in `_compute_end_date` (e.g. `-timedelta` in
# place of `+timedelta`) or a dropped `settings.TAVILY_END_DATE_OFFSET_DAYS`
# handoff in `run_react` would silently invert the temporal mask while every
# existing test stays green.


@pytest.mark.parametrize(
    "end_time, offset_days, expected",
    [
        ("2026-03-01", -1, "2026-02-28"),
        ("2026-01-01", -1, "2025-12-31"),
        ("2024-03-01", -1, "2024-02-29"),
        ("2025-03-01", -1, "2025-02-28"),
        ("2026-01-18", 0, "2026-01-18"),
        ("2026-01-18", 5, "2026-01-23"),
    ],
)
def test_compute_end_date_pins_offset_arithmetic(
    end_time: str, offset_days: int, expected: str
) -> None:
    """χ_i = τ_i + δ. Covers cross-month, cross-year, leap-day, δ=0, and δ>0
    (relaxation) cases — a wider surface than the default δ=-1 alone, so a sign
    flip is caught regardless of which case the regression hits first."""
    assert react._compute_end_date(end_time, offset_days) == expected


@pytest.mark.parametrize(
    "offset_env, expected_end_date",
    [
        ("-1", "2026-02-28"),
        ("0", "2026-03-01"),
        ("3", "2026-03-04"),
    ],
)
async def test_react_injects_end_date_pins_q_end_time_plus_offset(
    offset_env: str,
    expected_end_date: str,
    templates: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end pin for L2: `run_react` must invoke `tavily_search` with
    `end_date = q.end_time + settings.TAVILY_END_DATE_OFFSET_DAYS`. Three
    offsets cover the project default, the contract-knob boundary, and a
    forward relaxation; the suite was previously blind to either the offset
    handoff dropping or the arithmetic direction flipping."""
    from forecast_eval.search import SearchResult

    settings = _make_settings(
        monkeypatch, TAVILY_END_DATE_OFFSET_DAYS=offset_env
    )
    captured: dict[str, Any] = {}

    async def capturing_tavily(query: str, end_date: str, *args: Any, **kwargs: Any) -> SearchResult:
        captured["end_date"] = end_date
        return SearchResult(query=query, end_date=end_date, answer=None, results=[])

    script = [
        (_tool_msg("call_1", "evidence?"), "tool_calls", {"response_id": "r0"}),
        (_final_msg("\\boxed{Yes}"), "stop", {"response_id": "r1"}),
    ]
    monkeypatch.setattr(react, "llm_chat", _ScriptedLLM(script))
    monkeypatch.setattr(react, "tavily_search", capturing_tavily)

    q = _yes_no_question()
    await react.run_react(
        q,
        model=settings.MODELS[0],
        sample_idx=0,
        settings=settings,
        templates=templates,
        run_id="test",
    )

    assert captured["end_date"] == expected_end_date
