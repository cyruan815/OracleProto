"""Regression: `_run_task_with_retry` must hand `run_react` the *real* model
slug, not the virtual `{real}::r{R}::c{C}` slug. Upstream LLM providers don't
recognise the virtual suffix and reply with `model_not_found`.

The dispatcher uses the virtual slug for .db routing / writers / progress
logging; only the LLM API call needs the unwrapped name.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from forecast_eval import db as dbmod
from forecast_eval import runner as runner_mod
from forecast_eval.config import Settings
from forecast_eval.runner import Task, _run_task_with_retry
from forecast_eval.types import Question, SampleResult


def _question() -> Question:
    return Question(
        id="q1",
        choice_type="single",
        question_type="yes_no",
        event="ev",
        options=json.dumps(["Yes", "No"]),
        answer="A",
        end_time="2026-05-01",
    )


def _settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setenv("MODELS", "qwen3.5-plus-2026-02-15")
    monkeypatch.setenv("MODEL_TRAINING_CUTOFFS", "")
    monkeypatch.setenv("TAVILY_MAX_RESULTS", "5,10")
    monkeypatch.setenv("REACT_MAX_SEARCH_CALLS", "1,3")
    monkeypatch.setenv("REACT_MIN_SEARCH_CALLS", "0")
    monkeypatch.setenv("SAMPLING_N", "1")
    return Settings(_env_file=None)


def _stub_sample_result(model: str) -> SampleResult:
    return SampleResult(
        run_id="run",
        question_id="q1",
        model=model,
        sample_idx=0,
        final_answer_letters="A",
        final_answer_raw="A",
        correct=1,
        parse_ok=1,
        tool_calls_count=0,
        react_steps=1,
        prompt_tokens=10,
        completion_tokens=2,
        reasoning_tokens=0,
        latency_ms=1,
        user_prompt="RENDERED",
        messages_trace=None,
        search_calls=None,
        error=None,
        created_at=dbmod.utcnow_iso(),
        finish_reason="stop",
        nudges_used=0,
        step_metrics=None,
        response_id=None,
        system_fingerprint=None,
        service_tier=None,
        belief_final=None,
        belief_trace=None,
        belief_parse_ok=0,
    )


def test_virtual_slug_is_unwrapped_to_real_model_for_run_react(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    real = "qwen3.5-plus-2026-02-15"
    virtual = dbmod.compose_virtual_slug(real, 5, 3)

    captured: dict[str, Any] = {}

    async def fake_run_react(q: Question, **kwargs: Any) -> SampleResult:
        captured["model"] = kwargs["model"]
        return _stub_sample_result(kwargs["model"])

    monkeypatch.setattr(runner_mod, "run_react", fake_run_react)

    task = Task(question=_question(), model=virtual, sample_idx=0, settings=settings)
    sem = asyncio.Semaphore(1)
    row = asyncio.run(
        _run_task_with_retry(
            task,
            _global_settings=settings,
            templates={},
            run_id="run",
            llm_semaphore=sem,
            search_semaphore=sem,
        )
    )

    # The LLM provider must receive the real slug, never the virtual one.
    assert captured["model"] == real
    assert "::r" not in captured["model"]
    assert row["error"] is None


def test_non_virtual_slug_passes_through_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Legacy single-cell callers don't use the virtual slug. They must keep
    # working byte-identically — model goes straight to run_react.
    settings = _settings(monkeypatch)
    plain = "qwen3.5-plus-2026-02-15"
    captured: dict[str, Any] = {}

    async def fake_run_react(q: Question, **kwargs: Any) -> SampleResult:
        captured["model"] = kwargs["model"]
        return _stub_sample_result(kwargs["model"])

    monkeypatch.setattr(runner_mod, "run_react", fake_run_react)

    task = Task(question=_question(), model=plain, sample_idx=0, settings=settings)
    sem = asyncio.Semaphore(1)
    asyncio.run(
        _run_task_with_retry(
            task,
            _global_settings=settings,
            templates={},
            run_id="run",
            llm_semaphore=sem,
            search_semaphore=sem,
        )
    )

    assert captured["model"] == plain
