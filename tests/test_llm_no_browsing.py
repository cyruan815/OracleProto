from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
import respx
from openai import AsyncOpenAI

from forecast_eval.llm import AuthError, chat, get_client
from forecast_eval.tools import WEB_SEARCH_SCHEMA


@dataclass
class _StubSettings:
    OPENROUTER_API_KEY: str = "sk-or-v1-TEST"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_TEMPERATURE: float = 0.7
    LLM_TOP_P: float = 1.0
    LLM_MAX_TOKENS: int = 128
    LLM_TIMEOUT_S: int = 30
    LLM_BACKOFF_NETWORK_S: list[int] = None  # type: ignore[assignment]
    LLM_BACKOFF_RATE_LIMIT_S: list[int] = None  # type: ignore[assignment]
    LLM_BACKOFF_SERVER_5XX_S: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.LLM_BACKOFF_NETWORK_S is None:
            self.LLM_BACKOFF_NETWORK_S = [0, 0, 0]
        if self.LLM_BACKOFF_RATE_LIMIT_S is None:
            self.LLM_BACKOFF_RATE_LIMIT_S = [0, 0, 0]
        if self.LLM_BACKOFF_SERVER_5XX_S is None:
            self.LLM_BACKOFF_SERVER_5XX_S = [0, 0, 0]


def _success_body() -> dict[str, object]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "openai/gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "\\boxed{Yes}"},
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4},
    }


def _new_client(settings: _StubSettings) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
    )


@respx.mock
async def test_outbound_request_has_no_browsing_knobs() -> None:
    route = respx.post(re.compile(r"https://openrouter\.ai/api/v1/chat/completions")).mock(
        return_value=httpx.Response(200, json=_success_body())
    )
    settings = _StubSettings()
    client = _new_client(settings)

    resp = await chat(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        settings=settings,
        client=client,
    )

    assert route.called
    body = json.loads(route.calls.last.request.content.decode("utf-8"))

    # (1) no provider-native browsing knobs
    assert "plugins" not in body
    # (2) tools entry is only our web_search schema
    assert isinstance(body["tools"], list) and len(body["tools"]) == 1
    assert body["tools"][0]["function"]["name"] == "web_search"
    # (3) model slug is untouched — no ':online' suffix appended
    assert body["model"] == "openai/gpt-4o-mini"
    assert not body["model"].endswith(":online")

    # Response shape
    assert resp.message["role"] == "assistant"
    assert resp.message["content"] == "\\boxed{Yes}"
    assert resp.usage.prompt_tokens == 12
    assert resp.usage.completion_tokens == 4
    assert resp.usage.reasoning_tokens == 0


async def test_online_suffix_rejected_pre_flight() -> None:
    settings = _StubSettings()
    with pytest.raises(ValueError, match=":online"):
        await chat(
            model="openai/gpt-4o-mini:online",
            messages=[{"role": "user", "content": "x"}],
            settings=settings,
            client=_new_client(settings),
        )


async def test_extra_tool_rejected_pre_flight() -> None:
    settings = _StubSettings()
    # Second tool is not allowed.
    second_tool = {
        "type": "function",
        "function": {"name": "python", "parameters": {"type": "object", "properties": {}}},
    }
    with pytest.raises(ValueError, match="exactly one"):
        await chat(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "x"}],
            settings=settings,
            tools=[WEB_SEARCH_SCHEMA, second_tool],
            client=_new_client(settings),
        )


@respx.mock
async def test_auth_error_raises_auth_error_not_retried() -> None:
    route = respx.post(re.compile(r"https://openrouter\.ai/api/v1/chat/completions")).mock(
        return_value=httpx.Response(401, json={"error": {"message": "bad key"}})
    )
    settings = _StubSettings()
    with pytest.raises(AuthError):
        await chat(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "x"}],
            settings=settings,
            client=_new_client(settings),
        )
    assert route.call_count == 1
