from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger
from openai import AsyncOpenAI

from .config import Settings
from .errors import (
    AuthError,
    ErrorKind,
    backoff_seconds,
    classify,
    parse_retry_after,
    should_retry,
)
from .tools import WEB_SEARCH_SCHEMA


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass
class ChatResponse:
    """Minimal wrapper around an OpenAI-compatible chat completion response.

    Callers read `.message` (the serialisable assistant message dict), `.usage`
    (tokens, with reasoning defaulting to 0 when unsupported) and
    `.response_headers` (raw headers, needed for Retry-After parsing).
    """

    message: dict[str, Any]
    usage: Usage
    response_headers: httpx.Headers = field(default_factory=httpx.Headers)
    raw: Any = None


_client: AsyncOpenAI | None = None


def get_client(settings: Settings) -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_BASE_URL,
        )
    return _client


def _is_reasoning_model(model: str, patterns: list[str]) -> bool:
    """推理模型 slug 子串匹配. 匹配到的模型调用时会跳过 temperature / top_p."""
    lower = model.lower()
    return any(p.lower() in lower for p in patterns if p)


def _assert_no_browsing(*, model: str, tools: list[dict[str, Any]], extra_body: dict[str, Any] | None) -> None:
    """Hard gate: refuse to send a request that would leak information via
    provider-native browsing. These checks duplicate the startup-time Settings
    validation, but we still assert at send-time so a test fixture or partial
    config drift cannot bypass the barrier.
    """
    if model.endswith(":online"):
        raise ValueError(
            f"model {model!r} ends with ':online' — provider-native browsing is not allowed"
        )
    # Empty tools is allowed (ENABLE_WEB_SEARCH=false) — the LLM gets no tool
    # schema at all, which is strictly stricter than the single-tool baseline.
    if tools:
        if len(tools) != 1:
            raise ValueError("tools must contain at most one schema: web_search")
        schema = tools[0]
        if schema is not WEB_SEARCH_SCHEMA:
            # allow callers to pass the same dict by identity; otherwise check shape
            func = schema.get("function", {}) if isinstance(schema, dict) else {}
            if func.get("name") != "web_search":
                raise ValueError(
                    "the only allowed tool schema is web_search; provider-native retrieval is forbidden"
                )
    if extra_body and "plugins" in extra_body:
        raise ValueError("plugins field is forbidden (provider-native browsing)")


def _serialise_message(msg: Any) -> dict[str, Any]:
    if hasattr(msg, "model_dump"):
        return msg.model_dump(exclude_unset=True)
    if isinstance(msg, dict):
        return msg
    raise TypeError(f"cannot serialise message of type {type(msg)!r}")


def _extract_usage(raw_usage: Any) -> Usage:
    if raw_usage is None:
        return Usage()
    if hasattr(raw_usage, "model_dump"):
        d = raw_usage.model_dump()
    elif isinstance(raw_usage, dict):
        d = raw_usage
    else:
        return Usage()
    prompt = int(d.get("prompt_tokens") or 0)
    completion = int(d.get("completion_tokens") or 0)
    reasoning = 0
    ctd = d.get("completion_tokens_details") or {}
    if isinstance(ctd, dict):
        reasoning = int(ctd.get("reasoning_tokens") or 0)
    if not reasoning:
        reasoning = int(d.get("reasoning_tokens") or 0)
    return Usage(prompt_tokens=prompt, completion_tokens=completion, reasoning_tokens=reasoning)


async def chat(
    *,
    model: str,
    messages: list[dict[str, Any]],
    settings: Settings,
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    client: AsyncOpenAI | None = None,
) -> ChatResponse:
    """Send one request to OpenRouter with the web_search tool attached.

    Handles layered retry by ErrorKind: NETWORK / RATE_LIMIT / SERVER_5XX get
    backed off, AUTH raises AuthError immediately, BAD_REQUEST / CONTENT_POLICY
    propagate so the runner can record the right `error` kind and move on.
    """
    tools = tools if tools is not None else [WEB_SEARCH_SCHEMA]
    extra_body: dict[str, Any] = {}
    _assert_no_browsing(model=model, tools=tools, extra_body=extra_body)

    c = client or get_client(settings)
    attempt = 0
    last_exc: BaseException | None = None

    # 推理模型不接受自定义 temperature / top_p (o-series / deepseek-r1 / qwq 等会返回 400),
    # 命中 pattern 时显式不传这两个字段.
    reasoning_patterns = getattr(settings, "LLM_REASONING_MODEL_PATTERNS", []) or []
    skip_sampling = _is_reasoning_model(model, reasoning_patterns)
    base_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS,
        "timeout": timeout if timeout is not None else settings.LLM_TIMEOUT_S,
    }
    # Omit the `tools` key entirely when empty — some OpenAI-compatible providers
    # reject an empty list with 400, and "no tools" is exactly what we want when
    # ENABLE_WEB_SEARCH=false.
    if tools:
        base_kwargs["tools"] = tools
    if not skip_sampling:
        base_kwargs["temperature"] = (
            temperature if temperature is not None else settings.LLM_TEMPERATURE
        )
        base_kwargs["top_p"] = top_p if top_p is not None else settings.LLM_TOP_P

    while True:
        attempt += 1
        try:
            raw = await c.chat.completions.with_raw_response.create(**base_kwargs)
        except BaseException as exc:  # noqa: BLE001 — we re-raise after classification
            kind = classify(exc)
            if kind is ErrorKind.AUTH:
                raise AuthError(str(exc)) from exc
            last_exc = exc
            if not should_retry(kind):
                raise
            retry_after = parse_retry_after(getattr(getattr(exc, "response", None), "headers", None))
            wait = backoff_seconds(kind, attempt, settings, retry_after=retry_after)
            if wait is None:
                raise
            logger.warning(
                "llm.chat retry model={} attempt={} kind={} wait={}s err={}",
                model,
                attempt,
                kind,
                wait,
                exc,
            )
            await asyncio.sleep(wait)
            continue

        parsed = raw.parse()
        headers = getattr(raw, "headers", httpx.Headers())
        message = _serialise_message(parsed.choices[0].message)
        usage = _extract_usage(getattr(parsed, "usage", None))
        return ChatResponse(message=message, usage=usage, response_headers=headers, raw=parsed)
