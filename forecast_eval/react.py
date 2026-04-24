from __future__ import annotations

import asyncio
import json
import time
from datetime import date, timedelta
from typing import Any

import httpx
from loguru import logger

from .config import Settings
from .db import utcnow_iso
from .llm import ChatResponse, chat as llm_chat
from .parser import is_correct, parse_answer, parse_gt
from .prompts import render_user_prompt
from .search import SearchResult, tavily_search
from .tools import (
    WEB_SEARCH_SCHEMA,
    extract_query,
    parse_tool_arguments,
    tool_error_message,
    tool_result_message,
)
from .types import Question, SampleResult


def _compute_end_date(end_time: str, offset_days: int) -> str:
    return (date.fromisoformat(end_time) + timedelta(days=offset_days)).isoformat()


def _accumulate_tokens(totals: dict[str, int], resp: ChatResponse) -> None:
    totals["prompt"] += resp.usage.prompt_tokens
    totals["completion"] += resp.usage.completion_tokens
    totals["reasoning"] += resp.usage.reasoning_tokens


def _record_search_call(
    search_calls: list[dict[str, Any]],
    *,
    query: str,
    end_date: str,
    result: SearchResult | None,
) -> None:
    if result is None or not result.ok:
        search_calls.append(
            {
                "query": query,
                "end_date": end_date,
                "n_results": 0,
                "published_dates": [],
                "error_kind": result.error_kind if result else "pre_call_error",
            }
        )
        return
    search_calls.append(
        {
            "query": query,
            "end_date": end_date,
            "n_results": len(result.results),
            "published_dates": [r.published_date for r in result.results],
        }
    )


async def run_react(
    q: Question,
    *,
    model: str,
    sample_idx: int,
    settings: Settings,
    templates: dict[str, str],
    run_id: str,
    search_semaphore: asyncio.Semaphore | None = None,
    httpx_client: httpx.AsyncClient | None = None,
) -> SampleResult:
    """One sample of the ReAct loop.

    The LLM receives a single `user` message with no system role. The only tool
    it sees is `web_search(query)` — `end_date` is computed from `q.end_time`
    by the tool layer and never exposed.

    This function NEVER writes `error` on success — the runner is responsible
    for wrapping it and recording retry-exhausted / content-policy errors.
    """
    end_date = _compute_end_date(q.end_time, settings.TAVILY_END_DATE_OFFSET_DAYS)
    user_prompt = render_user_prompt(q, templates)
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]

    search_calls: list[dict[str, Any]] = []
    tokens = {"prompt": 0, "completion": 0, "reasoning": 0}
    t0 = time.monotonic()
    final_raw = ""
    steps_executed = 0

    for step in range(settings.REACT_MAX_STEPS):
        steps_executed = step + 1
        resp = await llm_chat(
            model=model,
            messages=messages,
            settings=settings,
            tools=[WEB_SEARCH_SCHEMA],
        )
        _accumulate_tokens(tokens, resp)
        assistant_msg = resp.message
        messages.append(assistant_msg)

        tool_calls = assistant_msg.get("tool_calls") or []
        if not tool_calls:
            final_raw = assistant_msg.get("content") or ""
            break

        for tc in tool_calls:
            tc_id = tc.get("id") or ""
            fn = tc.get("function") or {}
            fn_name = fn.get("name")
            raw_args = fn.get("arguments")

            if fn_name != "web_search":
                messages.append(tool_error_message(tc_id, f"unknown tool: {fn_name}"))
                continue

            if len(search_calls) >= settings.REACT_MAX_SEARCH_CALLS:
                messages.append(tool_error_message(tc_id, "search budget exceeded"))
                continue

            args, err = parse_tool_arguments(raw_args if isinstance(raw_args, str) else json.dumps(raw_args or {}))
            if err is not None or args is None:
                messages.append(tool_error_message(tc_id, err or "invalid arguments"))
                continue

            query, qerr = extract_query(args)
            if qerr is not None or query is None:
                messages.append(tool_error_message(tc_id, qerr or "missing query"))
                continue

            if search_semaphore is not None:
                await search_semaphore.acquire()
            try:
                result = await tavily_search(query, end_date, settings, client=httpx_client)
            finally:
                if search_semaphore is not None:
                    search_semaphore.release()

            _record_search_call(search_calls, query=query, end_date=end_date, result=result)
            messages.append(tool_result_message(tc_id, result.to_llm_payload()))

    # Loop exited either by break or by exhausting REACT_MAX_STEPS.
    parsed = parse_answer(final_raw, q)
    try:
        gt = parse_gt(q.answer)
    except ValueError:
        logger.error("question {} has invalid answer field: {!r}", q.id, q.answer)
        gt = frozenset()
    correct = is_correct(parsed, gt) if parsed is not None else None

    return SampleResult(
        run_id=run_id,
        question_id=q.id,
        model=model,
        sample_idx=sample_idx,
        final_answer_letters=json.dumps(sorted(parsed)) if parsed is not None else None,
        final_answer_raw=final_raw,
        correct=int(correct) if isinstance(correct, bool) else None,
        parse_ok=1 if parsed is not None else 0,
        tool_calls_count=len(search_calls),
        react_steps=steps_executed,
        prompt_tokens=tokens["prompt"],
        completion_tokens=tokens["completion"],
        reasoning_tokens=tokens["reasoning"],
        latency_ms=int((time.monotonic() - t0) * 1000),
        user_prompt=user_prompt,
        messages_trace=json.dumps(messages, ensure_ascii=False) if settings.WRITE_MESSAGES_TRACE else None,
        search_calls=json.dumps(search_calls, ensure_ascii=False),
        error=None,
        created_at=utcnow_iso(),
    )
