from __future__ import annotations

import json
from dataclasses import dataclass

import httpx
import pytest
import respx

from forecast_eval.search import SearchResult, TAVILY_ENDPOINT, tavily_search
from forecast_eval.tools import (
    WEB_SEARCH_SCHEMA,
    extract_query,
    parse_tool_arguments,
    tool_error_message,
    tool_result_message,
)


@dataclass
class _StubSettings:
    TAVILY_API_KEY: str = "tvly-TEST"
    TAVILY_MAX_RESULTS: int = 5
    TAVILY_INCLUDE_RAW_CONTENT: bool = False
    SEARCH_RETRY_MAX: int = 3
    SEARCH_BACKOFF_S: list[int] = None  # type: ignore[assignment]
    LLM_TIMEOUT_S: int = 30

    def __post_init__(self) -> None:
        if self.SEARCH_BACKOFF_S is None:
            self.SEARCH_BACKOFF_S = [0, 0, 0]  # tests run fast, no real backoff


def test_web_search_schema_whitelist() -> None:
    props = WEB_SEARCH_SCHEMA["function"]["parameters"]["properties"]
    assert set(props.keys()) == {"query"}
    assert WEB_SEARCH_SCHEMA["function"]["parameters"]["required"] == ["query"]
    # no end_date / date / max_results leaked into LLM-visible schema
    for forbidden in ("end_date", "date", "time", "max_results"):
        assert forbidden not in props


def test_parse_tool_arguments_happy_path() -> None:
    args, err = parse_tool_arguments('{"query": "foo"}')
    assert err is None and args == {"query": "foo"}

    args, err = parse_tool_arguments(None)
    assert err is None and args == {}


def test_parse_tool_arguments_invalid_json() -> None:
    args, err = parse_tool_arguments("{not json")
    assert args is None
    assert err is not None and "invalid arguments JSON" in err


def test_parse_tool_arguments_not_object() -> None:
    args, err = parse_tool_arguments("[1, 2]")
    assert args is None
    assert err is not None


def test_extract_query_strips_extra_fields() -> None:
    args = {"query": "who won?", "end_date": "2099-01-01"}
    q, err = extract_query(args)
    assert err is None
    assert q == "who won?"


def test_extract_query_missing() -> None:
    q, err = extract_query({})
    assert q is None and err is not None


def test_tool_error_and_result_messages() -> None:
    err = tool_error_message("call-1", "bad")
    assert err["role"] == "tool" and err["tool_call_id"] == "call-1"
    parsed = json.loads(err["content"])
    assert parsed == {"error": "bad"}

    ok = tool_result_message("call-2", {"answer": "yes", "results": []})
    parsed = json.loads(ok["content"])
    assert parsed == {"answer": "yes", "results": []}


@respx.mock
async def test_tavily_search_success_injects_end_date() -> None:
    route = respx.post(TAVILY_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "answer": "Team A won.",
                "results": [
                    {
                        "title": "Match recap",
                        "url": "https://example.com/a",
                        "content": "Short summary",
                        "published_date": "2026-01-16",
                    },
                    {
                        "title": "Follow-up",
                        "url": "https://example.com/b",
                        "content": "details",
                    },
                ],
            },
        )
    )

    settings = _StubSettings()
    result = await tavily_search("who won yesterday?", "2026-01-17", settings)

    assert route.called
    body = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert body["query"] == "who won yesterday?"
    assert body["end_date"] == "2026-01-17"
    assert body["max_results"] == 5
    assert body["include_raw_content"] is False

    assert result.ok
    assert result.answer == "Team A won."
    assert len(result.results) == 2
    assert result.results[0].published_date == "2026-01-16"
    assert result.results[1].published_date is None

    payload = result.to_llm_payload()
    assert "raw_content" not in json.dumps(payload)
    assert payload["answer"] == "Team A won."


@respx.mock
async def test_tavily_search_retries_then_succeeds() -> None:
    route = respx.post(TAVILY_ENDPOINT).mock(
        side_effect=[
            httpx.Response(503, text="unavailable"),
            httpx.Response(
                200,
                json={"answer": None, "results": []},
            ),
        ]
    )
    settings = _StubSettings()
    result = await tavily_search("q", "2026-01-17", settings)
    assert route.call_count == 2
    assert result.ok
    assert result.answer is None


@respx.mock
async def test_tavily_search_exhausted_returns_error_payload() -> None:
    respx.post(TAVILY_ENDPOINT).mock(return_value=httpx.Response(503, text="down"))
    settings = _StubSettings(SEARCH_RETRY_MAX=2, SEARCH_BACKOFF_S=[0, 0])
    result = await tavily_search("q", "2026-01-17", settings)
    assert not result.ok
    assert result.error_kind == "tavily_error"
    payload = result.to_llm_payload()
    assert payload["error"] == "tavily_error"
    assert payload["results"] == []


@respx.mock
async def test_tavily_search_network_error_retries() -> None:
    route = respx.post(TAVILY_ENDPOINT).mock(
        side_effect=[
            httpx.ConnectError("boom"),
            httpx.Response(200, json={"answer": "x", "results": []}),
        ]
    )
    settings = _StubSettings()
    result = await tavily_search("q", "2026-01-17", settings)
    assert route.call_count == 2
    assert result.ok


def test_search_result_payload_drops_raw_content() -> None:
    from forecast_eval.search import SearchResultItem

    r = SearchResult(
        query="q",
        end_date="2026-01-17",
        answer=None,
        results=[
            SearchResultItem(
                title="t", url="u", content="short", published_date=None
            )
        ],
    )
    # raw_content should never be in the LLM-bound payload even if the dataclass
    # later grows such a field; test on the current output shape.
    p = r.to_llm_payload()
    assert "raw_content" not in json.dumps(p)
