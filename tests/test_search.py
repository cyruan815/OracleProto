from __future__ import annotations

import json
from dataclasses import dataclass

import httpx
import pytest
import respx

from forecast_eval.search import (
    SearchResult,
    SearchResultItem,
    TAVILY_ENDPOINT,
    _build_request_payload,
    _truncate_raw_content,
    tavily_search,
)
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
    TAVILY_SEARCH_DEPTH: str = "basic"
    TAVILY_INCLUDE_RAW_CONTENT: str = "false"
    TAVILY_RAW_CONTENT_MAX_CHARS: int = 8000
    TAVILY_INCLUDE_ANSWER: str = "false"
    SEARCH_RETRY_MAX: int = 3
    SEARCH_BACKOFF_S: list[int] = None  # type: ignore[assignment]
    LLM_TIMEOUT_S: int = 30

    def __post_init__(self) -> None:
        if self.SEARCH_BACKOFF_S is None:
            self.SEARCH_BACKOFF_S = [0, 0, 0]  # tests run fast, no real backoff


# ==================== tools schema/argument ====================

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


# ==================== _build_request_payload (enum mapping) ====================

def test_payload_default_uses_bool_false_and_omits_answer() -> None:
    settings = _StubSettings()
    p = _build_request_payload(query="q", end_date="2026-01-17", settings=settings)
    assert p["search_depth"] == "basic"
    # "false" 字符串需映射到 JSON bool false (Tavily 协议: bool | "markdown" | "text")
    assert p["include_raw_content"] is False
    # include_answer 默认关闭时整字段不应进入 payload (Tavily 默认即 false)
    assert "include_answer" not in p


def test_payload_include_raw_content_markdown() -> None:
    settings = _StubSettings(TAVILY_INCLUDE_RAW_CONTENT="markdown")
    p = _build_request_payload(query="q", end_date="2026-01-17", settings=settings)
    assert p["include_raw_content"] == "markdown"


def test_payload_include_raw_content_text() -> None:
    settings = _StubSettings(TAVILY_INCLUDE_RAW_CONTENT="text")
    p = _build_request_payload(query="q", end_date="2026-01-17", settings=settings)
    assert p["include_raw_content"] == "text"


def test_payload_search_depth_advanced() -> None:
    settings = _StubSettings(TAVILY_SEARCH_DEPTH="advanced")
    p = _build_request_payload(query="q", end_date="2026-01-17", settings=settings)
    assert p["search_depth"] == "advanced"


@pytest.mark.parametrize("answer_mode", ["basic", "advanced"])
def test_payload_emits_include_answer_when_enabled(answer_mode: str) -> None:
    settings = _StubSettings(TAVILY_INCLUDE_ANSWER=answer_mode)
    p = _build_request_payload(query="q", end_date="2026-01-17", settings=settings)
    assert p["include_answer"] == answer_mode


# ==================== _truncate_raw_content ====================

def test_truncate_raw_content_under_limit_returns_intact() -> None:
    assert _truncate_raw_content("hello", 10) == "hello"
    assert _truncate_raw_content("hello", 5) == "hello"  # 等长不截断


def test_truncate_raw_content_over_limit_appends_marker() -> None:
    out = _truncate_raw_content("a" * 100, 10)
    assert out is not None
    assert out.startswith("a" * 10)
    assert "truncated to 10 chars" in out


def test_truncate_raw_content_zero_means_no_truncation() -> None:
    long = "a" * 50000
    assert _truncate_raw_content(long, 0) == long


def test_truncate_raw_content_none_passthrough() -> None:
    assert _truncate_raw_content(None, 1000) is None


# ==================== tavily_search end-to-end ====================

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
                        "score": 0.71,
                        "raw_content": "## Heading\nfull markdown body...",
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

    settings = _StubSettings(TAVILY_INCLUDE_RAW_CONTENT="markdown")
    result = await tavily_search("who won yesterday?", "2026-01-17", settings)

    assert route.called
    body = json.loads(route.calls.last.request.content.decode("utf-8"))
    assert body["query"] == "who won yesterday?"
    assert body["end_date"] == "2026-01-17"
    assert body["max_results"] == 5
    assert body["search_depth"] == "basic"
    assert body["include_raw_content"] == "markdown"
    assert "include_answer" not in body  # default false → 不发送

    assert result.ok
    assert result.answer == "Team A won."
    assert len(result.results) == 2
    assert result.results[0].published_date == "2026-01-16"
    assert result.results[0].score == 0.71
    assert result.results[0].raw_content == "## Heading\nfull markdown body..."
    assert result.results[1].published_date is None
    assert result.results[1].score is None
    assert result.results[1].raw_content is None

    payload = result.to_llm_payload()
    assert payload["answer"] == "Team A won."
    items = payload["results"]
    assert items[0]["score"] == 0.71
    assert items[0]["raw_content"].startswith("## Heading")
    # 第二条没 score / raw_content / published_date, 不应作为 null 字段进入 payload
    assert "score" not in items[1]
    assert "raw_content" not in items[1]
    assert "published_date" not in items[1]


@respx.mock
async def test_tavily_search_truncates_long_raw_content() -> None:
    huge = "x" * 50000
    respx.post(TAVILY_ENDPOINT).mock(
        return_value=httpx.Response(
            200,
            json={
                "answer": None,
                "results": [
                    {
                        "title": "t",
                        "url": "https://example.com/a",
                        "content": "c",
                        "raw_content": huge,
                    }
                ],
            },
        )
    )
    settings = _StubSettings(
        TAVILY_INCLUDE_RAW_CONTENT="markdown",
        TAVILY_RAW_CONTENT_MAX_CHARS=8000,
    )
    result = await tavily_search("q", "2026-01-17", settings)
    rc = result.results[0].raw_content
    assert rc is not None
    assert rc.startswith("x" * 8000)
    assert "truncated to 8000 chars" in rc


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


# ==================== to_llm_payload conditional emission ====================

def test_payload_emits_score_and_raw_content_when_present() -> None:
    r = SearchResult(
        query="q",
        end_date="2026-01-17",
        answer="cached",
        results=[
            SearchResultItem(
                title="t",
                url="u",
                content="c",
                published_date="2026-01-16",
                score=0.6,
                raw_content="page body",
            )
        ],
    )
    p = r.to_llm_payload()
    assert p["answer"] == "cached"
    item = p["results"][0]
    assert item["score"] == 0.6
    assert item["raw_content"] == "page body"
    assert item["published_date"] == "2026-01-16"


def test_payload_omits_optional_fields_when_none() -> None:
    r = SearchResult(
        query="q",
        end_date="2026-01-17",
        answer=None,
        results=[SearchResultItem(title="t", url="u", content="c")],
    )
    p = r.to_llm_payload()
    # answer=None → 整个 answer 字段不应出现
    assert "answer" not in p
    item = p["results"][0]
    assert "score" not in item
    assert "raw_content" not in item
    assert "published_date" not in item
