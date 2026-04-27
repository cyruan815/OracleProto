from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

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
from forecast_eval.tavily_keys import (
    AllKeysExhausted,
    TavilyKeyPool,
    reset_pool_cache,
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
    # 升级后 settings.TAVILY_API_KEY 是 list[str] (CSV 多 key); 测试 stub 沿用同语义.
    TAVILY_API_KEY: list[str] = field(default_factory=lambda: ["tvly-TEST"])
    TAVILY_KEY_COOLDOWN_S: float = 60.0
    TAVILY_MAX_RESULTS: int = 5
    TAVILY_SEARCH_DEPTH: str = "basic"
    TAVILY_INCLUDE_RAW_CONTENT: str = "false"
    TAVILY_RAW_CONTENT_MAX_CHARS: int = 8000
    TAVILY_INCLUDE_ANSWER: str = "false"
    SEARCH_RETRY_MAX: int = 3
    SEARCH_BACKOFF_S: list[int] = None  # type: ignore[assignment]
    LLM_TIMEOUT_S: int = 30
    # search-leak-filter-v1: 默认关闭, 走非-detector 路径; 单测里需要时显式开启.
    ENABLE_SEARCH_LEAK_FILTER: bool = False

    def __post_init__(self) -> None:
        if self.SEARCH_BACKOFF_S is None:
            self.SEARCH_BACKOFF_S = [0, 0, 0]  # tests run fast, no real backoff


@pytest.fixture(autouse=True)
def _reset_pool_cache() -> None:
    """每个测试独立的 pool cache, 避免一个测试拉黑 key 串到下个测试.

    模块级 cache 在 prod 跨 grid cell 共享是 feature; 在测试隔离中是 bug 源.
    """
    reset_pool_cache()


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
    p = _build_request_payload(
        query="q", end_date="2026-01-17", settings=settings, api_key="tvly-PAYLOAD"
    )
    # api_key 来自调用方注入的池, 而非 settings.TAVILY_API_KEY[0].
    assert p["api_key"] == "tvly-PAYLOAD"
    assert p["search_depth"] == "basic"
    # "false" 字符串需映射到 JSON bool false (Tavily 协议: bool | "markdown" | "text")
    assert p["include_raw_content"] is False
    # include_answer 默认关闭时整字段不应进入 payload (Tavily 默认即 false)
    assert "include_answer" not in p


def test_payload_include_raw_content_markdown() -> None:
    settings = _StubSettings(TAVILY_INCLUDE_RAW_CONTENT="markdown")
    p = _build_request_payload(
        query="q", end_date="2026-01-17", settings=settings, api_key="tvly-T"
    )
    assert p["include_raw_content"] == "markdown"


def test_payload_include_raw_content_text() -> None:
    settings = _StubSettings(TAVILY_INCLUDE_RAW_CONTENT="text")
    p = _build_request_payload(
        query="q", end_date="2026-01-17", settings=settings, api_key="tvly-T"
    )
    assert p["include_raw_content"] == "text"


def test_payload_search_depth_advanced() -> None:
    settings = _StubSettings(TAVILY_SEARCH_DEPTH="advanced")
    p = _build_request_payload(
        query="q", end_date="2026-01-17", settings=settings, api_key="tvly-T"
    )
    assert p["search_depth"] == "advanced"


@pytest.mark.parametrize("answer_mode", ["basic", "advanced"])
def test_payload_emits_include_answer_when_enabled(answer_mode: str) -> None:
    settings = _StubSettings(TAVILY_INCLUDE_ANSWER=answer_mode)
    p = _build_request_payload(
        query="q", end_date="2026-01-17", settings=settings, api_key="tvly-T"
    )
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


@respx.mock
async def test_tavily_search_remote_protocol_error_retries() -> None:
    """v5.1 (harness-resilience): Tavily server hangs up mid-response. Pre-v5.1
    except clause (ConnectError / ReadTimeout / ConnectTimeout / asyncio.TimeoutError)
    let RemoteProtocolError bubble out — sample failed without retry. New
    clause treats it as a normal network blip and retries."""
    route = respx.post(TAVILY_ENDPOINT).mock(
        side_effect=[
            httpx.RemoteProtocolError("Server disconnected without sending a response."),
            httpx.Response(200, json={"answer": "ok", "results": []}),
        ]
    )
    settings = _StubSettings()
    result = await tavily_search("q", "2026-01-17", settings)
    assert route.call_count == 2
    assert result.ok
    assert result.answer == "ok"


@respx.mock
async def test_tavily_search_remote_protocol_error_exhausts_to_payload() -> None:
    """RemoteProtocolError repeating up to SEARCH_RETRY_MAX MUST return an
    error payload (not raise). The ReAct loop relies on this so the LLM sees
    a tool_result error and the sample stays alive."""
    respx.post(TAVILY_ENDPOINT).mock(
        side_effect=httpx.RemoteProtocolError(
            "Server disconnected without sending a response."
        )
    )
    settings = _StubSettings(SEARCH_RETRY_MAX=2, SEARCH_BACKOFF_S=[0, 0])
    result = await tavily_search("q", "2026-01-17", settings)
    assert not result.ok
    assert result.error_kind == "tavily_error"
    payload = result.to_llm_payload()
    assert payload["error"] == "tavily_error"
    assert payload["results"] == []


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


# ==================== TavilyKeyPool unit tests ====================


def _used_for(pool: TavilyKeyPool) -> dict[str, int]:
    return {st.key: st.used for st in pool.states}


def _key_state(pool: TavilyKeyPool, key: str):
    for st in pool.states:
        if st.key == key:
            return st
    raise AssertionError(f"key {key!r} not in pool")


async def test_pool_least_used_balances_load() -> None:
    # 强制 offset=0 以让计数从 0,0,0 开始, 测试 least-used 选择是否对称.
    pool = TavilyKeyPool.from_keys(
        ["k-a", "k-b", "k-c"], rng=random.Random(0), cooldown_s=10.0
    )
    # 抹掉随机起点, 让初始计数都是 0 — 测试 least-used 决策本身.
    for st in pool.states:
        st.used = 0

    counts: dict[str, int] = {"k-a": 0, "k-b": 0, "k-c": 0}
    for _ in range(9):
        k = await pool.acquire()
        counts[k] += 1
    # 9 / 3 keys → 每把 key 各 3 次 (least-used + 顺序 tie-breaker 决定均衡).
    assert counts == {"k-a": 3, "k-b": 3, "k-c": 3}


async def test_pool_random_starting_offset_rotates_first_key() -> None:
    # 不同种子应给出不同的初始用量分布 (起点偏移生效).
    seeds = [1, 2, 3, 4, 5, 6, 7, 8]
    first_keys = []
    for seed in seeds:
        pool = TavilyKeyPool.from_keys(
            ["k-a", "k-b", "k-c", "k-d"], rng=random.Random(seed), cooldown_s=10.0
        )
        first_keys.append(await pool.acquire())
    # 不要求每个 seed 都不同, 但至少不应 8 次都命中 keys[0] (那就是没随机).
    assert len(set(first_keys)) >= 2


async def test_pool_auth_failure_blacklists_permanently() -> None:
    pool = TavilyKeyPool.from_keys(
        ["k-a", "k-b"], rng=random.Random(0), cooldown_s=10.0
    )
    for st in pool.states:
        st.used = 0
    bad = await pool.acquire()  # k-a (顺序 tie-breaker)
    await pool.report_failure(bad, "auth")
    assert _key_state(pool, bad).blacklisted is True

    # 接下来无论 acquire 多少次都不会再返回 bad.
    for _ in range(10):
        k = await pool.acquire()
        assert k != bad


async def test_pool_rate_limit_cooldown_then_recover() -> None:
    fake_now = [1000.0]

    pool = TavilyKeyPool.from_keys(
        ["k-a", "k-b"], rng=random.Random(0), cooldown_s=30.0
    )
    pool._now = lambda: fake_now[0]
    for st in pool.states:
        st.used = 0

    rate_limited = await pool.acquire()  # k-a
    await pool.report_failure(rate_limited, "rate_limit")
    # cooldown 内不应再返回该 key.
    for _ in range(5):
        k = await pool.acquire()
        assert k != rate_limited

    # cooldown 过后应恢复 (least-used 选择会优先回到它, 因为另一把 key 已用了 6 次).
    fake_now[0] += 31.0
    saw_recovered = False
    for _ in range(20):
        k = await pool.acquire()
        if k == rate_limited:
            saw_recovered = True
            break
    assert saw_recovered, "key 未在 cooldown 后恢复"


async def test_pool_all_blacklisted_raises() -> None:
    pool = TavilyKeyPool.from_keys(["k-a", "k-b"], rng=random.Random(0))
    for k in ["k-a", "k-b"]:
        await pool.report_failure(k, "auth")
    with pytest.raises(AllKeysExhausted):
        await pool.acquire()


async def test_pool_other_failure_does_not_blacklist() -> None:
    pool = TavilyKeyPool.from_keys(["k-a"], rng=random.Random(0))
    await pool.report_failure("k-a", "other")
    # 5xx / 网络错不应拉黑 key — 重试还能拿到它.
    k = await pool.acquire()
    assert k == "k-a"


async def test_pool_dedups_keys() -> None:
    pool = TavilyKeyPool.from_keys(["k-a", "k-a", "k-b"], rng=random.Random(0))
    keys_in_pool = [s.key for s in pool.states]
    assert keys_in_pool == ["k-a", "k-b"]


# ==================== tavily_search × multi-key integration ====================


@respx.mock
async def test_tavily_search_distributes_across_multiple_keys() -> None:
    seen_keys: list[str] = []

    def _capture(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen_keys.append(body["api_key"])
        return httpx.Response(200, json={"answer": None, "results": []})

    respx.post(TAVILY_ENDPOINT).mock(side_effect=_capture)

    settings = _StubSettings(
        TAVILY_API_KEY=["tvly-K1", "tvly-K2", "tvly-K3"],
    )
    pool = TavilyKeyPool.from_keys(
        list(settings.TAVILY_API_KEY), rng=random.Random(0), cooldown_s=10.0
    )
    # 抹掉随机起点, 让 6 次调用稳定切成 2/2/2.
    for st in pool.states:
        st.used = 0

    for _ in range(6):
        result = await tavily_search("q", "2026-01-17", settings, pool=pool)
        assert result.ok

    counts = {k: seen_keys.count(k) for k in ["tvly-K1", "tvly-K2", "tvly-K3"]}
    assert counts == {"tvly-K1": 2, "tvly-K2": 2, "tvly-K3": 2}


@respx.mock
async def test_tavily_search_401_blacklists_then_uses_other_key() -> None:
    seen: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen.append(body["api_key"])
        if body["api_key"] == "tvly-BAD":
            return httpx.Response(401, text="invalid api key")
        return httpx.Response(200, json={"answer": "ok", "results": []})

    respx.post(TAVILY_ENDPOINT).mock(side_effect=_handler)

    pool = TavilyKeyPool.from_keys(
        ["tvly-BAD", "tvly-GOOD"], rng=random.Random(0), cooldown_s=10.0
    )
    for st in pool.states:
        st.used = 0  # tie-breaker → 先选 tvly-BAD

    settings = _StubSettings(
        TAVILY_API_KEY=["tvly-BAD", "tvly-GOOD"], SEARCH_RETRY_MAX=2
    )
    result = await tavily_search("q", "2026-01-17", settings, pool=pool)
    assert result.ok
    assert result.answer == "ok"
    # 第一次 401 不算网络重试, 立刻换 key 再试; 共 2 次请求.
    assert seen == ["tvly-BAD", "tvly-GOOD"]
    assert _key_state(pool, "tvly-BAD").blacklisted is True

    # 之后再调用应直接走 tvly-GOOD, 不再尝试拉黑的 key.
    seen.clear()
    result2 = await tavily_search("q2", "2026-01-17", settings, pool=pool)
    assert result2.ok
    assert seen == ["tvly-GOOD"]


@respx.mock
async def test_tavily_search_429_cools_down_then_uses_other_key() -> None:
    seen: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        seen.append(body["api_key"])
        if body["api_key"] == "tvly-FULL":
            return httpx.Response(429, text="rate limit exceeded")
        return httpx.Response(200, json={"answer": None, "results": []})

    respx.post(TAVILY_ENDPOINT).mock(side_effect=_handler)

    pool = TavilyKeyPool.from_keys(
        ["tvly-FULL", "tvly-OK"], rng=random.Random(0), cooldown_s=60.0
    )
    for st in pool.states:
        st.used = 0

    settings = _StubSettings(TAVILY_API_KEY=["tvly-FULL", "tvly-OK"])
    result = await tavily_search("q", "2026-01-17", settings, pool=pool)
    assert result.ok
    # 429 不计网络重试, 立即换到 tvly-OK 成功.
    assert seen == ["tvly-FULL", "tvly-OK"]
    # tvly-FULL 应处于 cooldown.
    assert _key_state(pool, "tvly-FULL").cooldown_until > pool._now()


@respx.mock
async def test_tavily_search_all_keys_exhausted_returns_error() -> None:
    respx.post(TAVILY_ENDPOINT).mock(return_value=httpx.Response(401, text="invalid"))
    pool = TavilyKeyPool.from_keys(
        ["tvly-A", "tvly-B"], rng=random.Random(0), cooldown_s=10.0
    )
    settings = _StubSettings(
        TAVILY_API_KEY=["tvly-A", "tvly-B"], SEARCH_RETRY_MAX=1
    )
    result = await tavily_search("q", "2026-01-17", settings, pool=pool)
    assert not result.ok
    assert result.error_kind == "tavily_error"
    # 两把 key 都应被永久拉黑.
    for st in pool.states:
        assert st.blacklisted is True


@respx.mock
async def test_tavily_search_5xx_uses_network_retry_does_not_blacklist() -> None:
    """5xx 仍走 SEARCH_BACKOFF_S 重试, key 不拉黑 (服务器问题, 非 key 问题)."""
    route = respx.post(TAVILY_ENDPOINT).mock(
        side_effect=[
            httpx.Response(503, text="down"),
            httpx.Response(200, json={"answer": None, "results": []}),
        ]
    )
    pool = TavilyKeyPool.from_keys(["tvly-only"], rng=random.Random(0))
    settings = _StubSettings(TAVILY_API_KEY=["tvly-only"])
    result = await tavily_search("q", "2026-01-17", settings, pool=pool)
    assert route.call_count == 2
    assert result.ok
    assert _key_state(pool, "tvly-only").blacklisted is False


# ==================== search-leak-filter-v1: detector wiring ====================


@dataclass
class _DetectorStubSettings(_StubSettings):
    """Extend the search _StubSettings with detector knobs the leak_filter reads."""

    ENABLE_SEARCH_LEAK_FILTER: bool = True
    LEAK_DETECTOR_API_KEY: str = "sk-detector-test-1234"
    LEAK_DETECTOR_BASE_URL: str = "https://api.detector.test/v1"
    LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
    LEAK_DETECTOR_MODEL: str = "anthropic/claude-sonnet-4.6"
    LEAK_DETECTOR_TIMEOUT_S: int = 60
    LEAK_DETECTOR_TEMPERATURE: float = 0.0
    LEAK_DETECTOR_MAX_TOKENS: int = 512
    LEAK_DETECTOR_RETRY_MAX: int = 0
    LEAK_DETECTOR_BACKOFF_S: list[int] = field(default_factory=lambda: [0, 0, 0])
    LEAK_DETECTOR_FAIL_ACTION: str = "drop"
    LEAK_DETECTOR_CONCURRENCY: int = 5


def _patch_detector(monkeypatch: pytest.MonkeyPatch, verdicts: list[str]) -> dict:
    """Install a fake detector by monkey-patching ``leak_filter._detect_one``.

    Returns a dict containing ``calls`` (list of (item.title, cutoff_date)) so
    tests can inspect ordering / count.
    """
    from forecast_eval import leak_filter

    state = {"calls": [], "verdicts": list(verdicts), "i": 0}

    async def fake_detect_one(item, cutoff_date, settings, client):  # noqa: ANN001
        state["calls"].append((item.title, cutoff_date))
        v = state["verdicts"][state["i"]]
        state["i"] += 1
        return v, "stub"

    monkeypatch.setattr(leak_filter, "_detect_one", fake_detect_one)
    return state


def _five_results_payload() -> dict:
    return {
        "answer": "synthesised answer",
        "results": [
            {
                "title": f"t{i}",
                "url": f"https://example.com/{i}",
                "content": f"snippet {i}",
                "published_date": "2026-01-16",
            }
            for i in range(5)
        ],
    }


@respx.mock
async def test_tavily_search_with_leak_filter_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """启用 leak filter: 5 条 + verdicts [keep,drop,keep,keep,drop] → 返回 3 条;
    result.audit 含 5 字段."""
    respx.post(TAVILY_ENDPOINT).mock(
        return_value=httpx.Response(200, json=_five_results_payload())
    )
    state = _patch_detector(monkeypatch, ["keep", "drop", "keep", "keep", "drop"])
    settings = _DetectorStubSettings()
    result = await tavily_search("q", "2026-01-17", settings)
    assert result.ok
    assert len(result.results) == 3
    assert [r.title for r in result.results] == ["t0", "t2", "t3"]
    # audit 字段齐全: 五个 spec 字段 + published_dates_raw (供 react 写入
    # search_calls.published_dates 用, 长度 == n_results_raw).
    assert result.audit is not None
    assert {
        "n_results_raw",
        "n_results_kept",
        "detector_verdicts",
        "detector_latency_ms",
        "detector_error_kind",
    } <= set(result.audit.keys())
    assert "published_dates_raw" in result.audit
    assert len(result.audit["published_dates_raw"]) == 5
    assert result.audit["n_results_raw"] == 5
    assert result.audit["n_results_kept"] == 3
    # detector 收到的 cutoff_date 与 Tavily end_date 同源.
    assert all(c[1] == "2026-01-17" for c in state["calls"])


@respx.mock
async def test_tavily_search_with_leak_filter_disabled() -> None:
    """开关关闭: 字节级与本提案前一致, audit is None, detector 不被调用."""
    respx.post(TAVILY_ENDPOINT).mock(
        return_value=httpx.Response(200, json=_five_results_payload())
    )
    settings = _DetectorStubSettings(ENABLE_SEARCH_LEAK_FILTER=False)
    result = await tavily_search("q", "2026-01-17", settings)
    assert result.ok
    assert len(result.results) == 5  # 原始数量, 未经 detector 裁剪
    assert result.audit is None  # 开关关闭则 audit 缺省


@respx.mock
async def test_tavily_search_failed_skips_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tavily 重试耗尽 → leak_filter 不被调用, error_kind=tavily_error."""
    from forecast_eval import leak_filter

    respx.post(TAVILY_ENDPOINT).mock(return_value=httpx.Response(503, text="down"))
    detector_called = {"n": 0}

    async def fake_filter(*args, **kwargs):  # noqa: ANN001
        detector_called["n"] += 1
        raise AssertionError("filter_search_result MUST NOT be called on Tavily failure")

    monkeypatch.setattr(leak_filter, "filter_search_result", fake_filter)
    settings = _DetectorStubSettings(SEARCH_RETRY_MAX=1, SEARCH_BACKOFF_S=[0])
    result = await tavily_search("q", "2026-01-17", settings)
    assert not result.ok
    assert result.error_kind == "tavily_error"
    assert result.audit is None
    assert detector_called["n"] == 0


@respx.mock
async def test_tavily_search_all_dropped_no_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """5 条全 drop → ok=True, results=[], answer=None (answer 由 drop 内容合成)."""
    respx.post(TAVILY_ENDPOINT).mock(
        return_value=httpx.Response(200, json=_five_results_payload())
    )
    _patch_detector(monkeypatch, ["drop"] * 5)
    settings = _DetectorStubSettings()
    result = await tavily_search("q", "2026-01-17", settings)
    assert result.ok
    assert result.error_kind is None
    assert result.results == []
    assert result.answer is None
    assert result.audit["n_results_kept"] == 0


@respx.mock
async def test_tavily_search_partial_drop_keeps_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """部分 drop → answer 保留 (已知折中, 见 spec)."""
    respx.post(TAVILY_ENDPOINT).mock(
        return_value=httpx.Response(200, json=_five_results_payload())
    )
    _patch_detector(monkeypatch, ["keep", "drop", "keep", "drop", "keep"])
    settings = _DetectorStubSettings()
    result = await tavily_search("q", "2026-01-17", settings)
    assert len(result.results) == 3
    assert result.answer == "synthesised answer"


def test_to_llm_payload_excludes_audit() -> None:
    """to_llm_payload() MUST 不输出 audit 字段, 防止它进入主 LLM 可见路径."""
    r = SearchResult(
        query="q",
        end_date="2026-01-17",
        answer="ok",
        results=[SearchResultItem(title="t", url="u", content="c")],
        audit={
            "n_results_raw": 1,
            "n_results_kept": 1,
            "detector_verdicts": ["keep"],
            "detector_latency_ms": 12,
            "detector_error_kind": None,
        },
    )
    payload = r.to_llm_payload()
    assert "audit" not in payload
    # 同时检查 results 内的每一项也不含 audit (defensive).
    for item in payload["results"]:
        assert "audit" not in item
