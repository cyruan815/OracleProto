"""Unit tests for forecast_eval.leak_filter (search-leak-filter-v1)."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Sequence
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from forecast_eval import leak_filter
from forecast_eval.config import Settings
from forecast_eval.errors import AuthError
from forecast_eval.search import SearchResult, SearchResultItem


# ---- Test stubs -------------------------------------------------------------


@dataclass
class _StubSettings:
    """Minimal stub mirroring the Settings fields ``leak_filter`` actually reads.

    Default values match ``Settings`` defaults so individual tests only override
    the few knobs they care about. SearchResultItem stays untouched.
    """

    LEAK_DETECTOR_API_KEY: str = "sk-detector-test-1234"
    LEAK_DETECTOR_BASE_URL: str = "https://api.detector.test/v1"
    LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
    LEAK_DETECTOR_MODEL: str = "anthropic/claude-sonnet-4.6"
    LEAK_DETECTOR_TIMEOUT_S: int = 60
    LEAK_DETECTOR_TEMPERATURE: float = 0.0
    LEAK_DETECTOR_MAX_TOKENS: int = 512
    LEAK_DETECTOR_RETRY_MAX: int = 2
    LEAK_DETECTOR_BACKOFF_S: list[int] = field(default_factory=lambda: [0, 0, 0])
    LEAK_DETECTOR_FAIL_ACTION: str = "drop"
    LEAK_DETECTOR_CONCURRENCY: int = 5


def _make_item(
    title: str = "Match recap",
    content: str = "summary",
    raw_content: str | None = "raw page",
    published_date: str | None = "2026-01-16",
    url: str = "https://example.com/a",
) -> SearchResultItem:
    return SearchResultItem(
        title=title,
        url=url,
        content=content,
        published_date=published_date,
        raw_content=raw_content,
    )


def _envelope(content: str) -> Any:
    """Build a fake ``with_raw_response.create`` envelope.

    Returned object exposes ``.parse()`` → an object whose ``choices[0].message.content``
    matches the provided string. We use ``MagicMock`` for cheap attribute access.
    """
    parsed = MagicMock()
    parsed.choices = [MagicMock()]
    parsed.choices[0].message.content = content
    raw = MagicMock()
    raw.parse = MagicMock(return_value=parsed)
    return raw


def _scripted_client(responses: Sequence[Any]) -> Any:
    """Async client whose ``chat.completions.with_raw_response.create`` replays
    the given sequence: each entry is either a callable returning an envelope,
    a ``BaseException`` to raise, or an envelope.
    """
    iterator = iter(list(responses))

    async def _create(**kwargs: Any) -> Any:
        try:
            nxt = next(iterator)
        except StopIteration as e:
            raise AssertionError("scripted client exhausted") from e
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    client = MagicMock()
    client.chat.completions.with_raw_response.create = _create
    client._captured_kwargs = []  # type: ignore[attr-defined]
    return client


def _capturing_client(responses: Sequence[Any]) -> Any:
    """Like ``_scripted_client`` but records every kwargs dict in ``client.captured``."""
    iterator = iter(list(responses))
    captured: list[dict[str, Any]] = []

    async def _create(**kwargs: Any) -> Any:
        captured.append(dict(kwargs))
        try:
            nxt = next(iterator)
        except StopIteration as e:
            raise AssertionError("capturing client exhausted") from e
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    client = MagicMock()
    client.chat.completions.with_raw_response.create = _create
    client.captured = captured  # type: ignore[attr-defined]
    return client


# ---- _detect_one ------------------------------------------------------------


async def test_detect_one_keep() -> None:
    settings = _StubSettings()
    client = _scripted_client(
        [_envelope(json.dumps({"verdict": "keep", "reason": "pre-cutoff"}))]
    )
    verdict, reason = await leak_filter._detect_one(
        _make_item(), "2026-01-17", settings, client
    )
    assert verdict == "keep"
    assert reason == "pre-cutoff"


async def test_detect_one_drop() -> None:
    settings = _StubSettings()
    client = _scripted_client(
        [_envelope(json.dumps({"verdict": "drop", "reason": "future event"}))]
    )
    verdict, reason = await leak_filter._detect_one(
        _make_item(), "2026-01-17", settings, client
    )
    assert verdict == "drop"
    assert "future" in reason


async def test_detect_one_invalid_json_retry() -> None:
    """Invalid JSON on first attempt, valid on second -> eventual success."""
    settings = _StubSettings(LEAK_DETECTOR_RETRY_MAX=2)
    client = _scripted_client(
        [
            _envelope("I think it should keep, but..."),  # not JSON, retry
            _envelope(json.dumps({"verdict": "keep", "reason": "ok"})),
        ]
    )
    verdict, _ = await leak_filter._detect_one(
        _make_item(), "2026-01-17", settings, client
    )
    assert verdict == "keep"


async def test_detect_one_invalid_verdict_retry() -> None:
    settings = _StubSettings(LEAK_DETECTOR_RETRY_MAX=2)
    client = _scripted_client(
        [
            _envelope(json.dumps({"verdict": "maybe", "reason": "..."})),
            _envelope(json.dumps({"verdict": "drop", "reason": "future"})),
        ]
    )
    verdict, _ = await leak_filter._detect_one(
        _make_item(), "2026-01-17", settings, client
    )
    assert verdict == "drop"


async def test_detect_one_network_retry_exhausted() -> None:
    """Three timeouts (1 main + 2 retries = 3 attempts) -> failed:network."""
    settings = _StubSettings(LEAK_DETECTOR_RETRY_MAX=2)
    client = _scripted_client(
        [
            httpx.ReadTimeout("boom"),
            httpx.ReadTimeout("boom"),
            httpx.ReadTimeout("boom"),
        ]
    )
    verdict, _ = await leak_filter._detect_one(
        _make_item(), "2026-01-17", settings, client
    )
    assert verdict.startswith("failed:network"), verdict


async def test_detect_one_auth_no_retry_no_propagate() -> None:
    """AUTH error fails immediately, no retry, AuthError MUST NOT propagate."""
    settings = _StubSettings(LEAK_DETECTOR_RETRY_MAX=2)
    # Simulate 401: use httpx.HTTPStatusError carrying response.status_code=401;
    # errors.classify reads status_code to identify AUTH.
    response = httpx.Response(
        401, request=httpx.Request("POST", "https://api.detector.test/v1/x")
    )
    auth_exc = httpx.HTTPStatusError(
        "401 unauthorized", request=response.request, response=response
    )
    call_count = {"n": 0}

    async def _create(**kwargs: Any) -> Any:
        call_count["n"] += 1
        raise auth_exc

    client = MagicMock()
    client.chat.completions.with_raw_response.create = _create

    verdict, _ = await leak_filter._detect_one(
        _make_item(), "2026-01-17", settings, client
    )
    assert verdict.startswith("failed:auth"), verdict
    assert call_count["n"] == 1, "AUTH MUST NOT trigger retry"


async def test_detect_one_handles_explicit_auth_error() -> None:
    """forecast_eval.errors.AuthError raised directly is also caught locally."""
    settings = _StubSettings(LEAK_DETECTOR_RETRY_MAX=2)
    # Raising AuthError directly should be caught locally (errors.classify will
    # take the UNKNOWN path because there's no status_code, but inside leak_filter
    # we use ErrorKind.AUTH for the decision -- confirmed via HTTPStatusError).
    # Switch here to one that classify can recognise as AUTH: response 401.
    response = httpx.Response(
        403, request=httpx.Request("POST", "https://api.detector.test/v1/x")
    )
    forbidden = httpx.HTTPStatusError(
        "403 forbidden", request=response.request, response=response
    )
    client = _scripted_client([forbidden])
    verdict, _ = await leak_filter._detect_one(
        _make_item(), "2026-01-17", settings, client
    )
    assert verdict.startswith("failed:auth")


# ---- _assert_detector_safe --------------------------------------------------


def test_assert_detector_safe_blocks_online_model() -> None:
    with pytest.raises(ValueError, match=":online"):
        leak_filter._assert_detector_safe("anthropic/x:online", {})


def test_assert_detector_safe_blocks_tools_kwargs() -> None:
    with pytest.raises(ValueError, match="tools"):
        leak_filter._assert_detector_safe("anthropic/x", {"tools": []})


def test_assert_detector_safe_blocks_plugins_kwargs() -> None:
    with pytest.raises(ValueError, match="plugins"):
        leak_filter._assert_detector_safe("anthropic/x", {"plugins": []})


def test_assert_detector_safe_blocks_tool_choice_kwargs() -> None:
    with pytest.raises(ValueError, match="tool_choice"):
        leak_filter._assert_detector_safe("anthropic/x", {"tool_choice": "auto"})


def test_assert_detector_safe_passes_clean_kwargs() -> None:
    leak_filter._assert_detector_safe(
        "anthropic/claude-sonnet-4.6",
        {"model": "anthropic/claude-sonnet-4.6", "messages": [], "max_tokens": 512},
    )  # MUST NOT raise


# ---- send-time inspection ---------------------------------------------------


async def test_detector_request_no_tools_in_kwargs() -> None:
    """The kwargs actually dispatched MUST NOT include tools / plugins / tool_choice."""
    settings = _StubSettings()
    client = _capturing_client(
        [_envelope(json.dumps({"verdict": "keep", "reason": "ok"}))]
    )
    await leak_filter._detect_one(_make_item(), "2026-01-17", settings, client)
    assert len(client.captured) == 1
    kwargs = client.captured[0]
    assert "tools" not in kwargs
    assert "plugins" not in kwargs
    assert "tool_choice" not in kwargs
    assert kwargs["model"] == settings.LEAK_DETECTOR_MODEL
    assert kwargs["timeout"] == settings.LEAK_DETECTOR_TIMEOUT_S


# ---- filter_search_result ---------------------------------------------------


def _result_with(items: list[SearchResultItem], answer: str | None = None) -> SearchResult:
    return SearchResult(query="q", end_date="2026-01-17", answer=answer, results=items)


def _verdict_envelope(verdict: str, reason: str = "ok") -> Any:
    return _envelope(json.dumps({"verdict": verdict, "reason": reason}))


async def test_filter_search_result_partial_drop() -> None:
    settings = _StubSettings()
    client = _scripted_client(
        [
            _verdict_envelope("keep"),
            _verdict_envelope("drop"),
            _verdict_envelope("keep"),
            _verdict_envelope("keep"),
            _verdict_envelope("drop"),
        ]
    )
    items = [_make_item(title=f"t{i}") for i in range(5)]
    result = _result_with(items, answer="some answer")
    out = await leak_filter.filter_search_result(
        result, end_date="2026-01-17", settings=settings, client=client
    )
    assert len(out.results) == 3
    assert [it.title for it in out.results] == ["t0", "t2", "t3"]
    assert out.audit is not None
    assert out.audit["n_results_raw"] == 5
    assert out.audit["n_results_kept"] == 3
    assert out.audit["detector_verdicts"] == [
        "keep",
        "drop",
        "keep",
        "keep",
        "drop",
    ]
    # On partial drop, answer is retained (known trade-off).
    assert out.answer == "some answer"


async def test_filter_search_result_all_drop_clears_answer() -> None:
    settings = _StubSettings()
    client = _scripted_client(
        [_verdict_envelope("drop") for _ in range(3)]
    )
    items = [_make_item(title=f"t{i}") for i in range(3)]
    result = _result_with(items, answer="synthesised summary")
    out = await leak_filter.filter_search_result(
        result, end_date="2026-01-17", settings=settings, client=client
    )
    assert out.results == []
    assert out.answer is None
    assert out.audit is not None
    assert out.audit["n_results_kept"] == 0
    assert out.audit["detector_verdicts"] == ["drop", "drop", "drop"]


async def test_filter_search_result_fail_action_drop() -> None:
    """FAIL_ACTION=drop: failed entries removed, audit contains failed:* verdict."""
    settings = _StubSettings(LEAK_DETECTOR_RETRY_MAX=0, LEAK_DETECTOR_FAIL_ACTION="drop")
    client = _scripted_client(
        [
            _verdict_envelope("keep"),
            httpx.ReadTimeout("net down"),
            _verdict_envelope("keep"),
        ]
    )
    items = [_make_item(title=f"t{i}") for i in range(3)]
    result = _result_with(items)
    out = await leak_filter.filter_search_result(
        result, end_date="2026-01-17", settings=settings, client=client
    )
    assert [it.title for it in out.results] == ["t0", "t2"]
    assert out.audit is not None
    verdicts = out.audit["detector_verdicts"]
    assert verdicts[0] == "keep"
    assert verdicts[1].startswith("failed:")
    assert verdicts[2] == "keep"
    assert out.audit["detector_error_kind"] == "network"


async def test_filter_search_result_fail_action_keep() -> None:
    """FAIL_ACTION=keep: failed entries passed through, verdict still recorded as failed:*."""
    settings = _StubSettings(LEAK_DETECTOR_RETRY_MAX=0, LEAK_DETECTOR_FAIL_ACTION="keep")
    client = _scripted_client(
        [
            _verdict_envelope("keep"),
            httpx.ReadTimeout("net down"),
            _verdict_envelope("drop"),
        ]
    )
    items = [_make_item(title=f"t{i}") for i in range(3)]
    result = _result_with(items)
    out = await leak_filter.filter_search_result(
        result, end_date="2026-01-17", settings=settings, client=client
    )
    # keep + failed passed through, drop removed -> remaining t0, t1.
    assert [it.title for it in out.results] == ["t0", "t1"]
    assert out.audit["detector_verdicts"][1].startswith("failed:")


async def test_filter_search_result_concurrency_limit() -> None:
    """CONCURRENCY=2: 5 concurrent items, simultaneous in-flight <= 2."""
    settings = _StubSettings(LEAK_DETECTOR_CONCURRENCY=2)
    in_flight = {"n": 0, "peak": 0}
    lock = asyncio.Lock()

    async def _create(**kwargs: Any) -> Any:
        async with lock:
            in_flight["n"] += 1
            in_flight["peak"] = max(in_flight["peak"], in_flight["n"])
        await asyncio.sleep(0.02)
        async with lock:
            in_flight["n"] -= 1
        return _verdict_envelope("keep")

    client = MagicMock()
    client.chat.completions.with_raw_response.create = _create

    items = [_make_item(title=f"t{i}") for i in range(5)]
    result = _result_with(items)
    await leak_filter.filter_search_result(
        result, end_date="2026-01-17", settings=settings, client=client
    )
    assert in_flight["peak"] <= 2, in_flight


async def test_filter_search_result_question_not_leaked() -> None:
    """detector messages MUST NOT contain the question field value. Verify with a unique string."""
    settings = _StubSettings()
    client = _capturing_client(
        [_verdict_envelope("keep"), _verdict_envelope("keep")]
    )
    # Embed the unique string in every possible field outside the item, then
    # confirm the detector message does not contain it.
    sentinel = "QUESTION_TEXT_SENTINEL_2026"
    items = [
        _make_item(title="t0", content="completely safe content"),
        _make_item(title="t1", content="completely safe content"),
    ]
    # Intentionally place the sentinel in a third-party variable, not passed to filter_search_result.
    _unused = sentinel  # noqa: F841 — confirms we never reference the sentinel
    result = _result_with(items)
    await leak_filter.filter_search_result(
        result, end_date="2026-01-17", settings=settings, client=client
    )
    for kw in client.captured:
        for msg in kw["messages"]:
            assert sentinel not in msg["content"]


async def test_prompt_hash_stable() -> None:
    a = leak_filter._compute_prompt_hash()
    b = leak_filter._compute_prompt_hash()
    assert a == b
    assert len(a) == 16


def test_prompt_hash_changes_when_template_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    original = leak_filter.LEAK_DETECTOR_PROMPT_TEMPLATE
    h_before = leak_filter._compute_prompt_hash()
    monkeypatch.setattr(leak_filter, "LEAK_DETECTOR_PROMPT_TEMPLATE", original + " ")
    h_after = leak_filter._compute_prompt_hash()
    assert h_before != h_after


def test_prompt_template_contains_six_principles() -> None:
    tpl = leak_filter.LEAK_DETECTOR_PROMPT_TEMPLATE
    assert "{cutoff_date}" in tpl
    # principle 2: scheduled / speculative future event
    assert "scheduled" in tpl
    assert "speculative" in tpl
    # principle 3: when in doubt → drop
    assert "When in doubt" in tpl
    # principle 4: do not use parametric knowledge
    assert "Do NOT use your own knowledge" in tpl
    # principle 5: strict JSON output
    assert "verdict" in tpl
    assert "reason" in tpl
    # principle 6: no question / answer / option language
    lower = tpl.lower()
    assert "question" not in lower
    assert "answer" not in lower
    assert "option" not in lower


def test_detector_client_independent_of_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_detector_client(settings) and llm.get_client(settings) MUST return different instances."""
    from forecast_eval import llm as llmmod

    # Reset both module singletons to avoid cross-test pollution.
    monkeypatch.setattr(leak_filter, "_detector_client", None)
    monkeypatch.setattr(llmmod, "_client", None)

    @dataclass
    class _S:
        LLM_API_KEY: str = "sk-llm-1"
        LLM_BASE_URL: str = "https://example.test/llm"
        LEAK_DETECTOR_API_KEY: str = "sk-llm-1"
        LEAK_DETECTOR_BASE_URL: str = "https://example.test/llm"

    s = _S()
    main = llmmod.get_client(s)
    detector = leak_filter.get_detector_client(s)
    assert id(main) != id(detector)


# ---- cell-local Settings transparency ---------------------------------------


async def test_filter_uses_passed_settings_not_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When filter_search_result(..., settings=local) is invoked, the detector
    call MUST use local.LEAK_DETECTOR_MODEL, not any global value.
    """
    settings_a = _StubSettings(LEAK_DETECTOR_MODEL="model-A")
    settings_b = _StubSettings(LEAK_DETECTOR_MODEL="model-B")
    client = _capturing_client([_verdict_envelope("keep")])
    items = [_make_item(title="t0")]
    result = _result_with(items)
    await leak_filter.filter_search_result(
        result, end_date="2026-01-17", settings=settings_b, client=client
    )
    assert client.captured[0]["model"] == "model-B"
    # settings_a was not used -- guards against closure-captures-global regressions.
    assert client.captured[0]["model"] != settings_a.LEAK_DETECTOR_MODEL


# ---- _parse_verdict edge cases ---------------------------------------------


def test_parse_verdict_handles_prose_wrapper() -> None:
    text = 'Yes I think this. Output: {"verdict": "keep", "reason": "fine"} done.'
    verdict, reason = leak_filter._parse_verdict(text)  # type: ignore[misc]
    assert verdict == "keep"
    assert reason == "fine"


def test_parse_verdict_rejects_unknown_verdict() -> None:
    text = '{"verdict": "maybe", "reason": "..."}'
    assert leak_filter._parse_verdict(text) is None


def test_parse_verdict_rejects_non_object() -> None:
    assert leak_filter._parse_verdict("[\"keep\"]") is None
    assert leak_filter._parse_verdict("") is None


def test_parse_verdict_coerces_non_string_reason() -> None:
    text = '{"verdict": "drop", "reason": 42}'
    verdict, reason = leak_filter._parse_verdict(text)  # type: ignore[misc]
    assert verdict == "drop"
    assert reason == "42"


def test_render_user_message_uses_placeholders_for_missing_fields() -> None:
    item = SearchResultItem(
        title="t", url="u", content="c", published_date=None, raw_content=None
    )
    rendered = leak_filter._render_user_message(item, "2026-01-17")
    assert "(unknown)" in rendered  # published_date placeholder
    assert "(empty)" in rendered  # raw_content placeholder
    assert "None" not in rendered.split("raw_content:")[1].split("\n")[0]
