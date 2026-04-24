from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from forecast_eval.errors import (
    AuthError,
    ErrorKind,
    backoff_seconds,
    classify,
    parse_retry_after,
    should_retry,
)


@dataclass
class _StubSettings:
    LLM_BACKOFF_NETWORK_S: list[int]
    LLM_BACKOFF_RATE_LIMIT_S: list[int]
    LLM_BACKOFF_SERVER_5XX_S: list[int]


def _settings() -> _StubSettings:
    return _StubSettings(
        LLM_BACKOFF_NETWORK_S=[2, 5, 15, 30, 60],
        LLM_BACKOFF_RATE_LIMIT_S=[10, 30, 60, 120, 300],
        LLM_BACKOFF_SERVER_5XX_S=[5, 15, 30, 60, 120],
    )


def _http_error(status: int, body: str = "") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.com/x")
    response = httpx.Response(status_code=status, request=request, text=body)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


def test_network_errors_map_to_network() -> None:
    req = httpx.Request("GET", "https://example.com")
    assert classify(httpx.ConnectError("boom", request=req)) is ErrorKind.NETWORK
    assert classify(httpx.ReadTimeout("slow", request=req)) is ErrorKind.NETWORK
    assert classify(httpx.ConnectTimeout("slow", request=req)) is ErrorKind.NETWORK
    assert classify(asyncio.TimeoutError()) is ErrorKind.NETWORK


def test_status_codes_classification() -> None:
    assert classify(_http_error(401)) is ErrorKind.AUTH
    assert classify(_http_error(403)) is ErrorKind.AUTH
    assert classify(_http_error(429)) is ErrorKind.RATE_LIMIT
    for code in (500, 502, 503, 504):
        assert classify(_http_error(code)) is ErrorKind.SERVER_5XX
    assert classify(_http_error(400, '{"error": {"code": "model_not_found"}}')) is ErrorKind.BAD_REQUEST
    assert classify(_http_error(400, '{"error": {"code": "content_policy_violation"}}')) is ErrorKind.CONTENT_POLICY
    assert classify(_http_error(418)) is ErrorKind.UNKNOWN


def test_should_retry_rules() -> None:
    assert should_retry(ErrorKind.NETWORK)
    assert should_retry(ErrorKind.RATE_LIMIT)
    assert should_retry(ErrorKind.SERVER_5XX)
    assert not should_retry(ErrorKind.AUTH)
    assert not should_retry(ErrorKind.BAD_REQUEST)
    assert not should_retry(ErrorKind.CONTENT_POLICY)
    assert not should_retry(ErrorKind.UNKNOWN)


def test_backoff_sequences() -> None:
    s = _settings()
    assert backoff_seconds(ErrorKind.NETWORK, 1, s) == 2.0
    assert backoff_seconds(ErrorKind.NETWORK, 5, s) == 60.0
    # exhausted -> None
    assert backoff_seconds(ErrorKind.NETWORK, 6, s) is None


def test_rate_limit_retry_after_wins() -> None:
    s = _settings()
    assert backoff_seconds(ErrorKind.RATE_LIMIT, 1, s, retry_after=30) == 30.0
    # no header -> fall back to sequence
    assert backoff_seconds(ErrorKind.RATE_LIMIT, 1, s) == 10.0
    assert backoff_seconds(ErrorKind.RATE_LIMIT, 6, s) is None


def test_auth_error_type_exists() -> None:
    with pytest.raises(AuthError):
        raise AuthError("nope")


def test_parse_retry_after_numeric() -> None:
    class H:
        def __init__(self, d: dict[str, str]) -> None:
            self._d = d

        def get(self, k: str) -> Any:
            return self._d.get(k)

    assert parse_retry_after(H({"Retry-After": "15"})) == 15.0
    assert parse_retry_after(H({"retry-after": "7.5"})) == 7.5
    assert parse_retry_after(H({})) is None
    assert parse_retry_after(H({"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"})) is None
    assert parse_retry_after(None) is None
