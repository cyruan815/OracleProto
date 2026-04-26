"""Tests for `db.compose_virtual_slug` / `db.parse_virtual_slug` / filename safety.

These three pure helpers live at the boundary between the dispatcher (which
encodes (real_model, R, C) tuples) and the analysis layer (which decodes
them). The DB / runner / react / search layers all treat virtual slugs as
opaque strings, so these helpers are the single point where the encoding
contract is enforced.
"""
from __future__ import annotations

import pytest

from forecast_eval.db import (
    compose_virtual_slug,
    model_slug_safe,
    parse_virtual_slug,
)


def test_compose_round_trip_simple() -> None:
    slug = compose_virtual_slug("openai/gpt-5", 5, 3)
    assert slug == "openai/gpt-5::r5::c3"
    assert parse_virtual_slug(slug) == ("openai/gpt-5", 5, 3)


def test_compose_round_trip_with_dots_and_dashes() -> None:
    slug = compose_virtual_slug("anthropic/claude-sonnet-4.5", 10, 8)
    assert slug == "anthropic/claude-sonnet-4.5::r10::c8"
    assert parse_virtual_slug(slug) == ("anthropic/claude-sonnet-4.5", 10, 8)


def test_compose_round_trip_with_dated_slug() -> None:
    # qwen3.5-plus-2026-02-15 contains dots, hyphens, digits — all valid in
    # provider slugs. The non-greedy regex must keep the entire stem intact.
    slug = compose_virtual_slug("qwen3.5-plus-2026-02-15", 10, 8)
    assert slug == "qwen3.5-plus-2026-02-15::r10::c8"
    assert parse_virtual_slug(slug) == ("qwen3.5-plus-2026-02-15", 10, 8)


def test_compose_rejects_real_model_with_double_colon() -> None:
    # Defensive: real_model containing `::` would break parse_virtual_slug
    # round-trip — fail fast with a clear error.
    with pytest.raises(ValueError, match="must not contain '::'"):
        compose_virtual_slug("foo::bar", 5, 3)


def test_parse_returns_none_for_plain_slug() -> None:
    assert parse_virtual_slug("openai/gpt-5") is None
    assert parse_virtual_slug("anthropic/claude-sonnet-4.5") is None


def test_parse_returns_none_for_malformed_tail() -> None:
    # Trailing junk after the c{C} segment is rejected.
    assert parse_virtual_slug("m::r5::c3-extra") is None
    # Missing one segment.
    assert parse_virtual_slug("m::r5") is None
    # Wrong order.
    assert parse_virtual_slug("m::c3::r5") is None
    # Non-integer numbers.
    assert parse_virtual_slug("m::rx::c3") is None


def test_parse_handles_non_string_input() -> None:
    # Defensive: never raise for non-string input.
    assert parse_virtual_slug(None) is None  # type: ignore[arg-type]
    assert parse_virtual_slug(123) is None  # type: ignore[arg-type]


def test_model_slug_safe_for_virtual_slug_is_grep_able() -> None:
    # `:` is normalized to `_` by `_UNSAFE_CHARS`, `/` to `__`. Result is
    # a single grep-friendly filename stem with cell coordinates trailing.
    assert model_slug_safe("openai/gpt-5::r5::c3") == "openai__gpt-5__r5__c3"
    assert (
        model_slug_safe("anthropic/claude-sonnet-4.5::r10::c8")
        == "anthropic__claude-sonnet-4.5__r10__c8"
    )


def test_model_slug_safe_real_slug_byte_compat() -> None:
    # Real-slug behavior MUST be byte-compatible with v4. (Pre-change
    # callers that pass plain real slugs see the same output as before.)
    assert model_slug_safe("anthropic/claude-sonnet-4.5") == "anthropic__claude-sonnet-4.5"
    assert model_slug_safe("openai/gpt-5") == "openai__gpt-5"
    assert model_slug_safe("deepseek/deepseek-r1") == "deepseek__deepseek-r1"
