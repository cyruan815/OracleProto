"""Settings validation tests for search-leak-filter-v1 namespace."""
from __future__ import annotations

import pytest

from forecast_eval.config import Settings


def _base_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Fill in the required main LLM / Tavily fields so other fields can be tested independently."""
    monkeypatch.setenv("LLM_API_KEY", "sk-or-v1-ABCDEFGHIJKLMNOP0123")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-ABCDEFGHIJK0123")
    monkeypatch.setenv("MODELS", "openai/gpt-5")
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("SOURCE_DB", str(tmp_path / "forecast_eval_set_example.db"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))


def test_leak_filter_enabled_requires_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """When ENABLE_SEARCH_LEAK_FILTER=true, missing LEAK_DETECTOR_API_KEY must fail startup."""
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ENABLE_SEARCH_LEAK_FILTER", "true")
    monkeypatch.setenv("LEAK_DETECTOR_MODEL", "anthropic/claude-sonnet-4.6")
    # LEAK_DETECTOR_API_KEY not set
    with pytest.raises(ValueError, match="LEAK_DETECTOR_API_KEY"):
        Settings(_env_file=None)


def test_leak_filter_enabled_requires_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ENABLE_SEARCH_LEAK_FILTER", "true")
    monkeypatch.setenv("LEAK_DETECTOR_API_KEY", "sk-detector-real-key-1234")
    # LEAK_DETECTOR_MODEL not set
    with pytest.raises(ValueError, match="LEAK_DETECTOR_MODEL"):
        Settings(_env_file=None)


def test_leak_filter_enabled_rejects_placeholder_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ENABLE_SEARCH_LEAK_FILTER", "true")
    monkeypatch.setenv("LEAK_DETECTOR_API_KEY", "REPLACE_ME")
    monkeypatch.setenv("LEAK_DETECTOR_MODEL", "anthropic/claude-sonnet-4.6")
    with pytest.raises(ValueError, match="placeholder"):
        Settings(_env_file=None)


def test_leak_filter_disabled_skips_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """When ENABLE_SEARCH_LEAK_FILTER=false, startup should succeed even if all LEAK_DETECTOR_* are empty."""
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ENABLE_SEARCH_LEAK_FILTER", "false")
    s = Settings(_env_file=None)
    assert s.ENABLE_SEARCH_LEAK_FILTER is False
    assert s.LEAK_DETECTOR_API_KEY == ""
    assert s.LEAK_DETECTOR_MODEL == ""
    assert s.LEAK_DETECTOR_FAIL_ACTION == "drop"
    assert s.LEAK_DETECTOR_CONCURRENCY == 5
    assert s.LEAK_DETECTOR_BACKOFF_S == [2, 5, 15]


def test_leak_detector_repr_redacts_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """repr(settings) MUST display LEAK_DETECTOR_API_KEY as the literal '<redacted>'."""
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ENABLE_SEARCH_LEAK_FILTER", "true")
    monkeypatch.setenv("LEAK_DETECTOR_API_KEY", "sk-detector-supersecret-123456")
    monkeypatch.setenv("LEAK_DETECTOR_MODEL", "anthropic/claude-sonnet-4.6")
    s = Settings(_env_file=None)
    blob = repr(s)
    assert "sk-detector-supersecret-123456" not in blob
    assert "<redacted>" in blob
    # __repr__ existing path uses suffix replacement, the detector key is handled alongside LLM_API_KEY.
    assert "sk-or-v1-ABCDEFGHIJKLMNOP0123" not in blob


def test_leak_detector_fail_action_invalid_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ENABLE_SEARCH_LEAK_FILTER", "false")
    monkeypatch.setenv("LEAK_DETECTOR_FAIL_ACTION", "silent")
    with pytest.raises(ValueError, match="drop, keep"):
        Settings(_env_file=None)


def test_leak_detector_model_no_online_suffix(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ENABLE_SEARCH_LEAK_FILTER", "true")
    monkeypatch.setenv("LEAK_DETECTOR_API_KEY", "sk-detector-real-key-1234")
    monkeypatch.setenv("LEAK_DETECTOR_MODEL", "anthropic/claude-sonnet-4.6:online")
    with pytest.raises(ValueError, match=":online"):
        Settings(_env_file=None)


def test_leak_filter_requires_web_search(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """ENABLE_WEB_SEARCH=false + ENABLE_SEARCH_LEAK_FILTER=true are mutually exclusive, must fail startup."""
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "false")
    monkeypatch.setenv("ENABLE_SEARCH_LEAK_FILTER", "true")
    monkeypatch.setenv("LEAK_DETECTOR_API_KEY", "sk-detector-real-key-1234")
    monkeypatch.setenv("LEAK_DETECTOR_MODEL", "anthropic/claude-sonnet-4.6")
    with pytest.raises(ValueError, match="ENABLE_WEB_SEARCH"):
        Settings(_env_file=None)


def test_leak_detector_concurrency_must_be_positive(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ENABLE_SEARCH_LEAK_FILTER", "false")
    monkeypatch.setenv("LEAK_DETECTOR_CONCURRENCY", "0")
    with pytest.raises(ValueError, match="LEAK_DETECTOR_CONCURRENCY"):
        Settings(_env_file=None)


def test_leak_detector_backoff_parses_csv(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ENABLE_SEARCH_LEAK_FILTER", "false")
    monkeypatch.setenv("LEAK_DETECTOR_BACKOFF_S", "1,3,8")
    s = Settings(_env_file=None)
    assert s.LEAK_DETECTOR_BACKOFF_S == [1, 3, 8]


def test_leak_detector_temperature_must_be_non_negative(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ENABLE_SEARCH_LEAK_FILTER", "false")
    monkeypatch.setenv("LEAK_DETECTOR_TEMPERATURE", "-0.1")
    with pytest.raises(ValueError, match="LEAK_DETECTOR_TEMPERATURE"):
        Settings(_env_file=None)


def test_leak_detector_max_tokens_must_be_positive(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("ENABLE_SEARCH_LEAK_FILTER", "false")
    monkeypatch.setenv("LEAK_DETECTOR_MAX_TOKENS", "0")
    with pytest.raises(ValueError, match="LEAK_DETECTOR_MAX_TOKENS"):
        Settings(_env_file=None)
