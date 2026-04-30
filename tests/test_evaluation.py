"""Tests for evaluation.py dispatcher: detector fingerprint + cell-local view."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import evaluation
from forecast_eval import db as dbmod
from forecast_eval import leak_filter
from forecast_eval.config import Settings
from forecast_eval.types import QFilter


SOURCE_DB = Path(__file__).resolve().parents[1] / "forecast_eval_set_example.db"


def _settings_env(monkeypatch: pytest.MonkeyPatch, tmp_path, **overrides: str) -> None:
    base = {
        "LLM_API_KEY": "sk-or-v1-ABCDEFGHIJKLMNOP0123",
        "TAVILY_API_KEY": "tvly-ABCDEFGHIJK0123",
        "MODELS": "openai/gpt-5",
        "RUNS_ROOT": str(tmp_path / "runs"),
        "SOURCE_DB": str(SOURCE_DB),
        "LOG_DIR": str(tmp_path / "logs"),
        "ENABLE_SEARCH_LEAK_FILTER": "false",
    }
    base.update(overrides)
    for k, v in base.items():
        monkeypatch.setenv(k, v)


def _call_init_model_db(
    settings: Settings,
    db_path: Path,
    *,
    capture: dict[str, Any],
) -> None:
    """Wrap evaluation._init_model_db with monkeypatched register_run_meta capture."""
    captured: list[dict[str, Any]] = []
    real_register = dbmod.register_run_meta

    def fake_register(conn, **kwargs):  # noqa: ANN001
        captured.append(dict(kwargs))
        return real_register(conn, **kwargs)

    capture["register_calls"] = captured

    cell_settings = settings.model_copy(
        update={
            "TAVILY_MAX_RESULTS": int(settings.TAVILY_MAX_RESULTS[0]),
            "REACT_MAX_SEARCH_CALLS": int(settings.REACT_MAX_SEARCH_CALLS[0]),
        }
    )

    import evaluation as evalmod  # noqa: PLC0415

    # monkeypatch register_run_meta on the evaluation module's reference
    orig = evalmod.dbmod.register_run_meta
    evalmod.dbmod.register_run_meta = fake_register
    try:
        evalmod._init_model_db(
            db_path=db_path,
            cell_settings=cell_settings,
            run_id="20260424-120000-abcd",
            virtual_model="openai/gpt-5::r5::c8",
            real_model="openai/gpt-5",
            R=int(cell_settings.TAVILY_MAX_RESULTS),
            C=int(cell_settings.REACT_MAX_SEARCH_CALLS),
            effective_min_search_calls=0,
            source_path=settings.source_db_path(),
            filters=QFilter(),
            filters_snapshot={},
            source_db_hash="a" * 64,
            metadata_hash="b" * 64,
            prompt_templates_hash="c" * 64,
            reflection_protocol_text=None,
            reflection_protocol_hash=None,
            belief_protocol_text=None,
            belief_protocol_hash=None,
        )
    finally:
        evalmod.dbmod.register_run_meta = orig


def test_init_model_db_injects_detector_fingerprint_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the leak filter is enabled, config_snapshot contains the detector's three keys."""
    _settings_env(
        monkeypatch,
        tmp_path,
        ENABLE_SEARCH_LEAK_FILTER="true",
        LEAK_DETECTOR_API_KEY="sk-detector-real-key-1234",
        LEAK_DETECTOR_MODEL="anthropic/claude-sonnet-4.6",
    )
    settings = Settings(_env_file=None)
    capture: dict[str, Any] = {}
    _call_init_model_db(settings, tmp_path / "m.db", capture=capture)

    assert len(capture["register_calls"]) == 1
    cs = capture["register_calls"][0]["config_snapshot"]
    assert cs["leak_detector_enabled"] is True
    assert cs["leak_detector_model"] == "anthropic/claude-sonnet-4.6"
    prompt_hash = cs["leak_detector_prompt_hash"]
    assert isinstance(prompt_hash, str) and len(prompt_hash) == 16
    # prompt hash matches leak_filter._compute_prompt_hash().
    assert prompt_hash == leak_filter._compute_prompt_hash()


def test_init_model_db_disabled_writes_default_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the toggle is off, the three keys still appear, but with default values (False / "" / "")."""
    _settings_env(monkeypatch, tmp_path, ENABLE_SEARCH_LEAK_FILTER="false")
    settings = Settings(_env_file=None)
    capture: dict[str, Any] = {}
    _call_init_model_db(settings, tmp_path / "m.db", capture=capture)

    cs = capture["register_calls"][0]["config_snapshot"]
    assert cs["leak_detector_enabled"] is False
    assert cs["leak_detector_model"] == ""
    assert cs["leak_detector_prompt_hash"] == ""


def test_init_model_db_does_not_change_register_signature(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The detector's three keys are injected via the config_snapshot dict without changing the register_run_meta signature."""
    _settings_env(monkeypatch, tmp_path, ENABLE_SEARCH_LEAK_FILTER="false")
    settings = Settings(_env_file=None)
    capture: dict[str, Any] = {}
    _call_init_model_db(settings, tmp_path / "m.db", capture=capture)
    kwargs = capture["register_calls"][0]
    # Sanity: signature kwargs unchanged — no 'leak_detector_*' kwarg surfaces.
    for k in kwargs:
        assert not k.startswith("leak_detector_"), k


def test_init_model_db_writes_redacted_detector_key_to_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """LEAK_DETECTOR_API_KEY appears in the snapshot as a redact_api_key dict; the plaintext is absent."""
    _settings_env(
        monkeypatch,
        tmp_path,
        ENABLE_SEARCH_LEAK_FILTER="true",
        LEAK_DETECTOR_API_KEY="sk-detector-supersecret-7890",
        LEAK_DETECTOR_MODEL="anthropic/claude-sonnet-4.6",
    )
    settings = Settings(_env_file=None)
    capture: dict[str, Any] = {}
    db_path = tmp_path / "m.db"
    _call_init_model_db(settings, db_path, capture=capture)
    cs = capture["register_calls"][0]["config_snapshot"]
    blob = json.dumps(cs)
    assert "sk-detector-supersecret-7890" not in blob
    assert isinstance(cs["LEAK_DETECTOR_API_KEY"], dict)
    assert cs["LEAK_DETECTOR_API_KEY"]["provider"] == "leak_detector"


def test_cell_local_settings_carries_leak_detector_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cell-local Settings derived by the grid dispatcher MUST pass through all LEAK_DETECTOR_* fields."""
    _settings_env(
        monkeypatch,
        tmp_path,
        ENABLE_SEARCH_LEAK_FILTER="true",
        LEAK_DETECTOR_API_KEY="sk-detector-supersecret-7890",
        LEAK_DETECTOR_MODEL="anthropic/claude-sonnet-4.6",
        TAVILY_MAX_RESULTS="5,10",
        REACT_MAX_SEARCH_CALLS="3,8",
    )
    settings = Settings(_env_file=None)

    factory = evaluation._make_settings_factory(settings)
    for slug, R, C in [
        ("openai/gpt-5::r5::c3", 5, 3),
        ("openai/gpt-5::r10::c8", 10, 8),
    ]:
        view = factory(slug, R, C)
        assert view.LEAK_DETECTOR_MODEL == settings.LEAK_DETECTOR_MODEL
        assert view.LEAK_DETECTOR_API_KEY == settings.LEAK_DETECTOR_API_KEY
        assert view.LEAK_DETECTOR_BASE_URL == settings.LEAK_DETECTOR_BASE_URL
        assert view.ENABLE_SEARCH_LEAK_FILTER == settings.ENABLE_SEARCH_LEAK_FILTER
        assert view.LEAK_DETECTOR_FAIL_ACTION == settings.LEAK_DETECTOR_FAIL_ACTION
        assert view.LEAK_DETECTOR_CONCURRENCY == settings.LEAK_DETECTOR_CONCURRENCY
        assert view.LEAK_DETECTOR_RETRY_MAX == settings.LEAK_DETECTOR_RETRY_MAX
        assert view.LEAK_DETECTOR_BACKOFF_S == settings.LEAK_DETECTOR_BACKOFF_S
        # cell-local sub-view downcasts R/C to single int per cell.
        assert view.TAVILY_MAX_RESULTS == R
        assert view.REACT_MAX_SEARCH_CALLS == C


def test_other_fingerprints_unaffected_by_leak_filter_toggle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Toggling the leak filter switch leaves other fingerprints (prompt / source / metadata) byte-identical."""
    # First pass: disable the leak filter
    _settings_env(monkeypatch, tmp_path, ENABLE_SEARCH_LEAK_FILTER="false")
    s_off = Settings(_env_file=None)

    # Second pass: enable
    monkeypatch.setenv("ENABLE_SEARCH_LEAK_FILTER", "true")
    monkeypatch.setenv("LEAK_DETECTOR_API_KEY", "sk-detector-real-key-1234")
    monkeypatch.setenv("LEAK_DETECTOR_MODEL", "anthropic/claude-sonnet-4.6")
    s_on = Settings(_env_file=None)

    # snapshot_settings stays stable for other fields (does not affect the main LLM / Tavily fingerprint).
    snap_off = dbmod.snapshot_settings(s_off)
    snap_on = dbmod.snapshot_settings(s_on)
    # MODELS / MODEL_TRAINING_CUTOFFS / TAVILY_* fields are all equal.
    assert snap_off["MODELS"] == snap_on["MODELS"]
    assert snap_off["TAVILY_INCLUDE_RAW_CONTENT"] == snap_on["TAVILY_INCLUDE_RAW_CONTENT"]
    assert snap_off["TAVILY_RAW_CONTENT_MAX_CHARS"] == snap_on["TAVILY_RAW_CONTENT_MAX_CHARS"]
    # Only the ENABLE_SEARCH_LEAK_FILTER + LEAK_DETECTOR_* fields differ.
    assert snap_off["ENABLE_SEARCH_LEAK_FILTER"] != snap_on["ENABLE_SEARCH_LEAK_FILTER"]
