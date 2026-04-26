"""Tests for multi-value Settings parsing + per-cell sub-view derivation.

`config.Settings` now treats `TAVILY_MAX_RESULTS` and `REACT_MAX_SEARCH_CALLS`
as multi-value grid axes. The dispatcher (`evaluation._make_settings_factory`)
derives per-cell sub-views via `model_copy(update=...)` — this test suite
locks in the parser, the post-validate semantics (MIN-vs-C, GRID_DEFAULT
membership, real_model `::` defense), and the immutability of the global
instance under sub-view derivation.
"""
from __future__ import annotations

import pytest

from forecast_eval.config import Settings


def _required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "sk-or-v1-TEST_ABCDEFGH")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-TEST_ABCDEFGH")
    monkeypatch.setenv("MODELS", "openai/gpt-5")
    monkeypatch.setenv("MODEL_TRAINING_CUTOFFS", "")
    monkeypatch.setenv("ENABLE_WEB_SEARCH", "true")


def test_single_value_parses_to_length_one_list(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("TAVILY_MAX_RESULTS", "5")
    monkeypatch.setenv("REACT_MAX_SEARCH_CALLS", "8")
    s = Settings(_env_file=None)
    assert s.TAVILY_MAX_RESULTS == [5]
    assert s.REACT_MAX_SEARCH_CALLS == [8]


def test_multi_value_csv_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("TAVILY_MAX_RESULTS", "5,10")
    monkeypatch.setenv("REACT_MAX_SEARCH_CALLS", "1,3,5,8")
    s = Settings(_env_file=None)
    assert s.TAVILY_MAX_RESULTS == [5, 10]
    assert s.REACT_MAX_SEARCH_CALLS == [1, 3, 5, 8]


def test_empty_value_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("TAVILY_MAX_RESULTS", "")
    monkeypatch.setenv("REACT_MAX_SEARCH_CALLS", "8")
    with pytest.raises(ValueError, match="non-empty CSV"):
        Settings(_env_file=None)


def test_non_positive_value_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("TAVILY_MAX_RESULTS", "5,0")
    monkeypatch.setenv("REACT_MAX_SEARCH_CALLS", "8")
    with pytest.raises(ValueError, match="must be > 0"):
        Settings(_env_file=None)


def test_grid_default_r_must_be_in_list(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("TAVILY_MAX_RESULTS", "5,10")
    monkeypatch.setenv("REACT_MAX_SEARCH_CALLS", "1,3")
    monkeypatch.setenv("GRID_DEFAULT_R", "7")
    with pytest.raises(ValueError) as excinfo:
        Settings(_env_file=None)
    msg = str(excinfo.value)
    assert "7" in msg
    assert "[5, 10]" in msg


def test_grid_default_c_must_be_in_list(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("TAVILY_MAX_RESULTS", "5,10")
    monkeypatch.setenv("REACT_MAX_SEARCH_CALLS", "1,3")
    monkeypatch.setenv("GRID_DEFAULT_C", "8")
    with pytest.raises(ValueError) as excinfo:
        Settings(_env_file=None)
    msg = str(excinfo.value)
    assert "8" in msg
    assert "[1, 3]" in msg


def test_grid_default_unset_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("TAVILY_MAX_RESULTS", "5,10")
    monkeypatch.setenv("REACT_MAX_SEARCH_CALLS", "1,3")
    s = Settings(_env_file=None)
    assert s.GRID_DEFAULT_R is None
    assert s.GRID_DEFAULT_C is None


def test_min_above_all_c_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("TAVILY_MAX_RESULTS", "5")
    monkeypatch.setenv("REACT_MAX_SEARCH_CALLS", "1,3")
    monkeypatch.setenv("REACT_MIN_SEARCH_CALLS", "5")
    with pytest.raises(ValueError, match="REACT_MIN_SEARCH_CALLS"):
        Settings(_env_file=None)


def test_min_below_some_c_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    # MIN=3, C_list=[1,3,5,8] — min(C)=1 < MIN, so per-cell silent clamp is
    # the dispatcher's responsibility (effective_min). Settings construction
    # itself must not raise here, otherwise scanning low-budget cells would
    # require manual .env edits.
    _required_env(monkeypatch)
    monkeypatch.setenv("TAVILY_MAX_RESULTS", "5")
    monkeypatch.setenv("REACT_MAX_SEARCH_CALLS", "1,3,5,8")
    monkeypatch.setenv("REACT_MIN_SEARCH_CALLS", "3")
    # Hard-error only when MIN exceeds the smallest C in the list — here
    # MIN=3, min(C)=1, so MIN > min(C) → MUST raise.
    with pytest.raises(ValueError, match="REACT_MIN_SEARCH_CALLS"):
        Settings(_env_file=None)
    # Counter-case: MIN=1 stays under min(C)=1 (equal), which is fine.
    monkeypatch.setenv("REACT_MIN_SEARCH_CALLS", "1")
    s = Settings(_env_file=None)
    assert s.REACT_MIN_SEARCH_CALLS == 1
    assert s.REACT_MAX_SEARCH_CALLS == [1, 3, 5, 8]


def test_real_model_with_double_colon_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    # Defensive guard: a user pasting `m::sneaky` into MODELS would collide
    # with the virtual slug encoding. _post_validate rejects early.
    _required_env(monkeypatch)
    monkeypatch.setenv("MODELS", "openai/gpt-5,evil::injection")
    monkeypatch.setenv("TAVILY_MAX_RESULTS", "5")
    monkeypatch.setenv("REACT_MAX_SEARCH_CALLS", "8")
    with pytest.raises(ValueError, match="reserved for grid-search virtual slugs"):
        Settings(_env_file=None)


def test_model_copy_subview_does_not_pollute_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    # Core C-plan invariant: dispatcher derives per-cell sub-views via
    # model_copy without mutating the original Settings. After deriving N
    # views, the global Settings must still expose its list-form R/C.
    _required_env(monkeypatch)
    monkeypatch.setenv("TAVILY_MAX_RESULTS", "5,10")
    monkeypatch.setenv("REACT_MAX_SEARCH_CALLS", "1,3")
    s = Settings(_env_file=None)

    view_5_1 = s.model_copy(
        update={"TAVILY_MAX_RESULTS": 5, "REACT_MAX_SEARCH_CALLS": 1}
    )
    view_10_3 = s.model_copy(
        update={"TAVILY_MAX_RESULTS": 10, "REACT_MAX_SEARCH_CALLS": 3}
    )

    # Sub-views carry single ints
    assert view_5_1.TAVILY_MAX_RESULTS == 5
    assert view_5_1.REACT_MAX_SEARCH_CALLS == 1
    assert view_10_3.TAVILY_MAX_RESULTS == 10
    assert view_10_3.REACT_MAX_SEARCH_CALLS == 3

    # Original is untouched (still list-form)
    assert s.TAVILY_MAX_RESULTS == [5, 10]
    assert s.REACT_MAX_SEARCH_CALLS == [1, 3]


def test_effective_min_clamp_via_subview(monkeypatch: pytest.MonkeyPatch) -> None:
    # Dispatcher's silent clamp: effective_min = min(global_min, C). For C=1
    # cell with global MIN=1 (min of C list = 1, so MIN<=min(C)), effective
    # equals MIN; no nudge dead loop because react.py treats MIN==MAX as
    # "saturated → no nudge needed".
    _required_env(monkeypatch)
    monkeypatch.setenv("TAVILY_MAX_RESULTS", "5")
    monkeypatch.setenv("REACT_MAX_SEARCH_CALLS", "1,3")
    monkeypatch.setenv("REACT_MIN_SEARCH_CALLS", "1")
    s = Settings(_env_file=None)

    view_c1 = s.model_copy(
        update={
            "TAVILY_MAX_RESULTS": 5,
            "REACT_MAX_SEARCH_CALLS": 1,
            "REACT_MIN_SEARCH_CALLS": min(s.REACT_MIN_SEARCH_CALLS, 1),
        }
    )
    assert view_c1.REACT_MIN_SEARCH_CALLS == 1
    assert view_c1.REACT_MAX_SEARCH_CALLS == 1
