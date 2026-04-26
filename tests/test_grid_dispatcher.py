"""Tests for the grid dispatcher in `evaluation.py`.

These tests bypass the real LLM / Tavily layers and directly exercise the
dispatcher's two responsibilities:
  1. Cartesian-expand `(MODELS × R_list × C_list)` into virtual slugs and
     write one .db file per cell with a coherent `config_snapshot.grid_origin`.
  2. Persist the `grid` section into the run-level `manifest.json`.

We invoke `_init_model_db` + `_write_manifest` directly (the same primitives
`_run_async` uses) so we don't need to script LLM responses. The runner /
react / search layers are exercised end-to-end by `test_smoke_dry_run.py`.
"""
from __future__ import annotations

import itertools
import json
import sqlite3
from pathlib import Path

import pytest

from evaluation import _init_model_db, _make_settings_factory, _write_manifest
from forecast_eval import db as dbmod
from forecast_eval.config import Settings
from forecast_eval.types import QFilter


SOURCE_DB = Path(__file__).resolve().parents[1] / "forecast_eval_set_example.db"


def _grid_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **overrides: str) -> Settings:
    monkeypatch.setenv("LLM_API_KEY", "sk-or-v1-TEST_ABCDEFGH")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-TEST_ABCDEFGH")
    monkeypatch.setenv("MODELS", "openai/gpt-4o-mini")
    monkeypatch.setenv("MODEL_TRAINING_CUTOFFS", "")
    monkeypatch.setenv("SAMPLING_N", "1")
    monkeypatch.setenv("SOURCE_DB", str(SOURCE_DB))
    monkeypatch.setenv("RUNS_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("TAVILY_MAX_RESULTS", "5,10")
    monkeypatch.setenv("REACT_MAX_SEARCH_CALLS", "1,3")
    monkeypatch.setenv("REACT_MIN_SEARCH_CALLS", "0")
    for k, v in overrides.items():
        monkeypatch.setenv(k, v)
    return Settings(_env_file=None)


def _cell_index(settings: Settings) -> dict[str, tuple[str, int, int, int]]:
    """Mirror evaluation._run_async's cell_index construction."""
    real_models = list(settings.MODELS)
    r_list = list(settings.TAVILY_MAX_RESULTS)
    c_list = list(settings.REACT_MAX_SEARCH_CALLS)
    global_min = settings.REACT_MIN_SEARCH_CALLS
    out: dict[str, tuple[str, int, int, int]] = {}
    for real, R, C in itertools.product(real_models, r_list, c_list):
        slug = dbmod.compose_virtual_slug(real, R, C)
        out[slug] = (real, R, C, min(global_min, C))
    return out


def test_cartesian_expansion_count(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _grid_settings(monkeypatch, tmp_path)
    cells = _cell_index(settings)
    # 1 model × 2 R × 2 C = 4 virtual slugs
    assert len(cells) == 4
    # Multi-model
    monkeypatch.setenv("MODELS", "m_a,m_b")
    s2 = Settings(_env_file=None)
    cells2 = _cell_index(s2)
    assert len(cells2) == 2 * 2 * 2  # 2 × 2 × 2 = 8


def test_settings_factory_returns_distinct_subviews(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _grid_settings(monkeypatch, tmp_path)
    factory = _make_settings_factory(settings)
    cells = _cell_index(settings)

    seen: dict[tuple[int, int], Settings] = {}
    for slug, (real, R, C, _eff) in cells.items():
        view = factory(slug, R, C)
        # Sub-view downcasts list -> int
        assert view.TAVILY_MAX_RESULTS == R
        assert view.REACT_MAX_SEARCH_CALLS == C
        # Distinct cell coordinates -> distinct settings views
        assert (R, C) not in seen
        seen[(R, C)] = view
    # Original settings remain list-form (not polluted)
    assert settings.TAVILY_MAX_RESULTS == [5, 10]
    assert settings.REACT_MAX_SEARCH_CALLS == [1, 3]


def test_factory_clamps_effective_min(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Global MIN=1 with min(C)=1 in list=[1,3]; cell C=1 keeps MIN=1, cell
    # C=3 also keeps MIN=1 (no clamp needed). Now flip global MIN=3 with
    # C list [3,5] (min(C)=3 == MIN, so Settings construction passes).
    settings = _grid_settings(
        monkeypatch,
        tmp_path,
        REACT_MAX_SEARCH_CALLS="3,5",
        REACT_MIN_SEARCH_CALLS="3",
    )
    factory = _make_settings_factory(settings)
    view_c3 = factory("m::r5::c3", 5, 3)
    view_c5 = factory("m::r5::c5", 5, 5)
    # Both cells inherit global MIN=3 since both C values >= MIN.
    assert view_c3.REACT_MIN_SEARCH_CALLS == 3
    assert view_c5.REACT_MIN_SEARCH_CALLS == 3


def test_init_model_db_writes_grid_origin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _grid_settings(monkeypatch, tmp_path)
    factory = _make_settings_factory(settings)
    cells = _cell_index(settings)

    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True)

    filters = QFilter()
    filters_snapshot = {**filters.snapshot(), "question_count": 0, "question_ids": []}

    real_model = list(settings.MODELS)[0]
    R, C = 5, 3
    virtual_slug = dbmod.compose_virtual_slug(real_model, R, C)
    db_path = db_dir / f"{dbmod.model_slug_safe(virtual_slug)}.db"
    cell_settings = factory(virtual_slug, R, C)
    effective_min = cells[virtual_slug][3]

    conn, _templates, _qs = _init_model_db(
        db_path=db_path,
        cell_settings=cell_settings,
        run_id="20260424-120000-abcd",
        virtual_model=virtual_slug,
        real_model=real_model,
        R=R,
        C=C,
        effective_min_search_calls=effective_min,
        source_path=SOURCE_DB,
        filters=filters,
        filters_snapshot=filters_snapshot,
        source_db_hash="srchash",
        metadata_hash="metahash",
        prompt_templates_hash="tmplhash",
        reflection_protocol_text=None,
        reflection_protocol_hash=None,
        belief_protocol_text=None,
        belief_protocol_hash=None,
    )

    row = conn.execute(
        "SELECT model, config_snapshot FROM run_meta WHERE run_id=?",
        ("20260424-120000-abcd",),
    ).fetchone()
    conn.close()

    assert row["model"] == virtual_slug
    snapshot = json.loads(row["config_snapshot"])
    # cell-local single-int values, not lists
    assert snapshot["TAVILY_MAX_RESULTS"] == 5
    assert snapshot["REACT_MAX_SEARCH_CALLS"] == 3
    # grid_origin field is the new self-describing audit record
    assert snapshot["grid_origin"] == {
        "real_model": real_model,
        "R": 5,
        "C": 3,
        "effective_min_search_calls": effective_min,
    }


def test_legacy_path_does_not_write_grid_origin(tmp_path: Path) -> None:
    # When register_run_meta is called with grid_origin=None, the persisted
    # config_snapshot must NOT contain a `grid_origin` key (back-compat).
    db_path = tmp_path / "legacy.db"
    conn = dbmod.connect(db_path)
    dbmod.init_schema(conn, sampling_n=1)
    dbmod.register_run_meta(
        conn,
        run_id="20260424-120000-abcd",
        model="openai/gpt-4o-mini",
        sampling_n=1,
        filters_snapshot={},
        config_snapshot={"TAVILY_MAX_RESULTS": 5},
        source_db_hash="x",
        metadata_hash="y",
        prompt_templates_hash="z",
    )
    row = conn.execute("SELECT config_snapshot FROM run_meta").fetchone()
    conn.close()
    snapshot = json.loads(row["config_snapshot"])
    assert "grid_origin" not in snapshot


def test_manifest_grid_section_complete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _grid_settings(monkeypatch, tmp_path)
    cells = _cell_index(settings)
    real_models = list(settings.MODELS)
    virtual_models = list(cells.keys())
    model_files = {
        slug: f"{dbmod.model_slug_safe(slug)}.db" for slug in virtual_models
    }
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        run_id="20260424-120000-abcd",
        settings=settings,
        filters=QFilter(),
        question_count=0,
        question_ids=[],
        virtual_models=virtual_models,
        real_models=real_models,
        model_files=model_files,
        source_db_hash="x",
        metadata_hash="y",
        prompt_templates_hash="z",
        reflection_protocol_hash=None,
        belief_protocol_hash=None,
        started_at="2026-04-24T12:00:00+00:00",
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["models"] == virtual_models
    grid = payload["grid"]
    # 6 mandatory fields per spec
    assert set(grid.keys()) == {
        "r_list",
        "c_list",
        "default_r",
        "default_c",
        "real_models",
        "n_cells",
    }
    assert grid["r_list"] == [5, 10]
    assert grid["c_list"] == [1, 3]
    # No GRID_DEFAULT in env -> first element of each list
    assert grid["default_r"] == 5
    assert grid["default_c"] == 1
    assert grid["real_models"] == real_models
    assert grid["n_cells"] == len(virtual_models) == 4
    # n_cells must equal len(manifest.models)
    assert grid["n_cells"] == len(payload["models"])


def test_manifest_grid_default_when_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _grid_settings(
        monkeypatch,
        tmp_path,
        GRID_DEFAULT_R="10",
        GRID_DEFAULT_C="3",
    )
    cells = _cell_index(settings)
    real_models = list(settings.MODELS)
    virtual_models = list(cells.keys())
    model_files = {slug: "x.db" for slug in virtual_models}
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        run_id="20260424-120000-abcd",
        settings=settings,
        filters=QFilter(),
        question_count=0,
        question_ids=[],
        virtual_models=virtual_models,
        real_models=real_models,
        model_files=model_files,
        source_db_hash="x",
        metadata_hash="y",
        prompt_templates_hash="z",
        reflection_protocol_hash=None,
        belief_protocol_hash=None,
        started_at="2026-04-24T12:00:00+00:00",
    )
    grid = json.loads(manifest_path.read_text(encoding="utf-8"))["grid"]
    assert grid["default_r"] == 10
    assert grid["default_c"] == 3


def test_n_cells_multi_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _grid_settings(
        monkeypatch,
        tmp_path,
        MODELS="openai/gpt-5,anthropic/claude-sonnet-4.5",
        TAVILY_MAX_RESULTS="5,10",
        REACT_MAX_SEARCH_CALLS="1,3,5,8",
    )
    cells = _cell_index(settings)
    # 2 models × 2 R × 4 C = 16 cells
    assert len(cells) == 16
    # All real_models from cells map back to settings.MODELS
    real_models_from_cells = {meta[0] for meta in cells.values()}
    assert real_models_from_cells == set(settings.MODELS)


def test_single_value_env_still_produces_one_cell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _grid_settings(
        monkeypatch,
        tmp_path,
        TAVILY_MAX_RESULTS="5",
        REACT_MAX_SEARCH_CALLS="8",
    )
    cells = _cell_index(settings)
    assert len(cells) == 1
    only_slug = next(iter(cells))
    assert only_slug.endswith("::r5::c8")
