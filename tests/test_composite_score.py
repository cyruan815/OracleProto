"""composite-score-by-subtype 单元测试。

覆盖：
* 默认权重数值正确（手算对一个 model + metric）；
* override 起作用（同时其他指标走默认）；
* 某桶 None → 被剔除并按比例归一化；
* 全 None → composite = None；
* 权重和 != 1（如 60/40）→ 正确归一化；
* 权重某桶为 0 → 该桶被剔除（与 None 同效）；
* CSV 列与 ``per_model_summary.csv`` 一一对齐；
* 不在 ``KNOWN_METRICS`` 内的 override metric 名 → ``compute_composite`` raise；
* 配置非法值 → ``Settings`` 启动期 raise；
* 端到端: 用 fixture run 跑 ``run_analysis``，断言三个新文件都生成且合理。
"""
from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from forecast_eval import analysis
from forecast_eval import db as dbmod
from forecast_eval.analysis.composite import (
    DEFAULT_WEIGHTS_CTYPE,
    DEFAULT_WEIGHTS_QTYPE,
    KNOWN_METRICS,
    CompositeReport,
    compute_composite,
)
from forecast_eval.analysis.writers import _SUMMARY_FIELDS
from forecast_eval.config import Settings


# --------------------------------------------------------------------------- #
# 纯函数: compute_composite
# --------------------------------------------------------------------------- #


def _make_bucket_values(
    metric: str, by_bucket: dict[str, float | None]
) -> dict[str, dict[str, dict[str, float | None]]]:
    """构造 ``{model: {metric: {bucket: value}}}``，单 model 单 metric 简化版。"""
    return {"m1": {metric: dict(by_bucket)}}


def test_default_weights_yield_expected_value() -> None:
    bv = _make_bucket_values(
        "fss",
        {"yes_no": 0.8, "binary_named": 0.6, "multiple_choice": 0.4},
    )
    rep = compute_composite(
        dimension="question_type",
        bucket_values_per_model=bv,
        weights_default=DEFAULT_WEIGHTS_QTYPE,
        overrides={},
    )
    info = rep.per_model["m1"]["fss"]
    expected = (0.15 * 0.8 + 0.15 * 0.6 + 0.70 * 0.4) / 1.0
    assert info.value == pytest.approx(expected)
    assert info.weights_kind == "default"
    assert tuple(sorted(info.buckets_used)) == (
        "binary_named",
        "multiple_choice",
        "yes_no",
    )


def test_overrides_apply_to_one_metric_only() -> None:
    bv = {
        "m1": {
            "fss": {"yes_no": 1.0, "binary_named": 0.5, "multiple_choice": 0.0},
            "pass_at_1_avg": {
                "yes_no": 1.0,
                "binary_named": 1.0,
                "multiple_choice": 0.0,
            },
        }
    }
    overrides = {"fss": {"multiple_choice": 1.0}}  # 只关心多选
    rep = compute_composite(
        dimension="question_type",
        bucket_values_per_model=bv,
        weights_default=DEFAULT_WEIGHTS_QTYPE,
        overrides=overrides,
    )
    fss_info = rep.per_model["m1"]["fss"]
    assert fss_info.value == pytest.approx(0.0)  # 只看多选 = 0
    assert fss_info.weights_kind == "overridden"
    assert fss_info.buckets_used == ("multiple_choice",)

    pass_info = rep.per_model["m1"]["pass_at_1_avg"]
    expected = 0.15 * 1.0 + 0.15 * 1.0 + 0.70 * 0.0
    assert pass_info.value == pytest.approx(expected)
    assert pass_info.weights_kind == "default"


def test_none_bucket_dropped_and_renormalized() -> None:
    """binary_named=None 时, 该桶被剔除, 剩余 yes_no + mc 重新归一化。"""
    bv = _make_bucket_values(
        "fss",
        {"yes_no": 0.8, "binary_named": None, "multiple_choice": 0.4},
    )
    rep = compute_composite(
        dimension="question_type",
        bucket_values_per_model=bv,
        weights_default=DEFAULT_WEIGHTS_QTYPE,
        overrides={},
    )
    info = rep.per_model["m1"]["fss"]
    expected = (0.15 * 0.8 + 0.70 * 0.4) / (0.15 + 0.70)
    assert info.value == pytest.approx(expected)
    assert tuple(sorted(info.buckets_used)) == ("multiple_choice", "yes_no")
    # 归一化权重和应为 1.0
    assert sum(info.weights_used_normalized.values()) == pytest.approx(1.0)
    # 各桶归一化权重
    assert info.weights_used_normalized["yes_no"] == pytest.approx(
        0.15 / 0.85
    )
    assert info.weights_used_normalized["multiple_choice"] == pytest.approx(
        0.70 / 0.85
    )


def test_all_none_yields_none() -> None:
    bv = _make_bucket_values(
        "fss",
        {"yes_no": None, "binary_named": None, "multiple_choice": None},
    )
    rep = compute_composite(
        dimension="question_type",
        bucket_values_per_model=bv,
        weights_default=DEFAULT_WEIGHTS_QTYPE,
        overrides={},
    )
    info = rep.per_model["m1"]["fss"]
    assert info.value is None
    assert info.buckets_used == ()


def test_unnormalized_weights_still_correct() -> None:
    """权重和 != 1 时也要正确归一化。"""
    bv = _make_bucket_values("fss", {"single": 0.5, "multi": 0.0})
    rep = compute_composite(
        dimension="choice_type",
        bucket_values_per_model=bv,
        weights_default={"single": 60.0, "multi": 40.0},  # 非归一化
        overrides={},
    )
    info = rep.per_model["m1"]["fss"]
    expected = (60.0 * 0.5 + 40.0 * 0.0) / 100.0
    assert info.value == pytest.approx(expected)


def test_zero_weight_bucket_excluded() -> None:
    """权重 0 的桶等同 None, 不参与合成。"""
    bv = _make_bucket_values("fss", {"single": 0.5, "multi": 0.9})
    rep = compute_composite(
        dimension="choice_type",
        bucket_values_per_model=bv,
        weights_default={"single": 1.0, "multi": 0.0},
        overrides={},
    )
    info = rep.per_model["m1"]["fss"]
    assert info.value == pytest.approx(0.5)
    assert info.buckets_used == ("single",)


def test_unknown_metric_in_override_raises() -> None:
    bv = _make_bucket_values("fss", {"yes_no": 0.5})
    overrides = {"unknown_metric": {"yes_no": 1.0}}
    with pytest.raises(ValueError, match="not a known metric"):
        compute_composite(
            dimension="question_type",
            bucket_values_per_model=bv,
            weights_default=DEFAULT_WEIGHTS_QTYPE,
            overrides=overrides,
        )


def test_known_metrics_align_with_summary_fields() -> None:
    """``KNOWN_METRICS`` 必须与 ``_SUMMARY_FIELDS`` 去掉元数据列后一致。"""
    summary_data = {f for f in _SUMMARY_FIELDS if f not in ("model", "sampling_n")}
    assert KNOWN_METRICS == summary_data


# --------------------------------------------------------------------------- #
# Settings 启动期校验
# --------------------------------------------------------------------------- #


def _settings_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("MODELS", "m1")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")


def test_settings_default_composite_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings_env(monkeypatch)
    s = Settings()
    assert s.COMPOSITE_WEIGHTS_QTYPE == DEFAULT_WEIGHTS_QTYPE
    assert s.COMPOSITE_WEIGHTS_CTYPE == DEFAULT_WEIGHTS_CTYPE
    assert s.COMPOSITE_WEIGHT_OVERRIDES_QTYPE == {}
    assert s.COMPOSITE_WEIGHT_OVERRIDES_CTYPE == {}


def test_settings_parses_csv_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _settings_env(monkeypatch)
    monkeypatch.setenv(
        "COMPOSITE_WEIGHTS_QTYPE",
        "yes_no=0.10,multiple_choice=0.90",
    )
    monkeypatch.setenv(
        "COMPOSITE_WEIGHT_OVERRIDES_CTYPE",
        "fss=single=0.3,multi=0.7;cohen_kappa=multi=1.0",
    )
    s = Settings()
    assert s.COMPOSITE_WEIGHTS_QTYPE == {"yes_no": 0.10, "multiple_choice": 0.90}
    assert s.COMPOSITE_WEIGHT_OVERRIDES_CTYPE == {
        "fss": {"single": 0.3, "multi": 0.7},
        "cohen_kappa": {"multi": 1.0},
    }


@pytest.mark.parametrize(
    "env_key,env_value,expected_msg",
    [
        ("COMPOSITE_WEIGHTS_QTYPE", "wrong_bucket=0.5", "not in"),
        ("COMPOSITE_WEIGHTS_CTYPE", "single=-0.1,multi=1.0", "must be >= 0"),
        ("COMPOSITE_WEIGHTS_CTYPE", "single=0,multi=0", "at least one weight"),
        (
            "COMPOSITE_WEIGHT_OVERRIDES_QTYPE",
            "fss=unknown=0.5",
            "not in",
        ),
    ],
)
def test_settings_rejects_invalid_composite_config(
    monkeypatch: pytest.MonkeyPatch,
    env_key: str,
    env_value: str,
    expected_msg: str,
) -> None:
    _settings_env(monkeypatch)
    monkeypatch.setenv(env_key, env_value)
    with pytest.raises(ValueError, match=expected_msg):
        Settings()


# --------------------------------------------------------------------------- #
# 端到端: run_analysis 写文件
# --------------------------------------------------------------------------- #


def _seed_questions(conn: sqlite3.Connection) -> None:
    rows = [
        ("q1", "single", "yes_no",          "ev1", json.dumps(["Yes", "No"]),     "A", "2026-03-01"),
        ("q2", "single", "binary_named",    "ev2", json.dumps(["Alpha", "Beta"]), "B", "2026-03-02"),
        ("q3", "multi",  "multiple_choice", "ev3", json.dumps(["x", "y", "z"]),   "A, C", "2026-03-03"),
    ]
    now = dbmod.utcnow_iso()
    conn.executemany(
        "INSERT INTO questions (id, choice_type, question_type, event, options, answer, end_time, imported_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        [(*r, now) for r in rows],
    )


def _sample_dict(
    *,
    question_id: str,
    sample_idx: int,
    correct: int | None,
    parse_ok: int,
    error: str | None,
    letters: list[str] | None = None,
) -> dict:
    return {
        "question_id": question_id,
        "sample_idx": sample_idx,
        "user_prompt": "P",
        "final_answer_letters": json.dumps(sorted(letters)) if letters is not None else None,
        "final_answer_raw": "raw",
        "correct": correct,
        "parse_ok": parse_ok,
        "tool_calls_count": 2,
        "react_steps": 3,
        "prompt_tokens": 100,
        "completion_tokens": 40,
        "reasoning_tokens": 0,
        "latency_ms": 1000,
        "messages_trace": None,
        "search_calls": None,
        "error": error,
        "created_at": dbmod.utcnow_iso(),
        "finish_reason": "stop",
        "nudges_used": 0,
        "step_metrics": json.dumps([
            {"step": 0, "prompt": 100, "completion": 40, "reasoning": 0,
             "latency_ms": 1000, "finish_reason": "stop", "n_tool_calls": 2},
        ]),
        "response_id": "resp_test",
        "system_fingerprint": "fp_test",
        "service_tier": "default",
        "belief_final": None,
        "belief_trace": None,
        "belief_parse_ok": 0,
        "final_answer_retry_used": 0,
    }


def _build_fixture_run(tmp_path: Path) -> Path:
    """两个模型, K=3。Model A 单选都对、多选两对一错; Model B 全错。"""
    run_dir = tmp_path / "run1"
    db_dir = run_dir / "db"
    db_dir.mkdir(parents=True)

    def _make_conn(path: Path, model: str) -> sqlite3.Connection:
        conn = dbmod.connect(path)
        dbmod.init_schema(conn, sampling_n=3)
        _seed_questions(conn)
        dbmod.register_run_meta(
            conn,
            run_id="run1",
            model=model,
            sampling_n=3,
            filters_snapshot={},
            config_snapshot={},
            source_db_hash="a" * 64,
            metadata_hash="b" * 64,
            prompt_templates_hash="c" * 64,
        )
        return conn

    conn_a = _make_conn(db_dir / "m__a.db", "m/a")
    for qid, gt_letters, _multi in [
        ("q1", ["A"], False),  # yes_no, gt=A
        ("q2", ["B"], False),  # binary_named, gt=B
    ]:
        for i in range(3):
            dbmod.upsert_sample_sync(
                conn_a,
                3,
                _sample_dict(
                    question_id=qid, sample_idx=i, correct=1,
                    parse_ok=1, error=None, letters=gt_letters,
                ),
            )
    # q3 (multi): K=3, two correct {A,C}, one wrong {A,B}
    for i, letters in enumerate([["A", "C"], ["A", "C"], ["A", "B"]]):
        correct = 1 if set(letters) == {"A", "C"} else 0
        dbmod.upsert_sample_sync(
            conn_a,
            3,
            _sample_dict(
                question_id="q3", sample_idx=i, correct=correct,
                parse_ok=1, error=None, letters=letters,
            ),
        )
    conn_a.commit()
    conn_a.close()

    conn_b = _make_conn(db_dir / "m__b.db", "m/b")
    for qid, wrong_letters in [
        ("q1", ["B"]),  # 全错
        ("q2", ["A"]),
    ]:
        for i in range(3):
            dbmod.upsert_sample_sync(
                conn_b,
                3,
                _sample_dict(
                    question_id=qid, sample_idx=i, correct=0,
                    parse_ok=1, error=None, letters=wrong_letters,
                ),
            )
    for i in range(3):
        dbmod.upsert_sample_sync(
            conn_b,
            3,
            _sample_dict(
                question_id="q3", sample_idx=i, correct=0,
                parse_ok=1, error=None, letters=["B"],
            ),
        )
    conn_b.commit()
    conn_b.close()

    manifest = {
        "run_id": "run1",
        "models": ["m/a", "m/b"],
        "model_files": {"m/a": "m__a.db", "m/b": "m__b.db"},
        "sampling_n": 3,
        "analysis_schema": "v5",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def test_run_analysis_writes_composite_files(tmp_path: Path) -> None:
    run_dir = _build_fixture_run(tmp_path)
    paths = analysis.run_analysis(run_dir)
    names = {p.name for p in paths}
    assert "per_model_composite_by_question_type.csv" in names
    assert "per_model_composite_by_choice_type.csv" in names
    assert "composite_meta.json" in names


def test_composite_csv_columns_match_summary(tmp_path: Path) -> None:
    """``per_model_composite_*.csv`` 列序 = ``model + sampling_n + weights_kind +
    [_SUMMARY_FIELDS 数据列]``。"""
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)
    expected_data = [
        f for f in _SUMMARY_FIELDS if f not in ("model", "sampling_n")
    ]
    expected = ["model", "sampling_n", "weights_kind", *expected_data]
    for path in (
        run_dir / "analysis" / "per_model_composite_by_question_type.csv",
        run_dir / "analysis" / "per_model_composite_by_choice_type.csv",
    ):
        with path.open() as f:
            header = next(csv.reader(f))
        assert header == expected, f"{path.name} header drift"


def test_composite_meta_records_used_buckets(tmp_path: Path) -> None:
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)
    meta = json.loads(
        (run_dir / "analysis" / "composite_meta.json").read_text()
    )
    assert "question_type" in meta and "choice_type" in meta
    qtype_section = meta["question_type"]
    assert qtype_section["weights_default"] == DEFAULT_WEIGHTS_QTYPE

    # m/a 的 fss 在三个 qtype 桶都有数据 → buckets_used 应为全 3 桶
    info = qtype_section["per_model"]["m/a"]["fss"]
    assert sorted(info["buckets_used"]) == sorted(DEFAULT_WEIGHTS_QTYPE)
    # 归一化权重和 = 1
    total = sum(info["weights_used_normalized"].values())
    assert abs(total - 1.0) < 1e-9
    # weights_kind 默认
    assert info["weights_kind"] == "default"


def test_composite_overrides_propagate_to_csv(tmp_path: Path) -> None:
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(
        run_dir,
        composite_overrides_qtype={"fss": {"multiple_choice": 1.0}},
    )
    # composite csv 的 weights_kind 列应为 overridden（任一指标命中 override）
    with (
        run_dir / "analysis" / "per_model_composite_by_question_type.csv"
    ).open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        assert r["weights_kind"] == "overridden"

    meta = json.loads(
        (run_dir / "analysis" / "composite_meta.json").read_text()
    )
    fss_info = meta["question_type"]["per_model"]["m/a"]["fss"]
    assert fss_info["weights_kind"] == "overridden"
    assert fss_info["buckets_used"] == ["multiple_choice"]
    # 其他指标仍是 default
    pass_info = meta["question_type"]["per_model"]["m/a"]["pass_at_1_avg"]
    assert pass_info["weights_kind"] == "default"


def test_slice_csv_now_carries_v5_columns(tmp_path: Path) -> None:
    """既有 ``per_model_by_question_type.csv`` / ``per_model_by_choice_type.csv``
    现在应该带 v5 列 (FSS / Cohen κ / Hamming / Fleiss κ / mean_entropy /
    VCI / MVG); 至少一行该有非 NULL 值。"""
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)
    for fname in (
        "per_model_by_question_type.csv",
        "per_model_by_choice_type.csv",
    ):
        path = run_dir / "analysis" / fname
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert rows, f"{fname} empty"
        # 至少一行 fss 非空
        fss_seen = any(r.get("fss") not in ("", None) for r in rows)
        assert fss_seen, f"{fname} has no fss values"


def test_overall_json_includes_composite(tmp_path: Path) -> None:
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)
    overall = json.loads(
        (run_dir / "analysis" / "overall.json").read_text()
    )
    assert "composite" in overall
    assert "question_type" in overall["composite"]
    assert "choice_type" in overall["composite"]
    qt_per = overall["composite"]["question_type"]["per_model"]["m/a"]
    assert "fss" in qt_per
    assert "weights_kind" in qt_per["fss"]
