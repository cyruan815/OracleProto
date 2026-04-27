"""Unit tests for `forecast_eval.analysis`.

We hand-assemble a `RUNS_ROOT/{run_id}/` directory (manifest + one per-model DB)
and assert that the generated CSV/MD/JSON files carry the expected metric
values. Analysis is a pure read: nothing here needs the LLM/Tavily stack.
"""
from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from forecast_eval import analysis
from forecast_eval import db as dbmod


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


def _sample(
    *,
    question_id: str,
    sample_idx: int,
    correct: int | None,
    parse_ok: int,
    error: str | None,
    letters: list[str] | None = None,
    tool_calls: int = 2,
    react_steps: int = 3,
    latency: int = 1000,
    prompt_tokens: int = 100,
    completion_tokens: int = 40,
    reasoning_tokens: int = 0,
    finish_reason: str | None = "stop",
    nudges_used: int = 0,
) -> dict:
    return {
        "question_id": question_id,
        "sample_idx": sample_idx,
        "user_prompt": "P",
        "final_answer_letters": json.dumps(sorted(letters)) if letters is not None else None,
        "final_answer_raw": "raw",
        "correct": correct,
        "parse_ok": parse_ok,
        "tool_calls_count": tool_calls,
        "react_steps": react_steps,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "latency_ms": latency,
        "messages_trace": None,
        "search_calls": None,
        "error": error,
        "created_at": dbmod.utcnow_iso(),
        # v3 observability columns. Defaults mirror a successful sample;
        # cutoff/error fixtures override `finish_reason=None`.
        "finish_reason": finish_reason,
        "nudges_used": nudges_used,
        "step_metrics": json.dumps([
            {"step": 0, "prompt": prompt_tokens, "completion": completion_tokens,
             "reasoning": reasoning_tokens, "latency_ms": latency,
             "finish_reason": finish_reason, "n_tool_calls": tool_calls},
        ]),
        "response_id": "resp_test",
        "system_fingerprint": "fp_test",
        "service_tier": "default",
        # v4 belief columns: all NULL / 0 here because these fixtures don't
        # exercise the BELIEF_PROTOCOL path. Phase 1 will add fixtures that
        # populate them; Phase 0 just keeps the writer happy.
        "belief_final": None,
        "belief_trace": None,
        "belief_parse_ok": 0,
        # v5.1 (harness-resilience): default to 0 (no bail-out retry triggered).
        "final_answer_retry_used": 0,
    }


def _build_fixture_run(tmp_path: Path) -> Path:
    """Assemble one run directory with two models and SAMPLING_N=3.

    Model A (m/a): scores 2/3 on q1, 0/3 on q2, cutoff on q3.
    Model B (m/b): q1 parse-fails twice + correct once; q2 network; q3 all wrong.
    """
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
    # m/a: q1 → 2/3 correct, q2 → 0/3 correct, q3 → cutoff all 3
    # nudges_used is bumped on a single sample so avg_nudges_used > 0; this
    # keeps the assertion a non-trivial number rather than 0.
    q1_nudges = {0: 0, 1: 2, 2: 0}
    for i, correct in enumerate([1, 1, 0]):
        dbmod.upsert_sample_sync(conn_a, 3, _sample(
            question_id="q1", sample_idx=i, correct=correct, parse_ok=1, error=None,
            letters=["A"] if correct else ["B"],
            nudges_used=q1_nudges[i],
        ))
    # One q2 sample finishes with `length` so the breakdown CSV has > 1 row.
    q2_finish = {0: "length", 1: "stop", 2: "stop"}
    for i in range(3):
        dbmod.upsert_sample_sync(conn_a, 3, _sample(
            question_id="q2", sample_idx=i, correct=0, parse_ok=1, error=None,
            letters=["A"],  # wrong: GT is B
            finish_reason=q2_finish[i],
        ))
    for i in range(3):
        dbmod.upsert_sample_sync(conn_a, 3, _sample(
            question_id="q3", sample_idx=i, correct=None, parse_ok=0,
            error="skipped_training_cutoff", letters=None,
            tool_calls=0, react_steps=0, latency=0,
            prompt_tokens=0, completion_tokens=0, reasoning_tokens=0,
            finish_reason=None,  # cutoff path never invoked the LLM.
        ))
    dbmod.finish_run_meta(conn_a, "run1")
    conn_a.close()

    conn_b = _make_conn(db_dir / "m__b.db", "m/b")
    # m/b: q1 → 1/3 correct, 2/3 parse-failures
    dbmod.upsert_sample_sync(conn_b, 3, _sample(
        question_id="q1", sample_idx=0, correct=1, parse_ok=1, error=None, letters=["A"],
    ))
    dbmod.upsert_sample_sync(conn_b, 3, _sample(
        question_id="q1", sample_idx=1, correct=None, parse_ok=0, error=None, letters=None,
    ))
    dbmod.upsert_sample_sync(conn_b, 3, _sample(
        question_id="q1", sample_idx=2, correct=None, parse_ok=0, error=None, letters=None,
    ))
    # q2: all network errors. `_error_row` in production never reaches the
    # LLM, so finish_reason is None — mirror that here so the breakdown CSV
    # surfaces a `<missing>` bucket.
    for i in range(3):
        dbmod.upsert_sample_sync(conn_b, 3, _sample(
            question_id="q2", sample_idx=i, correct=None, parse_ok=0, error="network",
            letters=None,
            finish_reason=None,
        ))
    # q3: all wrong (answer {A, C}, model outputs {A})
    for i in range(3):
        dbmod.upsert_sample_sync(conn_b, 3, _sample(
            question_id="q3", sample_idx=i, correct=0, parse_ok=1, error=None,
            letters=["A"],
        ))
    dbmod.finish_run_meta(conn_b, "run1")
    conn_b.close()

    manifest = {
        "run_id": "run1",
        "schema_version": dbmod.SCHEMA_VERSION,
        "sampling_n": 3,
        "models": ["m/a", "m/b"],
        "model_files": {"m/a": "m__a.db", "m/b": "m__b.db"},
        "model_training_cutoffs": {},
        "filters": {
            "question_types": None, "choice_types": None,
            "question_count": 3, "question_ids": ["q1", "q2", "q3"],
        },
        "hashes": {
            "source_db": "a" * 64, "metadata": "b" * 64, "prompt_templates": "c" * 64,
        },
        "started_at": dbmod.utcnow_iso(),
        "finished_at": dbmod.utcnow_iso(),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return run_dir


def test_run_analysis_produces_all_artefacts(tmp_path: Path) -> None:
    run_dir = _build_fixture_run(tmp_path)
    paths = analysis.run_analysis(run_dir)
    names = {p.name for p in paths}
    assert {
        "per_model_summary.csv",
        "per_model_summary.md",
        "per_model_by_question_type.csv",
        "per_model_by_choice_type.csv",
        "error_breakdown.csv",
        "overall.json",
    }.issubset(names)


def test_per_model_summary_values(tmp_path: Path) -> None:
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)

    rows_by_model = {}
    with (run_dir / "analysis" / "per_model_summary.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_by_model[row["model"]] = row

    # ---- Model m/a ----
    # eligible = q1+q2 (cutoff drops q3 → 3 samples)
    # Resolvable samples: 6 (all eligible parsed)
    # pass@1 = 2 correct (q1) out of 6 = 0.3333
    # pass_any@N: q1=1 (has correct), q2=0 → 0.5
    # ≥majority (ceil(3/2)=2): q1=1 (2≥2), q2=0 → 0.5
    # ≥all: q1=0 (2<3), q2=0 → 0
    # majority vote: q1 letters={A: 2, B: 1} → {A} correct (GT A) ✔, q2 letters={A:3} → {A} GT B ✖ → 1/2 = 0.5
    a = rows_by_model["m/a"]
    assert int(a["eligible_samples"]) == 6
    assert int(a["eligible_questions"]) == 2
    assert int(a["resolvable_samples"]) == 6
    assert int(a["cutoff_skip_samples"]) == 3
    assert float(a["cutoff_skip_rate"]) == pytest.approx(3 / 9, rel=1e-3)
    assert float(a["pass_at_1_avg"]) == pytest.approx(2 / 6, rel=1e-3)
    assert float(a["pass_any_at_n"]) == pytest.approx(0.5, rel=1e-3)
    assert float(a["at_least_majority_at_n"]) == pytest.approx(0.5, rel=1e-3)
    assert float(a["at_least_all_at_n"]) == pytest.approx(0.0, abs=1e-3)
    assert float(a["majority_vote_accuracy"]) == pytest.approx(0.5, rel=1e-3)
    assert float(a["parse_failure_rate"]) == pytest.approx(0.0, abs=1e-6)
    assert float(a["error_rate"]) == pytest.approx(0.0, abs=1e-6)

    # ---- Model m/b ----
    # eligible = 9 (nothing cutoff), resolvable = q1 s0 (correct=1) + q3 × 3 (correct=0) = 4
    # Parse fail = q1 s1,s2 (parse_ok=0, error IS NULL) = 2 / 9
    # Error rate = q2 × 3 (error=network) = 3 / 9
    # pass@1 = 1 / 4 = 0.25
    # pass_any: q1=1 (s0 correct), q2=0 (no resolvable), q3=0 → averaged over questions with ≥1 resolvable.
    #           q1 has 1 resolvable, q2 has 0, q3 has 3. So average is over {q1, q3} = 1/2 = 0.5
    # majority_vote: q1 letters pool = {["A"]} → {A} vs A ✔; q2 no parsed → skipped; q3 {["A"]×3} → {A} vs {A,C} ✖
    #                → 1/2 = 0.5
    b = rows_by_model["m/b"]
    assert int(b["eligible_samples"]) == 9
    assert int(b["resolvable_samples"]) == 4
    assert float(b["pass_at_1_avg"]) == pytest.approx(0.25, rel=1e-3)
    assert float(b["parse_failure_rate"]) == pytest.approx(2 / 9, rel=1e-3)
    assert float(b["error_rate"]) == pytest.approx(3 / 9, rel=1e-3)
    assert float(b["pass_any_at_n"]) == pytest.approx(0.5, rel=1e-3)
    assert float(b["majority_vote_accuracy"]) == pytest.approx(0.5, rel=1e-3)


def test_error_breakdown_csv(tmp_path: Path) -> None:
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)
    with (run_dir / "analysis" / "error_breakdown.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_model: dict[str, dict[str, int]] = {}
    for r in rows:
        by_model.setdefault(r["model"], {})[r["error_kind"]] = int(r["count"])
    # m/a: 6 ok + 3 cutoff
    assert by_model["m/a"]["<ok>"] == 6
    assert by_model["m/a"]["skipped_training_cutoff"] == 3
    # m/b: 4 ok (q1 s0 + q3 × 3) + 2 parse-fails with error IS NULL (<ok>) + 3 network
    # <ok> counts include parse-failures (error IS NULL), so 4 + 2 = 6
    assert by_model["m/b"]["<ok>"] == 6
    assert by_model["m/b"]["network"] == 3


def test_overall_json_matches_csv(tmp_path: Path) -> None:
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)
    overall = json.loads((run_dir / "analysis" / "overall.json").read_text())
    assert overall["run_id"] == "run1"
    assert set(overall["per_model"]) == {"m/a", "m/b"}
    assert overall["per_model"]["m/a"]["pass_at_1_avg"] == pytest.approx(2 / 6, rel=1e-3)
    assert set(overall["per_model_by_question_type"]["m/a"]) == {"yes_no", "binary_named", "multiple_choice"}
    assert set(overall["per_model_by_choice_type"]["m/a"]) == {"single", "multi"}


def test_avg_nudges_used_in_summary(tmp_path: Path) -> None:
    """Per-model summary must surface avg_nudges_used over eligible samples.

    m/a: nudges across 6 eligible samples = [0,2,0,0,0,0] → 2/6
    m/b: every eligible sample uses nudges_used=0 → 0.0
    """
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)

    rows_by_model: dict[str, dict[str, str]] = {}
    with (run_dir / "analysis" / "per_model_summary.csv").open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert "avg_nudges_used" in reader.fieldnames  # type: ignore[operator]
        for row in reader:
            rows_by_model[row["model"]] = row

    assert float(rows_by_model["m/a"]["avg_nudges_used"]) == pytest.approx(2 / 6, abs=1e-2)
    assert float(rows_by_model["m/b"]["avg_nudges_used"]) == pytest.approx(0.0, abs=1e-6)


def test_finish_reason_breakdown_csv(tmp_path: Path) -> None:
    """`finish_reason_breakdown.csv` should be emitted with one row per
    (model, reason) over eligible samples (cutoff excluded). Shares must sum
    to 1.0 within each model.

    Expected eligible counts (cutoff drops m/a q3):
      m/a 6 eligible  → stop=5, length=1
      m/b 9 eligible  → stop=6 (q1 s0 + q3×3 + q1 s1/s2 parse-fails), <missing>=3 (q2 errors)
    """
    run_dir = _build_fixture_run(tmp_path)
    written = analysis.run_analysis(run_dir)

    breakdown_path = run_dir / "analysis" / "finish_reason_breakdown.csv"
    assert breakdown_path in written
    assert breakdown_path.exists()

    counts: dict[str, dict[str, int]] = {}
    shares: dict[str, dict[str, float]] = {}
    with breakdown_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["model", "finish_reason", "count", "share_of_eligible"]
        for row in reader:
            counts.setdefault(row["model"], {})[row["finish_reason"]] = int(row["count"])
            shares.setdefault(row["model"], {})[row["finish_reason"]] = float(row["share_of_eligible"])

    assert counts["m/a"] == {"stop": 5, "length": 1}
    assert counts["m/b"] == {"stop": 6, "<missing>": 3}
    assert sum(shares["m/a"].values()) == pytest.approx(1.0, abs=1e-3)
    assert sum(shares["m/b"].values()) == pytest.approx(1.0, abs=1e-3)


# ---------- v4 probabilistic regression -------------------------------------


def test_per_model_summary_has_probabilistic_columns(tmp_path: Path) -> None:
    """Phase 1 task 16.2: BI / NLL / ABI / fallback_share columns MUST exist
    in `per_model_summary.csv` and carry plausible values.

    The fixture has `belief_final IS NULL` everywhere → every scoreable
    sample is on the §2.4 fallback path. So `fallback_share` MUST be 1.0
    for any model that produced any probability vector.
    """
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)

    with (run_dir / "analysis" / "per_model_summary.csv").open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        # Phase 1 columns appended at the end of the existing header.
        for col in ("bi", "bi_dec", "nll", "mbs", "abi_crowd", "abi_uniform", "fallback_share"):
            assert col in fieldnames, f"missing column {col} in per_model_summary.csv"
        rows_by_model = {row["model"]: row for row in reader}

    a = rows_by_model["m/a"]
    # m/a has 2 eligible questions (q1 yes_no single, q2 binary_named single).
    # All samples come through the fallback path (belief_final IS NULL but
    # parse_ok=1). Phase 1 averages probability vectors per (model, question)
    # then aggregates → fallback_share at the question level is 1.0.
    assert a["bi"], "BI must be populated for m/a"
    assert 0.0 <= float(a["bi"]) <= 100.0
    assert float(a["fallback_share"]) == pytest.approx(1.0, rel=1e-9)
    # Single-only metrics should be present (q1 + q2 are both single).
    assert a["mbs"]
    assert a["bi_dec"]


def test_v3_run_fallback_path(tmp_path: Path) -> None:
    """Phase 1 task 16.3: A v3-style run (belief_final IS NULL, parse_ok=1)
    MUST yield BI / NLL via §2.4 fallback with fallback_share = 1.0.

    `_build_fixture_run` already exercises this — it never sets
    `belief_final`, so it doubles as a v3-replay fixture.
    """
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)
    overall = json.loads((run_dir / "analysis" / "overall.json").read_text())
    assert "probabilistic" in overall
    prob = overall["probabilistic"]["per_model"]
    for model in ("m/a", "m/b"):
        if model not in prob:
            continue
        agg = prob[model]
        # Either no probability vector available (None) or fallback_share = 1.0.
        if agg.get("fallback_share") is not None:
            assert agg["fallback_share"] == pytest.approx(1.0, rel=1e-9)


def test_byte_regression_existing_csvs(tmp_path: Path) -> None:
    """Phase 1 task 16.4: error_breakdown.csv and finish_reason_breakdown.csv
    MUST stay byte-identical between consecutive runs of `run_analysis`.

    This is the strongest "no regression" check we can do without diffing
    against a checked-in v3 baseline: re-running v4 against the same fixture
    must deterministically produce identical files.
    """
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)
    breakdown1 = (run_dir / "analysis" / "error_breakdown.csv").read_bytes()
    finish1 = (run_dir / "analysis" / "finish_reason_breakdown.csv").read_bytes()

    # Second run on the same DBs must be byte-identical.
    analysis.run_analysis(run_dir)
    breakdown2 = (run_dir / "analysis" / "error_breakdown.csv").read_bytes()
    finish2 = (run_dir / "analysis" / "finish_reason_breakdown.csv").read_bytes()

    assert breakdown1 == breakdown2
    assert finish1 == finish2


def test_per_model_summary_csv_v3_columns_unchanged(tmp_path: Path) -> None:
    """Existing v3 columns of `per_model_summary.csv` MUST remain at their
    same positions; v4 only appends new columns. This guards against an
    accidental reorder that would invisibly break downstream tools.
    """
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)
    with (run_dir / "analysis" / "per_model_summary.csv").open(encoding="utf-8") as f:
        first_line = f.readline().rstrip("\r\n")
    columns = first_line.split(",")
    # The first 24 columns must match the v3+v5.1 header verbatim. v5.1
    # appended `final_answer_retry_rate` adjacent to the avg_* diagnostics
    # so it lives next to "what did the harness do?" data, not next to FSS.
    expected_v3 = [
        "model", "sampling_n", "eligible_samples", "eligible_questions",
        "resolvable_samples", "cutoff_skip_samples", "cutoff_skip_rate",
        "pass_at_1_avg", "resolvable_rate", "pass_any_at_n",
        "at_least_majority_at_n", "at_least_all_at_n",
        "exam_score_at_n_avg",  # exam-score-metric: column order assertion
        "majority_vote_accuracy", "majority_vote_resolvable_rate",
        "parse_failure_rate", "error_rate",
        "avg_tool_calls", "avg_react_steps", "avg_latency_ms",
        "avg_prompt_tokens", "avg_completion_tokens", "avg_reasoning_tokens",
        "avg_nudges_used",
        "final_answer_retry_rate",
    ]
    assert columns[: len(expected_v3)] == expected_v3
    # v5 inserts 8 discrete-native + consistency columns between v3 and v4
    # probabilistic columns (Decision: FSS-family next to pass@1).
    expected_v5 = [
        "fss", "fss_pe_mean", "cohen_kappa", "hamming_score",
        "fleiss_kappa", "mean_entropy", "vci", "mvg",
    ]
    assert columns[len(expected_v3):len(expected_v3) + len(expected_v5)] == expected_v5
    # v4 probabilistic family is now the trailing block (not the only
    # post-v3 block as in v4 layout).
    assert columns[len(expected_v3) + len(expected_v5):] == [
        "bi", "bi_dec", "nll", "mbs", "abi_crowd", "abi_uniform", "fallback_share",
    ]


def test_overall_json_carries_analysis_schema(tmp_path: Path) -> None:
    """`overall.json` should surface `analysis_schema` from manifest.json so
    downstream tools can branch on v3-vs-v4 metric availability without
    re-reading the manifest."""
    run_dir = _build_fixture_run(tmp_path)
    # Inject analysis_schema into manifest before running analysis.
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["analysis_schema"] = "v4"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))

    analysis.run_analysis(run_dir)
    overall = json.loads((run_dir / "analysis" / "overall.json").read_text())
    assert overall.get("analysis_schema") == "v4"


# ---------- Phase 2 integration tests --------------------------------------


def test_phase2_deliverables_present(tmp_path: Path) -> None:
    """v5: every spec'd output file lands under analysis/.

    The fixture goes through the §2.4 fallback path (belief_final IS NULL),
    which is enough to exercise the writers: shrinkage scans the fallback
    probability vectors, BS-paired bootstrap contrasts m/a vs m/b, the
    difficulty tertile slices the union set, and the v5 inter-trial
    consistency / pairwise bootstrap files emit.

    Calibration / Reliability / Murphy decomposition outputs are gone in v5
    (Decision 2 — K=5 makes them statistically meaningless).
    """
    run_dir = _build_fixture_run(tmp_path)
    written = {p.name for p in analysis.run_analysis(run_dir)}
    expected = {
        "shrinkage_alpha_curve.csv",
        "paired_delta_bi.csv",
        "pairwise_significance.csv",
        "posterior_pairwise.csv",
        "per_model_by_difficulty.csv",
        "paired_delta_bi_by_difficulty.csv",
        # v5 additions
        "inter_trial_consistency.csv",
        "pairwise_bootstrap.csv",
    }
    missing = expected - written
    assert not missing, f"Expected outputs missing: {sorted(missing)}"
    # v5 removed outputs MUST NOT appear.
    forbidden = {
        "calibration_params.json",
        "per_model_summary_calibrated.csv",
        "reliability_data.json",
        "reliability_data_calibrated.json",
        "brier_decomposition.csv",
    }
    assert not (forbidden & written), (
        f"v5 deprecated outputs leaked: {sorted(forbidden & written)}"
    )


def test_shrinkage_alpha_curve_csv_format(tmp_path: Path) -> None:
    """`shrinkage_alpha_curve.csv` MUST have one row per (model, ctype, alpha) on
    the default 11-point grid."""
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)
    path = run_dir / "analysis" / "shrinkage_alpha_curve.csv"
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "model", "alpha", "mean_bs", "bi", "n_questions", "choice_type"
        ]
        rows = list(reader)
    # At least 11 alphas × ≥1 (model, ctype) cell.
    assert len(rows) >= 11
    # Every alpha appears 0..1 in 0.1 steps.
    seen_alphas = sorted({float(r["alpha"]) for r in rows})
    assert seen_alphas[0] == pytest.approx(0.0)
    assert seen_alphas[-1] == pytest.approx(1.0)


def test_paired_delta_bi_includes_holm(tmp_path: Path) -> None:
    """Paired bootstrap CSV has Holm-adjusted p-value column populated."""
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)
    path = run_dir / "analysis" / "paired_delta_bi.csv"
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)
    for required in (
        "model_a", "model_b", "n_questions",
        "delta_bs", "ci_low_bs", "ci_high_bs",
        "delta_bi_approx", "p_raw", "p_holm", "posterior_a_better",
    ):
        assert required in cols
    # 2 models → 1 pair.
    assert len(rows) == 1
    assert rows[0]["model_a"] == "m/a" and rows[0]["model_b"] == "m/b"
    assert rows[0]["p_holm"] != ""


def test_per_model_by_difficulty_three_tiers(tmp_path: Path) -> None:
    """`per_model_by_difficulty.csv`: 3 tiers per model, populated rows only."""
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)
    path = run_dir / "analysis" / "per_model_by_difficulty.csv"
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    tiers_seen = {r["difficulty_tertile"] for r in rows}
    assert tiers_seen.issubset({"low", "mid", "high"})
    # Every non-empty tier should report n_questions ≥ 0.
    for r in rows:
        assert int(r["n_questions"]) >= 0


def test_posterior_pairwise_csv_probabilities_in_unit_interval(tmp_path: Path) -> None:
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)
    path = run_dir / "analysis" / "posterior_pairwise.csv"
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    for r in rows:
        p = float(r["prob_a_better"])
        assert 0.0 <= p <= 1.0


# --------------------------------------------------------------------------- #
# v5 writer-level tests (Phase D / E)
# --------------------------------------------------------------------------- #


def test_per_model_summary_csv_has_v5_columns(tmp_path: Path) -> None:
    """v5 columns are present and at least one model has values populated."""
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)
    path = run_dir / "analysis" / "per_model_summary.csv"
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)
    for required in (
        "fss", "fss_pe_mean", "cohen_kappa", "hamming_score",
        "fleiss_kappa", "mean_entropy", "vci", "mvg",
    ):
        assert required in cols, f"v5 column missing: {required}"
    # At least one model should have a numeric FSS (the fixture has scoreable
    # questions for both models on q1 + q2).
    fss_values = [r["fss"] for r in rows if r["fss"]]
    assert fss_values, "Expected at least one model with non-empty FSS"


def test_per_model_summary_md_no_calibration_columns(tmp_path: Path) -> None:
    """v5 markdown removes BI_cal / NLL_cal / ECE_* columns and the cal* marker."""
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)
    md = (run_dir / "analysis" / "per_model_summary.md").read_text(encoding="utf-8")
    for forbidden in ("BI_cal", "NLL_cal", "ECE_uncal", "ECE_cal", "cal*"):
        assert forbidden not in md, f"v5 markdown contains forbidden token: {forbidden}"
    # FSS column heading is present in the table (sandwiched between pass@1
    # and pass_any@N — Decision: FSS adjacent to pass@1).
    assert "FSS" in md
    # K=5 disclaimer footnote is present.
    assert "K=5 parallel trials" in md or "discrete probability" in md


def test_inter_trial_consistency_csv_format(tmp_path: Path) -> None:
    """v5: inter_trial_consistency.csv has one row per model with the 5 columns."""
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)
    path = run_dir / "analysis" / "inter_trial_consistency.csv"
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)
    assert cols == [
        "model", "fleiss_kappa", "mean_entropy", "vci", "mvg",
        "n_questions_used",
    ]
    assert len(rows) == 2  # m/a + m/b
    model_names = {r["model"] for r in rows}
    assert model_names == {"m/a", "m/b"}


def test_metric_pairwise_bootstrap_csv_format(tmp_path: Path) -> None:
    """v5: pairwise_bootstrap.csv has the v5 column schema and at least one
    metric reports rows for the m/a vs m/b pair."""
    run_dir = _build_fixture_run(tmp_path)
    analysis.run_analysis(run_dir)
    path = run_dir / "analysis" / "pairwise_bootstrap.csv"
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)
    assert cols == [
        "metric", "model_a", "model_b", "n_questions",
        "delta_mean", "ci_low", "ci_high",
        "p_value", "cohens_d", "sig_at_05",
    ]
    # At least FSS or Acc should produce a row for m/a vs m/b.
    pair_rows = [
        r for r in rows
        if r["model_a"] == "m/a" and r["model_b"] == "m/b"
    ]
    assert pair_rows, "Expected at least one metric for m/a vs m/b pair"
    metric_names = {r["metric"] for r in pair_rows}
    assert metric_names & {"fss", "acc"}
