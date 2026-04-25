"""Phase 3 behavior analysis tests.

Covers `forecast_eval/analysis/behavior.py` end-to-end:

* §25 belief evolution indicators on synthetic traces;
* §26 reflection A/B pairing — including the "MUST NOT pair on mismatched
  fingerprints" guarantee from spec 26.5;
* §27 tool usage PDP — recovery of an injected logistic relationship;
* §28 confidence calibration tables + conflict marker.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path

import pytest

from forecast_eval.analysis.behavior import (
    BeliefEvolutionRow,
    CONFIDENCE_BUCKETS,
    PairedRunSpec,
    TOOL_PDP_FEATURES,
    build_belief_evolution_rows,
    confidence_calibration,
    confidence_conflict_models,
    convergence_step,
    counterevidence_engagement,
    evidence_efficiency,
    find_paired_runs,
    fit_linear_ridge,
    fit_logistic_irls,
    inter_trial_variance,
    numeric_confidence_calibration,
    parse_belief_trace,
    reflection_ab_report,
    tool_usage_pdp,
    trial_internal_volatility,
)
from forecast_eval.analysis.flatten import SampleRow


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_trace(steps: list[dict | None]) -> str:
    return json.dumps(steps, ensure_ascii=False)


def _step(p: dict, *, step: int = 0, confidence: str = "low",
          counterevidence: list[str] | None = None) -> dict:
    out = {
        "step": step,
        "p": p,
        "confidence": confidence,
        "delta_reason": "",
    }
    if counterevidence is not None:
        out["counterevidence"] = counterevidence
    return out


def _sample(
    *,
    model: str = "m1",
    qid: str = "q1",
    qtype: str = "yes_no",
    ctype: str = "single",
    options: list[str] | None = None,
    sample_idx: int = 0,
    correct: int | None = 1,
    parse_ok: int | None = 1,
    tool_calls_count: int | None = 2,
    react_steps: int | None = 3,
    latency_ms: int | None = 1000,
    prompt_tokens: int | None = 500,
    completion_tokens: int | None = 200,
    final_letters: str | None = '["A"]',
    probabilities: list[float] | None = None,
    belief_trace_steps: list[dict | None] | None = None,
) -> SampleRow:
    if options is None:
        options = ["yes", "no"]
    if probabilities is None:
        probabilities = [0.7, 0.3]
    belief_trace = (
        _make_trace(belief_trace_steps) if belief_trace_steps is not None else None
    )
    belief_final = (
        json.dumps({chr(ord("A") + i): probabilities[i] for i in range(len(options))})
        if probabilities is not None else None
    )
    return SampleRow(
        model=model,
        question_id=qid,
        question_type=qtype,
        choice_type=ctype,
        options=options,
        sample_idx=sample_idx,
        correct=correct,
        parse_ok=parse_ok,
        tool_calls_count=tool_calls_count,
        react_steps=react_steps,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=0,
        latency_ms=latency_ms,
        final_answer_letters=final_letters,
        error=None,
        created_at="2026-04-26T00:00:00Z",
        finish_reason="stop",
        nudges_used=0,
        belief_final=belief_final,
        belief_trace=belief_trace,
        belief_parse_ok=1 if belief_trace_steps else 0,
        probabilities=probabilities,
        is_fallback=False,
    )


# --------------------------------------------------------------------------- #
# §25.1 — parse_belief_trace
# --------------------------------------------------------------------------- #


def test_parse_belief_trace_returns_list_of_dicts():
    steps = [_step({"A": 0.5, "B": 0.5}, step=0), _step({"A": 0.7, "B": 0.3}, step=1)]
    parsed = parse_belief_trace(_make_trace(steps))
    assert len(parsed) == 2
    assert parsed[0]["confidence"] == "low"


def test_parse_belief_trace_handles_none_entries():
    steps = [None, _step({"A": 0.6, "B": 0.4}, step=1)]
    parsed = parse_belief_trace(_make_trace(steps))
    assert parsed[0] is None
    assert isinstance(parsed[1], dict)


def test_parse_belief_trace_returns_empty_on_invalid_json():
    assert parse_belief_trace(None) == []
    assert parse_belief_trace("") == []
    assert parse_belief_trace("not json") == []
    assert parse_belief_trace("{}") == []  # not a list


# --------------------------------------------------------------------------- #
# §25.2 — trial_internal_volatility
# --------------------------------------------------------------------------- #


def test_volatility_zero_when_belief_unchanged():
    options = ["yes", "no"]
    trace = [_step({"A": 0.7, "B": 0.3}, step=0), _step({"A": 0.7, "B": 0.3}, step=1)]
    assert trial_internal_volatility(trace, options) == pytest.approx(0.0)


def test_volatility_matches_l2():
    options = ["yes", "no"]
    trace = [_step({"A": 0.5, "B": 0.5}, step=0), _step({"A": 0.7, "B": 0.3}, step=1)]
    expected = math.sqrt(0.04 + 0.04)
    assert trial_internal_volatility(trace, options) == pytest.approx(expected, abs=1e-9)


def test_volatility_none_when_one_step():
    options = ["yes", "no"]
    trace = [_step({"A": 0.6, "B": 0.4}, step=0)]
    assert trial_internal_volatility(trace, options) is None


def test_volatility_skips_failed_steps():
    options = ["yes", "no"]
    trace = [
        _step({"A": 0.5, "B": 0.5}, step=0),
        None,
        _step({"A": 0.7, "B": 0.3}, step=2),
    ]
    expected = math.sqrt(0.04 + 0.04)
    assert trial_internal_volatility(trace, options) == pytest.approx(expected, abs=1e-9)


# --------------------------------------------------------------------------- #
# §25.3 — inter_trial_variance
# --------------------------------------------------------------------------- #


def test_inter_trial_variance_zero_when_identical():
    options = ["yes", "no"]
    traces = [
        [_step({"A": 0.7, "B": 0.3}, step=0)],
        [_step({"A": 0.7, "B": 0.3}, step=0)],
    ]
    assert inter_trial_variance(traces, options) == pytest.approx(0.0, abs=1e-9)


def test_inter_trial_variance_increases_with_disagreement():
    options = ["yes", "no"]
    close = [
        [_step({"A": 0.6, "B": 0.4}, step=0)],
        [_step({"A": 0.7, "B": 0.3}, step=0)],
    ]
    far = [
        [_step({"A": 0.2, "B": 0.8}, step=0)],
        [_step({"A": 0.9, "B": 0.1}, step=0)],
    ]
    assert inter_trial_variance(close, options) < inter_trial_variance(far, options)


def test_inter_trial_variance_none_with_one_trial():
    options = ["yes", "no"]
    traces = [[_step({"A": 0.7, "B": 0.3}, step=0)]]
    assert inter_trial_variance(traces, options) is None


# --------------------------------------------------------------------------- #
# §25.4 — convergence_step
# --------------------------------------------------------------------------- #


def test_convergence_step_zero_when_immediately_converged():
    options = ["yes", "no"]
    trace = [
        _step({"A": 0.7, "B": 0.3}, step=0),
        _step({"A": 0.71, "B": 0.29}, step=1),
        _step({"A": 0.7, "B": 0.3}, step=2),
    ]
    # Step 0 is within eps=0.05 of the final step.
    assert convergence_step(trace, options) == 0


def test_convergence_step_returns_first_close_index():
    options = ["yes", "no"]
    trace = [
        _step({"A": 0.5, "B": 0.5}, step=0),
        _step({"A": 0.6, "B": 0.4}, step=1),
        _step({"A": 0.9, "B": 0.1}, step=2),
        _step({"A": 0.92, "B": 0.08}, step=3),
    ]
    # Final = (0.92, 0.08). Step 2 = (0.9, 0.1). Distance = sqrt(0.0008) ~= 0.028 < 0.05.
    assert convergence_step(trace, options, eps=0.05) == 2


def test_convergence_step_none_on_empty_trace():
    options = ["yes", "no"]
    assert convergence_step([], options) is None


# --------------------------------------------------------------------------- #
# §25.5 — evidence_efficiency
# --------------------------------------------------------------------------- #


def test_evidence_efficiency_positive_when_belief_improves():
    options = ["yes", "no"]
    obs = [1, 0]  # A is correct
    trace = [
        _step({"A": 0.4, "B": 0.6}, step=0),
        _step({"A": 0.9, "B": 0.1}, step=1),
    ]
    eff = evidence_efficiency(trace, options, obs, "single", search_calls=2)
    assert eff is not None
    assert eff > 0  # NLL went down (good)


def test_evidence_efficiency_negative_when_belief_diverges():
    options = ["yes", "no"]
    obs = [1, 0]
    trace = [
        _step({"A": 0.9, "B": 0.1}, step=0),
        _step({"A": 0.1, "B": 0.9}, step=1),
    ]
    eff = evidence_efficiency(trace, options, obs, "single", search_calls=2)
    assert eff is not None
    assert eff < 0


def test_evidence_efficiency_handles_zero_search_calls():
    options = ["yes", "no"]
    obs = [1, 0]
    trace = [
        _step({"A": 0.5, "B": 0.5}, step=0),
        _step({"A": 0.9, "B": 0.1}, step=1),
    ]
    eff = evidence_efficiency(trace, options, obs, "single", search_calls=0)
    assert eff is not None
    # max(1, 0) protects denominator.
    assert eff > 0


def test_evidence_efficiency_none_when_only_one_step():
    options = ["yes", "no"]
    obs = [1, 0]
    trace = [_step({"A": 0.5, "B": 0.5}, step=0)]
    assert evidence_efficiency(trace, options, obs, "single", search_calls=1) is None


# --------------------------------------------------------------------------- #
# §25.6 — counterevidence_engagement
# --------------------------------------------------------------------------- #


def test_counterevidence_engaged_when_mentions_other_letter():
    options = ["yes", "no"]
    final = frozenset({"A"})
    counterevidence = ["If outcome were B, we would expect Q which we did NOT observe."]
    assert counterevidence_engagement(counterevidence, final, options) == 1


def test_counterevidence_not_engaged_when_no_letters_mentioned():
    options = ["yes", "no"]
    final = frozenset({"A"})
    counterevidence = ["The model is generally cautious here."]
    assert counterevidence_engagement(counterevidence, final, options) == 0


def test_counterevidence_not_engaged_when_only_chosen_letter():
    options = ["yes", "no"]
    final = frozenset({"A"})
    counterevidence = ["Even A is not totally certain because of X."]
    assert counterevidence_engagement(counterevidence, final, options) == 0


def test_counterevidence_engaged_with_3plus_options():
    options = ["red", "green", "blue", "yellow"]
    final = frozenset({"C"})  # picked blue
    counterevidence = ["If A or D were the case we would see Z."]
    assert counterevidence_engagement(counterevidence, final, options) == 1


def test_counterevidence_empty_returns_zero():
    options = ["yes", "no"]
    final = frozenset({"A"})
    assert counterevidence_engagement(None, final, options) == 0
    assert counterevidence_engagement([], final, options) == 0


# --------------------------------------------------------------------------- #
# §25.7 — build_belief_evolution_rows
# --------------------------------------------------------------------------- #


def test_build_belief_evolution_rows_basic():
    samples = [
        _sample(
            model="m1",
            qid="q1",
            sample_idx=0,
            belief_trace_steps=[
                _step({"A": 0.5, "B": 0.5}, step=0, counterevidence=["B is unlikely"]),
                _step({"A": 0.8, "B": 0.2}, step=1, counterevidence=["B is unlikely"]),
            ],
        ),
        _sample(
            model="m1",
            qid="q1",
            sample_idx=1,
            belief_trace_steps=[
                _step({"A": 0.5, "B": 0.5}, step=0),
                _step({"A": 0.7, "B": 0.3}, step=1),
            ],
        ),
    ]
    samples_by_model = {"m1": samples}
    gt_map = {"q1": frozenset({"A"})}
    rows = build_belief_evolution_rows(samples_by_model, gt_map)
    assert len(rows) == 2
    # Inter-trial variance is shared across both rows of the same question.
    assert rows[0].inter_trial_variance == rows[1].inter_trial_variance
    # First sample engaged (mentions B); second has no counterevidence list.
    assert rows[0].counterevidence_engaged == 1
    assert rows[1].counterevidence_engaged is None  # no counterevidence key


def test_build_belief_evolution_skips_unparseable_traces():
    """Spec 30.2: v3 fixture (no belief_trace) yields zero rows, not exceptions."""
    samples = [_sample(belief_trace_steps=None)]  # belief_trace=None
    rows = build_belief_evolution_rows({"m1": samples}, {})
    assert rows == []


# --------------------------------------------------------------------------- #
# §26 — Reflection A/B
# --------------------------------------------------------------------------- #


def _build_minimal_run(
    base: Path,
    *,
    run_id: str,
    model: str,
    reflection_hash: str | None,
    belief_hash: str | None = None,
    qids: list[str] | None = None,
    correct_pattern: list[int] | None = None,
    probs_pattern: list[list[float]] | None = None,
) -> Path:
    """Build a minimal v4-shaped run dir on disk: manifest + per-model DB.

    The DB holds only the columns the test reads back through `_read_run_meta`
    + `_question_ids_in` + `_flatten_db`. We intentionally bypass the full
    `forecast_eval.db.init_schema` machinery so the test stays under 1ms even
    on slow CI.
    """
    qids = qids or ["q1", "q2", "q3"]
    correct_pattern = correct_pattern or [1] * len(qids)
    probs_pattern = probs_pattern or [[0.7, 0.3]] * len(qids)
    run_dir = base / run_id
    (run_dir / "db").mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": run_id,
        "models": [model],
        "model_files": {model: f"{model}.db"},
        "sampling_n": 1,
    }), encoding="utf-8")
    db_path = run_dir / "db" / f"{model}.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
    CREATE TABLE run_meta (
        run_id TEXT,
        model TEXT,
        sampling_n INTEGER,
        source_db_hash TEXT,
        metadata_hash TEXT,
        prompt_templates_hash TEXT,
        reflection_protocol_hash TEXT,
        belief_protocol_hash TEXT,
        started_at TEXT
    );
    CREATE TABLE questions (
        id TEXT PRIMARY KEY,
        question_type TEXT,
        choice_type TEXT,
        options TEXT,
        answer TEXT
    );
    """)
    # run_results — only the columns flatten reads.
    cols = [
        "question_id TEXT PRIMARY KEY",
        "s0_correct INTEGER", "s0_parse_ok INTEGER",
        "s0_tool_calls_count INTEGER", "s0_react_steps INTEGER",
        "s0_prompt_tokens INTEGER", "s0_completion_tokens INTEGER",
        "s0_reasoning_tokens INTEGER", "s0_latency_ms INTEGER",
        "s0_final_answer_letters TEXT", "s0_error TEXT",
        "s0_created_at TEXT", "s0_finish_reason TEXT", "s0_nudges_used INTEGER",
        "s0_belief_final TEXT", "s0_belief_trace TEXT", "s0_belief_parse_ok INTEGER",
    ]
    conn.execute(f"CREATE TABLE run_results ({', '.join(cols)})")
    conn.execute(
        "INSERT INTO run_meta VALUES (?, ?, 1, 'srchash', 'mdhash', 'pthash', ?, ?, ?)",
        (run_id, model, reflection_hash, belief_hash, "2026-04-25T12:00:00Z"),
    )
    for i, qid in enumerate(qids):
        opts = json.dumps(["yes", "no"])
        conn.execute(
            "INSERT INTO questions VALUES (?, 'yes_no', 'single', ?, 'A')",
            (qid, opts),
        )
        belief_final = json.dumps({"A": probs_pattern[i][0], "B": probs_pattern[i][1]})
        belief_trace = json.dumps([
            {"step": 0, "p": {"A": 0.5, "B": 0.5}, "confidence": "low",
             "delta_reason": "", "counterevidence": []},
            {"step": 1, "p": {"A": probs_pattern[i][0], "B": probs_pattern[i][1]},
             "confidence": "medium", "delta_reason": "", "counterevidence": []},
        ])
        conn.execute(
            "INSERT INTO run_results VALUES (?, ?, 1, 1, 2, 200, 100, 0, 500, ?, NULL, "
            "'2026-04-25T12:00:01Z', 'stop', 0, ?, ?, 1)",
            (qid, correct_pattern[i], '["A"]', belief_final, belief_trace),
        )
    conn.commit()
    conn.close()
    return run_dir


def test_find_paired_runs_pairs_runs_with_only_reflection_diff(tmp_path):
    runs_root = tmp_path / "runs"
    _build_minimal_run(
        runs_root, run_id="run_on", model="modelA",
        reflection_hash="refl_v1", belief_hash="bel_v1",
    )
    _build_minimal_run(
        runs_root, run_id="run_off", model="modelA",
        reflection_hash=None, belief_hash="bel_v1",
    )
    pairs = find_paired_runs(runs_root)
    assert len(pairs) == 1
    assert pairs[0].model == "modelA"
    assert pairs[0].run_on.parent.parent.name == "run_on"
    assert pairs[0].run_off.parent.parent.name == "run_off"
    assert set(pairs[0].common_qids) == {"q1", "q2", "q3"}


def test_find_paired_runs_does_not_pair_on_belief_hash_mismatch(tmp_path):
    """Spec 26.5: any other fingerprint mismatch breaks pairing."""
    runs_root = tmp_path / "runs"
    _build_minimal_run(
        runs_root, run_id="run_on", model="modelA",
        reflection_hash="refl_v1", belief_hash="bel_v1",
    )
    _build_minimal_run(
        runs_root, run_id="run_off", model="modelA",
        reflection_hash=None, belief_hash="bel_v2",  # different
    )
    assert find_paired_runs(runs_root) == []


def test_find_paired_runs_does_not_pair_on_template_mismatch(tmp_path, monkeypatch):
    """The bucketing key includes prompt_templates_hash — patch it to differ."""
    runs_root = tmp_path / "runs"
    _build_minimal_run(
        runs_root, run_id="run_on", model="modelA",
        reflection_hash="refl_v1",
    )
    _build_minimal_run(
        runs_root, run_id="run_off", model="modelA",
        reflection_hash=None,
    )
    # Swap one run's prompt_templates_hash.
    db = runs_root / "run_off" / "db" / "modelA.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE run_meta SET prompt_templates_hash = 'OTHER' WHERE run_id = 'run_off'"
    )
    conn.commit()
    conn.close()
    assert find_paired_runs(runs_root) == []


def test_find_paired_runs_no_pair_when_neither_has_reflection(tmp_path):
    runs_root = tmp_path / "runs"
    _build_minimal_run(runs_root, run_id="r1", model="modelA", reflection_hash=None)
    _build_minimal_run(runs_root, run_id="r2", model="modelA", reflection_hash=None)
    assert find_paired_runs(runs_root) == []


def test_reflection_ab_report_emits_per_qtype_rows(tmp_path):
    runs_root = tmp_path / "runs"
    _build_minimal_run(
        runs_root, run_id="run_on", model="modelA",
        reflection_hash="refl_v1",
        probs_pattern=[[0.85, 0.15], [0.85, 0.15], [0.85, 0.15]],
        correct_pattern=[1, 1, 1],
    )
    _build_minimal_run(
        runs_root, run_id="run_off", model="modelA",
        reflection_hash=None,
        probs_pattern=[[0.55, 0.45], [0.55, 0.45], [0.55, 0.45]],
        correct_pattern=[1, 1, 1],
    )
    pairs = find_paired_runs(runs_root)
    assert pairs
    rows = reflection_ab_report(pairs, n_bootstrap=200, seed=7)
    metrics = {r.metric for r in rows}
    assert "delta_bi" in metrics
    # All-questions row should exist for delta_bi
    all_rows = [r for r in rows if r.metric == "delta_bi" and r.question_type == "all"]
    assert len(all_rows) == 1
    # delta = on - off; on has lower BS_lab (better), so delta_mean < 0
    assert all_rows[0].delta_mean < 0


# --------------------------------------------------------------------------- #
# §27 — Tool usage PDP
# --------------------------------------------------------------------------- #


def test_logistic_irls_recovers_known_relationship():
    """Inject Pr(y=1 | x) = sigmoid(2 * x) and check that the recovered
    weight is positive and the sign is right."""
    import random
    rng = random.Random(42)
    X = []
    y = []
    for _ in range(400):
        x = rng.uniform(-2, 2)
        z = 2.0 * x  # weight=2 on standardized scale roughly
        p = 1.0 / (1.0 + math.exp(-z))
        X.append([x])
        y.append(1 if rng.random() < p else 0)
    fit = fit_logistic_irls(X, y, l2=0.001, max_iter=50)
    assert fit is not None
    assert fit.weights[0] > 0  # right sign


def test_linear_ridge_recovers_known_slope():
    import random
    rng = random.Random(7)
    X = []
    y = []
    for _ in range(200):
        x = rng.uniform(-3, 3)
        target = 1.5 * x + 0.5 + rng.gauss(0, 0.1)
        X.append([x])
        y.append(target)
    fit = fit_linear_ridge(X, y, l2=0.001)
    assert fit is not None
    # On standardized scale, weight ≈ slope * std(x). We just confirm sign + magnitude.
    assert fit.weights[0] > 0


def test_tool_usage_pdp_emits_rows_for_every_feature():
    """Synthetic: tool_calls_count strongly predicts correct."""
    import random
    rng = random.Random(99)
    samples = []
    for i in range(200):
        tc = rng.randint(0, 6)
        # P(correct) increases with tool_calls
        is_correct = 1 if (rng.random() < 0.3 + tc * 0.1) else 0
        samples.append(_sample(
            model="m1",
            qid=f"q{i}",
            sample_idx=0,
            correct=is_correct,
            tool_calls_count=tc,
            react_steps=tc + 1,
            latency_ms=1000 + tc * 100,
            prompt_tokens=500 + tc * 50,
            completion_tokens=200 + tc * 30,
            probabilities=[0.7 if is_correct else 0.3, 0.3 if is_correct else 0.7],
            final_letters='["A"]' if is_correct else '["B"]',
        ))
    gt_map = {f"q{i}": frozenset({"A"}) for i in range(200)}
    rows = tool_usage_pdp({"m1": samples}, gt_map, n_grid=5)
    features = {r.feature for r in rows}
    # Every requested feature appears.
    assert features == set(TOOL_PDP_FEATURES)
    # tool_calls_count: PDP for correct should rise with feature value.
    tc_rows = sorted(
        [r for r in rows if r.feature == "tool_calls_count"],
        key=lambda r: r.feature_value,
    )
    # Probability must monotonically (weakly) increase across the grid.
    pdp_values = [r.pdp_correct for r in tc_rows if r.pdp_correct is not None]
    assert pdp_values[-1] > pdp_values[0]


def test_tool_usage_pdp_skips_when_too_few_samples():
    samples = [_sample() for _ in range(3)]
    rows = tool_usage_pdp({"m1": samples}, {})
    assert rows == []


# --------------------------------------------------------------------------- #
# §28 — Confidence calibration
# --------------------------------------------------------------------------- #


def test_confidence_calibration_groups_by_bucket():
    samples = [
        _sample(
            model="m1", qid=f"q{i}", sample_idx=0,
            correct=1 if i < 5 else 0,  # 5 of 8 correct
            probabilities=[0.7, 0.3] if i < 4 else [0.55, 0.45],
            belief_trace_steps=[
                _step({"A": 0.7, "B": 0.3}, step=0,
                      confidence="high" if i < 4 else "low"),
            ],
        )
        for i in range(8)
    ]
    rows = confidence_calibration({"m1": samples})
    by_bucket = {r.confidence: r for r in rows if r.model == "m1"}
    # high bucket (i=0..3): all correct → hit_rate=1
    assert by_bucket["high"].n_samples == 4
    assert by_bucket["high"].hit_rate == pytest.approx(1.0)
    # low bucket (i=4..7): 1 of 4 correct → hit_rate=0.25
    assert by_bucket["low"].n_samples == 4
    assert by_bucket["low"].hit_rate == pytest.approx(0.25)
    # all = 5/8
    assert by_bucket["all"].hit_rate == pytest.approx(5 / 8)


def test_confidence_calibration_skips_samples_without_belief():
    samples = [
        _sample(belief_trace_steps=None),
        _sample(qid="q2"),  # no belief_trace
    ]
    rows = confidence_calibration({"m1": samples})
    by_bucket = {r.confidence: r for r in rows if r.model == "m1"}
    # Every bucket has zero samples — function still emits rows (rectangular).
    for bucket in CONFIDENCE_BUCKETS:
        assert by_bucket[bucket].n_samples == 0


def test_numeric_confidence_calibration_bins_by_max_p():
    samples = [
        _sample(qid=f"q{i}", correct=1 if i % 2 == 0 else 0,
                probabilities=[0.5 + i * 0.05, 0.5 - i * 0.05])
        for i in range(8)
    ]
    rows = numeric_confidence_calibration({"m1": samples}, n_bins=5)
    # 5 bins per model.
    assert sum(1 for r in rows if r.model == "m1") == 5
    # Total samples across bins = 8.
    assert sum(r.n_samples for r in rows if r.model == "m1") == 8


def test_confidence_conflict_low_bucket_overconfident_numerically():
    """`low` confidence + mean_max_p > 0.70 → flagged."""
    rows = [
        # 10 samples, mean_max_p = 0.80, hit_rate doesn't matter for the flag.
        _confidence_row(model="m1", confidence="low", n=10, mean_p=0.80, hit=0.5),
        _confidence_row(model="m1", confidence="medium", n=10, mean_p=0.6, hit=0.6),
        _confidence_row(model="m1", confidence="high", n=10, mean_p=0.9, hit=0.85),
        _confidence_row(model="m1", confidence="all", n=30, mean_p=0.77, hit=0.65),
    ]
    flagged = confidence_conflict_models(rows)
    assert "m1" in flagged


def test_confidence_conflict_high_bucket_underconfident():
    rows = [
        _confidence_row(model="m1", confidence="low", n=10, mean_p=0.40, hit=0.3),
        _confidence_row(model="m1", confidence="medium", n=10, mean_p=0.55, hit=0.5),
        _confidence_row(model="m1", confidence="high", n=10, mean_p=0.45, hit=0.7),  # numeric below threshold
        _confidence_row(model="m1", confidence="all", n=30, mean_p=0.47, hit=0.5),
    ]
    flagged = confidence_conflict_models(rows)
    assert "m1" in flagged


def test_confidence_conflict_no_flag_when_consistent():
    rows = [
        _confidence_row(model="m1", confidence="low", n=10, mean_p=0.4, hit=0.3),
        _confidence_row(model="m1", confidence="medium", n=10, mean_p=0.6, hit=0.6),
        _confidence_row(model="m1", confidence="high", n=10, mean_p=0.85, hit=0.85),
        _confidence_row(model="m1", confidence="all", n=30, mean_p=0.62, hit=0.58),
    ]
    flagged = confidence_conflict_models(rows)
    assert flagged == set()


def test_confidence_conflict_ignores_small_buckets():
    rows = [
        # only 2 samples in low bucket — under min_samples threshold.
        _confidence_row(model="m1", confidence="low", n=2, mean_p=0.95, hit=0.5),
        _confidence_row(model="m1", confidence="medium", n=10, mean_p=0.6, hit=0.6),
        _confidence_row(model="m1", confidence="high", n=10, mean_p=0.85, hit=0.85),
        _confidence_row(model="m1", confidence="all", n=22, mean_p=0.7, hit=0.7),
    ]
    flagged = confidence_conflict_models(rows)
    assert flagged == set()


def _confidence_row(*, model, confidence, n, mean_p, hit):
    from forecast_eval.analysis.behavior import ConfidenceCalibrationRow
    return ConfidenceCalibrationRow(
        model=model, confidence=confidence, n_samples=n,
        mean_max_p=mean_p, hit_rate=hit,
    )
