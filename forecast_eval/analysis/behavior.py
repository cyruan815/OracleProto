"""Phase 3 behavior analysis.

Five families of indicators built on top of `s{i}_belief_trace` and friends:

1. **Belief evolution** (§25): per-trial volatility, inter-trial variance,
   convergence step, evidence efficiency, counterevidence engagement.
2. **Reflection A/B** (§26): pair runs that differ only in
   `reflection_protocol_hash` and report ΔBI / Δσ / ΔC / Δη with paired
   bootstrap 95% CI, optionally stratified by question_type.
3. **Tool usage PDP** (§27): hand-rolled multi-feature logistic / linear
   regression to recover partial-dependence of `Pr(correct | x)` and
   `E[NLL | x]` over `[tool_calls_count, react_steps, latency_ms,
   prompt_tokens, completion_tokens]`.
4. **Confidence calibration** (§28): subjective (low/medium/high) vs numeric
   (max_p bin) hit-rate tables, plus a `conflict*` marker for models whose
   linguistic and numeric confidence disagree.

The module has zero hard dependency on numpy / scipy / sklearn; everything
runs on plain Python lists and `math.*`. This keeps Phase 3 inside the same
"no new core deps" envelope as Phase 0–2.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..parser import parse_gt
from ..prompts import index_to_letter, letter_to_index
from .. import db as dbmod
from .flatten import (
    SampleRow,
    _answer_gt_for,
    _flatten_db,
    gt_vector,
)
from .inference import paired_bootstrap
from .proper_score import brier_index, brier_score_lab, nll


# --------------------------------------------------------------------------- #
# §25.1 — belief_trace parsing
# --------------------------------------------------------------------------- #


def parse_belief_trace(trace_json: str | None) -> list[dict[str, Any] | None]:
    """Decode persisted belief_trace JSON.

    Each entry is either a step dict (`{step, p, confidence, delta_reason,
    counterevidence?}`) or `None` for steps where parsing failed. Returns `[]`
    when the trace is missing or malformed — caller treats that as "no
    behavioral signal" rather than an error.
    """
    if not trace_json:
        return []
    try:
        data = json.loads(trace_json)
    except (TypeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any] | None] = []
    for item in data:
        if isinstance(item, dict):
            out.append(item)
        else:
            out.append(None)
    return out


def _belief_step_vector(
    step: dict[str, Any] | None, options: list[str]
) -> list[float] | None:
    """Project a step's `p` dict onto the per-letter vector or return None."""
    if not isinstance(step, dict):
        return None
    p = step.get("p")
    if not isinstance(p, dict):
        return None
    out: list[float] = []
    for i in range(len(options)):
        letter = index_to_letter(i)
        v = p.get(letter)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None
        v = float(v)
        if v < 0.0 or v > 1.0:
            return None
        out.append(v)
    return out


def _l2(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _last_valid_vector(
    trace: list[dict[str, Any] | None], options: list[str]
) -> list[float] | None:
    for s in reversed(trace):
        v = _belief_step_vector(s, options)
        if v is not None:
            return v
    return None


# --------------------------------------------------------------------------- #
# §25.2-25.6 — five belief evolution indicators
# --------------------------------------------------------------------------- #


def trial_internal_volatility(
    trace: list[dict[str, Any] | None], options: list[str]
) -> float | None:
    """V_{q,k} = (1/(T-1)) Σ_t ||b_t − b_{t-1}||₂.

    Averaged over consecutive valid step pairs. `None` when fewer than 2 valid
    steps survive parsing — the model never expressed enough information to
    measure within-trial drift.
    """
    if not options:
        return None
    vecs = [_belief_step_vector(s, options) for s in trace]
    valid = [v for v in vecs if v is not None]
    if len(valid) < 2:
        return None
    deltas = [_l2(valid[i], valid[i - 1]) for i in range(1, len(valid))]
    return sum(deltas) / len(deltas)


def inter_trial_variance(
    traces: list[list[dict[str, Any] | None]], options: list[str]
) -> float | None:
    """σ_q = std_k b^{(q,k)}_T as Euclidean stdev around the centroid.

    `np.std`-style with `ddof=0`: σ = sqrt( mean_k ||b_T^{(k)} − μ||₂² ).
    `None` when fewer than 2 trials carry a valid final belief — variance is
    undefined with one observation.
    """
    if not options:
        return None
    finals = [_last_valid_vector(t, options) for t in traces]
    finals = [v for v in finals if v is not None]
    if len(finals) < 2:
        return None
    k = len(finals[0])
    centroid = [sum(v[i] for v in finals) / len(finals) for i in range(k)]
    sq = sum(_l2(v, centroid) ** 2 for v in finals) / len(finals)
    return math.sqrt(sq)


def convergence_step(
    trace: list[dict[str, Any] | None], options: list[str], eps: float = 0.05
) -> int | None:
    """C_{q,k} = min{t : ||b_T − b_t||₂ < eps}.

    Index is into the original trace (so steps with parse failures still count
    against the timestep budget). `None` when no step has a valid belief.
    """
    if not options:
        return None
    vecs = [_belief_step_vector(s, options) for s in trace]
    valid_indices = [i for i, v in enumerate(vecs) if v is not None]
    if not valid_indices:
        return None
    final = vecs[valid_indices[-1]]
    for i in valid_indices:
        if _l2(vecs[i], final) < eps:
            return i
    return valid_indices[-1]


def evidence_efficiency(
    trace: list[dict[str, Any] | None],
    options: list[str],
    obs: list[int],
    choice_type: str,
    search_calls: int,
) -> float | None:
    """η_{q,k} = (NLL(b_initial) − NLL(b_final)) / max(1, search_calls).

    Total drop in negative log-likelihood per search call — higher means each
    search bought more information about the eventual answer. `None` when we
    don't have ≥2 valid beliefs or when the obs vector is unusable.
    """
    if not options or not obs or len(obs) != len(options):
        return None
    vecs = [_belief_step_vector(s, options) for s in trace]
    valid = [v for v in vecs if v is not None]
    if len(valid) < 2:
        return None
    try:
        nll_initial = nll(valid[0], obs, choice_type)
        nll_final = nll(valid[-1], obs, choice_type)
    except (ValueError, ZeroDivisionError):
        return None
    denom = max(1, int(search_calls or 0))
    return (nll_initial - nll_final) / denom


_LETTER_RE = re.compile(r"\b([A-Z])\b")


def counterevidence_engagement(
    counterevidence: list[str] | None,
    final_choice: frozenset[str],
    options: list[str],
) -> int:
    """1 iff at least one counterevidence string mentions an option letter
    that's NOT in `final_choice`. Pure letter matching, no NLP.

    Intent: catch BLF "stress-test the opposite" reasoning. If the model
    listed counter-arguments but they only reference the chosen letter, that's
    rationalisation, not engagement. An empty/missing counterevidence list is
    "no engagement" (returns 0) — same outcome as a list of irrelevant items.
    """
    if not counterevidence:
        return 0
    valid_letters = {index_to_letter(i) for i in range(len(options))}
    counter_letters: set[str] = set()
    for s in counterevidence:
        if not isinstance(s, str):
            continue
        for m in _LETTER_RE.findall(s):
            if m in valid_letters:
                counter_letters.add(m)
    return int(any(letter not in final_choice for letter in counter_letters))


# --------------------------------------------------------------------------- #
# §25.7 — belief_evolution.csv row builder
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BeliefEvolutionRow:
    model: str
    question_id: str
    question_type: str
    choice_type: str
    sample_idx: int
    n_steps: int
    volatility: float | None
    convergence_step: int | None
    evidence_efficiency: float | None
    counterevidence_engaged: int | None
    inter_trial_variance: float | None  # constant across trials of the same q


def build_belief_evolution_rows(
    samples_by_model: dict[str, list[SampleRow]],
    gt_map: dict[str, frozenset[str]],
) -> list[BeliefEvolutionRow]:
    """Compute the 5 indicators per (model, q, k). Skips samples whose
    `belief_trace` is missing or unparseable — those don't contribute behavior
    rows at all (which is what `test_belief_trace_missing` exercises)."""
    out: list[BeliefEvolutionRow] = []
    for model, samples in samples_by_model.items():
        per_q: dict[str, list[SampleRow]] = {}
        for s in samples:
            if not s.is_eligible:
                continue
            per_q.setdefault(s.question_id, []).append(s)
        for qid, ss in per_q.items():
            options = ss[0].options if ss else []
            gt = gt_map.get(qid)
            obs_vec = (
                gt_vector(gt, len(options)) if (gt and options) else None
            )
            traces = [parse_belief_trace(s.belief_trace) for s in ss]
            sigma = inter_trial_variance(traces, options)
            for s, trace in zip(ss, traces):
                # `n_steps` counts the number of parsed (non-None) steps so a
                # reviewer can see "this trial died at step 3" without parsing
                # the JSON themselves.
                n_steps_valid = sum(1 for st in trace if isinstance(st, dict))
                if n_steps_valid == 0:
                    # Nothing to report; skipping keeps the CSV row count
                    # equal to "trials with at least one parsed belief".
                    continue
                vol = trial_internal_volatility(trace, options)
                cstep = convergence_step(trace, options)
                eff: float | None = None
                if obs_vec is not None and s.tool_calls_count is not None:
                    eff = evidence_efficiency(
                        trace,
                        options,
                        obs_vec,
                        s.choice_type,
                        s.tool_calls_count or 0,
                    )
                counter_engagement: int | None = None
                last_step_counter: list[str] | None = None
                for st in reversed(trace):
                    if isinstance(st, dict):
                        counter = st.get("counterevidence")
                        if isinstance(counter, list):
                            last_step_counter = counter
                            break
                        # last parsed step has no counterevidence key (older
                        # trace schema before Phase 3) — keep walking; this
                        # is rare since react.py ALWAYS writes the key now.
                if last_step_counter is not None:
                    final_choice = s.parsed_letters or frozenset()
                    counter_engagement = counterevidence_engagement(
                        last_step_counter, final_choice, options
                    )
                out.append(
                    BeliefEvolutionRow(
                        model=model,
                        question_id=qid,
                        question_type=s.question_type,
                        choice_type=s.choice_type,
                        sample_idx=s.sample_idx,
                        n_steps=n_steps_valid,
                        volatility=vol,
                        convergence_step=cstep,
                        evidence_efficiency=eff,
                        counterevidence_engaged=counter_engagement,
                        inter_trial_variance=sigma,
                    )
                )
    return out


# --------------------------------------------------------------------------- #
# §26 — Reflection A/B
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PairedRunSpec:
    """One A/B pair. `run_on` has reflection enabled; `run_off` has it disabled."""

    model: str
    run_on: Path
    run_off: Path
    common_qids: tuple[str, ...]


def _read_run_meta(db_path: Path) -> dict[str, Any] | None:
    """Pull (model, source_db_hash, metadata_hash, prompt_templates_hash,
    reflection_protocol_hash, belief_protocol_hash, sampling_n) from a
    per-model DB. Returns None if the run hasn't been initialised."""
    try:
        conn = dbmod.connect(db_path)
    except Exception:
        return None
    try:
        row = conn.execute(
            "SELECT model, source_db_hash, metadata_hash, prompt_templates_hash, "
            "reflection_protocol_hash, belief_protocol_hash, sampling_n "
            "FROM run_meta ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return {k: row[k] for k in row.keys()}
    finally:
        conn.close()


def _iter_run_dbs(runs_root: Path) -> list[tuple[Path, Path, dict[str, Any]]]:
    """Yield (run_dir, db_path, meta) for every per-model DB under runs_root.

    Skips run dirs without `manifest.json` (incomplete) or without `db/`.
    """
    out: list[tuple[Path, Path, dict[str, Any]]] = []
    if not runs_root.exists():
        return out
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        if not (run_dir / "manifest.json").exists():
            continue
        db_dir = run_dir / "db"
        if not db_dir.exists():
            continue
        for db_path in sorted(db_dir.glob("*.db")):
            meta = _read_run_meta(db_path)
            if meta is None:
                continue
            out.append((run_dir, db_path, meta))
    return out


def find_paired_runs(runs_root: Path) -> list[PairedRunSpec]:
    """Find (reflection-on, reflection-off) DB pairs that match on every
    other fingerprint.

    Matching rule (spec 26.1 + 26.5): two DBs pair iff
        - same `model`,
        - same `source_db_hash`,
        - same `metadata_hash`,
        - same `prompt_templates_hash`,
        - same `belief_protocol_hash` (None == None counts as same),
        - exactly one has `reflection_protocol_hash IS NULL` and the other
          has it non-NULL.
    Mismatch on ANY hash → no pair, even if the same model is on both sides.
    Question intersection MUST be non-empty.
    """
    runs_root = Path(runs_root)
    entries = _iter_run_dbs(runs_root)
    # Bucket by every fingerprint EXCEPT reflection_protocol_hash.
    buckets: dict[tuple, list[tuple[Path, Path, dict[str, Any]]]] = {}
    for run_dir, db_path, meta in entries:
        key = (
            meta.get("model"),
            meta.get("source_db_hash"),
            meta.get("metadata_hash"),
            meta.get("prompt_templates_hash"),
            meta.get("belief_protocol_hash"),
        )
        buckets.setdefault(key, []).append((run_dir, db_path, meta))

    out: list[PairedRunSpec] = []
    for entries_in_bucket in buckets.values():
        # Inside a bucket, partition by reflection presence.
        on = [e for e in entries_in_bucket if e[2].get("reflection_protocol_hash")]
        off = [e for e in entries_in_bucket if not e[2].get("reflection_protocol_hash")]
        for on_entry in on:
            for off_entry in off:
                qids_on = _question_ids_in(on_entry[1])
                qids_off = _question_ids_in(off_entry[1])
                common = sorted(qids_on & qids_off)
                if not common:
                    continue
                out.append(
                    PairedRunSpec(
                        model=on_entry[2]["model"],
                        run_on=on_entry[1],
                        run_off=off_entry[1],
                        common_qids=tuple(common),
                    )
                )
    return out


def _question_ids_in(db_path: Path) -> set[str]:
    conn = dbmod.connect(db_path)
    try:
        rows = conn.execute("SELECT id FROM questions").fetchall()
        return {r["id"] for r in rows}
    finally:
        conn.close()


@dataclass(frozen=True)
class ReflectionABRow:
    model: str
    question_type: str  # "all" for the global slice
    metric: str  # delta_bi / delta_sigma / delta_convergence / delta_eta
    n_questions: int
    delta_mean: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None


def _per_question_metrics_for_db(
    db_path: Path,
) -> tuple[
    dict[str, float],          # bs_lab_per_q (mean across the K trials)
    dict[str, float],          # sigma_per_q (inter-trial variance)
    dict[str, float],          # mean convergence step per q
    dict[str, float],          # mean evidence efficiency per q
    dict[str, str],            # qtype_per_q
]:
    """Aggregate the four per-question signals reflection A/B compares.

    Each signal is the mean across the question's K trials (so the paired
    bootstrap is over questions, not over trials — same protocol as Phase 2).
    """
    conn = dbmod.connect(db_path)
    try:
        meta = conn.execute(
            "SELECT sampling_n, model FROM run_meta ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        if meta is None:
            return {}, {}, {}, {}, {}
        sampling_n = int(meta["sampling_n"])
        model = meta["model"]
        samples = _flatten_db(conn, sampling_n, model)
        gt_map = _answer_gt_for(conn)
    finally:
        conn.close()

    per_q_samples: dict[str, list[SampleRow]] = {}
    qtype: dict[str, str] = {}
    for s in samples:
        if not s.is_eligible:
            continue
        per_q_samples.setdefault(s.question_id, []).append(s)
        qtype[s.question_id] = s.question_type

    bs_lab: dict[str, float] = {}
    sigma: dict[str, float] = {}
    cstep: dict[str, float] = {}
    eff: dict[str, float] = {}
    for qid, ss in per_q_samples.items():
        options = ss[0].options
        gt = gt_map.get(qid)
        if not options or gt is None:
            continue
        obs = gt_vector(gt, len(options))
        # Per-question BS_lab: mean across trials of brier_score_lab(probs, obs)
        per_trial_bs = []
        per_trial_eff = []
        per_trial_cstep = []
        traces = []
        for s in ss:
            trace = parse_belief_trace(s.belief_trace)
            traces.append(trace)
            if s.probabilities is not None:
                try:
                    per_trial_bs.append(brier_score_lab(s.probabilities, obs))
                except ValueError:
                    pass
            if s.tool_calls_count is not None:
                e = evidence_efficiency(
                    trace, options, obs, s.choice_type, s.tool_calls_count or 0
                )
                if e is not None:
                    per_trial_eff.append(e)
            c = convergence_step(trace, options)
            if c is not None:
                per_trial_cstep.append(c)
        if per_trial_bs:
            bs_lab[qid] = sum(per_trial_bs) / len(per_trial_bs)
        sig = inter_trial_variance(traces, options)
        if sig is not None:
            sigma[qid] = sig
        if per_trial_cstep:
            cstep[qid] = sum(per_trial_cstep) / len(per_trial_cstep)
        if per_trial_eff:
            eff[qid] = sum(per_trial_eff) / len(per_trial_eff)
    return bs_lab, sigma, cstep, eff, qtype


def _paired_delta_rows(
    bs_on: dict[str, float],
    bs_off: dict[str, float],
    common: list[str],
    *,
    model: str,
    metric: str,
    qtype_per_q: dict[str, str] | None,
    n_bootstrap: int,
    seed: int,
) -> list[ReflectionABRow]:
    """For one metric, emit one "all" row plus per-qtype rows."""
    rows: list[ReflectionABRow] = []
    if not common:
        return rows
    a_all = [bs_on[q] for q in common]
    b_all = [bs_off[q] for q in common]
    if a_all and b_all:
        result = paired_bootstrap(a_all, b_all, n_bootstrap=n_bootstrap, seed=seed)
        rows.append(
            ReflectionABRow(
                model=model,
                question_type="all",
                metric=metric,
                n_questions=len(common),
                delta_mean=result.delta_mean,
                ci_low=result.ci_low,
                ci_high=result.ci_high,
                p_value=result.p_two_sided,
            )
        )
    if qtype_per_q:
        per_qtype: dict[str, list[str]] = {}
        for q in common:
            per_qtype.setdefault(qtype_per_q.get(q, "unknown"), []).append(q)
        for qt, qs in sorted(per_qtype.items()):
            a = [bs_on[q] for q in qs]
            b = [bs_off[q] for q in qs]
            if not a or not b:
                continue
            try:
                r = paired_bootstrap(a, b, n_bootstrap=n_bootstrap, seed=seed)
            except ValueError:
                continue
            rows.append(
                ReflectionABRow(
                    model=model,
                    question_type=qt,
                    metric=metric,
                    n_questions=len(qs),
                    delta_mean=r.delta_mean,
                    ci_low=r.ci_low,
                    ci_high=r.ci_high,
                    p_value=r.p_two_sided,
                )
            )
    return rows


def reflection_ab_report(
    pairs: Iterable[PairedRunSpec],
    *,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> list[ReflectionABRow]:
    """For each paired run, compute Δ paired bootstrap CI + per-qtype slices.

    Sign convention: Δ = on − off. Negative Δ on `delta_bi` (a Brier signal)
    means reflection improves BI; positive Δ on `delta_eta` means reflection
    is more efficient per search.
    """
    rows: list[ReflectionABRow] = []
    for spec in pairs:
        bs_on, sig_on, cs_on, eff_on, qt_on = _per_question_metrics_for_db(spec.run_on)
        bs_off, sig_off, cs_off, eff_off, qt_off = _per_question_metrics_for_db(
            spec.run_off
        )
        # qtype map prefers the on-side; off-side fills gaps.
        qt = dict(qt_off)
        qt.update(qt_on)
        for metric, on_map, off_map in (
            ("delta_bi", bs_on, bs_off),
            ("delta_sigma", sig_on, sig_off),
            ("delta_convergence", cs_on, cs_off),
            ("delta_eta", eff_on, eff_off),
        ):
            common = sorted(
                set(spec.common_qids) & set(on_map.keys()) & set(off_map.keys())
            )
            rows.extend(
                _paired_delta_rows(
                    on_map,
                    off_map,
                    common,
                    model=spec.model,
                    metric=metric,
                    qtype_per_q=qt,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                )
            )
    return rows


# --------------------------------------------------------------------------- #
# §27 — Tool usage PDP (hand-rolled multi-feature regression)
# --------------------------------------------------------------------------- #


TOOL_PDP_FEATURES: tuple[str, ...] = (
    "tool_calls_count",
    "react_steps",
    "latency_ms",
    "prompt_tokens",
    "completion_tokens",
)


def _solve_linear_system(A: list[list[float]], b: list[float]) -> list[float] | None:
    """Gauss–Jordan solver for small dense linear systems.

    Returns None when the matrix is numerically singular; caller treats that
    as "PDP unavailable for this fit". Pivots by largest absolute value to
    reduce round-off on near-singular Hessians.
    """
    n = len(A)
    if n == 0 or any(len(row) != n for row in A) or len(b) != n:
        return None
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        # Partial pivoting.
        pivot_row = i
        for r in range(i + 1, n):
            if abs(M[r][i]) > abs(M[pivot_row][i]):
                pivot_row = r
        if abs(M[pivot_row][i]) < 1e-12:
            return None
        if pivot_row != i:
            M[i], M[pivot_row] = M[pivot_row], M[i]
        pivot = M[i][i]
        for k in range(i, n + 1):
            M[i][k] /= pivot
        for r in range(n):
            if r == i:
                continue
            factor = M[r][i]
            if factor == 0:
                continue
            for k in range(i, n + 1):
                M[r][k] -= factor * M[i][k]
    return [M[i][n] for i in range(n)]


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _zscore_columns(
    X: list[list[float]],
) -> tuple[list[list[float]], list[float], list[float]]:
    """Return (Z, means, stds) — stds are clipped to 1.0 when zero so we
    don't divide by zero. Z[i][j] = (X[i][j] − mean_j) / std_j.
    """
    if not X:
        return [], [], []
    p = len(X[0])
    means = [0.0] * p
    for row in X:
        for j in range(p):
            means[j] += row[j]
    for j in range(p):
        means[j] /= len(X)
    var = [0.0] * p
    for row in X:
        for j in range(p):
            var[j] += (row[j] - means[j]) ** 2
    for j in range(p):
        var[j] /= len(X)
    stds = [math.sqrt(v) if v > 1e-12 else 1.0 for v in var]
    Z = [[(row[j] - means[j]) / stds[j] for j in range(p)] for row in X]
    return Z, means, stds


@dataclass(frozen=True)
class LogisticFit:
    """Coefficients for `Pr(y=1 | x) = σ(b + Σ_j w_j · z_j)` on z-scored features."""

    weights: list[float]
    bias: float
    feature_means: list[float]
    feature_stds: list[float]
    converged: bool


def fit_logistic_irls(
    X: list[list[float]],
    y: list[int],
    *,
    l2: float = 0.01,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> LogisticFit | None:
    """Hand-rolled IRLS Newton method for logistic regression.

    Standardises features in-place (z-score), fits on the standardised matrix
    so the L2 penalty is comparable across features, and reports convergence
    via the loss-delta tolerance. Returns None for empty data or failed
    Hessian solve — caller falls back to "PDP unavailable".
    """
    if not X or not y or len(X) != len(y):
        return None
    Z, means, stds = _zscore_columns(X)
    n = len(Z)
    p = len(Z[0])
    w = [0.0] * p
    b = 0.0
    prev_loss = float("inf")
    for _ in range(max_iter):
        probs = []
        for zi in Z:
            zlin = b + sum(w[j] * zi[j] for j in range(p))
            probs.append(_sigmoid(zlin))
        # gradient (size p+1, last is bias)
        grad = [0.0] * (p + 1)
        for i in range(n):
            err = probs[i] - y[i]
            grad[p] += err
            for j in range(p):
                grad[j] += Z[i][j] * err
        for j in range(p):
            grad[j] += l2 * w[j]
        # Hessian (p+1) x (p+1)
        H = [[0.0] * (p + 1) for _ in range(p + 1)]
        for i in range(n):
            wi = probs[i] * (1.0 - probs[i])
            if wi < 1e-12:
                wi = 1e-12
            H[p][p] += wi
            for j in range(p):
                H[p][j] += wi * Z[i][j]
                H[j][p] += wi * Z[i][j]
                for k in range(p):
                    H[j][k] += wi * Z[i][j] * Z[i][k]
        for j in range(p):
            H[j][j] += l2
        delta = _solve_linear_system(H, grad)
        if delta is None:
            return None
        # Step clipping (Phase 2 calibration learned this lesson on saturated
        # sigmoids — divergence shows up as |delta|→∞).
        norm = math.sqrt(sum(d * d for d in delta))
        if norm > 5.0:
            scale = 5.0 / norm
            delta = [d * scale for d in delta]
        for j in range(p):
            w[j] -= delta[j]
        b -= delta[p]
        # Recompute loss for convergence check.
        loss = 0.0
        for i in range(n):
            zlin = b + sum(w[j] * Z[i][j] for j in range(p))
            pi = _sigmoid(zlin)
            pi = min(max(pi, 1e-12), 1 - 1e-12)
            loss -= y[i] * math.log(pi) + (1 - y[i]) * math.log(1 - pi)
        loss += 0.5 * l2 * sum(wi * wi for wi in w)
        if abs(prev_loss - loss) < tol:
            return LogisticFit(
                weights=w,
                bias=b,
                feature_means=means,
                feature_stds=stds,
                converged=True,
            )
        prev_loss = loss
    return LogisticFit(
        weights=w, bias=b, feature_means=means, feature_stds=stds, converged=False
    )


@dataclass(frozen=True)
class LinearFit:
    weights: list[float]
    bias: float
    feature_means: list[float]
    feature_stds: list[float]


def fit_linear_ridge(
    X: list[list[float]], y: list[float], *, l2: float = 0.01
) -> LinearFit | None:
    """Closed-form ridge regression on z-scored features.

    Solves `(Z^T Z + λI) β = Z^T y` plus a separate bias term so the penalty
    only applies to slopes. Returns None when the linear system is singular.
    """
    if not X or not y or len(X) != len(y):
        return None
    Z, means, stds = _zscore_columns(X)
    n = len(Z)
    p = len(Z[0])
    # Add bias column.
    ZtZ = [[0.0] * (p + 1) for _ in range(p + 1)]
    Zty = [0.0] * (p + 1)
    for i in range(n):
        for j in range(p):
            Zty[j] += Z[i][j] * y[i]
            for k in range(p):
                ZtZ[j][k] += Z[i][j] * Z[i][k]
            ZtZ[j][p] += Z[i][j]
            ZtZ[p][j] += Z[i][j]
        ZtZ[p][p] += 1.0
        Zty[p] += y[i]
    for j in range(p):
        ZtZ[j][j] += l2
    coefs = _solve_linear_system(ZtZ, Zty)
    if coefs is None:
        return None
    return LinearFit(
        weights=coefs[:p],
        bias=coefs[p],
        feature_means=means,
        feature_stds=stds,
    )


def _predict_logistic(fit: LogisticFit, X: list[list[float]]) -> list[float]:
    out = []
    for row in X:
        zlin = fit.bias
        for j in range(len(fit.weights)):
            zj = (row[j] - fit.feature_means[j]) / fit.feature_stds[j]
            zlin += fit.weights[j] * zj
        out.append(_sigmoid(zlin))
    return out


def _predict_linear(fit: LinearFit, X: list[list[float]]) -> list[float]:
    out = []
    for row in X:
        zlin = fit.bias
        for j in range(len(fit.weights)):
            zj = (row[j] - fit.feature_means[j]) / fit.feature_stds[j]
            zlin += fit.weights[j] * zj
        out.append(zlin)
    return out


def _grid_for_feature(
    column: list[float], n_points: int = 9
) -> list[float]:
    """Quantile-based PDP grid. n_points equally-spaced quantiles, deduped
    (so a feature with few unique values yields a small grid)."""
    if not column:
        return []
    sorted_col = sorted(column)
    n = len(sorted_col)
    if n_points < 2:
        n_points = 2
    out: list[float] = []
    seen: set[float] = set()
    for i in range(n_points):
        idx = int(round(i * (n - 1) / (n_points - 1)))
        v = sorted_col[idx]
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


@dataclass(frozen=True)
class PDPRow:
    model: str
    feature: str
    feature_value: float
    pdp_correct: float | None
    pdp_nll: float | None
    n_samples: int


def _collect_pdp_inputs(
    samples: Iterable[SampleRow],
    gt_map: dict[str, frozenset[str]],
) -> tuple[list[list[float]], list[int], list[float]]:
    """Extract (X, y_correct, y_nll) from samples, dropping rows with missing
    features or unscoreable probabilities."""
    X: list[list[float]] = []
    y_correct: list[int] = []
    y_nll: list[float] = []
    for s in samples:
        if not s.is_resolvable or s.probabilities is None:
            continue
        if any(
            getattr(s, f) is None
            for f in TOOL_PDP_FEATURES
        ):
            continue
        gt = gt_map.get(s.question_id)
        if gt is None:
            continue
        try:
            obs = gt_vector(gt, len(s.options))
            sample_nll = nll(s.probabilities, obs, s.choice_type)
        except (ValueError, ZeroDivisionError):
            continue
        X.append([float(getattr(s, f)) for f in TOOL_PDP_FEATURES])
        y_correct.append(int(s.correct or 0))
        y_nll.append(float(sample_nll))
    return X, y_correct, y_nll


def tool_usage_pdp(
    samples_by_model: dict[str, list[SampleRow]],
    gt_map: dict[str, frozenset[str]],
    *,
    n_grid: int = 9,
) -> list[PDPRow]:
    """Per-model partial-dependence rows for the 5 cost/usage features.

    For each feature `j`:
        - Sweep `j` over a quantile grid (`n_grid` points, deduped).
        - For each grid point, set `j` to that value across all training
          samples while keeping the OTHER features at their observed values,
          predict, then average.
    Returns flat rows so the writer can dump a single `tool_usage_pdp.csv`.
    """
    out: list[PDPRow] = []
    for model, samples in samples_by_model.items():
        X, y_correct, y_nll = _collect_pdp_inputs(samples, gt_map)
        if len(X) < max(20, len(TOOL_PDP_FEATURES) + 2):
            # Not enough rows to fit a sensible LR — skip rather than crash.
            continue
        # Skip if y_correct is degenerate (all 0 or all 1) — IRLS Hessian
        # collapses to zero.
        if len(set(y_correct)) < 2:
            continue
        log_fit = fit_logistic_irls(X, y_correct)
        lin_fit = fit_linear_ridge(X, y_nll)
        n = len(X)
        for j, feat in enumerate(TOOL_PDP_FEATURES):
            grid = _grid_for_feature([row[j] for row in X], n_points=n_grid)
            for v in grid:
                X_pdp = [row[:] for row in X]
                for row in X_pdp:
                    row[j] = v
                pdp_correct: float | None = None
                pdp_nll_val: float | None = None
                if log_fit is not None:
                    preds = _predict_logistic(log_fit, X_pdp)
                    pdp_correct = sum(preds) / len(preds) if preds else None
                if lin_fit is not None:
                    preds = _predict_linear(lin_fit, X_pdp)
                    pdp_nll_val = sum(preds) / len(preds) if preds else None
                out.append(
                    PDPRow(
                        model=model,
                        feature=feat,
                        feature_value=v,
                        pdp_correct=pdp_correct,
                        pdp_nll=pdp_nll_val,
                        n_samples=n,
                    )
                )
    return out


# --------------------------------------------------------------------------- #
# §28 — Confidence calibration joint diagnosis
# --------------------------------------------------------------------------- #


CONFIDENCE_BUCKETS: tuple[str, ...] = ("low", "medium", "high")


def _final_confidence(trace: list[dict[str, Any] | None]) -> str | None:
    """Take the LAST step's confidence — that's the one that drove the
    final boxed answer. Failed steps are skipped backwards."""
    for s in reversed(trace):
        if isinstance(s, dict):
            c = s.get("confidence")
            if c in CONFIDENCE_BUCKETS:
                return c
    return None


def _max_p(probs: list[float]) -> float:
    return max(probs)


@dataclass(frozen=True)
class ConfidenceCalibrationRow:
    model: str
    confidence: str  # "low", "medium", "high", or "all"
    n_samples: int
    mean_max_p: float | None
    hit_rate: float | None  # mean(correct) where correct is in {0, 1}


@dataclass(frozen=True)
class NumericConfidenceCalibrationRow:
    model: str
    bin_low: float
    bin_high: float
    n_samples: int
    mean_max_p: float | None
    hit_rate: float | None


def confidence_calibration(
    samples_by_model: dict[str, list[SampleRow]],
) -> list[ConfidenceCalibrationRow]:
    """Group samples by self-reported confidence (last-step value of the
    parsed trace). Samples with unparseable belief / no max_p / no correct
    label are skipped (they wouldn't contribute meaningful joint info)."""
    out: list[ConfidenceCalibrationRow] = []
    for model, samples in samples_by_model.items():
        # bucket -> (n, sum_max_p, sum_correct)
        per_bucket: dict[str, list[tuple[float, int]]] = {b: [] for b in CONFIDENCE_BUCKETS}
        all_pairs: list[tuple[float, int]] = []
        for s in samples:
            if not s.is_resolvable or s.probabilities is None:
                continue
            trace = parse_belief_trace(s.belief_trace)
            conf = _final_confidence(trace)
            if conf is None:
                continue
            if s.correct is None:
                continue
            mx = _max_p(s.probabilities)
            per_bucket[conf].append((mx, int(s.correct)))
            all_pairs.append((mx, int(s.correct)))
        for bucket in CONFIDENCE_BUCKETS:
            pairs = per_bucket[bucket]
            n = len(pairs)
            mean_p = sum(p for p, _ in pairs) / n if n else None
            hit = sum(c for _, c in pairs) / n if n else None
            out.append(
                ConfidenceCalibrationRow(
                    model=model,
                    confidence=bucket,
                    n_samples=n,
                    mean_max_p=mean_p,
                    hit_rate=hit,
                )
            )
        # "all" row aggregates the buckets — useful for sanity-checking that
        # the bucketing isn't dropping data on the floor.
        n_all = len(all_pairs)
        out.append(
            ConfidenceCalibrationRow(
                model=model,
                confidence="all",
                n_samples=n_all,
                mean_max_p=sum(p for p, _ in all_pairs) / n_all if n_all else None,
                hit_rate=sum(c for _, c in all_pairs) / n_all if n_all else None,
            )
        )
    return out


def numeric_confidence_calibration(
    samples_by_model: dict[str, list[SampleRow]],
    *,
    n_bins: int = 10,
) -> list[NumericConfidenceCalibrationRow]:
    """Bin samples by `max_p` (top-1 numeric confidence) into `n_bins`
    equal-width bins over [0, 1]. Output is per-model per-bin (n_bins rows
    each, including empty bins so the CSV is rectangular).
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    bin_edges = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    out: list[NumericConfidenceCalibrationRow] = []
    for model, samples in samples_by_model.items():
        per_bin: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
        for s in samples:
            if not s.is_resolvable or s.probabilities is None or s.correct is None:
                continue
            mx = _max_p(s.probabilities)
            # Place at edge: [low, high), but the last bin is [low, 1.0].
            idx = min(int(mx * n_bins), n_bins - 1)
            per_bin[idx].append((mx, int(s.correct)))
        for (low, high), pairs in zip(bin_edges, per_bin):
            n = len(pairs)
            out.append(
                NumericConfidenceCalibrationRow(
                    model=model,
                    bin_low=low,
                    bin_high=high,
                    n_samples=n,
                    mean_max_p=sum(p for p, _ in pairs) / n if n else None,
                    hit_rate=sum(c for _, c in pairs) / n if n else None,
                )
            )
    return out


def confidence_conflict_models(
    rows: list[ConfidenceCalibrationRow],
    *,
    low_max_p_threshold: float = 0.70,
    high_max_p_threshold: float = 0.55,
    min_samples: int = 5,
) -> set[str]:
    """Return models flagged with a `conflict*` marker for spec 28.3.

    Two diagnoses, both keyed off the SAME row dict-set:
        - "language conservative + numeric overconfident": `low` bucket has
          `mean_max_p > low_max_p_threshold` (model says low but the numbers
          are bullish).
        - "language confident + numeric underconfident": `high` bucket has
          `mean_max_p < high_max_p_threshold` (model says high but the
          numbers don't match).
    Buckets with fewer than `min_samples` are ignored — too noisy to flag.
    """
    by_model: dict[str, dict[str, ConfidenceCalibrationRow]] = {}
    for r in rows:
        by_model.setdefault(r.model, {})[r.confidence] = r
    flagged: set[str] = set()
    for model, by_bucket in by_model.items():
        low_row = by_bucket.get("low")
        if (
            low_row is not None
            and low_row.n_samples >= min_samples
            and low_row.mean_max_p is not None
            and low_row.mean_max_p > low_max_p_threshold
        ):
            flagged.add(model)
            continue
        high_row = by_bucket.get("high")
        if (
            high_row is not None
            and high_row.n_samples >= min_samples
            and high_row.mean_max_p is not None
            and high_row.mean_max_p < high_max_p_threshold
        ):
            flagged.add(model)
    return flagged


__all__ = [
    # §25
    "parse_belief_trace",
    "trial_internal_volatility",
    "inter_trial_variance",
    "convergence_step",
    "evidence_efficiency",
    "counterevidence_engagement",
    "BeliefEvolutionRow",
    "build_belief_evolution_rows",
    # §26
    "PairedRunSpec",
    "ReflectionABRow",
    "find_paired_runs",
    "reflection_ab_report",
    # §27
    "TOOL_PDP_FEATURES",
    "LogisticFit",
    "LinearFit",
    "PDPRow",
    "fit_logistic_irls",
    "fit_linear_ridge",
    "tool_usage_pdp",
    # §28
    "CONFIDENCE_BUCKETS",
    "ConfidenceCalibrationRow",
    "NumericConfidenceCalibrationRow",
    "confidence_calibration",
    "numeric_confidence_calibration",
    "confidence_conflict_models",
]
