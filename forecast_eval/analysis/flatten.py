"""Wide-table → per-sample row pivot.

Reads `run_results` (one row per question, with `s{i}_*` column groups for
$N$ samples) and emits a flat list of `SampleRow` joined with question
metadata. v4 additionally extracts the structured `belief_final` JSON into a
`probabilities` vector aligned with the question letter set, applying the
§2.4 fallback when belief is unavailable but `\\boxed{...}` succeeded.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from .. import db as dbmod
from ..parser import parse_gt
from ..prompts import index_to_letter


CUTOFF = "skipped_training_cutoff"

# Per-sample columns to read out of `run_results`. Order is the SELECT order:
# `_flatten_db` derives both the SQL column list AND the post-select offsets
# from this single tuple, so adding a v4 column means appending one entry
# here — no manual index bookkeeping. `created_at` keeps doubling as the
# "slot is populated" sentinel; finish_reason / nudges_used are v3 additions;
# belief_* are v4.
_ANALYSIS_FIELDS: tuple[str, ...] = (
    "correct",
    "parse_ok",
    "tool_calls_count",
    "react_steps",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "latency_ms",
    "final_answer_letters",
    "error",
    "created_at",
    "finish_reason",
    "nudges_used",
    "belief_final",
    "belief_trace",
    "belief_parse_ok",
)


# Fallback ε for the §2.4 "boxed-only" probability vector. Picked at the same
# order of magnitude as BLF §C.7's empirical-prior mixing weight; small enough
# to keep the boxed letter dominant, large enough to keep NLL finite.
FALLBACK_EPSILON: float = 0.05


@dataclass
class SampleRow:
    """Flattened per-sample view of one cell in the wide run_results table."""

    model: str
    question_id: str
    question_type: str
    choice_type: str
    options: list[str]
    sample_idx: int

    correct: int | None
    parse_ok: int | None
    tool_calls_count: int | None
    react_steps: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    latency_ms: int | None
    final_answer_letters: str | None
    error: str | None
    created_at: str | None
    finish_reason: str | None
    nudges_used: int | None

    # v4 belief columns (raw and derived).
    belief_final: str | None
    belief_trace: str | None
    belief_parse_ok: int | None
    # `probabilities` is the per-letter vector used by Phase 1 metrics:
    #   * If `belief_final` parses → the JSON probabilities, ordered by letter.
    #   * Else if `parse_ok=1` and we have a boxed letter set → §2.4 fallback.
    #   * Else None (sample excluded from probabilistic metrics).
    probabilities: list[float] | None
    # True iff `probabilities` came from the §2.4 fallback (NOT from a parsed
    # belief). Used by `per_model_summary.md` to surface fallback_share so a
    # reviewer can judge the calibration signal.
    is_fallback: bool

    @property
    def is_cutoff(self) -> bool:
        return self.error == CUTOFF

    @property
    def is_eligible(self) -> bool:
        return not self.is_cutoff

    @property
    def is_resolvable(self) -> bool:
        return self.is_eligible and self.correct is not None

    @property
    def parsed_letters(self) -> frozenset[str] | None:
        if not self.final_answer_letters:
            return None
        try:
            return frozenset(json.loads(self.final_answer_letters))
        except (TypeError, ValueError):
            return None


def _build_fallback_probabilities(
    parsed: frozenset[str] | None, options: list[str], epsilon: float = FALLBACK_EPSILON
) -> list[float] | None:
    """§2.4 fallback: $p_l = 1-\\epsilon$ for hits, $\\epsilon/(k-|\\text{hit}|)$ otherwise.

    Returns None if `parsed` is empty / None / contains a letter outside the
    option range — those cases are "completely unparsed" and MUST NOT pollute
    probabilistic averages.
    """
    if not parsed or not options:
        return None
    k = len(options)
    valid_letters = {index_to_letter(i) for i in range(k)}
    if not parsed.issubset(valid_letters):
        return None
    hit_count = len(parsed)
    if hit_count >= k:
        # Boxed answer covered every option — degenerate, treat as no signal.
        return None
    miss_share = epsilon / (k - hit_count)
    out: list[float] = []
    for i in range(k):
        letter = index_to_letter(i)
        if letter in parsed:
            out.append(1.0 - epsilon)
        else:
            out.append(miss_share)
    return out


def _parse_belief_probabilities(
    belief_json: str | None, options: list[str]
) -> list[float] | None:
    """Read the JSON in `belief_final` and project onto the letter order."""
    if not belief_json:
        return None
    try:
        data = json.loads(belief_json)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    out: list[float] = []
    for i in range(len(options)):
        letter = index_to_letter(i)
        v = data.get(letter)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None
        v_f = float(v)
        # The strict in-range / simplex / letter-set checks already happened
        # in `parser.parse_belief` before this string was persisted. We still
        # do a lightweight clamp here in case a belief blob slipped through
        # without strict validation (older tooling, manual SQL).
        if v_f < 0.0 or v_f > 1.0:
            return None
        out.append(v_f)
    return out


def _flatten_db(
    conn: sqlite3.Connection, sampling_n: int, model: str
) -> list[SampleRow]:
    """Pivot the wide run_results table into per-sample rows joined with question metadata."""
    cols: list[str] = [
        "q.id",
        "q.question_type",
        "q.choice_type",
        "q.options",
    ]
    for i in range(sampling_n):
        for name in _ANALYSIS_FIELDS:
            cols.append(f"r.{dbmod.sample_col(i, name)}")
    sql = (
        "SELECT " + ", ".join(cols) + " "
        "FROM questions q LEFT JOIN run_results r ON q.id = r.question_id"
    )
    offset = 4
    step = len(_ANALYSIS_FIELDS)
    field_idx = {name: i for i, name in enumerate(_ANALYSIS_FIELDS)}
    created_off = field_idx["created_at"]
    samples: list[SampleRow] = []
    for row in conn.execute(sql):
        qid, qtype, ctype, options_raw = row[0], row[1], row[2], row[3]
        try:
            options = json.loads(options_raw) if options_raw else []
        except (TypeError, ValueError):
            options = []
        for i in range(sampling_n):
            base = offset + step * i
            created = row[base + created_off]
            if created is None:
                # Sample slot is empty — no record written. Skip; we can still
                # judge pass_any_at_n with the other samples. Counting absent
                # slots as "error=unknown" would inflate error rates unfairly.
                continue
            final_letters_raw = row[base + field_idx["final_answer_letters"]]
            parse_ok_val = row[base + field_idx["parse_ok"]]
            belief_final_raw = row[base + field_idx["belief_final"]]
            belief_trace_raw = row[base + field_idx["belief_trace"]]
            belief_parse_ok_val = row[base + field_idx["belief_parse_ok"]]

            probs: list[float] | None = None
            is_fallback = False
            if belief_final_raw is not None:
                probs = _parse_belief_probabilities(belief_final_raw, options)
            if probs is None and parse_ok_val == 1 and final_letters_raw:
                # §2.4 fallback path — we still have a usable boxed answer,
                # so synthesize a deterministic probability vector for the
                # probabilistic metrics. `belief_parse_ok=0` regardless of
                # whether belief_final was just NULL or actively malformed.
                try:
                    parsed = frozenset(json.loads(final_letters_raw))
                except (TypeError, ValueError):
                    parsed = None
                fallback = _build_fallback_probabilities(parsed, options)
                if fallback is not None:
                    probs = fallback
                    is_fallback = True

            samples.append(
                SampleRow(
                    model=model,
                    question_id=qid,
                    question_type=qtype,
                    choice_type=ctype,
                    options=list(options),
                    sample_idx=i,
                    correct=row[base + field_idx["correct"]],
                    parse_ok=parse_ok_val,
                    tool_calls_count=row[base + field_idx["tool_calls_count"]],
                    react_steps=row[base + field_idx["react_steps"]],
                    prompt_tokens=row[base + field_idx["prompt_tokens"]],
                    completion_tokens=row[base + field_idx["completion_tokens"]],
                    reasoning_tokens=row[base + field_idx["reasoning_tokens"]],
                    latency_ms=row[base + field_idx["latency_ms"]],
                    final_answer_letters=final_letters_raw,
                    error=row[base + field_idx["error"]],
                    created_at=created,
                    finish_reason=row[base + field_idx["finish_reason"]],
                    nudges_used=row[base + field_idx["nudges_used"]],
                    belief_final=belief_final_raw,
                    belief_trace=belief_trace_raw,
                    belief_parse_ok=belief_parse_ok_val,
                    probabilities=probs,
                    is_fallback=is_fallback,
                )
            )
    return samples


def _group_by_question(samples: list[SampleRow]) -> dict[str, list[SampleRow]]:
    out: dict[str, list[SampleRow]] = {}
    for s in samples:
        out.setdefault(s.question_id, []).append(s)
    return out


def _answer_gt_for(conn: sqlite3.Connection) -> dict[str, frozenset[str]]:
    """Map question_id -> GT letter frozenset using the question-local parser."""
    rows = conn.execute("SELECT id, answer FROM questions").fetchall()
    out: dict[str, frozenset[str]] = {}
    for r in rows:
        try:
            out[r["id"]] = parse_gt(r["answer"])
        except ValueError:
            out[r["id"]] = frozenset()
    return out


def _question_options_for(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Map question_id -> options list (preserves letter ordering A, B, C, ...)."""
    rows = conn.execute("SELECT id, options FROM questions").fetchall()
    out: dict[str, list[str]] = {}
    for r in rows:
        try:
            opts = json.loads(r["options"]) if r["options"] else []
        except (TypeError, ValueError):
            opts = []
        out[r["id"]] = opts
    return out


def gt_vector(gt: frozenset[str], k: int) -> list[int]:
    """Project a GT letter set onto the per-letter Bernoulli observation vector."""
    return [1 if index_to_letter(i) in gt else 0 for i in range(k)]


__all__ = [
    "CUTOFF",
    "FALLBACK_EPSILON",
    "SampleRow",
    "_ANALYSIS_FIELDS",
    "_flatten_db",
    "_group_by_question",
    "_answer_gt_for",
    "_question_options_for",
    "gt_vector",
]
