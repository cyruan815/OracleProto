from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from .prompts import index_to_letter, letter_to_index
from .types import Question


BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
BELIEF_RE = re.compile(r"<belief>([\s\S]*?)</belief>")

_VALID_CONFIDENCE = ("low", "medium", "high")
_SIMPLEX_TOL = 1e-3
_KEY_EVIDENCE_MAX_CHARS = 280


@dataclass(frozen=True)
class Belief:
    """Structured belief block parsed from one assistant message.

    Mirrors the JSON schema declared in `prompts.BELIEF_PROTOCOL`. The
    `version` field defaults to the v4.0 protocol tag — newer protocol
    revisions bump it. Validation is performed by `parse_belief` BEFORE the
    dataclass is constructed, so any `Belief` instance is already simplex /
    range / letter-set checked.
    """

    probabilities: dict[str, float]
    confidence: str
    key_evidence: list[str]
    counterevidence: list[str]
    open_questions: list[str]
    decision_rule: str
    version: str = "v4.0"


def parse_answer(text: str, q: Question) -> Optional[frozenset[str]]:
    """Extract the final answer (last `\\boxed{...}`) and normalise to letters.

    Returns the canonical `frozenset[str]` of letter labels, or `None` when the
    LLM's response doesn't match the expected format (soft refusal / malformed
    payload / out-of-range letter). `None` is NOT an error — the caller records
    parse_ok=0 and moves on.
    """
    if not text:
        return None
    matches = BOXED_RE.findall(text)
    if not matches:
        return None
    payload = matches[-1].strip()
    if not payload:
        return None

    if q.question_type == "yes_no":
        v = payload.lower()
        if v == "yes":
            return frozenset({"A"})
        if v == "no":
            return frozenset({"B"})
        return None

    options = json.loads(q.options)

    if q.question_type == "binary_named":
        norm = payload.lower()
        for i, label in enumerate(options):
            if label.strip().lower() == norm:
                return frozenset({index_to_letter(i)})
        return None

    if q.question_type == "multiple_choice":
        tokens = [t.strip() for t in re.split(r"[,\s]+", payload) if t.strip()]
        if not tokens:
            return None
        n_opts = len(options)
        letters: set[str] = set()
        for t in tokens:
            if len(t) != 1:
                return None
            idx = letter_to_index(t)
            if not (0 <= idx < n_opts):
                return None
            letters.add(t)
        return frozenset(letters) if letters else None

    return None


def parse_gt(answer: str) -> frozenset[str]:
    """Turn the stored answer string (e.g. 'A' or 'A, B') into letter frozenset."""
    if answer is None or not answer.strip():
        raise ValueError("answer must be a non-empty letter CSV")
    letters = [tok.strip() for tok in answer.split(",") if tok.strip()]
    if not letters:
        raise ValueError(f"answer {answer!r} contains no letters")
    return frozenset(letters)


def is_correct(pred: Optional[frozenset[str]], gt: frozenset[str]) -> Optional[bool]:
    """Strict set equality (None -> None so callers can write NULL)."""
    if pred is None:
        return None
    return pred == gt


def _expected_letters(q: Question) -> tuple[str, ...]:
    """Letter set for a question, using the same `chr(ord('A') + i)` rule as
    the prompt rendering layer. Stays consistent with `parse_answer` for the
    >26-option corner cases (`[`, `\\`, `]`, `^`, `_`, `` ` ``, lowercase a..)."""
    options = json.loads(q.options)
    return tuple(index_to_letter(i) for i in range(len(options)))


def parse_belief(text: str, q: Question) -> Optional[Belief]:
    """Extract the final `<belief>...</belief>` JSON block and validate it.

    Returns a `Belief` on success, or `None` on ANY failure mode (missing
    tag, malformed JSON, schema mismatch, simplex violation, out-of-range
    probability, illegal `confidence`, empty `key_evidence`, etc.). NEVER
    raises — callers record `belief_parse_ok = 0` and continue.

    Independent of `parse_answer`: a sample can have `parse_ok = 1` but
    `belief_parse_ok = 0`, or vice versa. The two paths must not pollute
    each other.
    """
    if not text:
        return None
    matches = BELIEF_RE.findall(text)
    if not matches:
        return None
    payload = matches[-1].strip()
    if not payload:
        return None

    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None

    probs_raw = data.get("probabilities")
    if not isinstance(probs_raw, dict) or not probs_raw:
        return None

    expected_letters = _expected_letters(q)
    if set(probs_raw.keys()) != set(expected_letters):
        return None

    probabilities: dict[str, float] = {}
    for letter in expected_letters:
        v = probs_raw[letter]
        if isinstance(v, bool):  # bool is subclass of int — reject explicitly
            return None
        if not isinstance(v, (int, float)):
            return None
        f = float(v)
        if not (0.0 <= f <= 1.0):
            return None
        probabilities[letter] = f

    if q.choice_type == "single":
        total = sum(probabilities.values())
        if abs(total - 1.0) > _SIMPLEX_TOL:
            return None
    elif q.choice_type != "multi":
        return None  # unknown choice_type — refuse to validate

    confidence = data.get("confidence")
    if confidence not in _VALID_CONFIDENCE:
        return None

    key_evidence = data.get("key_evidence")
    if not isinstance(key_evidence, list) or len(key_evidence) < 1:
        return None
    for item in key_evidence:
        if not isinstance(item, str) or len(item) > _KEY_EVIDENCE_MAX_CHARS:
            return None

    counterevidence = data.get("counterevidence", [])
    if not isinstance(counterevidence, list):
        return None
    for item in counterevidence:
        if not isinstance(item, str) or len(item) > _KEY_EVIDENCE_MAX_CHARS:
            return None

    open_questions = data.get("open_questions", [])
    if not isinstance(open_questions, list):
        return None
    for item in open_questions:
        if not isinstance(item, str) or len(item) > _KEY_EVIDENCE_MAX_CHARS:
            return None

    decision_rule = data.get("decision_rule")
    if not isinstance(decision_rule, str) or not decision_rule.strip():
        return None

    version = data.get("version", "v4.0")
    if not isinstance(version, str):
        return None

    return Belief(
        probabilities=probabilities,
        confidence=confidence,
        key_evidence=list(key_evidence),
        counterevidence=list(counterevidence),
        open_questions=list(open_questions),
        decision_rule=decision_rule,
        version=version,
    )
