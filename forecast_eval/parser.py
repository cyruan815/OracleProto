from __future__ import annotations

import json
import re
from typing import Optional

from .prompts import index_to_letter, letter_to_index
from .types import Question


BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


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
