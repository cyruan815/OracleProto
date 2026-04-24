from __future__ import annotations

import json

from .types import Question


# 小写字母/符号标签在 markdown 里容易被反引号/下划线/星号吞。>26 选项时标签
# 会落到 `[` `\` `]` `^` `_` `` ` `` `a` `b` ... 这些字符上，因此我们统一给这些
# 非 A–Z 标签加反引号包裹，防止 LLM 看到的 prompt 里被 markdown 处理器吃掉。
_BACKTICK_SAFE_ASCII_RANGE = set(range(ord("A"), ord("Z") + 1))


def index_to_letter(i: int) -> str:
    if i < 0:
        raise ValueError(f"index must be >= 0, got {i}")
    return chr(ord("A") + i)


def letter_to_index(letter: str) -> int:
    if len(letter) != 1:
        raise ValueError(f"letter must be a single character, got {letter!r}")
    return ord(letter) - ord("A")


def _format_outcome_label(letter: str, label: str) -> str:
    """Wrap non-letter labels (>26 options: '[' '\\' ']' '^' '_' '`' 'a'..) in
    backticks so markdown doesn't swallow them, and pad with a leading space so
    a label that starts with a space / underscore still renders.
    """
    if ord(letter) in _BACKTICK_SAFE_ASCII_RANGE:
        return f"{letter}. {label}"
    return f"`{letter}`. {label}"


def _build_outcomes_block(options: list[str]) -> str:
    lines = [_format_outcome_label(index_to_letter(i), label) for i, label in enumerate(options)]
    return "\n" + "\n".join(lines)


def render_user_prompt(q: Question, templates: dict[str, str]) -> str:
    """Assemble the single user message handed to the LLM for one sample.

    All template text lives in `templates` (synced from dataset_metadata). This
    function only branches on `q.question_type` and handles the three
    rendering rules from the prompt-rendering spec.
    """
    options = json.loads(q.options)

    if q.question_type == "yes_no":
        outcomes_block = ""
        output_format = templates["yes_no_output_format"]

    elif q.question_type == "binary_named":
        if len(options) != 2:
            raise ValueError(
                f"binary_named question {q.id!r} must have exactly 2 options, got {len(options)}"
            )
        outcomes_block = ""
        output_format = (
            templates["binary_named_output_format"]
            .replace("<options[0]>", options[0])
            .replace("<options[1]>", options[1])
        )

    elif q.question_type == "multiple_choice":
        outcomes_block = _build_outcomes_block(options)
        output_format = templates["multiple_choice_output_format"]

    else:
        raise ValueError(f"unknown question_type: {q.question_type!r}")

    return templates["prompt_template"].format(
        agent_role=templates["agent_role"],
        event=q.event,
        end_time=q.end_time,
        outcomes_block=outcomes_block,
        output_format=output_format,
        guidance=templates["guidance"],
    )
