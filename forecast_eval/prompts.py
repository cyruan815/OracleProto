from __future__ import annotations

import json

from .types import Question


# 小写字母/符号标签在 markdown 里容易被反引号/下划线/星号吞。>26 选项时标签
# 会落到 `[` `\` `]` `^` `_` `` ` `` `a` `b` ... 这些字符上，因此我们统一给这些
# 非 A–Z 标签加反引号包裹，防止 LLM 看到的 prompt 里被 markdown 处理器吃掉。
_BACKTICK_SAFE_ASCII_RANGE = set(range(ord("A"), ord("Z") + 1))


# 反思协议. 作为 user message 末尾的附加段落, 不进 dataset_metadata.prompt_reconstruction
# (因此 prompt_templates_hash 不受影响), 但会作为 user_prompt 字段落库, 完全可复盘.
# 设计目标: 用 prompt-engineering 把模型从 "1 次 web_search 直接答" 拉到
# "≥3 次不同角度 + 每次反思 + 反方向自检", 主要靠协议本身驱动, 不需要硬性最低次数.
#
# v4 新增 BELIEF_PROTOCOL: 与 reflection 协议平行的尾部追加段, 要求 LLM 在
# \\boxed{...} 之前再输出一段 <belief>...</belief> 严格 JSON, 让概率族指标
# (Brier / NLL / MBS / ECE / 校准曲线) 能直接落地. 同样不进 prompt_templates_hash;
# 启用条件由 Settings.BELIEF_PROTOCOL 控制, 指纹独立记录在 run_meta.belief_protocol_*.
REFLECTION_PROTOCOL = """\

---
**Forecasting Protocol — follow every step before producing the final \\boxed{...} answer.**

1. **Decompose.** Privately list the sub-questions whose answers, taken together, settle this prediction. Note the entities, dates, quantities, and definitions that need to be pinned down.

2. **Plan distinct angles.** Before issuing any `web_search`, list at least three SEPARATE investigation angles that use different keywords and look at different evidence types — for example direct event reporting, official statements/data, betting markets or polls, expert analysis, base rates from similar past events, contradictory or skeptical sources. Re-phrasing the same query is NOT a new angle.

3. **Search iteratively, reflect after every result.** Each time `web_search` returns:
   - In your reasoning, paraphrase what the most relevant snippets actually said (do not copy them).
   - Tag each result as relevant / partially relevant / off-topic, and note its publication date.
   - Identify what new sub-question or contradiction the result raises.
   - Pick the next query to fill the largest remaining gap or cover an angle you have NOT yet exercised. Avoid near-duplicate queries.

4. **Cross-validate.** Do not commit to an answer until at least two independent sources (different domains, different publication dates) corroborate the key facts driving it. If only one source supports the conclusion, search again from a fresh angle.

5. **Stress-test the opposite.** Explicitly ask "What is the strongest case for the OPPOSITE outcome?" If you cannot articulate it from the evidence you have gathered, run a search aimed at that counter-case. Then weigh both sides.

6. **Calibrate, then commit.** State your confidence (low / medium / high), the single most likely failure mode of your prediction, and the most decisive piece of evidence. Only after this self-check, output the final answer in the exact required `\\boxed{...}` format on the last line.

**Quality bar.** A confident answer with only one search is almost always under-researched on this benchmark; multi-angle searches with explicit reflection consistently outperform one-shot guesses. Use the `web_search` tool generously — your search budget is large enough to support thorough investigation. Do NOT skip the protocol even if the answer feels obvious.
"""


# Belief protocol (v4). Appended AFTER reflection protocol when both are enabled.
# Like reflection it's a tail-attachment to the user message: NOT part of
# `prompt_templates`, so `prompt_templates_hash` is unchanged; the fingerprint
# lives in `run_meta.belief_protocol_text` / `belief_protocol_hash`.
#
# Design intent: convert the model's implicit "evidence -> chosen letter" into
# an explicit, scoreable probability vector, so the analysis layer can compute
# proper scoring rules (Brier / NLL / MBS), calibration (ECE / Murphy
# decomposition), and behavioral metrics (belief evolution, confidence
# diagnosis) without re-prompting. Boxed answer remains for backward compat.
BELIEF_PROTOCOL = """\

---
**Belief Protocol — emit a structured probability block BEFORE every `\\boxed{...}` answer.**

After completing the forecasting protocol above, output a `<belief>...</belief>` block on its own line(s), THEN the final `\\boxed{...}` answer. The belief block is the object that gets scored by proper scoring rules (Brier / NLL); the boxed answer is kept only for backward compatibility. Probabilities matter — being well-calibrated beats being overconfident, even when the boxed letter is the same.

**Required JSON schema** (strict; emit valid JSON only — no comments, no trailing commas):

```
<belief>{
  "version": "v4.0",
  "probabilities": { "<letter>": <float in [0, 1]>, ... },
  "confidence": "low" | "medium" | "high",
  "key_evidence":     [ "<= 280 chars per bullet, 1-4 bullets",  ... ],
  "counterevidence":  [ "<= 280 chars per bullet, 0-3 bullets",  ... ],
  "open_questions":   [ "<= 280 chars per bullet, 0-3 bullets",  ... ],
  "decision_rule": "argmax" | "multi-select@<threshold>"
}</belief>
```

**Probability rules.**
- Use the SAME letter labels (`A`, `B`, ...) shown in the question's outcomes block. Emit one entry per outcome — every letter present in the question MUST appear as a key.
- For single-answer questions (yes_no / binary_named / multiple_choice with one true label): the values MUST sum to `1.0` (tolerance `1e-3`).
- For multi-select questions: each value is an INDEPENDENT Bernoulli (probability that THAT outcome is in the true set); values do NOT need to sum to 1.
- Each value MUST lie in `[0, 1]`. Avoid the boundary unless you really mean it: `0.99` says "I'd accept long odds against this being wrong"; pick `0.85` if you would not.

**Calibration norms.**
- Calibrated probabilities are the goal. If you searched once and feel "maybe 60%", emit `0.6`, not `0.9`. Overconfidence is penalised quadratically (Brier) and exponentially (NLL).
- `confidence` is your subjective bucket; pair it honestly with the spread of `probabilities`. `"high"` + flat distribution is incoherent.
- `key_evidence` lists the strongest 1-4 facts that DROVE your top probability. `counterevidence` is the best case AGAINST your top choice — emit it even when you reject it; an empty `counterevidence` on a non-trivial question is a red flag.
- `decision_rule` is `"argmax"` for single-answer; for multi-select, use `"multi-select@<t>"` with `<t>` your inclusion threshold (e.g. `"multi-select@0.5"`).

**Output order (exact).**
1. Your reasoning / search reflection (free text — anywhere in the message).
2. The `<belief>{...}</belief>` JSON block.
3. The final `\\boxed{...}` answer on the LAST line.

If you cannot produce a valid belief block, emit your boxed answer anyway — the boxed path is independent and will still be scored. But a missing or malformed belief block costs you the probabilistic family of metrics for this sample, so default to emitting one whenever your reasoning supports it.
"""


def _build_nudge_message(*, searches_done: int, min_required: int) -> str:
    """User-side reminder injected when the LLM tries to finalise too early.

    Phrased as concrete process feedback so the model knows what behaviour to
    change, not as scolding. The message never reveals end_date or any other
    information-barrier internal — it only references the public search count
    and the public minimum from `.env`.
    """
    return (
        f"You attempted to give a final answer after only {searches_done} "
        f"web_search call(s), but the forecasting protocol requires consulting "
        f"at least {min_required} sources from DIFFERENT angles before "
        "committing. Please continue the investigation: pick a NEW angle you "
        "have not yet exercised (e.g. an opposing viewpoint, an independent "
        "data source, or a base-rate comparable), issue another `web_search`, "
        "then reflect on what it adds before you finalise. Do not repeat a "
        "near-duplicate of any previous query."
    )


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


def render_user_prompt(
    q: Question,
    templates: dict[str, str],
    *,
    reflection_protocol: str | None = None,
    belief_protocol: str | None = None,
) -> str:
    """Assemble the single user message handed to the LLM for one sample.

    All template text lives in `templates` (synced from dataset_metadata). This
    function only branches on `q.question_type` and handles the three
    rendering rules from the prompt-rendering spec.

    `reflection_protocol` and `belief_protocol`, if provided, are appended
    verbatim after the canonical template body — reflection first, belief
    last. Neither is part of `prompt_templates`, so the run's
    `prompt_templates_hash` stays bit-identical to runs without either
    protocol; their presence is recorded via `Settings.config_snapshot` in
    `run_meta` (plus the dedicated `..._protocol_text` / `..._protocol_hash`
    columns) and per-sample via the `user_prompt` field. Passing neither
    yields the v3 byte-identical rendering.
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

    rendered = templates["prompt_template"].format(
        agent_role=templates["agent_role"],
        event=q.event,
        end_time=q.end_time,
        outcomes_block=outcomes_block,
        output_format=output_format,
        guidance=templates["guidance"],
    )
    if reflection_protocol:
        rendered = rendered + reflection_protocol
    if belief_protocol:
        rendered = rendered + belief_protocol
    return rendered
