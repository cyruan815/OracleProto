# OracleProto — Technical Framework

> This document is the engineering specification of the OracleProto reference implementation.
> It is the bridge between the paper's formal run unit
> $\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)$ and the
> Python codebase: every formal symbol is traced to a module, function, environment variable,
> SQLite column, and the unit test that pins the invariant. Read alongside `paper/main.tex`
> (formal framework) and `DESIGN.md` (rationale behind each trade-off).

---

## 1. Project goal

This codebase is the reference implementation of **OracleProto**: a reproducible framework for
benchmarking the *native forecasting capability* of LLMs via knowledge cutoffs and temporal
masking, instantiated in this paper on a curated FutureX-Past subset and evaluated on six
contemporary LLMs. The goal in one paragraph:

> Materialise a run unit $\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi,
> \Gamma)$ such that the same configuration produces byte-equivalent intermediate artefacts
> and stochastic-only differences in final-answer text — and bind the auxiliary leakage
> detector $H_{\mathrm{aux}}$ via a SHA-256 fingerprint to the run metadata so the leakage
> barrier itself is byte-reproducible. Every section below answers a single question: *how
> does this realise some component of $\mathcal{R}$, and which test guarantees it?*

### 1.1 Run unit ↔ implementation grand map

The paper's tuple $\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi,
\Gamma)$, plus the auxiliary detector $H_{\mathrm{aux}}$ logged as run metadata, maps to the
codebase as follows. Every symbol resolves to *one* configuration knob, *one* code path, *one*
DB column (where applicable), and *one* test that pins the contract.

| Symbol             | Object                          | Env / config key                                | Code path                                                                      | DB column / artefact                              | Pin test                          |
| ------------------ | ------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------- | --------------------------------- |
| $\mathcal{D}$      | Discrete forecasting dataset    | `SOURCE_DB`, `SOURCE_TABLE`                     | `loader.sync_questions` (loader.py:L77)                                        | `questions` table; `manifest.hashes.source_db`     | `test_db.py`, `test_evaluation.py` |
| $M$                | Evaluated model (slug)          | `MODELS` (CSV)                                  | `runner._resolve_settings` (runner.py:L160); `llm.chat` per model              | `run_meta.model`; one DB file per model            | `test_runner_grid_model.py`        |
| $\kappa_M$         | Knowledge cutoff per model      | `MODEL_TRAINING_CUTOFFS=<slug>=YYYY-MM-DD,...`  | `runner.build_task_plan` admissibility filter (runner.py:L132–L199)            | `run_meta.training_cutoff`; `s{i}_error="skipped_training_cutoff"` | `test_training_cutoff.py`         |
| $\delta$           | Temporal masking offset (days)  | `TAVILY_END_DATE_OFFSET_DAYS` (default `-1`)    | `react._compute_end_date` (react.py:L182); `search.tavily_search`              | `s{i}_search_calls[*].end_date`                    | `test_search.py`, `test_react.py` |
| $T$                | Max ReAct steps                 | `REACT_MAX_STEPS` (default `12`)                | `react.run_react` outer loop (react.py:L248)                                   | `s{i}_react_steps`; `s{i}_step_metrics`            | `test_react.py`                   |
| $C$                | Max search calls (grid axis)    | `REACT_MAX_SEARCH_CALLS` (CSV; default `[8]`)   | budget gate `react.py:L276–L279`; tool-call validation L429–L503               | virtual slug `::c{C}`; `s{i}_tool_calls_count`      | `test_react.py`, `test_grid_slug.py` |
| $R$                | Input renderer                  | `dataset_metadata.features_json.prompt_reconstruction` | `prompts.render_user_prompt` (prompts.py:L447)                          | `s{i}_user_prompt`; `manifest.hashes.prompt_templates` | `test_prompts.py`                |
| $\Psi$             | Output parser & validity        | (no env knob)                                   | `parser.parse_answer` (parser.py:L40–L89)                                       | `s{i}_final_answer_letters`, `s{i}_parse_ok`        | `test_parser.py`                  |
| $\phi$             | Answer normalisation map        | (letter encoding rule; see §3.7)                | `parser.parse_gt`, `parser.is_correct` (parser.py:L102)                         | `s{i}_correct`                                     | `test_parser.py`                  |
| $\Gamma$           | Aggregation rule                | `COMPOSITE_WEIGHTS_*`, `SAMPLING_N`, etc.       | `forecast_eval/analysis/*` (auto-invoked from `evaluation.py`)                  | CSV / MD / JSON in `runs/{run_id}/analysis/`        | `test_analysis.py`                |
| $H_{\mathrm{aux}}$ | Leakage detector (Stage-2)      | `ENABLE_SEARCH_LEAK_FILTER`, `LEAK_DETECTOR_*`  | `leak_filter.filter_search_result` (leak_filter.py:L348)                        | `s{i}_search_calls[*].audit.detector_*`             | `test_leak_filter.py`             |
| $\hat{p}_{q,j}$    | Belief vector (v4 companion)    | `BELIEF_PROTOCOL` (default `False`)             | `parser.parse_belief`; `react.run_react` finalisation (react.py:L598–L632)      | `s{i}_belief_final`, `s{i}_belief_trace`, `s{i}_belief_parse_ok` | `test_parser_belief.py`, `test_react_reflection.py` |

The auxiliary axis $R_{\mathrm{tav}}$ — Tavily results-per-call, distinct from the renderer
symbol $R$ — corresponds to `TAVILY_MAX_RESULTS` (CSV, default `[5]`); it is grid-scannable
along with $C$ and contributes to the virtual slug `{real_model}::r{R}::c{C}` (§13).

### 1.2 Hard constraints derived from $\mathcal{R}$

These are the paper-level guarantees the implementation MUST hold; each is enforced by code
*and* by at least one test, so a failing test invalidates the run unit.

1. **The LLM never sees $\chi_i$.** The `web_search` tool schema exposed to the LLM has only
   a `query` parameter (tools.py:L7–L24); $\chi_i = \tau_i + \delta$ is hard-coded by the tool
   implementation layer (react.py:L182, search.py:L133). Pinned by `test_search.py` (the
   payload contract) and `test_react.py` (end-to-end injection).
2. **Provider-native browsing is forbidden.** No `:online` slug, no `plugins` field, no
   provider-specific web tool. `Settings` validation rejects `:online` and `::` at startup
   (config.py:L599–L614); `llm.chat` re-asserts on the wire (llm.py:L74–L98). Pinned by
   `test_llm_no_browsing.py` and `test_config.py`.
3. **Sample admission is upstream of LLM calls.** The $\kappa_M \le \chi_i$ check happens at
   task-plan generation; admissibility violations write
   `error="skipped_training_cutoff"` rows directly without consuming any LLM/Tavily budget
   (runner.py:L132–L199, L190–L193). Pinned by `test_training_cutoff.py`.
4. **Strict frozenset equality scores answers.** `parser.is_correct(pred, gt)` is one line:
   `pred == gt` (parser.py:L102–L106). All three question types reduce to this. Pinned by
   `test_parser.py`.
5. **DBs store raw observations only.** No aggregates, no derived metrics; every metric in
   §11 is computed post-hoc by `forecast_eval.analysis` reading the wide table. Pinned by
   `test_analysis.py` (analysis runs on a hand-crafted DB fixture without re-touching it).
6. **Stage-2 detector $H_{\mathrm{aux}}$ has a closed input whitelist.** Only `title / url /
   published_date / content / raw_content / cutoff_date` enter the detector prompt
   (leak_filter.py:L212–L227); the question text, options, and answer are *never* passed.
   Pinned by `test_leak_filter.py`.
7. **Three independent fingerprints rather than one.** `prompt_templates_hash`,
   `reflection_protocol_hash`, `belief_protocol_hash` are stored side-by-side in
   `run_meta` and at the manifest top level (db.py:L143–L150, evaluation.py:L171–L178), so
   ablations along {template, reflection, belief} axes do not collide.
8. **Composite accuracy is the headline.** The default subtype weights are `yes_no=0.15`,
   `binary_named=0.15`, `multiple_choice=0.70` (config.py:L365–L368); per-metric overrides
   are validated against a known-metrics allowlist that fails-fast on typos
   (config.py:L515–L535, composite.py:L77–L127). Pinned by `test_composite_score.py`.

### 1.3 Numerical defaults: ship-by-default vs paper main run

The example DB and the configuration ship with a *deeper search-budget configuration* than
the paper's main run. This is intentional: the paper's $R_{\mathrm{tav}}\cdot C = 5\cdot 4 =
20$ matches "two pages of Google search results" as a deliberately-tight budget for
discrimination, while the codebase defaults trade a wider budget for smoother behavioural
analysis.

| Knob                          | Paper main run       | Codebase default                | Notes                                                                |
| ----------------------------- | -------------------- | -------------------------------- | -------------------------------------------------------------------- |
| Dataset $|\mathcal{D}|$       | 80 (curated)         | 319 (`forecast_eval_set_example.db`) | Paper used a 80-question curated subset; example DB is broader and reproducible |
| Date range                    | 2026-03-11 – 2026-04-14 | 2026-01-15 – 2026-04-14         | Example DB spans the full quarter                                     |
| $T$ (max ReAct steps)         | 12                   | 12 (`REACT_MAX_STEPS`)           | Identical                                                             |
| $C$ (max search calls)        | 4                    | 8 (`REACT_MAX_SEARCH_CALLS=[8]`) | Codebase default doubles paper main; both are list-valued for grid    |
| $R_{\mathrm{tav}}$            | 5                    | 5 (`TAVILY_MAX_RESULTS=[5]`)     | Identical                                                             |
| $\delta$                      | 1 day (= `-1` offset)| `-1` (`TAVILY_END_DATE_OFFSET_DAYS`) | Identical sign convention; default strict                          |
| $n$ (samples per question)    | 3                    | 5 (`SAMPLING_N`)                 | Codebase default is 5; tighten to 3 to mirror paper                    |
| Belief protocol               | off (paper)          | off (`BELIEF_PROTOCOL=False`)    | Identical; v3 strict-letter mode                                       |
| Reflection protocol           | on                   | on (`REACT_REFLECTION_PROTOCOL=True`) | Identical                                                       |
| Stage-2 detector              | on                   | on (`ENABLE_SEARCH_LEAK_FILTER=True`) | Identical                                                      |
| Force-final-near-limit        | on                   | on (`REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=True`) | Identical, `LOOKAHEAD=2`                                |
| Budget-aware drop-tools       | on                   | on (`REACT_BUDGET_EXCEEDED_DROP_TOOLS=True`) | Identical                                                  |
| Final-answer-retry            | off                  | off (`REACT_FINAL_ANSWER_RETRY=False`) | v5.1 backstop; off by default for byte-equivalence with v5      |
| Min-search nudge              | off                  | off (`REACT_MIN_SEARCH_CALLS=0`)  | Soft floor; opt-in fallback                                            |

To exactly reproduce the paper's main run on the example DB:

```ini
SOURCE_DB=./forecast_eval_set_example.db
SOURCE_TABLE=forecast_eval_set_example
SAMPLING_N=3
REACT_MAX_STEPS=12
REACT_MAX_SEARCH_CALLS=4
TAVILY_MAX_RESULTS=5
TAVILY_END_DATE_OFFSET_DAYS=-1
REACT_REFLECTION_PROTOCOL=true
REACT_BUDGET_AWARENESS_PROTOCOL=true
REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=true
REACT_FORCE_FINAL_ANSWER_LOOKAHEAD=2
REACT_BUDGET_EXCEEDED_DROP_TOOLS=true
REACT_FINAL_ANSWER_RETRY=false
ENABLE_SEARCH_LEAK_FILTER=true
BELIEF_PROTOCOL=false
```

Then declare each model's $\kappa_M$ via `MODEL_TRAINING_CUTOFFS` per paper Table 2 (the
six published cutoffs cover $\kappa_M$ ∈ {2025-09-29, 2026-02-11, 2026-02-25, 2026-02-12,
2026-01-27, 2026-03-10}).

---

## 2. Data source $\mathcal{D}$

### 2.1 Source database `forecast_eval_set_example.db` (read-only)

The example dataset shipped with the repo is named `forecast_eval_set_example.db`, with main
table `forecast_eval_set_example`. Both are configurable via `.env`'s `SOURCE_DB` /
`SOURCE_TABLE`; with a custom dataset, keep the 7-column schema and `dataset_metadata`
structure. `SOURCE_TABLE` accepts only SQLite-legal identifiers
`^[A-Za-z_][A-Za-z0-9_]*$` and is validated at startup (config.py:L586–L595) — a
SQL-injection defence because it is interpolated into queries verbatim.

Main table `<SOURCE_TABLE>`, **N rows × 7 columns**:

| Field           | Type    | Description                                                                                                             |
| --------------- | ------- | ----------------------------------------------------------------------------------------------------------------------- |
| `id`            | TEXT PK | Unique question ID (sourced from HuggingFace).                                                                            |
| `choice_type`   | TEXT    | `single` \| `multi`, per `\| answer letter count \|` (1 → `single`, >1 → `multi`).                                       |
| `question_type` | TEXT    | `yes_no` \| `binary_named` \| `multiple_choice`; selects the prompt template family.                                    |
| `event`         | TEXT    | Event description $x_i$ — *no* options, *no* role-setting, *no* format requirements.                                    |
| `options`       | TEXT    | $\mathcal{A}_i$ as a JSON array. `yes_no`=`["Yes","No"]`; `binary_named`=two entity names; `multiple_choice`=A/B/C labels. |
| `answer`        | TEXT    | $Y_i$ encoded as letters: single `'A'`, multi `'A, B'` (comma + space). Letter↔index rule in §3.7.                       |
| `end_time`      | TEXT    | Resolution time $\tau_i$ (Asia/Shanghai), `YYYY-MM-DD`.                                                                  |

Indexes: `idx_<table>_choice_type` / `idx_<table>_question_type` / `idx_<table>_end_time`.

Auxiliary table `dataset_metadata` (single row), with `features_json` recording all prompt
templates, column descriptions, and conversion logs. The renderer $R$ reads templates from
this table at runtime; **do not hard-code them in source**. The `prompt_templates_hash`
fingerprint covers exactly the eight template keys listed in §2.3 — protocol additions
(reflection / budget-awareness / belief) live as runtime slots and do *not* enter the hash.

### 2.2 The example dataset

`forecast_eval_set_example.db` contains **319 questions** spanning 2026-01-15 to 2026-04-14:

| question_type / choice_type | single | multi | total |
| --------------------------- | -----: | ----: | ----: |
| `yes_no`                    |     93 |     0 |    93 |
| `binary_named`              |     11 |     0 |    11 |
| `multiple_choice`           |    181 |    34 |   215 |
| **total**                   |  **285** | **34** | **319** |

`multiple_choice` option count range: 3 ~ 35 (when > 26 the letter encoding enters the ASCII
continuation; see §3.7).

The paper's main experimental run uses a **curated 80-question subset of FutureX-Past** with
the same schema, to keep the leakage-audit cost bounded. The framework itself is
dataset-agnostic once the 7-column contract and `dataset_metadata` shape are met.

### 2.3 `dataset_metadata.features_json.prompt_reconstruction` contract

The renderer $R$ requires exactly these eight keys (loader.py:L13–L22; missing keys raise at
load time):

```text
agent_role
guidance
prompt_template
outcomes_block_rule
yes_no_output_format
binary_named_output_format
multiple_choice_single_output_format
multiple_choice_multi_output_format
```

`db.compute_prompt_templates_hash(templates)` (db.py:L397–L399) computes
`sha256(canonical_kv_string(templates))` over these keys *only*. Whether the run enabled
reflection / belief / budget-awareness is invisible to this hash; those texts are hashed
separately (§5.2).

### 2.4 Examples

`yes_no`:

```yaml
event:    "2026 a dream year for trump?"
options:  ["Yes","No"]
answer:   "B"            # B = No
end_time: "2026-01-31"
```

`binary_named`:

```yaml
event:    "Golden Knights vs. Kings"
options:  ["Golden Knights","Kings"]
answer:   "A"            # A = Golden Knights
end_time: "2026-01-15"
```

`multiple_choice` (single):

```yaml
event:    "Bank of Brazil decision in January?"
options:  ["No change in the Selic rate ...", "the Bank of Brazil raise ...", "the Bank of Brazil lower ..."]
answer:   "A"
end_time: "2026-01-27"
```

`multiple_choice` (multi):

```yaml
event:    "Oscars 2026: Achievement in Casting Nominations"
options:  [<12 nominee list entries>]
answer:   "A, B, D, E"
end_time: "2026-01-22"
```

> Convention: the `event` field carries no options or format requirements; those are
> spliced in at call time by the renderer $R$ (§3.6).

---

## 3. The information boundary in code (paper §2.3, §3, §4.4)

The paper organises leakage control around **three controlled information channels** plus a
documented residual surface. The codebase implements each channel at one specific layer.

### 3.1 Channel 1 — Parametric knowledge: admissibility filter $\kappa_M \le \chi_i < \tau_i$

**Motivation.** A question whose resolution time precedes the model's training cutoff is
likely already in the training corpus — the model "remembers" the answer rather than
forecasting it (paper §2.1, Eq. 4). Such samples cannot reflect native forecasting
capability and are removed from $\mathcal{D}^{\mathrm{pred}}_M$.

**Mechanism.**

* Per-model $\kappa_M$ is declared in `.env` via `MODEL_TRAINING_CUTOFFS` (CSV
  `<slug>=YYYY-MM-DD,...`), parsed by `config._parse_cutoffs` (config.py:L479).
* During task-plan generation, `runner.build_task_plan` filters every `(question, model)`:

  ```python
  cutoff = MODEL_TRAINING_CUTOFFS.get(real_model)   # None = not declared, no filtering
  if cutoff is not None and q.end_time <= cutoff:
      # day-rounded equivalent of χ_i < κ_M for δ = -1 day:
      # writes one row per sample_idx with error="skipped_training_cutoff"
      enqueue_skipped_cutoff_rows(q, model)
  ```

* Filtered `(question, model, sample_idx)` rows still land in `run_results` with
  `error="skipped_training_cutoff"`, `parse_ok=0`, `correct=NULL`, and all numeric fields 0.
  This makes "how many questions filtered per model" auditable (paper Table 2's
  "Excluded by Cutoff" column).
* Resume-from-checkpoint never re-attempts these rows (§5.3).

**Pin test.** `test_training_cutoff.py`: ① every N samples for `q.end_time <= cutoff` writes
`skipped_training_cutoff`; ② models without a declared cutoff are not filtered; ③ resume
takes precedence over cutoff (already-completed rows are not replaced).

### 3.2 Channel 2 — Tool-mediated knowledge: temporal masking $\delta$

**The LLM never proposes $\chi_i$.** The schema of `web_search` exposed to the LLM has only a
`query` parameter (tools.py:L7–L24):

```python
{
  "type": "function",
  "function": {
    "name": "web_search",
    "description": "Search the web for information relevant to the question.",
    "parameters": {
      "type": "object",
      "properties": {"query": {"type": "string", "description": "Search query"}},
      "required": ["query"],
      "additionalProperties": false
    }
  }
}
```

When Tavily is actually called, $\chi_i = \tau_i + \delta$ is hard-coded by the tool
implementation (react.py:L182; search.py:L133–L162):

```python
end_date = (date.fromisoformat(q.end_time)
            + timedelta(days=settings.TAVILY_END_DATE_OFFSET_DAYS)).isoformat()
result = await search.tavily_search(query=args["query"], end_date=end_date, settings=cfg)
```

**Default $\delta = -1$ day.** With `end_time` at `YYYY-MM-DD` granularity, the default
`TAVILY_END_DATE_OFFSET_DAYS=-1` excludes same-day information:

```text
question.end_time (τ_i)  = 2026-01-18
→ Tavily end_date (χ_i) = 2026-01-17
```

`δ = 0` (lenient) and `δ ∈ {-2, -3}` (more conservative) are valid, but reports default to
$\delta = -1$ comparisons; numbers under different offsets are not directly comparable.

**Pin tests.** `test_search.py` (the `web_search` schema does not contain `end_date`, and
`tavily_search` injects the right `end_date` when called); `test_react.py` (the in-loop
injection wires through correctly).

### 3.3 Channel 3 — Retrieval-content audit: Stage-2 detector $H_{\mathrm{aux}}$

Tool-level filtering constrains *request time*, but returned snippets, cached pages, or
aggregate summaries can still carry post-$\chi_i$ content. The Stage-2 detector
(`leak_filter.py`) audits each Tavily result *before* it enters the main LLM context.

**Independent client.** `_detector_client` (leak_filter.py:L108–L133) is a separate
`AsyncOpenAI` instance distinct from the main `llm._client`, configured by
`LEAK_DETECTOR_*` environment variables. If `LEAK_DETECTOR_BASE_URL` is empty, it falls
back to `LLM_BASE_URL`.

**Cut point.** End of the HTTP-200 path in `search.tavily_search`, before `return` —
verdicts are applied by `leak_filter.filter_search_result` (leak_filter.py:L348) which
walks the result list and drops items per verdict.

**Input whitelist (load-bearing).** The detector prompt receives ONLY these six fields per
result (leak_filter.py:L212–L227):

```text
title           — result.title
url             — result.url
published_date  — result.published_date or "(unknown)"
content         — result.content
raw_content     — result.raw_content or "(empty)"
cutoff_date     — the χ_i passed in by the caller
```

The question text, options, and gold answer are **never** passed; framing the detector as
an "answer auditor" would create second-order leakage (the detector might rationalise that
a fabricated piece of evidence "is consistent with the answer it knows" and let it through).

**Output schema.** Strict JSON, two fields:

```json
{"verdict": "keep" | "drop", "reason": "<sentence>"}
```

A `drop` removes the entire result (title / URL / content / raw_content all withdrawn)
before the main LLM sees anything; the audit fields are retained for post-hoc review.

**Failure mode (fail-closed default).** Retry sequence
`max_attempts = LEAK_DETECTOR_RETRY_MAX + 1` (default 3 retries, backoff `[2,5,15]`s).

* `401/403` (AUTH): caught locally, converted to `failed:auth`, never propagated; immediate
  fail-closed drop (leak_filter.py:L281–L288).
* Other retryables (network, rate-limit, server 5xx): exhaust retry sequence, then verdict
  becomes `failed:<kind>` and fall-back action applies.
* Fall-back action: `LEAK_DETECTOR_FAIL_ACTION=drop` (default) drops the item,
  `keep` lets it through. The default is fail-closed: detector hiccups (timeout, network)
  are uncorrelated with item content, so biasing the residual towards "drop on uncertainty"
  is the conservative choice.

**Audit fields appended to `s{i}_search_calls[*].audit`** (leak_filter.py:L380–L387):

| Field                  | Type         | Meaning                                               |
| ---------------------- | ------------ | ----------------------------------------------------- |
| `n_results_raw`        | int          | Count before filtering                                 |
| `n_results_kept`       | int          | Count after filtering                                  |
| `published_dates_raw`  | list[str]    | Original publish dates of all items (audit invariant)  |
| `detector_verdicts`    | list[str]    | Per-item verdict; values: `keep` / `drop` / `failed:*` |
| `detector_latency_ms`  | int          | Wall-clock detector latency                            |
| `detector_error_kind`  | str \| null  | Dominant failure kind across the batch                 |

**Three-key fingerprint** in `run_meta.config_snapshot` plus a top-level slot:

```text
leak_detector_enabled         — bool
leak_detector_model           — str
leak_detector_prompt_hash     — sha256[:16] of LEAK_DETECTOR_PROMPT_TEMPLATE
leak_detector_prompt_version  — human-readable label (default "v1")
```

When `ENABLE_SEARCH_LEAK_FILTER=False` the detector path is byte-level rolled back and
behaviour is identical to v5.1 without the detector.

**Audit results from the paper.** Manual sampling of N=270 items (30 questions × 3 models ×
3 trials, drawing one search result per test item) yields:

| Metric                                    | Value                |
| ----------------------------------------- | -------------------- |
| TP (real leak, detector dropped)          | 235 (87.0%)          |
| TN (no leak, detector kept)               | 31 (11.5%)           |
| FP (no leak, detector dropped)            | 1 (0.4%)             |
| FN (real leak, detector kept)             | 3 (1.1%)             |
| Recall (TP / (TP+FN))                     | **98.7%**             |
| Specificity (TN / (TN+FP))                | 96.9%                 |
| Per-audit-item residual leakage (FN/N)    | **1.1%**              |
| Wilson 95% upper bound on residual        | **3.2%**              |
| Leak-conditional pass-through (FN/(TP+FN)) | 1.3%                  |

By comparison, the no-detector baseline yields ≈ 100% leak-conditional pass-through, and
the Tavily-only baseline 3–16%; the detector reduces residual leakage by an order of
magnitude relative to Tavily alone.

**Pin test.** `test_leak_filter.py` (550 LOC): ① detector input fields whitelist enforced
(no `Question` fields leak in); ② fail-closed on retry exhaustion; ③ AUTH errors
immediate-fail-closed without propagation; ④ `search_calls` JSON entry contains
`n_results_raw / n_results_kept / detector_verdicts / detector_latency_ms /
detector_error_kind`; ⑤ disabled path is byte-equivalent to v5.1.

### 3.4 Channel 4 — Provider-side leak surface: `:online` ban + plugins ban

A model service may attach a built-in browsing tool that bypasses the Tavily layer.
Two defences:

1. **Slug ban.** Slugs ending in `:online` (OpenRouter's online-augmented variants) are
   rejected at startup by `Settings` validation (config.py:L599–L614) and re-asserted on
   the wire by `llm._assert_no_browsing` (llm.py:L74–L98). Pinned by `test_llm_no_browsing.py`.
2. **`plugins` field ban.** `extra_body.plugins` is rejected on the wire (llm.py:L97);
   only the single `[WEB_SEARCH_SCHEMA]` tool is permitted in `tools=[...]`.

A provider that forcibly attaches an unhideable browsing capability should be marked as
"unsuitable for strict evaluation" in the README/reports — the framework cannot defend
against capabilities the API does not expose.

### 3.5 Threat model and residual surface

| Leakage source                              | Controllable? | Mitigation in code                                                                       |
| ------------------------------------------- | ------------- | ---------------------------------------------------------------------------------------- |
| Tool search content (Tavily returned text)  | ✅            | `end_date = end_time + δ` injected by tool layer (§3.2)                                  |
| Provider-native browsing / web tool         | ✅            | `:online` ban; `plugins` ban; single-tool whitelist (§3.4)                               |
| Page bodies that mention post-$\chi_i$ events | ⚠️ Partial   | Stage-2 LLM detector with whitelisted inputs and fail-closed default (§3.3)              |
| Model parametric memory                     | ⚠️ Partial    | $\kappa_M$ admissibility filter excludes $\chi_i < \kappa_M$ samples (§3.1)              |
| Time clues in the question text             | ❌            | Inherent to the data; accepted as evaluation bias                                         |
| External knowledge backflow after training  | ❌            | Accepted as evaluation bias                                                               |

The first four lines are the *controlled channels* of paper §3.5; the last two are the
documented residual. The paper's framing is explicit: this is an auditable, reproducible,
comparable framework — not a proof that every leak is closed.

### 3.6 Prompt assembly: the renderer $R$

The source DB stores only raw material (`event` / `options` / `question_type` / `end_time`).
When the system spawns a sample, `prompts.render_user_prompt` reads templates from
`dataset_metadata.features_json.prompt_reconstruction` and assembles the user message
per `question_type`:

```text
{agent_role} The event to be predicted: "{event} (resolved around {end_time} (GMT+8)).{outcomes_block}"

IMPORTANT: Your final answer MUST end with this exact format:
{output_format}
{guidance}
```

Per-slot rendering rules:

| Slot              | Rendering logic                                                                                                                                 |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `agent_role`      | Constant `"You are an agent that can predict future events."`, inserted as-is                                                                    |
| `event`           | `<SOURCE_TABLE>.event` original text                                                                                                             |
| `end_time`        | `<SOURCE_TABLE>.end_time` original text (`YYYY-MM-DD`)                                                                                            |
| `outcomes_block`  | `yes_no` / `binary_named` → empty string (options implicit in `output_format`); `multiple_choice` → newline + `A. <opt[0]>\nB. <opt[1]>\n...` per §3.7 |
| `output_format`   | One of three (per `question_type`): `yes_no_output_format` / `binary_named_output_format` / `multiple_choice_output_format`. The `binary_named` template contains `<options[0]>` / `<options[1]>` placeholders that MUST be replaced with the actual two entity names |
| `guidance`        | Constant `"Do not use any other format. Do not refuse to make a prediction. ..."`, inserted as-is                                                  |

Output-format shapes:

* `yes_no` — requires `\boxed{Yes}` or `\boxed{No}`.
* `binary_named` — after rendering looks like `\boxed{Golden Knights} or \boxed{Kings}`.
* `multiple_choice` — requires `\boxed{A}` or `\boxed{B, C}`, with an example attached.

The reflection / budget-awareness / belief protocol additions are appended *at runtime*
when their switches are on (§4.2); they do not enter `dataset_metadata`, so
`prompt_templates_hash` is unaffected. The fully rendered user message lands in each
sample's `s{i}_user_prompt` field. Protocol-text fingerprints are persisted independently
in `run_meta.reflection_protocol_hash` and `run_meta.belief_protocol_hash`.

### 3.7 Answer encoding ↔ decoding (the map $\phi$)

The DB uniformly uses **letters** as the canonical answer; the LLM's output form varies by
`question_type`:

| question_type      | LLM output (inside `\boxed{}`)                                       | Parser normalisation target ($\phi$)                                |
| ------------------ | -------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `yes_no`           | `Yes` / `No` (case-insensitive)                                       | `frozenset({"A"})` / `frozenset({"B"})` — `Yes`=A, `No`=B            |
| `binary_named`     | one of the entries in `options` (trim + case-insensitive exact match) | look up the index in `options` → letter → frozenset                  |
| `multiple_choice`  | one or more letters, comma- or space-separated (`A` / `B, C` / `B,C`) | split → frozenset[str]                                                |

Letter ↔ index rule (parser.py:L420–L429), supports up to 35 options:

```text
index = ord(letter) - ord('A')
A=0, B=1, ..., Z=25
[ =26, \ =27, ] =28, ^ =29, _ =30, ` =31, a =32, b =33, c =34, ...
```

> ⚠️ **Compatibility-mode warning.** The example DB has 4 `multiple_choice` questions with
> > 26 options, and 3 of them have ground-truth letters landing on non-letter symbols like
> `[ \ ] ^ _ ` ` ` a b c …`. These ASCII-continuation labels are unfriendly to LLMs
> (backticks and underscores get swallowed by markdown/code blocks; lowercase `a` and
> uppercase `A` coexist and are easily confused). We keep this scheme only to preserve a
> one-to-one mapping with the source-data letter encoding for letter-set scoring. Mandatory
> defences: ① `prompts.render_user_prompt` quotes/escapes labels when generating the
> `outcomes_block` for > 26 options; ② `parser.parse_answer` has a round-trip unit test
> (label→letter→label) for `multiple_choice` with > 26 options; ③ logs/reports record
> letters and corresponding labels in parallel for manual review.

Ground-truth reverse lookup (`answer` letters → labels, for display or logging):

```python
opts    = json.loads(row["options"])
letters = [t.strip() for t in row["answer"].split(",")]
labels  = [opts[ord(L) - ord('A')] for L in letters]
```

---

## 4. End-to-end pipeline

```text
┌────────────────────────────────────────────────────────────────────────┐
│                    OracleProto Evaluation Pipeline                     │
└────────────────────────────────────────────────────────────────────────┘

[.env]  →  [python evaluation.py [--question-type ...] [--choice-type ...]]
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │  1. Load Settings      │
                          │  & init run_id         │
                          │  (mints YYYYMMDD-      │
                          │   HHMMSS-xxxx if empty)│
                          └────────────────────────┘
                                      │
                                      ▼
                          ┌──────────────────────────────────┐
                          │  2. Sync Source                  │
                          │  forecast_eval_set_example.db    │
                          │    → questions table             │
                          │    → prompt_templates table      │
                          │  (filtered by --filter)          │
                          │  ↓                               │
                          │  hashes: source_db / metadata    │
                          │  / prompt_templates / reflection │
                          │  / belief                        │
                          └──────────────────────────────────┘
                                      │
                                      ▼
                          ┌────────────────────────────────────┐
                          │  3. Resume Check                   │
                          │  load_completed_samples per model: │
                          │  skip rows where s{i}_created_at   │
                          │  NOT NULL & error in {NULL,        │
                          │  skipped_training_cutoff}          │
                          └────────────────────────────────────┘
                                      │
                                      ▼
                 ┌────────────────────────────────────────┐
                 │  4. Task Plan (D × M × N)              │
                 │  - apply κ_M admissibility filter      │
                 │    (§3.1): write skipped_training_     │
                 │    cutoff rows directly                │
                 │  - asyncio.Semaphore for LLM/Search/   │
                 │    Detector channels                   │
                 └────────────────────────────────────────┘
                                      │
                      ┌───────────────┼───────────────┐
                      ▼               ▼               ▼
                ┌──────────┐    ┌──────────┐    ┌──────────┐
                │ Worker 1 │    │ Worker 2 │ …  │ Worker N │
                └────┬─────┘    └────┬─────┘    └────┬─────┘
                     │               │               │
                     └───────────────┼───────────────┘
                                     ▼
                      ┌────────────────────────────┐
                      │  ReAct Loop F_M (per       │
                      │  sample, see §10):         │
                      │   render(q) → user_prompt  │
                      │   ↓                        │
                      │   while step < T:          │
                      │     pre-step injection     │
                      │     pick (1 of 4 priority) │
                      │     llm.chat(messages,     │
                      │               tools=...)   │
                      │     if no tool_call:       │
                      │       maybe nudge / break  │
                      │     for each tool_call:    │
                      │       u_t = tavily.search( │
                      │              query, χ_i)   │
                      │       ũ_t = AuxLeakFilter( │
                      │              u_t, χ_i)     │
                      │   ↓ (post-loop)            │
                      │   v5.1 final-answer-retry  │
                      │   parser.parse_answer      │
                      └──────────┬─────────────────┘
                                 │
                                 ▼
                      ┌────────────────────────┐
                      │  5. Score              │
                      │  Ψ ∘ φ on the parsed   │
                      │  letter set            │
                      └──────────┬─────────────┘
                                 │
                                 ▼
                      ┌────────────────────────┐
                      │  6. Enqueue → writer   │
                      │  Single AsyncWriter    │
                      │  per model; WAL +      │
                      │  batch UPSERT          │
                      └──────────┬─────────────┘
                                 │
                                 ▼
                      ┌────────────────────────┐
                      │  7. analysis.run       │
                      │  Aggregations Γ:       │
                      │  composite acc, FSS,   │
                      │  Cohen κ, Fleiss κ,    │
                      │  pass@k, BI, NLL …     │
                      │  → CSV / MD / JSON     │
                      └────────────────────────┘
```

---

## 5. Database design (`runs/{run_id}/db/<model_slug>.db`)

Each run × model corresponds to **one independent SQLite file**. The file self-contains
copies of `questions` / `prompt_templates`, so a single file replays independently.
Aggregations and statistics are **not persisted** — `forecast_eval.analysis` writes them
post-hoc into `analysis/`.

### 5.1 Schema (current = v5)

```sql
-- ⓪ schema version table
CREATE TABLE schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- ① source-question copy
CREATE TABLE questions (
    id            TEXT PRIMARY KEY,
    choice_type   TEXT NOT NULL CHECK (choice_type IN ('single','multi')),
    question_type TEXT NOT NULL CHECK (question_type IN ('yes_no','binary_named','multiple_choice')),
    event         TEXT NOT NULL,
    options       TEXT NOT NULL,             -- JSON array
    answer        TEXT NOT NULL,             -- comma-separated letters
    end_time      TEXT NOT NULL,             -- YYYY-MM-DD
    imported_at   TEXT NOT NULL
);
CREATE INDEX idx_questions_choice_type   ON questions(choice_type);
CREATE INDEX idx_questions_question_type ON questions(question_type);

-- ② prompt-templates copy
CREATE TABLE prompt_templates (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    imported_at TEXT NOT NULL
);

-- ③ unique (run, model) metadata; single row
CREATE TABLE run_meta (
    run_id                    TEXT PRIMARY KEY,
    model                     TEXT NOT NULL,
    sampling_n                INTEGER NOT NULL,
    config_snapshot           TEXT NOT NULL,   -- redacted .env JSON
    filters_snapshot          TEXT NOT NULL,   -- {"question_types":..., "choice_types":..., "question_ids":[...], "question_count":N}
    source_db_hash            TEXT NOT NULL,
    metadata_hash             TEXT NOT NULL,
    prompt_templates_hash     TEXT NOT NULL,
    reflection_protocol_text  TEXT,            -- v3+; full text of REFLECTION_PROTOCOL when on
    reflection_protocol_hash  TEXT,            -- v3+; sha256[:16] when on
    belief_protocol_text      TEXT,            -- v4+; full text of BELIEF_PROTOCOL when on
    belief_protocol_hash      TEXT,            -- v4+; sha256[:16] when on
    training_cutoff           TEXT,            -- κ_M (YYYY-MM-DD), NULL when not declared
    started_at                TEXT NOT NULL,
    finished_at               TEXT
);

-- ④ wide table: one row per question, one s{i}_* column group per sample.
-- 24 fields × SAMPLING_N columns generated dynamically by db.init_schema.
CREATE TABLE run_results (
    question_id TEXT PRIMARY KEY,
    user_prompt TEXT,                          -- shared across samples (COALESCE; first sample wins)

    -- v2 base (14 columns)
    s0_final_answer_letters TEXT,
    s0_final_answer_raw     TEXT,
    s0_correct              INTEGER,
    s0_parse_ok             INTEGER,
    s0_tool_calls_count     INTEGER,
    s0_react_steps          INTEGER,
    s0_prompt_tokens        INTEGER,
    s0_completion_tokens    INTEGER,
    s0_reasoning_tokens     INTEGER,
    s0_latency_ms           INTEGER,
    s0_messages_trace       TEXT,
    s0_search_calls         TEXT,
    s0_error                TEXT,
    s0_created_at           TEXT,
    -- v3 observability (6 columns)
    s0_finish_reason        TEXT,
    s0_nudges_used          INTEGER,
    s0_step_metrics         TEXT,              -- JSON array of per-step snapshots
    s0_response_id          TEXT,              -- ChatCompletion.id (last round)
    s0_system_fingerprint   TEXT,              -- ChatCompletion.system_fingerprint
    s0_service_tier         TEXT,              -- ChatCompletion.service_tier
    -- v4 belief (3 columns)
    s0_belief_final         TEXT,              -- final-step Belief.probabilities JSON; NULL if parse fails / off
    s0_belief_trace         TEXT,              -- per-step belief summary JSON array
    s0_belief_parse_ok      INTEGER,           -- whether final-step belief parses (0/1); independent of parse_ok
    -- v5 harness-resilience (1 column)
    s0_final_answer_retry_used INTEGER,        -- 0/1 — see §10.3

    -- ...same s1_* / s2_* / ... groups...

    FOREIGN KEY (question_id) REFERENCES questions(id)
);
CREATE INDEX idx_run_results_question ON run_results(question_id);
```

**Schema migrations** (db.py:L222–L345) are performed via `ALTER TABLE … ADD COLUMN`:

| Version | Change                                                                                          | Migration function          |
| ------- | ----------------------------------------------------------------------------------------------- | --------------------------- |
| v2      | Base 14 per-sample columns; bare `run_meta`                                                      | `_init_v2_schema`            |
| v2 → v3 | +6 per-sample observability fields; +2 reflection columns in `run_meta`                          | `_migrate_v2_to_v3` (L222–L267) |
| v3 → v4 | +3 per-sample belief columns; +2 belief columns in `run_meta`                                    | `_migrate_v3_to_v4` (L269–L310) |
| v4 → v5 | +1 per-sample column `final_answer_retry_used`                                                    | `_migrate_v4_to_v5` (L312–L345) |

SQLite's `ADD COLUMN` is metadata-only (O(1)); old rows default to NULL. On the resume path,
the first time an old DB is opened it is auto-migrated. When `Settings.BELIEF_PROTOCOL=False`,
all belief columns write NULL and the analysis pipeline early-exits the probabilistic family.

**Connection-init PRAGMA** (executed on every `sqlite3.connect`):

```sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;
```

### 5.2 Field write conventions (per-sample columns)

| Field                             | Source                                                                                                                                                          |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `s{i}_final_answer_letters`       | `frozenset[str]` from `parser.parse_answer(final_raw, q)`, written as `json.dumps(sorted(...))`                                                                  |
| `s{i}_final_answer_raw`           | Full `content` text of the LLM's last assistant message                                                                                                          |
| `s{i}_correct`                    | `frozenset == frozenset` → `int`; `NULL` when parse fails or sample is not in $\mathcal{S}$                                                                      |
| `s{i}_parse_ok`                   | `final_answer_letters is not None` (i.e. $v_{i,M}$ from paper §3.4)                                                                                              |
| `user_prompt`                     | Return value of `prompts.render_user_prompt(q, templates, …)`; rendered once per question, retained via COALESCE                                                  |
| `s{i}_messages_trace`             | Full `messages` list as JSON; NULL when `WRITE_MESSAGES_TRACE=False`                                                                                              |
| `s{i}_search_calls`               | List of metadata for each `web_search` call: `query / end_date / n_results / published_dates`; with leak filter: + `n_results_raw / n_results_kept / detector_verdicts / detector_latency_ms / detector_error_kind` |
| `s{i}_error`                      | Error classification after retries; NULL on normal completion (including refusal / parse fail)                                                                   |
| `s{i}_created_at`                 | UTC ISO-8601 at write time; the unique signal for "this slot has been filled"                                                                                    |
| `s{i}_finish_reason`              | Last round's `ChatCompletion.choices[0].finish_reason` (`stop` / `tool_calls` / `length` / `content_filter` …); NULL for error rows                              |
| `s{i}_nudges_used`                | Count of "strict floor not met → reminder injected" within this sample; capped by `REACT_MAX_NUDGES`                                                              |
| `s{i}_step_metrics`               | JSON array of per-round snapshots: `step / prompt / completion / reasoning / latency_ms / finish_reason / n_tool_calls`                                          |
| `s{i}_response_id`                | Last round's `ChatCompletion.id`                                                                                                                                  |
| `s{i}_system_fingerprint`         | Last round's `ChatCompletion.system_fingerprint` (when provider supplies it; detects provider-side model-routing changes)                                         |
| `s{i}_service_tier`               | Last round's `ChatCompletion.service_tier`                                                                                                                        |
| `s{i}_belief_final`               | v4. JSON-serialised `Belief.probabilities` from `parser.parse_belief` at the final step; NULL when parsing fails or `BELIEF_PROTOCOL=False`                       |
| `s{i}_belief_trace`               | v4. JSON array of belief summaries for every loop step                                                                                                            |
| `s{i}_belief_parse_ok`            | v4. Whether final-step belief parses legally (0/1); **independent** of `parse_ok`                                                                                  |
| `s{i}_final_answer_retry_used`    | v5. 0/1 — set when `REACT_FINAL_ANSWER_RETRY` mopped up an empty `final_raw`                                                                                       |

**Three independent protocol fingerprints** (paper §3.5; see DESIGN.md §5.6):

* `prompt_templates_hash` — main template (renderer $R$); hashed over the eight keys of
  §2.3.
* `reflection_protocol_hash` — switch on the search-behaviour prior; varies along
  {on/off, text edits, version}. Hashed over `prompts.REFLECTION_PROTOCOL` text only.
* `belief_protocol_hash` — switch on whether probabilistic-family metrics are populated.
  Hashed over `prompts.BELIEF_PROTOCOL` text only.

All three live both in `run_meta` and at the top level of `manifest.json`
(evaluation.py:L171–L178), so "grep the protocol fingerprint without opening the DB" covers
every protocol axis.

### 5.3 Resume

Each sample slot is judged independently:

```sql
-- Per i ∈ 0..SAMPLING_N-1:
SELECT question_id FROM run_results
 WHERE s{i}_created_at IS NOT NULL
   AND (s{i}_error IS NULL OR s{i}_error = 'skipped_training_cutoff');
```

Results merge into `set[(question_id, sample_idx)]` and are removed from the task queue.
Since each model's own DB contains one run, `run_id` does not enter the filter (the single
row in `run_meta` decides it).

State classification:

| `error` value                    | Meaning                            | Retry on next resume?                              |
| -------------------------------- | ---------------------------------- | -------------------------------------------------- |
| `NULL`                           | Completed normally                 | No                                                  |
| `'skipped_training_cutoff'`      | Actively excluded by §3.1          | No                                                  |
| `'network'` / `'server_5xx'`     | Still failing after backoff        | Yes                                                 |
| `'rate_limit'`                   | Rate-limit, backoff exhausted      | Yes                                                 |
| `'bad_request'`                  | model_not_found, etc.               | Yes (after config change)                           |
| `'content_policy'`               | Provider refusal                    | Optional: default retry once and overwrite the row  |

Rules:

* Re-running with the same `run_id` = resume; writes into the existing
  `runs/{run_id}/db/<slug>.db`.
* Changing `run_id` = a fresh run; creates a new `runs/{new_run_id}/`.
* Overwrite semantics: `INSERT ... ON CONFLICT(question_id) DO UPDATE SET s{i}_* =
  excluded.s{i}_*`; `user_prompt` is preserved with `COALESCE` to keep the first sample's
  value.

**Pin tests.** `test_runner_resume.py`: ① `load_completed_samples` excludes retryable
errors; ② `build_task_plan` deduplicates by per-model completed; ③ models not declared in
`completed` default to empty set (all enqueued).

### 5.4 Concurrent-write strategy

* Every DB connection executes the four PRAGMAs at startup.
* **One async writer task per model**: `runner.run` opens one `db.AsyncWriter` per
  model DB (runner.py:L362); every worker's result is enqueued via that model's writer.
* The writer flushes every `DB_COMMIT_BATCH` entries or every 1 second
  (`AsyncWriter.FLUSH_INTERVAL_S = 1.0`); short transactions; SQLite writes go through
  `await asyncio.to_thread(...)` to avoid blocking the event loop.
* A single-model DB has one writer and multiple readers; under WAL this is safe.
* `asyncio.Queue` is not cross-thread; for cross-thread consumption, use `queue.Queue` /
  `janus.Queue`. The current design stays fully async on a single thread.

---

## 6. Directory layout

```text
Forecast/
├── .env                           # gitignored, user-filled
├── .env.example                   # template, git-managed
├── .gitignore
├── environment.yml                # conda env definition
├── README.md                      # user-facing entry
├── DESIGN.md                      # rationale (this implements design)
├── FRAME.md                       # this document
├── paper/                         # paper source (LaTeX + bib + style)
├── evaluation.py                  # main entry: parse CLI → runner.run → analysis.run_analysis
├── forecast_eval_set_example.db   # source data (read-only, **checked into Git**)
├── runs/                          # all evaluation outputs (gitignored)
│   └── {run_id}/
│       ├── manifest.json          # run-level metadata + model_files map + grid block
│       ├── db/
│       │   └── {model_slug}.db    # one sqlite per model; self-contained replay
│       ├── analysis/              # statistical artefacts (post-hoc)
│       └── logs/{run_id}.log
├── forecast_eval/
│   ├── __init__.py
│   ├── config.py                  # pydantic-settings; Settings + grid axes + composite weights
│   ├── db.py                      # per-model wide-table schema + AsyncWriter + hashes
│   ├── loader.py                  # syncs questions + prompt_templates from SOURCE_DB
│   ├── prompts.py                 # renderer R + reflection / budget-awareness / belief / harness builders
│   ├── llm.py                     # OpenAI-compatible client + tiered retry; provider-native browsing forbidden
│   ├── search.py                  # Tavily wrapper + end_date injection + Stage-2 dispatch
│   ├── leak_filter.py             # Stage-2 detector H_aux (independent client, fail-closed)
│   ├── tavily_keys.py             # multi-key TavilyKeyPool (least-used + 401/403 blacklist + 429 cooldown)
│   ├── tools.py                   # web_search schema (LLM-visible; no date)
│   ├── react.py                   # ReAct loop F_M (single sample, 4-knob harness resilience)
│   ├── parser.py                  # parser Ψ + label normalisation φ + belief parser
│   ├── errors.py                  # error classification + backoff strategy
│   ├── runner.py                  # task orchestration + multi-model writer + κ_M filter
│   ├── types.py                   # dataclasses (Question / SampleResult / SearchResult / etc.)
│   └── analysis/                  # post-hoc statistics (Γ); read DB → CSV / MD / JSON
│       ├── __init__.py            #   `run_analysis(run_dir)` orchestrator
│       ├── accuracy.py            #   strict-equality + pass@k family + FSS / Cohen κ / Hamming
│       ├── exam_score.py          #   exam-style partial credit (paper Eq. 17)
│       ├── composite.py           #   subtype-weighted composite accuracy (paper Eq. 18)
│       ├── consistency.py         #   Fleiss κ, mean entropy, VCI, MVG (K-trial)
│       ├── proper_score.py        #   BI / NLL / MBS / ABI (probabilistic companion)
│       ├── aggregation.py         #   arithmetic / logit-space mean / LOO shrinkage
│       ├── inference.py           #   paired bootstrap, Holm-Bonferroni, posterior, multi-metric
│       ├── grid.py                #   grid-search analysis (virtual slug decode, marginal / pareto / winrate)
│       ├── behavior.py            #   reflection A/B, tool-usage PDP, confidence calibration, belief evolution
│       ├── probabilistic.py       #   probabilistic family report builder
│       ├── flatten.py             #   wide-table → SampleRow + per-question grouping
│       └── writers.py             #   CSV / MD / JSON serialisers; column rounding rules
├── scripts/                       # operator scripts
│   ├── build_forecast_eval_set.py #   dataset construction (stratified sample + topic cap)
│   ├── smoke_leak_filter.py       #   smoke test of Stage-2 detector pipeline
│   ├── verify_leak_filter_e2e.py  #   end-to-end leak-filter audit reproducer
│   ├── fss_sensitivity.py         #   FSS α/β sensitivity sweep
│   ├── plot_analysis.py           #   matplotlib renders for analysis/
│   └── migrate_split_mc_output_format.py  #   one-shot dataset-metadata migration
└── tests/                         # 33 unit/integration tests (~13K LOC), all offline (§14)
```

---

## 7. `.env` configuration reference

A condensed view of the most load-bearing knobs; for the full annotated block see
`.env.example`.

```ini
# -------- LLM Endpoint (OpenAI-compatible) --------
LLM_API_KEY=REPLACE_ME
LLM_BASE_URL=https://openrouter.ai/api/v1
MODELS=openai/gpt-5,anthropic/claude-sonnet-4.5,google/gemini-2.5-pro,deepseek/deepseek-r1
# κ_M per model — declare for every evaluated model
MODEL_TRAINING_CUTOFFS=openai/gpt-5=2024-10-01,anthropic/claude-sonnet-4.5=2025-03-01,...

# LLM call parameters
LLM_MAX_TOKENS=12000           # covers reasoning + completion (3-8k reasoning tokens common)
LLM_TIMEOUT_S=240
LLM_TEMPERATURE=0.7
LLM_TOP_P=1.0
# Reasoning models: matched slugs are called WITHOUT temperature / top_p
LLM_REASONING_MODEL_PATTERNS=o1,o3,o4,r1,qwq

# LLM concurrency & retry
LLM_MAX_CONCURRENCY=5
LLM_RETRY_MAX=5
LLM_BACKOFF_NETWORK_S=2,5,15,30,60
LLM_BACKOFF_RATE_LIMIT_S=10,30,60,120,300
LLM_BACKOFF_SERVER_5XX_S=5,15,30,60,120

# -------- Web Search Master Switch --------
ENABLE_WEB_SEARCH=true

# -------- Tavily Search --------
TAVILY_API_KEY=tvly-REPLACE_ME    # single value or CSV (multi-key pool)
TAVILY_KEY_COOLDOWN_S=60          # 429 cooldown for a single key
TAVILY_MAX_RESULTS=5              # R axis (paper main: R_tav = 5; multi-value → grid)
TAVILY_SEARCH_DEPTH=basic         # basic (1 credit) | advanced (2 credits)
TAVILY_INCLUDE_RAW_CONTENT=markdown # false | markdown (default) | text
TAVILY_RAW_CONTENT_MAX_CHARS=8000 # per-result raw_content truncation
TAVILY_INCLUDE_ANSWER=false       # off (avoid second-LLM contamination)
TAVILY_END_DATE_OFFSET_DAYS=-1    # δ; project default -1 (strict)
SEARCH_MAX_CONCURRENCY=5
SEARCH_RETRY_MAX=3
SEARCH_BACKOFF_S=2,5,15

# -------- ReAct Loop --------
REACT_MAX_STEPS=12                # T (paper main: 12)
REACT_MAX_SEARCH_CALLS=8          # C axis (paper main: C = 4; multi-value → grid)
REACT_REFLECTION_PROTOCOL=true    # 6-step decompose → angles → reflect → cross-validate → opposite → calibrate
REACT_BUDGET_AWARENESS_PROTOCOL=true # front-load total step / search count
REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=true
REACT_FORCE_FINAL_ANSWER_LOOKAHEAD=2 # soft reminder at second-to-last + hard switch on last
REACT_MIN_SEARCH_CALLS=0          # soft floor (off by default; rules are last resort)
REACT_MAX_NUDGES=2

# v5.1 harness-resilience
REACT_FINAL_ANSWER_RETRY=false       # mop up empty final_raw with a tools=[] retry
REACT_BUDGET_EXCEEDED_DROP_TOOLS=true # drop tool schema once C is hit

# -------- Search Leak Filter (Stage-2 detector) --------
ENABLE_SEARCH_LEAK_FILTER=true
LEAK_DETECTOR_API_KEY=REPLACE_ME
LEAK_DETECTOR_BASE_URL=              # empty → falls back to LLM_BASE_URL
LEAK_DETECTOR_MODEL=anthropic/claude-sonnet-4.6
LEAK_DETECTOR_TIMEOUT_S=60
LEAK_DETECTOR_TEMPERATURE=0.0
LEAK_DETECTOR_MAX_TOKENS=512
LEAK_DETECTOR_RETRY_MAX=3
LEAK_DETECTOR_BACKOFF_S=2,5,15
LEAK_DETECTOR_FAIL_ACTION=drop       # drop (fail-closed, default) | keep (A/B escape hatch)
LEAK_DETECTOR_CONCURRENCY=5
LEAK_DETECTOR_PROMPT_VERSION=v1      # human-readable label; sha256 hashed automatically

# -------- Composite score weights (paper Eq. 18) --------
COMPOSITE_WEIGHTS_QTYPE=yes_no=0.15,binary_named=0.15,multiple_choice=0.70
COMPOSITE_WEIGHTS_CTYPE=single=0.40,multi=0.60
COMPOSITE_WEIGHT_OVERRIDES_QTYPE=    # per-metric overrides; e.g. fss=yes_no=0.05,multiple_choice=0.95
COMPOSITE_WEIGHT_OVERRIDES_CTYPE=

# -------- Sampling --------
SAMPLING_N=5

# -------- Run / Resume --------
RUN_ID=                              # empty → mint YYYYMMDD-HHMMSS-xxxx; same → resume
RESUME=true

# -------- Database --------
SOURCE_DB=./forecast_eval_set_example.db
SOURCE_TABLE=forecast_eval_set_example
RUNS_ROOT=./runs
DB_COMMIT_BATCH=10
WRITE_MESSAGES_TRACE=true            # full ReAct trace; large but invaluable for debugging

# -------- Logging --------
LOG_LEVEL=INFO
LOG_DIR=./logs

# -------- Belief protocol (v4 probabilistic family, off by default) --------
BELIEF_PROTOCOL=false                # require <belief>{...}</belief> JSON before \boxed{}

# -------- Grid search anchors (optional; only when R / C are multi-valued) --------
GRID_DEFAULT_R=
GRID_DEFAULT_C=
```

### 7.1 Startup validation (fail-fast)

Before any LLM / Tavily call, `Settings()` enforces (config.py:L577–L851):

| Check                                                              | Where (line)        | Failure mode                                       |
| ------------------------------------------------------------------ | ------------------- | -------------------------------------------------- |
| `RUN_ID` matches `^\d{8}-\d{6}-[0-9a-f]{4}$` (when non-empty)      | L577–L584           | ValueError                                         |
| `SOURCE_TABLE` matches `^[A-Za-z_][A-Za-z0-9_]*$`                  | L586–L595           | ValueError (SQL injection defence)                 |
| `MODELS` non-empty; no `:online`; no `::`                          | L599–L614           | ValueError                                         |
| `LLM_API_KEY` non-empty; no placeholder tokens                     | L617–L622           | ValueError                                         |
| `TAVILY_API_KEY` non-empty when `ENABLE_WEB_SEARCH=True`           | L623–L636           | ValueError per key                                 |
| `LLM_MAX_CONCURRENCY` ≥ 1; `SAMPLING_N` ≥ 1; `REACT_MAX_STEPS` ≥ 1 | L641–L646           | ValueError                                         |
| `REACT_MAX_SEARCH_CALLS` items > 0; `TAVILY_MAX_RESULTS` items > 0 | L455–L460           | ValueError per cell                                |
| `REACT_MIN_SEARCH_CALLS` ≤ min(C)                                  | L661–L671           | ValueError                                         |
| `REACT_FORCE_FINAL_ANSWER_LOOKAHEAD` ∈ [1, T]                      | L696–L707           | ValueError                                         |
| `GRID_DEFAULT_R` ∈ `TAVILY_MAX_RESULTS` (when set)                  | L711–L715           | ValueError                                         |
| `GRID_DEFAULT_C` ∈ `REACT_MAX_SEARCH_CALLS` (when set)              | L716–L720           | ValueError                                         |
| `LEAK_DETECTOR_API_KEY` / `_MODEL` non-empty when filter enabled    | L758–L777           | ValueError; no `:online` on detector slug          |
| `COMPOSITE_WEIGHTS_*` buckets in known set; weights ≥ 0; ≥1 > 0    | L781–L851           | ValueError                                         |
| `COMPOSITE_WEIGHT_OVERRIDES_*` metric names in allowlist            | L515–L535 + composite.py:L77–L127 | ValueError on typo (no silent default fallback) |

**Pin tests.** `test_config.py` covers ~155 lines of boundary cases.

### 7.2 Redaction

Before writing `run_meta.config_snapshot`, `db.compute_redacted_config_snapshot` redacts
sensitive fields. Format: first 4 chars + length + `sha256[:12]`. `TAVILY_API_KEY` is
`list[str]` and persisted as `[{prefix, sha256_12, length, provider}, ...]` for "which keys
this run used" auditing. Sensitive plaintext is **never** persisted.

---

## 8. Core module responsibilities

| Module                  | Implements                                                                                                  | Key interfaces                                                                                                 | Pin tests                              |
| ----------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `config.py`             | Reads `.env` via pydantic-settings; validates types; parses CSV lists; runs all §7.1 startup checks         | `Settings` class (singleton); `_parse_csv`, `_parse_int_list`, `_parse_cutoffs`                                | `test_config.py`                        |
| `loader.py`             | Syncs `<SOURCE_TABLE>` → `questions`, `dataset_metadata.features_json.prompt_reconstruction` → `prompt_templates` | `sync_questions(source_db, conn, filters, table=...) -> list[Question]`; `sync_prompt_templates(...)`         | (covered by `test_db.py` + `test_evaluation.py`) |
| `prompts.py`            | Renderer $R$; reflection/belief/budget-awareness protocol bodies; harness status injection builders          | `render_user_prompt`, `REFLECTION_PROTOCOL`, `BELIEF_PROTOCOL`, `build_budget_awareness_protocol`, `build_*_warning`, `_build_status_header` | `test_prompts.py`                       |
| `tools.py`              | Defines the `web_search` OpenAI schema; LLM-visible part has no date                                         | `WEB_SEARCH_SCHEMA`, `parse_tool_arguments`, `extract_query`, `tool_error_message`, `tool_result_message`        | `test_search.py`                        |
| `search.py`             | Tavily wrapper; injects `end_date = q.end_time + δ`; truncates raw_content; dispatches Stage-2 detector       | `tavily_search(query, end_date, settings) -> SearchResult`                                                      | `test_search.py`                        |
| `leak_filter.py`        | Detector $H_{\mathrm{aux}}$: per-result `keep`/`drop`; whitelist; fail-closed                                  | `filter_search_result(result, cutoff_date, settings)`                                                           | `test_leak_filter.py`                   |
| `tavily_keys.py`        | Multi-key pool: least-used + 401/403 blacklist + 429 cooldown; least-used = healthy + min(used)                | `TavilyKeyPool.acquire / report_failure`; `get_pool(keys, cooldown_s)`                                          | (covered by `test_search.py`)           |
| `llm.py`                | OpenAI-compatible client; tiered retry by error kind; rejects `:online`, `plugins`, non-whitelist tools        | `chat(model, messages, tools, ...) -> ChatResponse`; `_assert_no_browsing`                                      | `test_llm_no_browsing.py`               |
| `react.py`              | Forecasting system $F_M$: ReAct loop with 4-knob harness resilience; per-step belief parsing                  | `run_react(q, model, sample_idx, settings) -> SampleResult`                                                     | `test_react.py` (1432 LOC), `test_react_reflection.py` |
| `parser.py`             | Parser $\Psi$ + normalisation $\phi$: `\boxed{}` extraction → letter `frozenset[str]`; belief JSON validator    | `parse_answer(text, q)`, `parse_gt(answer)`, `is_correct(pred, gt)`, `parse_belief(text, q)`                    | `test_parser.py`, `test_parser_belief.py` |
| `errors.py`             | Error classification + backoff lookup + AuthError                                                              | `ErrorKind`, `classify(exc)`, `should_retry(kind)`, `backoff_seconds(kind, attempt, settings, retry_after)`     | `test_errors.py`                        |
| `db.py`                 | Schema + AsyncWriter + hashes + redaction; v2→v5 migrations; resume queries; model-slug safety                | `init_schema`, `AsyncWriter.enqueue_result`, `load_completed_samples`, `register_run_meta`, `compute_*_hash`, `model_slug_safe` | `test_db.py`, `test_db_v4_migration.py`, `test_db_v5_migration.py` |
| `runner.py`             | Task orchestration: cartesian dedup → κ_M filter → asyncio concurrency → progress log → `finish_run_meta`     | `run(settings, filters, questions, templates, run_id, conns) -> RunStats`; `build_task_plan`                    | `test_runner_resume.py`, `test_runner_grid_model.py`, `test_training_cutoff.py` |
| `analysis/__init__.py`  | Aggregation $\Gamma$ orchestrator: walks DBs → runs metric stack → writes CSV/MD/JSON; auto-invoked or `python -m forecast_eval.analysis runs/{run_id}` | `run_analysis(run_dir: Path) -> list[Path]`                                                                      | `test_analysis.py`                      |

`QFilter` (types.py:L26–L51) is a dataclass with `question_types: frozenset[str] | None`
and `choice_types: frozenset[str] | None`; `None` means no filtering. `apply_sql()` returns
`(WHERE clause, params)` for SQLite parameterised execution; `snapshot()` returns a dict
for `manifest.filters_snapshot`.

### 8.1 `prompts.render_user_prompt` reference implementation

```python
def render_user_prompt(
    q: Question,
    templates: dict[str, str],
    reflection_protocol: str | None = None,
    budget_awareness: str | None = None,
    belief_protocol: str | None = None,
) -> str:
    options = json.loads(q.options)

    if q.question_type == "yes_no":
        outcomes_block = ""
        output_format = templates["yes_no_output_format"]

    elif q.question_type == "binary_named":
        outcomes_block = ""
        output_format = (
            templates["binary_named_output_format"]
            .replace("<options[0]>", options[0])
            .replace("<options[1]>", options[1])
        )

    elif q.question_type == "multiple_choice":
        outcomes_block = "\n" + "\n".join(
            f"{index_to_letter(i)}. {label}" for i, label in enumerate(options)
        )
        output_format = (templates["multiple_choice_single_output_format"]
                         if q.choice_type == "single"
                         else templates["multiple_choice_multi_output_format"])

    else:
        raise ValueError(f"unknown question_type: {q.question_type}")

    body = templates["prompt_template"].format(
        agent_role=templates["agent_role"],
        event=q.event,
        end_time=q.end_time,
        outcomes_block=outcomes_block,
        output_format=output_format,
        guidance=templates["guidance"],
    )
    # Protocol additions live as runtime slots; templates_hash is unaffected.
    # Order: budget-awareness → reflection → belief (matches react.py:L187 wiring).
    for protocol in (budget_awareness, reflection_protocol, belief_protocol):
        if protocol:
            body += "\n\n" + protocol
    return body
```

### 8.2 `parser.parse_answer` reference implementation

```python
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")

def index_to_letter(i: int) -> str:
    if i < 0:
        raise ValueError(f"index must be >= 0, got {i}")
    return chr(ord("A") + i)

def letter_to_index(letter: str) -> int:
    if len(letter) != 1:
        raise ValueError(f"letter must be a single character, got {letter!r}")
    return ord(letter) - ord("A")

def parse_answer(text: str, q: Question) -> frozenset[str] | None:
    matches = BOXED_RE.findall(text or "")
    if not matches:
        return None
    payload = matches[-1].strip()                       # take the LAST \boxed{...}

    if q.question_type == "yes_no":
        v = payload.lower()
        if v == "yes": return frozenset({"A"})
        if v == "no":  return frozenset({"B"})
        return None

    if q.question_type == "binary_named":
        opts = json.loads(q.options)
        norm = payload.strip().lower()
        for i, label in enumerate(opts):
            if label.strip().lower() == norm:
                return frozenset({index_to_letter(i)})
        return None

    if q.question_type == "multiple_choice":
        tokens = [t.strip() for t in re.split(r"[,\s]+", payload) if t.strip()]
        opts_n = len(json.loads(q.options))
        letters: set[str] = set()
        for t in tokens:
            if len(t) != 1:
                return None
            idx = letter_to_index(t)
            if not (0 <= idx < opts_n):
                return None
            letters.add(t)
        return frozenset(letters) if letters else None

    return None

def parse_gt(answer: str) -> frozenset[str]:
    return frozenset(t.strip() for t in answer.split(",") if t.strip())

def is_correct(pred: frozenset[str] | None, gt: frozenset[str]) -> bool | None:
    if pred is None:
        return None
    return pred == gt
```

---

## 9. Error tiering & backoff strategy

All exceptions are routed by the table below (`errors.py`):

| Error type                              | Identification                                                                  | Handling                                                                              |
| --------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Network / Timeout**                   | `httpx.ConnectError`, `httpx.ReadTimeout`, `asyncio.TimeoutError`, `RemoteProtocolError`, `WriteError`, `WriteTimeout`, `PoolTimeout` | Backoff per `LLM_BACKOFF_NETWORK_S`; on exhaustion → `error="network"`             |
| **Rate Limit (429)**                    | HTTP 429                                                                        | Prefer `Retry-After` header; otherwise `LLM_BACKOFF_RATE_LIMIT_S`                       |
| **Server 5xx**                          | HTTP 500/502/503/504                                                            | Backoff per `LLM_BACKOFF_SERVER_5XX_S`; on exhaustion → `error="server_5xx"`            |
| **Auth (401/403)**                      | HTTP 401/403                                                                    | **Fail immediately, abort entire run** (`AuthError`)                                    |
| **Bad Request (400)**                   | HTTP 400 + `model_not_found` / `invalid_request`                                | Skip immediately, `error="bad_request"`                                                 |
| **Content Policy**                      | HTTP 400 + match against `errors.CONTENT_POLICY_NEEDLES` (`content_policy / content_filter / safety / content_policy_violation / data_inspection_failed / inappropriate content / sensitive`) | **No retry**, `error="content_policy"`, `parse_ok=0`, `correct=NULL`                |
| **LLM soft refusal**                    | Normal return but `\boxed{...}` not found or parsed `frozenset` empty           | Not an error; `parse_ok=0`, `correct=NULL`                                              |
| **Exceed `REACT_MAX_STEPS`**            | ReAct loop exhausted without a final answer                                      | Not an error; `parse_ok=0`, `correct=NULL` (unless `REACT_FINAL_ANSWER_RETRY` mops up)  |
| **Tool arguments JSON parse fails**     | LLM's `arguments` are not legal JSON                                            | Tell the LLM the error and continue the loop (non-fatal)                                |
| **Tavily error itself**                 | Independent retry via `SEARCH_BACKOFF_S`; on exhaustion, error fed to LLM as `tool_result` | LLM can choose to retry or give up                                                      |
| **Detector error (Stage-2)**            | Retry via `LEAK_DETECTOR_BACKOFF_S`; AUTH errors immediate-fail-closed         | On `LEAK_DETECTOR_FAIL_ACTION=drop` (default) → drop the item; `keep` → pass through    |
| **Training-data contamination filter** | Detected during task-plan: `q.end_time <= κ_M` (see §3.1)                       | **Does not invoke the LLM**, directly writes `error="skipped_training_cutoff"`         |

### 9.1 Key boundaries

1. **Auth errors stop the entire run.** Continuing to burn budget on a wrong key is
   meaningless; early-stop saves money. `runner._run_task_with_retry` re-raises `AuthError`
   (runner.py:L245); the outer loop cancels all tasks, flushes the writer, and exits.
2. **Content policy is not retried.** Re-sending the same question yields the same result;
   tally how many each model was rejected on at the end.
3. **Refusal ≠ error.** The LLM returned a legal response but did not answer (missing
   boxed / letter outside option range) — this is part of model capability, counted in
   statistics but not in `error`.
4. **Tavily failure degrades to a tool_result error.** Let the LLM decide whether to retry
   the query or give up, without interrupting the whole sample.
5. **Detector failure is fail-closed by default.** Detector hiccups (timeout, network) are
   uncorrelated with item content; biasing the residual towards "drop on uncertainty" is the
   conservative choice.
6. **`skipped_training_cutoff` does not count toward error rate.** This is active data
   cleansing, not a model failure; reports tally "questions excluded / ratio" separately.

### 9.2 Error / parsing coupling rules (paper §4.2.4)

| State                                | `parse_ok` | `correct` | Counted in $\mathcal{S}$? | Counted in $\mathcal{D}^{\mathrm{eval}}$? |
| ------------------------------------ | ---------- | --------- | ------------------------- | ----------------------------------------- |
| Cutoff-excluded                       | 0          | NULL      | No                         | No (excluded)                              |
| Non-cutoff call error (network/5xx/policy) | 0     | NULL      | No                         | Yes (denominator), No (numerator)          |
| Parse failure / soft refusal         | 0          | 0         | Yes                        | Yes                                       |
| Strict equality match                | 1          | 1         | Yes                        | Yes                                       |
| Strict equality miss                 | 1          | 0         | Yes                        | Yes                                       |

This matrix is the contract between `react.py`'s output and `analysis/`'s denominators.

---

## 10. ReAct loop pseudocode (the forecasting system $F_M$)

The loop in `react.run_react` (react.py:L248–L632) is the heart of $F_M$. The four
harness-resilience knobs interleave with the per-step LLM call via a deterministic
priority chain. The pseudocode below preserves all v5.1 wiring; line numbers reference
`react.py` head-of-tree.

```python
async def run_react(q: Question, model: str, sample_idx: int, settings: Settings) -> SampleResult:
    # ① compute χ_i — invisible to the LLM
    end_date = (date.fromisoformat(q.end_time)
                + timedelta(days=settings.TAVILY_END_DATE_OFFSET_DAYS)).isoformat()

    # ② render m_0 = R(q^in): single user message, all enabled protocols appended
    user_prompt = prompts.render_user_prompt(
        q,
        settings.PROMPT_TEMPLATES,
        budget_awareness=(prompts.build_budget_awareness_protocol(
                            max_steps=settings.REACT_MAX_STEPS,
                            max_search_calls=settings.REACT_MAX_SEARCH_CALLS,
                          ) if settings.REACT_BUDGET_AWARENESS_PROTOCOL else None),
        reflection_protocol=(prompts.REFLECTION_PROTOCOL
                             if settings.REACT_REFLECTION_PROTOCOL else None),
        belief_protocol=(prompts.BELIEF_PROTOCOL
                         if settings.BELIEF_PROTOCOL else None),
    )
    messages: list[dict] = [{"role": "user", "content": user_prompt}]
    search_calls: list[dict] = []
    final_raw = ""
    step_metrics: list[dict] = []
    beliefs_per_step: list[Belief | None] = []
    nudges_used = 0
    final_answer_retry_used = 0
    budget_exhausted_notified = False
    pending_continuation = False
    t0 = time.monotonic()
    tokens = {"prompt": 0, "completion": 0, "reasoning": 0}
    effective_max_search_calls = (
        settings.REACT_MAX_SEARCH_CALLS if settings.ENABLE_WEB_SEARCH else 0
    )

    for step in range(settings.REACT_MAX_STEPS):
        steps_executed = step + 1
        searches_done_now = len(search_calls)

        # =====================================================================
        # PRE-STEP INJECTION DECISION (priority chain, at most ONE fires)
        # All four paths share `_build_status_header` (prompts.py:L128–L164):
        #   [Harness status] step k/N (R remaining) · web_search s/C used (M left).
        # Priority: (1) > (2) > (3) > (4)
        # =====================================================================
        force_final_active = (
            settings.REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT
            and (settings.REACT_MAX_STEPS - step) <= settings.REACT_FORCE_FINAL_ANSWER_LOOKAHEAD
        )
        budget_dropped = (
            settings.REACT_BUDGET_EXCEEDED_DROP_TOOLS
            and searches_done_now >= settings.REACT_MAX_SEARCH_CALLS
        )
        force_final_hard_cutoff = False
        injection: str | None = None

        if force_final_active:
            remaining = settings.REACT_MAX_STEPS - step
            if remaining == 1:
                # (1) LAST STEP HARD CUTOFF — tools=[] + force-finalise text
                injection = prompts.build_last_step_force_finalisation(
                    current_step=steps_executed, max_steps=settings.REACT_MAX_STEPS,
                    searches_done=searches_done_now,
                    max_search_calls=effective_max_search_calls,
                )
                force_final_hard_cutoff = True
            else:
                # (2) PENULTIMATE SOFT WARNING — tools still exposed inside LOOKAHEAD
                injection = prompts.build_penultimate_step_warning(
                    current_step=steps_executed, max_steps=settings.REACT_MAX_STEPS,
                    searches_done=searches_done_now,
                    max_search_calls=effective_max_search_calls,
                )
        elif budget_dropped and not budget_exhausted_notified:
            # (3) SEARCH BUDGET EXHAUSTED — fired ONCE per run
            injection = prompts.build_search_budget_exhausted_commit(
                current_step=steps_executed, max_steps=settings.REACT_MAX_STEPS,
                searches_done=searches_done_now,
                max_search_calls=effective_max_search_calls,
            )
            budget_exhausted_notified = True
        elif pending_continuation:
            # (4) CONTINUATION AFTER UN-BOXED TURN
            injection = prompts.build_continuation_after_unboxed_content(
                current_step=steps_executed, max_steps=settings.REACT_MAX_STEPS,
                searches_done=searches_done_now,
                max_search_calls=effective_max_search_calls,
            )

        if injection is not None:
            messages.append({"role": "user", "content": injection})
        pending_continuation = False  # reset whether or not we injected

        # =====================================================================
        # TOOL SCHEMA DECISION
        # =====================================================================
        if force_final_hard_cutoff or budget_dropped or not settings.ENABLE_WEB_SEARCH:
            tools_for_this_step = []
        else:
            tools_for_this_step = [WEB_SEARCH_SCHEMA]

        # =====================================================================
        # LLM CALL
        # =====================================================================
        resp = await llm.chat(
            model=model, messages=messages, tools=tools_for_this_step,
            temperature=settings.LLM_TEMPERATURE, top_p=settings.LLM_TOP_P,
            max_tokens=settings.LLM_MAX_TOKENS, timeout=settings.LLM_TIMEOUT_S,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_unset=True))
        _accumulate_tokens(tokens, resp.usage)
        step_metrics.append(_capture_step_metrics(step, resp))

        # Per-step belief parsing (independent of \boxed{} path)
        if settings.BELIEF_PROTOCOL:
            beliefs_per_step.append(parser.parse_belief(msg.content or "", q))
        else:
            beliefs_per_step.append(None)

        # =====================================================================
        # NO TOOL CALLS → maybe nudge or break
        # =====================================================================
        if not msg.tool_calls:
            content = msg.content or ""
            if (settings.REACT_MIN_SEARCH_CALLS > 0
                and settings.REACT_MAX_NUDGES > 0
                and searches_done_now < settings.REACT_MIN_SEARCH_CALLS
                and nudges_used < settings.REACT_MAX_NUDGES):
                # Soft floor not met → inject nudge, keep looping
                messages.append({"role": "user",
                                 "content": prompts._build_nudge_message(
                                    current_step=steps_executed, max_steps=settings.REACT_MAX_STEPS,
                                    searches_done=searches_done_now,
                                    max_search_calls=effective_max_search_calls,
                                    min_required=settings.REACT_MIN_SEARCH_CALLS,
                                 )})
                nudges_used += 1
                continue
            if "\\boxed{" not in content:
                pending_continuation = True
                continue
            final_raw = content
            break

        # =====================================================================
        # TOOL CALLS → validate + execute (one at a time)
        # =====================================================================
        for tc in msg.tool_calls:
            err = _validate_tool_call(tc, settings, searches_done=len(search_calls))
            if err is not None:
                messages.append(prompts.tool_error_message(tc, err, ...))
                continue
            args = parse_tool_arguments(tc.function.arguments)
            query = extract_query(args)

            # ③ inject χ_i (invisible to the LLM); Stage-2 detector audits results.
            result = await search.tavily_search(query=query, end_date=end_date,
                                                settings=settings)
            search_calls.append(result.to_search_call_record())  # incl. detector audit
            messages.append(prompts.tool_result_message(tc, result.to_llm_payload()))
    # exceeded REACT_MAX_STEPS; final_raw stays empty → parser will mark parse_ok=0

    # =========================================================================
    # v5.1 D1 FINAL-ANSWER RETRY — backstop for empty final_raw
    # =========================================================================
    if final_raw == "" and settings.REACT_FINAL_ANSWER_RETRY:
        messages.append({"role": "user",
                         "content": "Time to commit. Output your final \\boxed{...} answer "
                                    "now without further searches or tool calls."})
        resp = await llm.chat(model=model, messages=messages, tools=[], **kwargs)
        final_raw = resp.choices[0].message.content or ""
        step_metrics.append(_capture_step_metrics(step + 1, resp))
        final_answer_retry_used = 1

    # =========================================================================
    # ④ Ψ ∘ φ: parse + score
    # =========================================================================
    parsed = parser.parse_answer(final_raw, q)              # frozenset[str] | None
    gt = parser.parse_gt(q.answer)                          # frozenset[str]
    correct = parser.is_correct(parsed, gt)                  # bool | None

    # Belief finalisation (v4)
    belief_final, belief_trace, belief_parse_ok = _finalise_beliefs(beliefs_per_step)

    return SampleResult(
        run_id=settings.RUN_ID, question_id=q.id, model=model, sample_idx=sample_idx,
        final_answer_letters=json.dumps(sorted(parsed)) if parsed is not None else None,
        final_answer_raw=final_raw,
        correct=int(correct) if isinstance(correct, bool) else None,
        parse_ok=1 if parsed is not None else 0,
        tool_calls_count=len(search_calls),
        react_steps=steps_executed + final_answer_retry_used,
        prompt_tokens=tokens["prompt"], completion_tokens=tokens["completion"],
        reasoning_tokens=tokens["reasoning"],
        latency_ms=int((time.monotonic() - t0) * 1000),
        user_prompt=user_prompt,
        messages_trace=json.dumps(messages) if settings.WRITE_MESSAGES_TRACE else None,
        search_calls=json.dumps(search_calls),
        nudges_used=nudges_used, step_metrics=json.dumps(step_metrics),
        finish_reason=resp.choices[0].finish_reason,
        response_id=resp.id, system_fingerprint=resp.system_fingerprint,
        service_tier=resp.service_tier,
        belief_final=json.dumps(belief_final) if belief_final else None,
        belief_trace=json.dumps(belief_trace) if belief_trace else None,
        belief_parse_ok=belief_parse_ok,
        final_answer_retry_used=final_answer_retry_used,
        error=None, created_at=utcnow_iso(),
    )
```

### 10.1 Harness-resilience knob index

| Knob                                  | Default | Scope                  | Effect                                                                                       |
| ------------------------------------- | ------- | ---------------------- | -------------------------------------------------------------------------------------------- |
| `REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT` | True    | In-loop graded transition | Within last `LOOKAHEAD` steps: penultimate soft warning, last step hard `tools=[]` cutoff      |
| `REACT_FORCE_FINAL_ANSWER_LOOKAHEAD`  | 2       | Soft window              | How many steps before the limit to start intervening; clamped to $[1, T]$                     |
| `REACT_BUDGET_AWARENESS_PROTOCOL`     | True    | Prompt layer             | Append `T` and `C` to the user prompt so the model can plan holistically                       |
| `REACT_BUDGET_EXCEEDED_DROP_TOOLS`    | True    | In-loop budget gate      | Once cumulative `web_search ≥ C`, drop tools (`tools=[]`) for all subsequent rounds            |
| `REACT_FINAL_ANSWER_RETRY`            | False   | Post-loop backstop       | When loop ends with empty `final_raw`, call LLM once more with `tools=[]` to force `\boxed{}`  |
| `REACT_MIN_SEARCH_CALLS`              | 0       | Soft floor               | If LLM tries to commit before `MIN`, inject a nudge (capped by `REACT_MAX_NUDGES`)             |
| `REACT_MAX_NUDGES`                    | 2       | Soft floor cap           | Nudge budget per sample                                                                        |

### 10.2 Per-step belief processing (v4)

When `BELIEF_PROTOCOL=True`, every assistant turn (including the post-loop final-answer
retry) is parsed by `parser.parse_belief` (parser.py:L117–L213). Per-step results land in
`beliefs_per_step`, which is then aggregated into three persisted fields:

* `belief_final`: JSON of last-step probabilities when its belief parses; else NULL.
* `belief_trace`: JSON array of every step's belief summary (or `null` per step).
* `belief_parse_ok`: 1 iff last-step belief parsed; **independent** of `parse_ok`.

The belief JSON schema is strict (paper §C; prompts.py:L66–L105):

```json
{
  "version": "v4.0",
  "probabilities": { "<letter>": <float in [0, 1]>, ... },
  "confidence": "low" | "medium" | "high",
  "key_evidence":     [ "<= 280 chars per bullet, 1-4 bullets" ],
  "counterevidence":  [ "<= 280 chars per bullet, 0-3 bullets" ],
  "open_questions":   [ "<= 280 chars per bullet, 0-3 bullets" ],
  "decision_rule": "argmax" | "multi-select@<threshold>"
}
```

* Probabilities keys MUST exactly match the expected letter set (parser.py:L150).
* Single-choice: probabilities sum to 1.0 ± 1e-3 (parser.py:L167).
* Multi-select: each entry independent in [0, 1].
* Confidence must be one of `{low, medium, high}` (parser.py:L173).

A failed parse anywhere does **not** affect `parse_ok` — the `\boxed{}` path is the sole
correctness signal.

### 10.3 Reflection protocol (paper §4.2.3)

`prompts.REFLECTION_PROTOCOL` (prompts.py:L31–L53) is a 6-step reasoning scaffold appended
*at runtime* to the user message:

1. **Decompose** — list sub-questions whose joint answer settles the prediction.
2. **Plan distinct angles** — at least three different investigation angles before any
   `web_search`.
3. **Search iteratively, reflect after every result** — paraphrase, tag relevance, identify
   contradictions, pick the next query to fill the largest gap.
4. **Cross-validate** — at least two independent sources before committing.
5. **Stress-test the opposite** — articulate the strongest case for the opposite outcome.
6. **Calibrate, then commit** — state confidence, failure mode, decisive evidence; only
   then `\boxed{...}`.

The full text (~22 lines) is hashed to `reflection_protocol_hash` (sha256[:16]) and stored
verbatim in `run_meta.reflection_protocol_text`. This text is **not** in
`prompt_templates_hash`; toggling reflection on/off keeps the template hash invariant.

---

## 11. Evaluation metrics (the aggregation rule $\Gamma$)

Metrics are **computed entirely by `forecast_eval.analysis` after the run finishes**, never
stored in the DB. Artefacts land in `runs/{run_id}/analysis/`. The definitions below match
both the source implementation and the paper's notation (paper §3.1, §4.3).

### 11.1 Validity ($\mathcal{E}^{\mathrm{valid}}$, paper §3.1)

$v_{i,M} = \mathbb{1}[\Psi_i(o_{i,M}) \ne \bot]$ — whether the model's raw output yields a
parseable letter set.

| Metric                          | Definition                                                                              | DB column source              |
| ------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------ |
| `parse_failure_rate`            | $1 - \mathbb{E}[v_{i,M}]$ over the scorable set $\mathcal{S}$                            | `s{i}_parse_ok = 0`            |
| `final_answer_retry_rate`       | Share of samples where v5.1 D1 mopped up an empty `final_raw`                            | `s{i}_final_answer_retry_used = 1` |
| `error_rate`                    | Share of samples with non-cutoff `s{i}_error`                                            | `s{i}_error NOT IN (NULL, 'skipped_training_cutoff')` |
| `cutoff_skip_rate` (per model)  | `count(error='skipped_training_cutoff') / count(*)` per model                            | `s{i}_error = 'skipped_training_cutoff'` |
| `error_breakdown` (CSV)         | `Counter[error]` across all samples (cutoff included)                                    | `s{i}_error`                   |
| `finish_reason_breakdown` (CSV) | `Counter[finish_reason]` over eligible samples; spot abnormal `length` / `content_filter` | `s{i}_finish_reason`            |

### 11.2 Item-level ($\mathcal{E}^{\mathrm{item}}$, paper §3.2.1, §3.2.4, §3.5)

A `(question_id, model)` has $n$ samples ($n=$ `SAMPLING_N`). Tally **after excluding** rows
with `s{i}_error="skipped_training_cutoff"` — those are excluded questions, not the model
getting them wrong.

* **Strict equality** (paper Eq. 14):
  $r_{i,M} = \mathbb{1}[\widehat{G}_{i,M} = G_i]$ — `s{i}_correct` in the DB.

* **Exam-style partial credit** (paper Eq. 17, the project's headline per-sample score):

  $$
  \text{exam-score}(\hat S, G) = \begin{cases}
  \dfrac{|\hat S \cap G|}{|G|}, & \hat S \setminus G = \varnothing \\
  0, & \hat S \setminus G \ne \varnothing
  \end{cases}
  $$

  Single-answer questions degenerate to strict 0/1. Implementation:
  `analysis.exam_score.exam_score(s, gt)` (exam_score.py:L62). Decision tree (exam_score.py:L78–L91):

  ```
  is_cutoff           → None  (excluded)
  error is not None   → None  (excluded)
  parse_ok != 1       → 0.0   (parse failure counts as 0)
  FP > 0              → 0.0   (any false positive vetoes)
  otherwise           → |TP| / |G|
  ```

* **Tversky similarity** (paper Eq. 22, used for FSS):

  $$
  T(\hat S, G) = \frac{|\hat S \cap G|}{|\hat S \cap G| + \alpha\,|\hat S \setminus G| + \beta\,|G \setminus \hat S|}
  $$

  Project default $(\alpha, \beta) = (2.0, 0.5)$ — FP penalty 4× FN penalty. Implementation:
  `analysis.accuracy.tversky_score` (accuracy.py:L286).

* **Hamming score** (multi-only):

  $$
  \text{hamming}(\hat S, G, \mathcal{O}) = 1 - \frac{1}{k}\sum_{\ell\in\mathcal{O}}|\mathbb{1}[\ell\in\hat S] - \mathbb{1}[\ell\in G]|
  $$

  Symmetric in missing/wrong; multi-only (single degenerates to 0/1).

### 11.3 Question-level ($\mathcal{E}^{\mathrm{question}}$, paper §3.2.5–§3.2.6)

| Metric                          | Definition                                                                                            | Implementation                              |
| ------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `pass_at_1_avg` ($\passone$)    | Per-question intra-mean of strict hits, then equal-weight cross-question (paper Eq. 35)               | `accuracy._aggregate` (accuracy.py:L124)     |
| `pass_any_at_n` ($\passany$)    | $\mathbb{1}[\exists s: c_{q,s}=1]$ averaged across questions (paper Eq. 37; the standard `pass@k`)    | `accuracy._aggregate` (L134)                 |
| `at_least_all_at_n` ($\passall$)| $\prod_s c_{q,s}$ averaged (paper Eq. 38; repeated-consistency lower bound)                            | `accuracy._aggregate` (L141)                 |
| `at_least_majority_at_n`        | $\mathbb{1}[\sum_s c_{q,s} \ge \lceil n/2 \rceil]$ averaged                                            | `accuracy._aggregate`                        |
| `majority_vote_accuracy`        | Counter-based letter-set vote, single winner, then strict equality vs $G_q$                            | `accuracy._aggregate` (L150–L164)             |
| `exam_score_at_n_avg`           | Two-step (intra-question mean → inter-question mean) over scored index $\mathcal{J}_q^{\mathrm{cnt}}$ | `exam_score.exam_score_at_n_avg` (L94–L129)   |
| `cohen_kappa`                   | $(\text{acc} - p_e)/(1 - p_e)$; $p_e=1/k_q$ (single) or $0.5$ per-label (multi); see paper Eq. 39     | `accuracy.cohen_kappa` (L493–L532)            |
| `hamming_score`                 | Cross-question mean of per-question Hamming (multi only)                                                | `accuracy.hamming_score_per_question` (L535–L574) |

### 11.4 K-trial-only metrics ($n \ge 2$; paper §3.2.6, §A)

| Metric          | Definition                                                                                                              | Implementation                              |
| --------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `fleiss_kappa`  | $(\bar{P} - \bar{P}_e)/(1 - \bar{P}_e)$ on the $K_q^{\mathrm{eff}}$-trial vote matrix; stratified by $k_q$ for single, per-label for multi (paper Eq. 40–43) | `consistency.fleiss_kappa` (L257–L297)       |
| `mean_entropy`  | Per-question mean Shannon entropy of vote distribution; per-label binary mean for multi                                  | `consistency.prediction_entropy_*` (L305–L399) |
| `vci`           | $\text{VCI}_q = \max_\ell n_{q,\ell}/K_q^{\mathrm{eff}}$, cross-question mean (paper Eq. 70)                              | `consistency.mean_vci` (L401–L425)            |
| `mvg`           | $\text{MV-Acc} - \passone$; positive = self-consistency gain                                                              | `consistency.mvg` (L427–L450)                 |

### 11.5 Format Skill Score (FSS, paper §3.5)

The headline chance-corrected skill metric. For the $j$-th trial of question $q$:

$$
\bar{T}_q = \frac{1}{K_q^{\mathrm{eff}}}\sum_{j\in\mathcal{J}_q^{\mathrm{ok}}} T(P_{q,j}, G_q),
\qquad
\text{fss}_q = \frac{\bar{T}_q - T_q^{\mathrm{chance}}}{1 - T_q^{\mathrm{chance}}}
$$

Chance baseline closed form (paper Eq. 23):

$$
T_q^{\mathrm{chance}} = \begin{cases}
\dfrac{1}{k_q}, & \text{single-answer} \\[6pt]
2^{-k_q}\sum_{tp=1}^{m_q}\sum_{fp=0}^{k_q-m_q}\binom{m_q}{tp}\binom{k_q-m_q}{fp}\cdot\dfrac{tp}{tp+\alpha\,fp+\beta(m_q-tp)}, & \text{multi-answer}
\end{cases}
$$

Dataset-level: $\text{fss} = \frac{1}{|\mathcal{D}^{\mathrm{ok}}|}\sum_q \text{fss}_q$ where
$\mathcal{D}^{\mathrm{ok}} = \{q : \bar{T}_q \ne \text{None}\}$.

**Implementation.** `accuracy.fss` (accuracy.py:L386–L479), with closed-form chance via
`accuracy.tversky_baseline` (L316–L350). Returns
`{"fss", "n_valid", "mean_pe", "per_question"}` so downstream can decompose by question.

**Pin tests.** `test_fss.py` (528 LOC) covers correctness against analytical baselines;
`test_fss_sensitivity.py` covers $(\alpha, \beta)$ sweeps.

### 11.6 Composite Accuracy (paper §3.4, Eq. 18 — the headline)

The model-level summary metric. Substituting $\examavg^{(b)}$ as the per-bucket value:

$$
\text{Composite Accuracy}_m = \frac{\sum_{b\in B_{\mathrm{valid}}(m)} w_b \cdot \examavg^{(b),m}}{\sum_{b\in B_{\mathrm{valid}}(m)} w_b}
$$

where $B_{\mathrm{valid}}(m) = \{b\in B : v_{m,b}\ne\text{None} \wedge w_b > 0\}$. Missing
buckets are dropped and remaining weights renormalised. If $B_{\mathrm{valid}}(m) = \varnothing$
the composite is `None`.

**Default weights** (config.py:L365–L368, paper §3.3):

```text
yes_no          = 0.15
binary_named    = 0.15
multiple_choice = 0.70
```

choice-type weights (`single`/`multi`):

```text
single = 0.40
multi  = 0.60
```

**Per-metric overrides** via `COMPOSITE_WEIGHT_OVERRIDES_QTYPE` /
`COMPOSITE_WEIGHT_OVERRIDES_CTYPE` (CSV `metric=bucket=w,bucket=w;metric=...`); misspelled
metric names raise at runtime via the known-metrics allowlist (composite.py:L77–L127).

**Implementation.** `composite.compute_composite` (composite.py:L18–L28) +
`composite.slice_v5_metrics_by_bucket` (L151–L198).

### 11.7 Per-correct cost (paper §3.7, Eq. 25)

The cost-effectiveness scalar amortising the OpenRouter invoice across difficulty-weighted
notional correct count:

$$
C^{\mathrm{per\text{-}correct}}_m = \frac{C^{\mathrm{total}}_m}{|\mathcal{D}^{\mathrm{eval}}| \cdot n \cdot \text{Composite Accuracy}_m}
$$

The denominator $|\mathcal{D}^{\mathrm{eval}}| \cdot n \cdot \text{Composite Accuracy}_m$ is
the **difficulty-weighted notional correct-sample count**: when bucket weights coincide with
empirical question-type prevalence, it equals the raw correct-sample count; otherwise it
acts as a discrimination-aware reference count that up-weights harder buckets.

**Source of $C^{\mathrm{total}}_m$.** Read directly from OpenRouter's billing endpoint —
the platform invoice is the single financial fact verifiable by third parties. This avoids
divergences from "published unit price × token usage" calculations (reasoning-token billing,
prompt-cache discounts, tool-call billing, provider routing).

### 11.8 Probabilistic family (v4 companion; demoted under K=5)

`forecast_eval/analysis/proper_score.py` + `probabilistic.py`. Active only when
`BELIEF_PROTOCOL=True`.

| Metric                          | Formula                                                                                  | Applicable      |
| ------------------------------- | ---------------------------------------------------------------------------------------- | --------------- |
| **Brier Index (BI)**            | $100(1 - \sqrt{\overline{\text{BS}^{\mathrm{lab}}}})$, mean-then-square-root              | All qtypes      |
| **BI_dec**                      | Decision-wise Brier index                                                                  | Single only     |
| **NLL**                         | Single: $-\log p_{q,l^*}$; multi: per-label BCE; clip $\epsilon = 10^{-3}$                | All qtypes      |
| **MBS**                         | $100(\log_2 p_{q,l^*} + 1)$, clip same                                                     | Single only     |
| **ABI (crowd / uniform)**       | Sign-aware $100(1 \mp \sqrt{|\overline{\text{ABS}}|})$ vs LOO crowd / uniform baselines    | Crowd: multi-model |
| **fallback share**              | Share of questions through the §11.8.1 fallback                                            | All runs        |

> **K=5 disclaimer.** When `SAMPLING_N` is small (5, as the codebase default), the empirical
> probability $\hat p = n/K$ takes only 6 discrete values, making Reliability Diagram /
> Murphy three-decomposition / Platt LOO calibration statistically meaningless. v5 deletes
> `calibration.py` and its 5 artefacts; the probabilistic columns retain a `†` footnote in
> `per_model_summary.md`. To reintroduce calibration, raise $K$ to ≥ 30.

#### 11.8.1 Belief fallback (when `belief_final IS NULL` but `parse_ok = 1`)

Legacy v3 runs and v4 belief-parse-failures still benefit from a degenerate probability
vector for proper scoring:

$$
p_l = \begin{cases} 1 - \epsilon, & \ell \in \widehat{G}_{i,M} \\ \dfrac{\epsilon}{k - |\widehat{G}_{i,M}|}, & \text{otherwise} \end{cases},\quad \epsilon = 0.05
$$

The sample is recorded with `belief_parse_ok=0`. Samples with full failure
(`parse_ok=0`) MUST NOT enter probabilistic averaging — pollution defence (flatten.py:L126–L152).

### 11.9 Aggregation strategies (`aggregation.py`)

For probability vectors across $K$ samples per question:

| Strategy             | Formula                                                                                  | Use                                            |
| -------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------- |
| Arithmetic mean      | $\hat p_l = (1/K)\sum_k p_{k,l}$                                                          | Phase 1 default                                |
| Logit-space mean     | Single: softmax of mean log-prob; multi: per-label sigmoid of mean logit                 | Bayesian model average; paper §C.9             |
| LOO shrinkage        | Scan $\alpha \in \{0, 0.1, ..., 1.0\}$; blend toward uniform prior on logit              | Adaptive smoothing (`aggregation.loo_shrinkage`, L145–L199) |

### 11.10 Statistical inference (`inference.py`)

| Function                                  | Algorithm                                                                                | Output                                |
| ----------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------- |
| `paired_bootstrap(bs_a, bs_b)`            | $B=5000$ paired resampling (same indices index both A and B)                              | `delta_mean / ci_low / ci_high / p_two_sided` |
| `holm_bonferroni(p_values)`               | $(n-i) \cdot p_{(i)}$ then cumulative max                                                | Adjusted p-values                     |
| `difficulty_tertile(gammas)`              | Sort per-question $\gamma_q$, cut into tertiles                                          | `low / mid / high` buckets             |
| `posterior_a_better_than_b(bs_a, bs_b)`   | Monte-Carlo $\Pr(\overline{BS}_A < \overline{BS}_B)$ on paired bootstrap                  | $\Pr(\mathrm{BI}_A > \mathrm{BI}_B) \in [0,1]$ |
| `metric_paired_bootstrap(metric_fn, ...)` | Generic paired bootstrap on any metric (FSS / Acc / MV-Acc / Fleiss / EBI)               | `delta_mean / ci_low / ci_high / p_two_sided / cohens_d` |
| `pairwise_paired_bootstrap(...)`          | All-pairs application of `paired_bootstrap` over models                                   | `list[ModelPairResult]`               |

Multi-comparison control: Holm-Bonferroni at the FWER level. The paired bootstrap is
*same-indexed* — the same bootstrap draws the same question id to index both A's and B's
arrays — to control the question-level variance the paper §G.2 quantifies at 62% of total.

### 11.11 Behavioural-analysis family (`behavior.py`)

Active when `BELIEF_PROTOCOL=True`. Four diagnostic groups:

| Group                          | Metrics                                                                                   | Output                            |
| ------------------------------ | ----------------------------------------------------------------------------------------- | --------------------------------- |
| Belief evolution               | Per-trial volatility $V$, inter-trial variance $\sigma$, convergence step, evidence efficiency $\eta$, counter-evidence engagement | `belief_evolution.csv`             |
| Reflection A/B                 | Paired-bootstrap 95% CI of $\Delta\text{BI}$ / $\Delta\sigma$ / $\Delta C$ / $\Delta\eta$ under matched `reflection_protocol_hash` | `reflection_ab.csv`                |
| Tool-usage PDP                 | Logistic / linear regression of `Pr(correct \| x)` and `E[NLL \| x]` on `tool_calls_count / react_steps / latency_ms / prompt_tokens / completion_tokens` | `tool_usage_pdp.csv`               |
| Confidence calibration         | Subjective 3-bin (low/medium/high) and numeric max-$p$ binned hit-rate; conflict flag      | `confidence_calibration_*.csv`     |

### 11.12 Output artefacts (`writers.py`)

A run's `analysis/` directory contains:

| File                                            | Schema                                          | Contents                                |
| ----------------------------------------------- | ----------------------------------------------- | --------------------------------------- |
| `per_model_summary.csv` / `.md`                 | 24 v3 + 4 FSS + 4 consistency + 7 prob = 39 cols | One row per model                        |
| `per_model_by_question_type.csv`                | sliced summary                                   | Bucketed by `question_type`             |
| `per_model_by_choice_type.csv`                  | sliced summary                                   | Bucketed by `choice_type`               |
| `per_model_composite_by_question_type.csv`      | composite weights + per-bucket metrics           | Composite Accuracy with subtype weights  |
| `per_model_composite_by_choice_type.csv`        | composite weights + per-bucket metrics           | Composite Accuracy with choice weights   |
| `error_breakdown.csv`                           | `Counter[error]`                                 | All samples (incl. cutoff)               |
| `finish_reason_breakdown.csv`                   | `Counter[finish_reason]`                         | Eligible samples only                    |
| `paired_delta_bi.csv`                           | `ModelPairResult`                                | Paired-bootstrap deltas (BI units)      |
| `paired_delta_bi_by_difficulty.csv`             | per-tertile result                               | Difficulty-stratified pair tests        |
| `metric_pairwise_bootstrap.csv`                 | per-metric × per-pair result                     | v5 multi-metric pairwise                 |
| `belief_evolution.csv`                          | `BeliefEvolutionRow`                             | Volatility / variance / convergence      |
| `reflection_ab.csv`                             | `ReflectionABRow`                                | Reflection A/B paired CIs                |
| `tool_usage_pdp.csv`                            | `PDPRow`                                         | Feature importance                      |
| `confidence_calibration_subjective.csv`         | `ConfidenceCalibrationRow`                       | 3-bin calibration                        |
| `confidence_calibration_numeric.csv`            | `NumericConfidenceCalibrationRow`                | max-$p$ binned                           |
| `entropy_accuracy_bins.csv`                     | per-bucket entropy/acc/Fleiss                    | Per-tertile diagnostic                   |
| `overall.json`                                  | aggregated metrics + metadata                    | Single JSON for downstream tooling       |
| `grid_summary.csv` (when grid enabled)          | per `(real_model, R, C)` 17-col main             | Grid main table                          |
| `grid_marginal_C.csv`, `grid_marginal_R.csv`    | scan along axis with the other anchored          | Saturation curves                        |
| `grid_pareto.csv`                               | one row per cell + `dominated_by`                | Pareto frontier                          |
| `grid_winrate.csv`                              | per real-model pair × cross-(R,C) cell wins/ties + significance count | Winrate matrix |

Rounding default: 4 decimals (writers.py:L113–L116); `avg_react_steps` 2 decimals;
`avg_latency_ms` 1 decimal.

---

## 12. Grid search (the virtual-slug encoding)

`Settings.TAVILY_MAX_RESULTS` ($R_{\mathrm{tav}}$ axis) and `REACT_MAX_SEARCH_CALLS` ($C$
axis) accept CSV lists. When either has length > 1, the run becomes a Cartesian grid over
$R \times C \times M$ cells, each producing its own DB file via a *virtual slug*:

```text
{real_model}::r{R}::c{C}
```

Composition: `db.compose_virtual_slug(real_model, R, C)` (db.py:L477–L516).
Parsing: `db.parse_virtual_slug(slug)` returns `(real_model, R, C)` or `None` for legacy
single-cell runs. The `::` delimiter is chosen to avoid collision with provider slugs
(further enforced by config validation: `MODELS` may not contain `::`).

**Per-cell settings.** `runner._resolve_settings(slug)` (runner.py:L160) reads the slug,
clones `Settings` via `model_copy(update=...)` with the cell's `R`/`C` overrides, and hands
each cell its own settings view.

**Output (`grid_summary.csv` per §11.12):** real_model, R, C, n_eligible, n_total, acc_mean,
acc_ci_lo/hi, bi_mean, bi_ci_lo/hi, nll_mean, ece, mean_search_calls, mean_latency_ms,
parse_ok_rate, belief_parse_ok_rate. Bootstrap CIs at the cell level are computed by
`grid._bi_ci_from_bs_array` and `grid._acc_ci_for_samples` (grid.py:L122–L142, B=5000,
seed=42).

**Plot anchors.** `GRID_DEFAULT_R` / `GRID_DEFAULT_C` (config.py:L319–L322) pin the marginal
slices when the grid is multi-axis; if unset, `r_list[0]` / `c_list[0]` apply (validated
to belong to the lists; see §7.1).

**Pin tests.** `test_grid_slug.py`, `test_grid_dispatcher.py`, `test_grid_analysis.py`,
`test_grid_settings_view.py`, `test_runner_grid_model.py`.

---

## 13. CLI and how to run

### 13.1 Commands

```bash
# Run the entire dataset
python evaluation.py

# Filter by question_type (repeatable)
python evaluation.py --question-type yes_no --question-type binary_named

# Filter by choice_type (repeatable)
python evaluation.py --choice-type single

# Combined filter (AND): only multi-select multiple_choice
python evaluation.py --question-type multiple_choice --choice-type multi

# Do not generate analysis/ at run end (raw DBs still land in db/)
python evaluation.py --skip-analysis

# Refresh analysis/ independently (does not modify the DB)
python -m forecast_eval.analysis runs/{run_id}
```

`--question-type` values: `yes_no` / `binary_named` / `multiple_choice`, repeatable; if
not passed = no restriction. `--choice-type` values: `single` / `multi`, repeatable; if not
passed = no restriction. All tunables other than `--skip-analysis` go through `.env`.

### 13.2 Step-by-step run flow (`evaluation.py`)

1. `argparse` parses `--question-type` / `--choice-type` / `--skip-analysis`, assembling a
   `QFilter`.
2. `Settings()` loads and validates `.env` (incl. `MODEL_TRAINING_CUTOFFS` + `RUNS_ROOT`,
   running every check from §7.1).
3. Generate or reuse `run_id` → determine `run_dir = RUNS_ROOT/{run_id}`; create
   `db/` / `analysis/` / `logs/`.
4. Compute `source_db_hash` / `metadata_hash` / `prompt_templates_hash` and conditional
   `reflection_protocol_hash` / `belief_protocol_hash` (evaluation.py:L46–L75).
5. For each `MODELS[i]` (or virtual slug under grid):
   1. Open `conn = RUNS_ROOT/{run_id}/db/{safe_slug(model)}.db` (alphabet `[A-Za-z0-9._-]`,
      illegal characters replaced).
   2. `db.init_schema(conn, SAMPLING_N)` — dynamically create `s{i}_*` columns + apply
      v2→v5 migrations as needed.
   3. `loader.sync_prompt_templates(src, conn)` / `loader.sync_questions(src, conn,
      filter)`.
   4. `db.register_run_meta(conn, run_id, model, hashes, training_cutoff, ...)`.
6. `_write_manifest()` writes `manifest.json` (evaluation.py:L123–L192) containing:
   `run_id` / `schema_version` / `analysis_schema` / `sampling_n` / `models` (virtual) /
   `model_files` / `model_training_cutoffs` / `filters` / `hashes` /
   `reflection_protocol_hash` / `belief_protocol_hash` / `grid` / `started_at` /
   `finished_at: null`.
7. `runner.run(...)` starts the asyncio event loop:
   1. Per model: `db.load_completed_samples(conn, SAMPLING_N)` becomes the resume baseline.
   2. Generate the Cartesian product `questions × MODELS × range(SAMPLING_N)`; subtract
      the resume set.
   3. §3.1 admissibility filter: `(q, model, idx)` with `q.end_time <= cutoff` are written
      directly as `skipped_training_cutoff` rows; they never enter the LLM task queue.
   4. Remaining tasks run under three independent semaphores (LLM /
      Search / Detector concurrency), with `asyncio.create_task()` + `asyncio.as_completed()`
      polling.
   5. Each completion → routed to that model's writer → batch UPSERT into `s{i}_*`
      columns.
   6. One log line per completion (loguru):
      `[5/1610] q=69566c13 qt=binary_named ct=single model=openai/gpt-5 sample=2/5 correct=True steps=4 tool_calls=3 latency=8421ms`
8. Per model: `db.finish_run_meta(conn, run_id)`; `_finalise_manifest()` writes
   `finished_at`.
9. Unless `--skip-analysis`: `forecast_eval.analysis.run_analysis(run_dir)` walks DBs →
   metric stack → CSV/MD/JSON in `analysis/`.

---

## 14. Logging (`loguru`)

```python
from loguru import logger
import sys, os

logger.remove()
logger.add(
    sys.stderr,
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
)
logger.add(
    f"{LOG_DIR}/{run_id}.log",
    level="DEBUG",
    rotation="100 MB",
    retention=5,
)
```

### 14.1 Progress printing

```text
12:03:44 | INFO    | [run=20260424-120344-a7k3] [5/1610] q=69566c13 qt=binary_named ct=single model=openai/gpt-5 sample=2/5 correct=True steps=4 tool_calls=3 latency=8421ms
```

* `[5/1610]` denominator = `len(questions_after_filter) × len(MODELS) × SAMPLING_N`
  minus completed resume tasks.
* One line per sample completion.
* On error: `[x/xx] q=.. model=.. error=rate_limit retry_exhausted` at `ERROR`.

---

## 15. Test plan (`tests/` — 33 files, ~13 K LOC, all offline)

A single evaluation is costly (full dataset × models × N samples), so getting tests stable
saves significant API spend. All tests run **offline** and **do not burn the API**:
Tavily / OpenRouter exist as fixtures or mocked stand-ins.

### 15.1 Test → invariant mapping

The tests act as proofs that each component of $\mathcal{R}$ is implemented as advertised.

| Component of $\mathcal{R}$ / paper claim                                               | Pin tests                                                                                                                                       |
| -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| $\mathcal{D}$: dataset shape; templates contract; hashes deterministic                   | `test_db.py`, `test_evaluation.py`                                                                                                              |
| $M$: per-model DBs; virtual slugs; resume per model                                      | `test_runner_grid_model.py`, `test_runner_resume.py`                                                                                            |
| $\kappa_M$: admissibility filter; cutoff-row write contract                             | `test_training_cutoff.py`                                                                                                                       |
| $\delta$: tool-layer injection; LLM never sees `end_date`                                | `test_search.py`, `test_react.py`                                                                                                               |
| $T$, $C$: ReAct loop bounded; budget gates; harness priority chain; v5.1 switches        | `test_react.py` (1432 LOC), `test_react_reflection.py`                                                                                          |
| $R$: renderer correct for all three qtypes; protocol additions outside `prompt_templates_hash` | `test_prompts.py`                                                                                                                              |
| $\Psi$ + $\phi$: parser correctness; strict equality; >26 options round-trip              | `test_parser.py`, `test_parser_belief.py`                                                                                                       |
| $\Gamma$: aggregation correctness end-to-end                                             | `test_analysis.py` (670 LOC), `test_aggregation.py`, `test_consistency.py`, `test_inference.py`, `test_proper_score.py`                          |
| $H_{\mathrm{aux}}$: detector whitelist; fail-closed; AUTH immediate-drop                   | `test_leak_filter.py`                                                                                                                           |
| Composite: weights validation; per-metric overrides; allowlist                            | `test_composite_score.py`                                                                                                                       |
| FSS: closed-form chance baselines; $(\alpha,\beta)$ sensitivity                          | `test_fss.py`, `test_fss_sensitivity.py`                                                                                                        |
| Exam-score: paper Eq. 17 corner cases (FP-veto, parse-fail = 0, cutoff = None)            | `test_exam_score.py`                                                                                                                            |
| Behavioural metrics: belief evolution, reflection A/B, tool PDP, confidence calibration   | `test_behavior.py`                                                                                                                              |
| Grid: virtual-slug encoding; per-cell settings view; analysis pipeline                   | `test_grid_slug.py`, `test_grid_dispatcher.py`, `test_grid_analysis.py`, `test_grid_settings_view.py`, `test_plot_analysis_grid.py`             |
| DB schema migration: v2→v5 forward path                                                   | `test_db_v4_migration.py`, `test_db_v5_migration.py`                                                                                            |
| Information barrier: provider-native browsing forbidden                                  | `test_llm_no_browsing.py`                                                                                                                       |
| Error tiering: classification + backoff lookup                                            | `test_errors.py`                                                                                                                                |
| Configuration: every validator, every env-var contract                                    | `test_config.py`                                                                                                                                |
| End-to-end: dry-run replaces all transports with stubs                                    | `test_smoke_dry_run.py`                                                                                                                         |

### 15.2 CI redlines (must always be green)

Five tests map one-to-one to the components of $\mathcal{R}$ that, if broken, invalidate the
entire run unit. They MUST stay green on every commit:

1. `test_prompts.py` — guards $R$.
2. `test_parser.py` — guards $\Psi$ + $\phi$.
3. `test_training_cutoff.py` — guards $\kappa_M$ admissibility.
4. `test_llm_no_browsing.py` — guards the information barrier.
5. `test_analysis.py` — guards $\Gamma$.

If any of these fails, the run unit's contract is broken and no result downstream can be
trusted.

### 15.3 Heaviest test files

The implementation's complexity is reflected in test file sizes. The longest tests:

| File                           | LOC   | What it covers                                                               |
| ------------------------------ | ----- | ---------------------------------------------------------------------------- |
| `test_react.py`                | 1432  | Full ReAct loop: every harness branch, priority chain, finalisation, v5.1     |
| `test_search.py`               |  830  | Tavily wrapper + key rotation + raw-content truncation + audit metadata     |
| `test_behavior.py`             |  762  | Belief evolution, reflection A/B, tool PDP, confidence calibration           |
| `test_analysis.py`             |  670  | Phase 0–6 of the analysis orchestrator                                        |
| `test_db.py`                   |  630  | Schema, migrations, AsyncWriter, hashes, redaction                          |
| `test_inference.py`            |  630  | Paired bootstrap, Holm, posterior, multi-metric                              |
| `test_consistency.py`          |  595  | Fleiss κ stratification, entropy, VCI, MVG                                   |
| `test_grid_analysis.py`        |  605  | Virtual-slug grid analysis end-to-end                                        |
| `test_leak_filter.py`          |  550  | Whitelist, fail-closed, audit fields                                         |
| `test_fss.py`                  |  528  | FSS Tversky, chance baselines, edge cases                                    |
| `test_composite_score.py`      |  509  | Composite weights, allowlist, override parsing                              |
| `test_prompts.py`              |  447  | Renderer rules across all three qtypes + protocol toggles                    |
| `test_exam_score.py`           |  426  | exam-score corner cases per paper Eq. 17                                      |

Run all tests:

```bash
pytest tests/ -q
```

---

## 16. Conda environment (`environment.yml`)

```yaml
name: forecast
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - pip:
      - openai>=1.50            # OpenAI-compatible SDK (main LLM + detector)
      - tavily-python>=0.5
      - pydantic>=2.6
      - pydantic-settings>=2.2
      - python-dotenv>=1.0
      - loguru>=0.7
      - httpx>=0.27
      - tenacity>=9.0           # retry decorator
      - pytest>=8.0
      - pytest-asyncio>=0.23
      - respx>=0.21             # httpx mocking
```

Create the environment:

```bash
conda env create -f environment.yml
conda activate forecast
cp .env.example .env
# Edit .env: LLM_API_KEY, TAVILY_API_KEY, LEAK_DETECTOR_API_KEY, MODELS, MODEL_TRAINING_CUTOFFS
python evaluation.py --question-type yes_no
```

`matplotlib` is intentionally **not** in `environment.yml` — analysis stays
dependency-light; install it locally only to render the on-demand plot family
(`scripts/plot_analysis.py`).

---

## 17. Final-premise summary (last-review checklist)

The 30-item contract that, jointly satisfied, materialises the run unit $\mathcal{R}$.

1. **Source data has 7 fields**: `id / choice_type / question_type / event / options /
   answer / end_time`, with letter-encoded answers throughout.
2. **Source DB is checked into Git** (read-only example dataset; ensures `source_db_hash`
   reproducibility; `SOURCE_DB` / `SOURCE_TABLE` can point to a custom dataset).
3. **The LLM does not see $\chi_i$**; injection happens at the tool implementation layer
   (react.py:L182, search.py:L133).
4. **Tavily `end_date = end_time + δ`**; project default $\delta = -1$ as the strict
   baseline (all reports default to comparison under $\delta = -1$).
5. **Sample admission via $\kappa_M$**: samples with `q.end_time ≤ cutoff` are written as
   `error="skipped_training_cutoff"`, never invoke the LLM, and are not retried on resume.
6. **Three-layer leakage barrier** (paper §4.2.5): manual curation (L0) + tool-level
   `end_date` filter (L2 algorithmic) + Stage-2 LLM detector (L3 semantic, fail-closed).
   Audit (paper §4.3.4): N=270 → recall 98.7%, residual 1.1%, Wilson 95% UB 3.2%.
7. **Provider-native browsing forbidden**: no `:online`, no `plugins`, no provider-private
   web tool — startup-validated and wire-asserted; pinned by `test_llm_no_browsing.py`.
8. **Stage-2 detector input whitelist**: only `title / url / published_date / content /
   raw_content / cutoff_date`; `Question` fields never enter the detector.
9. **Prompt assembly** is performed by `prompts.py`: pull templates from
   `dataset_metadata` → render `outcomes_block` and `output_format` per `question_type`.
   Protocol additions (reflection / budget-awareness / belief) live as runtime slots;
   `prompt_templates_hash` is invariant under protocol toggles, but the rendered full
   user message lands in each sample's `user_prompt` field.
10. **Three independent protocol fingerprints** (`prompt_templates_hash` /
    `reflection_protocol_hash` / `belief_protocol_hash`) enable three-axis ablation studies
    without collisions; all three live both in `run_meta` and at the top level of
    `manifest.json`.
11. **Letter encoding $\phi$**: A=0, B=1, ..., Z=25, then ASCII continuation `[ \ ] ^ _
    ` ` ` a b c …`; round-trip pinned for >26-option questions.
12. **Evaluation = letter-set frozenset strict equality** (`pred == gt`). Soft-penalty
    companions (exam-score, FSS, Hamming) coexist for analysis but do not change strict
    equality.
13. **Parse failure ≠ error**; refusal / format-failure rate are tallied separately.
14. **Multi-model single-run Cartesian product**, with resume via `run_id`; one DB per
    model; `run_meta` records `filters_snapshot` + four hashes + `training_cutoff` +
    redacted `config_snapshot`.
15. **Auth errors stop the entire run**; other errors are retried with tiered backoff per
    `errors.py`.
16. **Content-policy violations are not retried**, just marked.
17. **All flexible parameters live in `.env`**; CLI exposes only `--question-type` /
    `--choice-type` / `--skip-analysis`.
18. **Main entrypoint `evaluation.py`**: creates `RUNS_ROOT/{run_id}/`, validates
    settings, runs the runner, then runs analysis (unless `--skip-analysis`).
19. **Conda + Python 3.12 + loguru**, with progress `[x/xx]` logged at INFO, DEBUG-level
    file log under `logs/{run_id}.log`.
20. **SQLite WAL + `PRAGMA foreign_keys=ON` + one async writer task per model**, avoiding
    concurrent-write lock contention.
21. **Each model DB is self-contained**: built-in `questions` + `prompt_templates` copies
    + `run_meta`, independently distributable and replayable.
22. **Schema versioning v2→v3→v4→v5** via O(1) `ADD COLUMN` migrations; `BELIEF_PROTOCOL`
    off keeps v3 byte-equivalent behaviour.
23. **Metric naming**: the standard `pass@k` corresponds to this project's `pass_any@N`;
    the legacy threshold-style metric is renamed `at_least_k_correct@N`.
24. **Recording and analysis are separated**: the DB stores only raw sample records;
    pass@1 / pass_any@N / FSS / Cohen κ / Fleiss κ / BI / per-correct cost etc. are
    computed post-hoc by `forecast_eval.analysis` and written to `analysis/` as
    CSV / MD / JSON.
25. **Grid search via virtual slug**: `(real_model, R, C)` encoded as
    `{real}::r{R}::c{C}`; runner / DB / main analysis pipeline are byte-unchanged for
    legacy single-cell runs.
26. **Composite Accuracy is the headline** (paper Eq. 18); default subtype weights follow
    "harder questions discriminate better" (yes_no=0.15 / binary_named=0.15 /
    multiple_choice=0.70; single=0.40 / multi=0.60); per-metric overrides supported with a
    known-metrics allowlist that fails-fast on typos.
27. **FSS** (Format Skill Score, paper §3.5): Tversky $(\alpha=2.0, \beta=0.5)$,
    chance-corrected, single-uniform-baseline / multi-Bern(0.5) baseline with closed-form
    expectation per paper Eq. 23.
28. **Per-correct cost** ($C^{\mathrm{per\text{-}correct}}_m$, paper Eq. 25) is the official
    cost-effectiveness axis; invoice-based amortisation across difficulty-weighted
    notional correct count places "expensive but accurate" and "cheap but reckless" on the
    same scale.
29. **Harness resilience** (v5.1) lives in four orthogonal switches:
    `REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT` (graded transition with `LOOKAHEAD`),
    `REACT_BUDGET_EXCEEDED_DROP_TOOLS` (in-loop budget gate), `REACT_FINAL_ANSWER_RETRY`
    (post-loop backstop), and `REACT_MIN_SEARCH_CALLS / REACT_MAX_NUDGES` (opt-in soft
    floor); their priority chain is deterministic and test-pinned.
30. **Belief protocol** (v4) is opt-in via `BELIEF_PROTOCOL=True`; when off, all
    probabilistic columns write NULL and the analysis pipeline skips the proper-scoring
    family (BI / NLL / MBS / ABI / behavioural traces). This preserves byte-equivalent
    accuracy outputs against v3 runs.

---

## 18. Suggested module landing order

For a from-scratch reimplementation, this order keeps each step locally verifiable; each
step lists the test that should pass before moving on.

| # | Module                          | Pin test                          |
| - | ------------------------------- | --------------------------------- |
| 1 | `environment.yml` + `.env.example` + `.gitignore`              | (smoke: `python -c 'import forecast_eval'`) |
| 2 | `forecast_eval/config.py`       | `test_config.py`                   |
| 3 | `forecast_eval/db.py`           | `test_db.py`, `test_db_v5_migration.py` |
| 4 | `forecast_eval/loader.py`       | (covered by `test_db.py`)          |
| 5 | `forecast_eval/prompts.py`      | `test_prompts.py`                  |
| 6 | `forecast_eval/parser.py`       | `test_parser.py`, `test_parser_belief.py` |
| 7 | `forecast_eval/errors.py`       | `test_errors.py`                   |
| 8 | `forecast_eval/search.py`       | `test_search.py`                   |
| 9 | `forecast_eval/leak_filter.py`  | `test_leak_filter.py`              |
| 10 | `forecast_eval/tools.py`       | (covered by `test_search.py`)      |
| 11 | `forecast_eval/llm.py`         | `test_llm_no_browsing.py`          |
| 12 | `forecast_eval/react.py`       | `test_react.py`, `test_react_reflection.py` |
| 13 | `forecast_eval/runner.py`      | `test_runner_resume.py`, `test_runner_grid_model.py`, `test_training_cutoff.py` |
| 14 | `forecast_eval/analysis/*`     | `test_analysis.py`, plus per-metric tests (`test_fss.py`, `test_consistency.py`, `test_inference.py`, `test_composite_score.py`, `test_exam_score.py`, `test_behavior.py`, `test_grid_analysis.py`) |
| 15 | `evaluation.py` (main entry)   | `test_evaluation.py`, `test_smoke_dry_run.py` |

Get a smoke test passing first via `--question-type yes_no` + `MODELS=openai/gpt-4o-mini` +
`SAMPLING_N=1`, verify that `prompts.render_user_prompt` output and `parser.parse_answer`
normalisation are correct, then open up to full evaluation.

---

> **One sentence.** This codebase is the contract that turns the paper's run unit
> $\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)$ — together
> with the auxiliary detector $H_{\mathrm{aux}}$ — into a Python module per symbol, a SQLite
> column per observation, a CSV column per metric, and a unit test per invariant: every
> number that ever appears in the report can be traced back to a row in a wide table, a
> hash in `run_meta`, an audit verdict in `search_calls`, or a green test in `tests/`.
