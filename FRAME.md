# OracleProto Technical Framework

> This document is the engineering specification of the OracleProto reference
> implementation. It traces every symbol of the run unit
> $`\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)`$,
> together with the auxiliary detector $`H_{\mathrm{aux}}`$, to a module,
> function, environment variable, SQLite column, and the unit test that pins
> the invariant. Read it alongside `DESIGN.md` for the rationale behind each
> trade-off; the two-page newcomer orientation lives in `README.md`.

## How to read this document

The document is a top-down reference. Sections 1–4 establish what is being
measured, sections 5–8 describe how the pipeline produces measurements, and
sections 9–12 cover the analytical and operational machinery that turns those
measurements into reportable numbers. Each section is independently navigable,
but later sections assume the symbol map of §1.

Two citation forms appear throughout:

* `module.py:Lnnn` references current head-of-tree line numbers. When a range
  is given, the cited symbol or contract spans those lines.
* `test_<name>.py` references the file under `tests/`. The repository ships 33
  test files containing roughly 560 individual cases, all offline.

The notation $`X \to Y`$ in textual prose means "X resolves to / produces Y";
$`X = Y`$ retains its mathematical meaning.

---

## 1. The run unit

This codebase is the reference implementation of **OracleProto**, a
reproducible framework for benchmarking the *native forecasting capability* of
LLMs through knowledge cutoffs and temporal masking.

The implementation has two responsibilities: it materialises the run unit
$`\mathcal{R}`$ so that the same configuration produces byte-equivalent
intermediate artefacts and stochastic-only differences in the final-answer
text, and it binds the auxiliary leakage detector $`H_{\mathrm{aux}}`$ to the run
metadata via a SHA-256 fingerprint so the leakage barrier itself is
byte-reproducible. Every section that follows answers a single question: how
does this realise some component of $`\mathcal{R}`$, and which test pins the
contract?

### 1.1 Symbol-to-implementation map

The run unit $`\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R,
\Psi, \phi, \Gamma)`$, together with the auxiliary detector
$`H_{\mathrm{aux}}`$ logged as run metadata, maps to the codebase as follows.
Each symbol resolves to one configuration knob, one code path, one DB column
where applicable, and one test that pins the contract.

| Symbol             | Object                          | Env / config key                                | Code path                                                                      | DB column / artefact                              | Pin test                          |
| ------------------ | ------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------- | --------------------------------- |
| $`\mathcal{D}`$      | Discrete forecasting dataset    | `SOURCE_DB`, `SOURCE_TABLE`                     | `loader.sync_questions` (loader.py:L77)                                        | `questions` table; `manifest.hashes.source_db`     | `test_db.py`, `test_evaluation.py` |
| $`M`$                | Evaluated model (slug)          | `MODELS` (CSV)                                  | `runner._resolve_settings` (runner.py:L160); `llm.chat` per model              | `run_meta.model`; one DB file per model            | `test_runner_grid_model.py`        |
| $`\kappa_M`$         | Knowledge cutoff per model      | `MODEL_TRAINING_CUTOFFS=<slug>=YYYY-MM-DD,...`  | `runner.build_task_plan` admissibility filter (runner.py:L132–L199)            | `run_meta.training_cutoff`; `s{i}_error="skipped_training_cutoff"` | `test_training_cutoff.py`         |
| $`\delta`$           | Temporal masking offset (days)  | `TAVILY_END_DATE_OFFSET_DAYS` (default `-1`)    | `react._compute_end_date` (react.py:L182); `search.tavily_search`              | `s{i}_search_calls[*].end_date`                    | `test_search.py`, `test_react.py` |
| $`T`$                | Max ReAct steps                 | `REACT_MAX_STEPS` (default `12`)                | `react.run_react` outer loop (react.py:L248)                                   | `s{i}_react_steps`; `s{i}_step_metrics`            | `test_react.py`                   |
| $`C`$                | Max search calls (grid axis)    | `REACT_MAX_SEARCH_CALLS` (CSV; default `[8]`)   | budget gate `react.py:L276–L279`; tool-call validation L429–L503               | virtual slug `::c{C}`; `s{i}_tool_calls_count`      | `test_react.py`, `test_grid_slug.py` |
| $`R`$                | Input renderer                  | `dataset_metadata.features_json.prompt_reconstruction` | `prompts.render_user_prompt` (prompts.py:L447)                          | `s{i}_user_prompt`; `manifest.hashes.prompt_templates` | `test_prompts.py`                |
| $`\Psi`$             | Output parser & validity        | (no env knob)                                   | `parser.parse_answer` (parser.py:L40)                                           | `s{i}_final_answer_letters`, `s{i}_parse_ok`        | `test_parser.py`                  |
| $`\phi`$             | Answer normalisation map        | (letter encoding rule, see §4.8)                | `parser.parse_gt`, `parser.is_correct` (parser.py:L102)                         | `s{i}_correct`                                     | `test_parser.py`                  |
| $`\Gamma`$           | Aggregation rule                | `COMPOSITE_WEIGHTS_*`, `SAMPLING_N`, etc.       | `forecast_eval/analysis/*` (auto-invoked from `evaluation.py`)                  | CSV / MD / JSON in `runs/{run_id}/analysis/`        | `test_analysis.py`                |
| $`H_{\mathrm{aux}}`$ | Leakage detector (Stage 2)      | `ENABLE_SEARCH_LEAK_FILTER`, `LEAK_DETECTOR_*`  | `leak_filter.filter_search_result` (leak_filter.py:L348)                        | `s{i}_search_calls[*].audit.detector_*`             | `test_leak_filter.py`             |
| $`\hat{p}_{q,j}`$    | Belief vector (v4 companion)    | `BELIEF_PROTOCOL` (default `False`)             | `parser.parse_belief` (parser.py:L117); `react.run_react` finalisation          | `s{i}_belief_final`, `s{i}_belief_trace`, `s{i}_belief_parse_ok` | `test_parser_belief.py`, `test_react_reflection.py` |

The auxiliary axis $`R_{\mathrm{tav}}`$ denotes Tavily results-per-call and is
distinct from the renderer symbol $`R`$. It corresponds to `TAVILY_MAX_RESULTS`,
which is CSV-valued with default `[5]`. Together with $`C`$ it spans the grid
encoded into the virtual slug `{real_model}::r{R}::c{C}` (§10).

### 1.2 Invariants

These eight statements are framework-level invariants that the implementation
must hold. Each is enforced by code and pinned by at least one test, so a
failing test invalidates the run unit.

1. **The LLM never sees $`\chi_i`$.** The `web_search` tool schema exposed to the
   LLM declares only a `query` parameter (tools.py:L7–L24); $`\chi_i = \tau_i +
   \delta`$ is hard-coded by the tool implementation layer (react.py:L182,
   search.py:L133). Pinned by `test_search.py` for the payload contract and
   `test_react.py` for end-to-end injection.
2. **Provider-native browsing is forbidden.** Slugs ending in `:online` are
   rejected at startup (config.py:L599–L614) and re-asserted on the wire
   (llm.py:L74–L98), as are `extra_body.plugins` and any tool schema other
   than the declared `web_search`. Pinned by `test_llm_no_browsing.py` and
   `test_config.py`.
3. **Sample admission precedes any LLM call.** The check $`\kappa_M \le \chi_i`$
   happens at task-plan generation; admissibility violations write
   `error="skipped_training_cutoff"` rows directly without consuming any LLM or
   Tavily budget (runner.py:L132–L199). Pinned by `test_training_cutoff.py`.
4. **Strict frozenset equality scores answers.** `parser.is_correct(pred, gt)`
   is one line of `pred == gt` (parser.py:L102–L106), and all three question
   types reduce to this equality. Pinned by `test_parser.py`.
5. **Databases store raw observations only.** No aggregates, no derived
   metrics; every metric in §9 is computed post-hoc by `forecast_eval.analysis`
   reading the wide table. Pinned by `test_analysis.py`, which runs analysis
   on a hand-crafted DB fixture without retouching it.
6. **The Stage-2 detector $`H_{\mathrm{aux}}`$ has a closed input whitelist.**
   Only `title`, `url`, `published_date`, `content`, `raw_content`, and
   `cutoff_date` enter the detector prompt (leak_filter.py:L212–L227); the
   question text, options, and gold answer are never passed. Pinned by
   `test_leak_filter.py`.
7. **Three independent fingerprints rather than one.**
   `prompt_templates_hash`, `reflection_protocol_hash`, and
   `belief_protocol_hash` live side by side in `run_meta` and at the manifest
   top level (db.py:L143–L150, evaluation.py:L171–L178), so ablations along the
   {template, reflection, belief} axes do not collide.
8. **Composite Accuracy is the headline.** The default subtype weights are
   `yes_no=0.15`, `binary_named=0.15`, `multiple_choice=0.70` (config.py:L365);
   per-metric overrides are validated against an allowlist of known metric
   names and fail fast on typos (config.py:L515–L535,
   composite.py:L77–L127). Pinned by `test_composite_score.py`.

---

## 2. The dataset $`\mathcal{D}`$

### 2.1 Source database

The example dataset ships as `forecast_eval_set_example.db`, with main table
`forecast_eval_set_example`. Both names are configurable via `SOURCE_DB` and
`SOURCE_TABLE` in `.env`. Custom datasets must keep the seven-column schema
and the `dataset_metadata` structure described below. `SOURCE_TABLE` accepts
only SQLite-legal identifiers matching `^[A-Za-z_][A-Za-z0-9_]*$`, and is
validated at startup (config.py:L586–L595) because it is interpolated into
queries verbatim and would otherwise be a SQL-injection vector.

The main table contains $`N`$ rows of seven columns:

| Field           | Type    | Description                                                                                                             |
| --------------- | ------- | ----------------------------------------------------------------------------------------------------------------------- |
| `id`            | TEXT PK | Unique question ID, sourced from HuggingFace.                                                                             |
| `choice_type`   | TEXT    | `single` or `multi`, computed from the answer-letter count.                                                              |
| `question_type` | TEXT    | `yes_no`, `binary_named`, or `multiple_choice`; selects the prompt-template family.                                       |
| `event`         | TEXT    | The event description $`x_i`$, carrying no options, role-setting, or format requirements.                                  |
| `options`       | TEXT    | The option set $`\mathcal{A}_i`$ as a JSON array. `yes_no` is `["Yes","No"]`; `binary_named` is two entity names; `multiple_choice` is the labelled options. |
| `answer`        | TEXT    | $`Y_i`$ encoded as letters: single is `"A"`, multi is `"A, B"` (comma + space). The letter-to-index rule is in §4.8.        |
| `end_time`      | TEXT    | Resolution time $`\tau_i`$ (Asia/Shanghai), formatted as `YYYY-MM-DD`.                                                     |

Indexes: `idx_<table>_choice_type`, `idx_<table>_question_type`,
`idx_<table>_end_time`.

The auxiliary table `dataset_metadata` holds a single row whose `features_json`
field records all prompt templates, column descriptions, and conversion logs.
The renderer $`R`$ reads templates from this table at runtime; they are
deliberately not hard-coded in source. The `prompt_templates_hash` fingerprint
covers exactly the eight template keys listed in §2.3, while protocol
additions for reflection, budget-awareness, and belief live as runtime slots
that do not enter the hash (§4.7).

### 2.2 The example dataset

`forecast_eval_set_example.db` contains 80 questions spanning 2026-03-12 to
2026-04-14:

| question_type / choice_type | single | multi | total |
| --------------------------- | -----: | ----: | ----: |
| `yes_no`                    |     37 |     0 |    37 |
| `binary_named`              |      3 |     0 |     3 |
| `multiple_choice`           |     32 |     8 |    40 |
| **total**                   |   **72** |  **8** |    **80** |

`multiple_choice` option counts in this example range from 3 to 14, but the
parser supports the full ASCII-continuation regime described in §4.8 so that
custom datasets with up to 35 options remain valid without code changes.

The framework itself is dataset-agnostic once the seven-column contract and
`dataset_metadata` shape are met.

### 2.3 The `prompt_reconstruction` contract

The renderer $`R`$ requires exactly these eight keys, and any missing key raises
at load time (loader.py:L13–L22):

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
$`\text{sha256}(\text{canonical\_kv\_string}(\text{templates}))`$ over these
keys only. Whether the run enabled reflection, belief, or budget-awareness is
invisible to this hash; those texts are hashed independently into
`reflection_protocol_hash` and `belief_protocol_hash` (§6.3).

### 2.4 Worked examples

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

By convention, the `event` field carries no options or format requirements;
those are spliced in at call time by the renderer $`R`$ (§4.7).

---

## 3. End-to-end pipeline

The pipeline runs in seven stages, from `.env` through the analysis writers.
A reader who wants to know where in the pipeline a given guarantee is enforced
should consult §4 for the information-boundary stages and §5 for the in-loop
control flow.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   OracleProto Evaluation Pipeline                      │
└────────────────────────────────────────────────────────────────────────┘

[.env]  →  [python evaluation.py [--question-type ...] [--choice-type ...]]
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │ 1. Load Settings       │
                          │    & init run_id       │
                          └────────────────────────┘
                                      │
                                      ▼
                          ┌──────────────────────────────────┐
                          │ 2. Sync Source                   │
                          │    forecast_eval_set_example.db  │
                          │      → questions table           │
                          │      → prompt_templates table    │
                          │    Compute four hashes           │
                          └──────────────────────────────────┘
                                      │
                                      ▼
                          ┌────────────────────────────────────┐
                          │ 3. Resume Check                    │
                          │    load_completed_samples per      │
                          │    model: skip rows with non-      │
                          │    retryable terminal state        │
                          └────────────────────────────────────┘
                                      │
                                      ▼
                 ┌────────────────────────────────────────┐
                 │ 4. Task Plan (D × M × N)               │
                 │    apply κ_M admissibility filter      │
                 │    write skipped_training_cutoff rows  │
                 │    asyncio.Semaphore for LLM/Search/   │
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
                      │ ReAct Loop F_M (per        │
                      │ sample, see §5):           │
                      │   render(q) → user_prompt  │
                      │   while step < T:          │
                      │     pre-step injection     │
                      │     llm.chat(messages,     │
                      │               tools=...)   │
                      │     for each tool_call:    │
                      │       u_t = tavily.search( │
                      │              query, χ_i)   │
                      │       ũ_t = AuxLeakFilter( │
                      │              u_t, χ_i)     │
                      │   v5.1 final-answer-retry  │
                      │   parser.parse_answer      │
                      └──────────┬─────────────────┘
                                 │
                                 ▼
                      ┌────────────────────────┐
                      │ 5. Score               │
                      │    Ψ ∘ φ on the        │
                      │    parsed letter set   │
                      └──────────┬─────────────┘
                                 │
                                 ▼
                      ┌────────────────────────┐
                      │ 6. Enqueue → writer    │
                      │    Single AsyncWriter  │
                      │    per model; WAL +    │
                      │    batch UPSERT        │
                      └──────────┬─────────────┘
                                 │
                                 ▼
                      ┌────────────────────────┐
                      │ 7. analysis.run        │
                      │    Aggregations Γ:     │
                      │    composite acc, FSS, │
                      │    Cohen κ, Fleiss κ,  │
                      │    pass@k, BI, NLL …   │
                      │    → CSV / MD / JSON   │
                      └────────────────────────┘
```

The seven stages, in narrative form:

1. **Load and validate.** `Settings()` parses `.env` via pydantic-settings and
   runs every check enumerated in §7.2. A `run_id` is minted (or reused if the
   user supplied one), and the run directory `RUNS_ROOT/{run_id}/` is created
   with `db/`, `analysis/`, and `logs/` subdirectories.
2. **Sync source.** `loader.sync_questions` and `loader.sync_prompt_templates`
   copy the source DB tables into each model's run-DB, after which the four
   reproducibility hashes (`source_db_hash`, `metadata_hash`,
   `prompt_templates_hash`, and conditionally `reflection_protocol_hash` and
   `belief_protocol_hash`) are computed and stored.
3. **Resume.** Per model, `db.load_completed_samples` scans the existing
   `run_results` rows and emits the `(question_id, sample_idx)` set whose
   slots are either filled normally or already marked
   `skipped_training_cutoff`.
4. **Task plan.** `runner.build_task_plan` produces the Cartesian product of
   $`\mathcal{D} \times M \times \{0, \dots, n-1\}`$, subtracts the resume set,
   then applies the $`\kappa_M`$ admissibility filter (§4.2). Inadmissible rows
   are written as `skipped_training_cutoff` directly without ever touching the
   LLM or Tavily.
5. **Worker fan-out.** Three independent `asyncio.Semaphore` objects bound LLM,
   Tavily, and detector concurrency. Each worker drives one sample through the
   ReAct loop $`F_M`$ (§5), which interleaves LLM turns with Tavily calls that
   are themselves audited by $`H_{\mathrm{aux}}`$.
6. **Score and persist.** Each completed `SampleResult` is enqueued to that
   model's `AsyncWriter` (§6.5), which batches UPSERTs and flushes either every
   `DB_COMMIT_BATCH` entries or once per second.
7. **Analyse.** Unless `--skip-analysis` was passed,
   `forecast_eval.analysis.run_analysis(run_dir)` walks every model DB, runs
   the metric stack of §9, and writes the artefacts of §9.12 into
   `analysis/`.

### 3.1 Concurrency model

Three semaphores partition the externally bounded resources:
`LLM_MAX_CONCURRENCY`, `SEARCH_MAX_CONCURRENCY`, and
`LEAK_DETECTOR_CONCURRENCY`. Each defaults to 5. The choice of three separate
semaphores rather than one shared budget reflects the fact that the three
backends rate-limit independently and that the detector's QPS budget is a
fraction of the main LLM's.

The writer side is single-async-thread per model: `runner.run`
(runner.py:L362) opens one `db.AsyncWriter` per model and routes that model's
results through it. A single-model DB consequently has one writer and many
readers, which is safe under SQLite's WAL mode.

### 3.2 Resume semantics

Resume is judged independently per sample slot. The query the writer performs
is

```sql
SELECT question_id FROM run_results
 WHERE s{i}_created_at IS NOT NULL
   AND (s{i}_error IS NULL OR s{i}_error = 'skipped_training_cutoff');
```

so every slot whose `created_at` is set and whose `error` is either NULL or a
deliberate exclusion is treated as completed and removed from the task queue.
The state classification used downstream is:

| `error` value                       | Meaning                              | Retry on next resume?                              |
| ----------------------------------- | ------------------------------------ | -------------------------------------------------- |
| `NULL`                              | Completed normally                   | No                                                  |
| `'skipped_training_cutoff'`         | Excluded by §4.2                     | No                                                  |
| `'network'`, `'server_5xx'`         | Still failing after backoff          | Yes                                                 |
| `'rate_limit'`                      | Rate limit, backoff exhausted        | Yes                                                 |
| `'bad_request'`                     | `model_not_found`, etc.              | Yes (after configuration fix)                       |
| `'content_policy'`                  | Provider refusal                     | Optional; default retries once and overwrites.      |

Re-running with the same `run_id` resumes into the existing
`runs/{run_id}/db/<slug>.db`; a different `run_id` produces a fresh
`runs/{new_run_id}/`. The overwrite primitive is `INSERT ... ON
CONFLICT(question_id) DO UPDATE SET s{i}_* = excluded.s{i}_*`, and
`user_prompt` is preserved with `COALESCE` so the first sample's value wins.
Pinned by `test_runner_resume.py`.

---

## 4. The information boundary

The framework organises leakage control around three controlled information
channels and one documented residual surface. The codebase implements each
channel at one specific layer of the pipeline; this section walks through
them in the order in which a sample encounters them.

### 4.1 Channels and residual

| Channel             | Object                                | Layer                       | Defence                                                                   |
| ------------------- | ------------------------------------- | --------------------------- | ------------------------------------------------------------------------- |
| 1. Parametric       | Knowledge that pre-existed training   | Sample admission            | $`\kappa_M`$ admissibility filter (§4.2)                                     |
| 2. Tool-mediated    | Tavily request payload                | Tool layer                  | $`\delta`$-offset injection on `end_date` (§4.3)                            |
| 3. Retrieval-content| Tavily response body                  | Stage-2 audit before LLM    | Independent detector $`H_{\mathrm{aux}}`$ with whitelist + fail-closed (§4.4)|
| 4. Provider-side    | Built-in browsing or augmentation     | Wire-protocol assertion     | `:online` ban, `plugins` ban, single-tool whitelist (§4.5)                |
| Residual A          | Time clues inside the question text   | None                        | Inherent to the data; accepted as evaluation bias (§4.6)                  |
| Residual B          | Knowledge backflow after training     | None                        | Accepted as evaluation bias (§4.6)                                        |

### 4.2 Channel 1: parametric knowledge

A question whose resolution time precedes the model's training cutoff is
likely already in the training corpus; the model "remembers" the answer
rather than forecasting it. Such samples cannot reflect native forecasting
capability and are removed from the model's evaluable subset
$`\mathcal{D}^{\mathrm{pred}}_M`$.

Per-model $`\kappa_M`$ is declared in `.env` via `MODEL_TRAINING_CUTOFFS`, a CSV
of `<slug>=YYYY-MM-DD` pairs parsed by `config._parse_cutoffs`
(config.py:L479). During task-plan generation, `runner.build_task_plan`
filters every `(question, model)` pair:

```python
cutoff = MODEL_TRAINING_CUTOFFS.get(real_model)   # None means not declared, no filtering
if cutoff is not None and q.end_time <= cutoff:
    # Day-rounded equivalent of χ_i < κ_M for δ = -1 day:
    # writes one row per sample_idx with error="skipped_training_cutoff"
    enqueue_skipped_cutoff_rows(q, model)
```

Filtered `(question, model, sample_idx)` rows still land in `run_results` with
`error="skipped_training_cutoff"`, `parse_ok=0`, `correct=NULL`, and all
numeric fields zero. This makes "how many questions were filtered per model"
auditable directly from the DB and feeds the per-model exclusion-count column
in reports. Resume never reattempts these rows.

`test_training_cutoff.py` pins the contract in three parts: every slot for a
question with `q.end_time <= cutoff` writes `skipped_training_cutoff`; models
without a declared cutoff are not filtered; and resume takes precedence over
cutoff so an already-completed row is not replaced by an exclusion row.

### 4.3 Channel 2: tool-mediated knowledge

The schema of `web_search` exposed to the LLM declares only a `query`
parameter (tools.py:L7–L24):

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

When Tavily is actually invoked, the cutoff $`\chi_i = \tau_i + \delta`$ is
hard-coded by the tool implementation (react.py:L182, search.py:L133–L162):

```python
end_date = (date.fromisoformat(q.end_time)
            + timedelta(days=settings.TAVILY_END_DATE_OFFSET_DAYS)).isoformat()
result = await search.tavily_search(query=args["query"], end_date=end_date, settings=cfg)
```

The default $`\delta = -1`$ day. With `end_time` at `YYYY-MM-DD` granularity,
this offset excludes same-day information:

```text
question.end_time (τ_i) = 2026-01-18
→ Tavily end_date (χ_i) = 2026-01-17
```

Both $`\delta = 0`$ (lenient) and $`\delta \in \{-2, -3\}`$ (more conservative)
are valid configurations, but reports default to comparisons under $`\delta =
-1`$, so numbers obtained under different offsets are not directly comparable.

`test_search.py` pins that the LLM-visible `web_search` schema does not
contain `end_date` and that `tavily_search` injects the right `end_date` when
called; `test_react.py` pins the in-loop wiring end-to-end.

### 4.4 Channel 3: retrieval-content audit

Tool-level filtering constrains the request side, but returned snippets,
cached pages, and aggregate summaries can still carry post-$`\chi_i`$ content.
The Stage-2 detector in `leak_filter.py` audits each Tavily result before it
enters the main LLM context.

The detector runs against an **independent client**: `_detector_client`
(leak_filter.py:L108–L133) is a separate `AsyncOpenAI` instance distinct from
the main `llm._client`, configured by `LEAK_DETECTOR_*` environment variables.
If `LEAK_DETECTOR_BASE_URL` is empty it falls back to `LLM_BASE_URL`. The cut
point is the end of the HTTP-200 path in `search.tavily_search`, before the
`return`; verdicts are then applied by `leak_filter.filter_search_result`
(leak_filter.py:L348), which walks the result list and drops items per
verdict.

The detector prompt has a **closed input whitelist** of six fields, which is
load-bearing for the contract (leak_filter.py:L212–L227):

```text
title           — result.title
url             — result.url
published_date  — result.published_date or "(unknown)"
content         — result.content
raw_content     — result.raw_content or "(empty)"
cutoff_date     — the χ_i passed in by the caller
```

The question text, options, and gold answer are never passed. Framing the
detector as an "answer auditor" would create second-order leakage, since the
detector might rationalise that fabricated evidence is consistent with the
answer it knows and let it through.

The output schema is strict JSON with two fields:

```json
{"verdict": "keep" | "drop", "reason": "<sentence>"}
```

A `drop` verdict removes the entire result, including title, URL, content,
and raw_content, before the main LLM sees anything; the audit fields are
retained for post-hoc review.

The detector is **fail-closed by default**. The retry sequence is `max_attempts
= LEAK_DETECTOR_RETRY_MAX + 1`, defaulting to three retries with backoff
`[2, 5, 15]` seconds. AUTH errors at status 401 or 403 are caught locally,
converted to `failed:auth`, and never propagated; the item is dropped
immediately (leak_filter.py:L281–L288). Other retryables exhaust the sequence,
after which the verdict becomes `failed:<kind>` and `LEAK_DETECTOR_FAIL_ACTION`
takes over: the default `drop` removes the item, while `keep` lets it through.
The default biases the residual towards "drop on uncertainty", since detector
hiccups are uncorrelated with item content.

Each call appends an audit dictionary to `s{i}_search_calls[*].audit`
(leak_filter.py:L380–L387):

| Field                  | Type         | Meaning                                               |
| ---------------------- | ------------ | ----------------------------------------------------- |
| `n_results_raw`        | int          | Count before filtering                                 |
| `n_results_kept`       | int          | Count after filtering                                  |
| `published_dates_raw`  | list[str]    | Original publish dates of all items (audit invariant)  |
| `detector_verdicts`    | list[str]    | Per-item verdict; values: `keep` / `drop` / `failed:*` |
| `detector_latency_ms`  | int          | Wall-clock detector latency                            |
| `detector_error_kind`  | str \| null  | Dominant failure kind across the batch                 |

Three keys describe the detector run inside `run_meta.config_snapshot`, plus a
top-level slot:

```text
leak_detector_enabled         — bool
leak_detector_model           — str
leak_detector_prompt_hash     — sha256[:16] of LEAK_DETECTOR_PROMPT_TEMPLATE
leak_detector_prompt_version  — human-readable label, default "v1"
```

When `ENABLE_SEARCH_LEAK_FILTER=False` the detector path is byte-level rolled
back and behaviour matches v5.1 without the detector.

`test_leak_filter.py` (550 LOC) pins five contracts: the detector input
whitelist, fail-closed on retry exhaustion, AUTH-immediate-fail-closed without
propagation, presence of every audit field on `search_calls`, and
byte-equivalence of the disabled path to v5.1.

### 4.5 Channel 4: provider-side residual

A model service may attach a built-in browsing tool that bypasses the Tavily
layer entirely. Two defences sit in the wire path.

The first is the **slug ban**. Slugs ending in `:online`, which is
OpenRouter's online-augmented variant naming, are rejected at startup by
`Settings` validation (config.py:L599–L614) and re-asserted on the wire by
`llm._assert_no_browsing` (llm.py:L74–L98). Pinned by `test_llm_no_browsing.py`.

The second is the **`plugins` field ban**. `extra_body.plugins` is rejected on
the wire (llm.py:L97), and only the single `[WEB_SEARCH_SCHEMA]` tool is
permitted in `tools=[...]`.

A provider that forcibly attaches an unhideable browsing capability should be
marked "unsuitable for strict evaluation" in the README and reports, since the
framework cannot defend against capabilities the API does not expose.

### 4.6 Threat model and residual surface

Of the six leakage sources the framework recognises, four are controllable via
the channels above. The remaining two are inherent to the data and to
post-training knowledge backflow, and are accepted as evaluation bias:

| Leakage source                               | Controllable? | Mitigation                                                                     |
| -------------------------------------------- | ------------- | ------------------------------------------------------------------------------ |
| Tool search content (Tavily returned text)   | Yes           | $`\delta`$-offset injection at the tool layer (§4.3)                              |
| Provider-native browsing or web tool         | Yes           | `:online` ban, `plugins` ban, single-tool whitelist (§4.5)                     |
| Page bodies that mention post-$`\chi_i`$ events | Partial      | Stage-2 detector with whitelisted inputs and fail-closed default (§4.4)        |
| Model parametric memory                      | Partial       | $`\kappa_M`$ admissibility filter excludes $`\chi_i < \kappa_M`$ samples (§4.2)     |
| Time clues in the question text              | No            | Inherent to the data; accepted as evaluation bias                                |
| External knowledge backflow after training   | No            | Accepted as evaluation bias                                                      |

The framing is explicit: this is an auditable, reproducible, and comparable
framework, not a proof that every leak is closed.

### 4.7 The renderer $`R`$

The source DB stores only raw material (`event`, `options`, `question_type`,
`end_time`). When the system spawns a sample,
`prompts.render_user_prompt` (prompts.py:L447) reads templates from
`dataset_metadata.features_json.prompt_reconstruction` and assembles the user
message per `question_type`:

```text
{agent_role} The event to be predicted: "{event} (resolved around {end_time} (GMT+8)).{outcomes_block}"

IMPORTANT: Your final answer MUST end with this exact format:
{output_format}
{guidance}
```

Per-slot rendering rules:

| Slot              | Rule                                                                                                                                                |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `agent_role`      | Constant `"You are an agent that can predict future events."`, inserted as-is.                                                                       |
| `event`           | `<SOURCE_TABLE>.event` original text.                                                                                                                |
| `end_time`        | `<SOURCE_TABLE>.end_time` original text, formatted `YYYY-MM-DD`.                                                                                      |
| `outcomes_block`  | Empty for `yes_no` and `binary_named`, since the options live inside `output_format`. For `multiple_choice` it is a newline plus `A. <opt[0]>\nB. <opt[1]>\n...` formatted via §4.8. |
| `output_format`   | One of three templates per `question_type`. The `binary_named` template contains `<options[0]>` and `<options[1]>` placeholders that must be replaced with the actual entity names. |
| `guidance`        | Constant `"Do not use any other format. Do not refuse to make a prediction. ..."`, inserted as-is.                                                    |

The output-format shapes that the parser ultimately matches against:

* `yes_no` requires `\boxed{Yes}` or `\boxed{No}`.
* `binary_named` after rendering looks like `\boxed{Golden Knights} or \boxed{Kings}`.
* `multiple_choice` requires `\boxed{A}` or `\boxed{B, C}`, with an example
  attached.

The reflection, budget-awareness, and belief protocol additions are appended
at runtime when their switches are on (§5.4); they do not enter
`dataset_metadata`, so `prompt_templates_hash` is invariant under these
toggles. The fully rendered user message lands in each sample's
`s{i}_user_prompt` field. Protocol-text fingerprints are persisted
independently in `run_meta.reflection_protocol_hash` and
`run_meta.belief_protocol_hash` (§6.3).

### 4.8 The letter map $`\phi`$

The DB uniformly uses **letters** as the canonical answer; the LLM's output
form varies by `question_type`:

| question_type      | LLM output (inside `\boxed{}`)                                       | Parser normalisation target ($`\phi`$)                                |
| ------------------ | -------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `yes_no`           | `Yes` / `No` (case-insensitive)                                       | `frozenset({"A"})` / `frozenset({"B"})`, with Yes=A, No=B            |
| `binary_named`     | one of the entries in `options` (trim + case-insensitive exact match) | look up the index in `options`, then map to letter and frozenset     |
| `multiple_choice`  | one or more letters, comma- or space-separated (`A`, `B, C`, `B,C`)   | split into tokens, then frozenset                                    |

The letter-to-index rule (parser.py:L420–L429) supports up to 35 options:

```text
index = ord(letter) - ord('A')
A=0, B=1, ..., Z=25
[ =26, \ =27, ] =28, ^ =29, _ =30, ` =31, a =32, b =33, c =34, ...
```

> ⚠️ **Compatibility warning.** When a question carries more than 26 options,
> the encoding lands on non-letter symbols such as `[`, `\`, `]`, `^`, `_`,
> `` ` ``, `a`, `b`, `c`. These ASCII-continuation labels are unfriendly to
> LLMs because backticks and underscores get swallowed by markdown and code
> blocks, and lowercase `a` and uppercase `A` are easily confused in inline
> rendering. The scheme is retained because it preserves a one-to-one mapping
> with the source-data letter encoding for letter-set scoring. Three mandatory
> defences apply: `prompts.render_user_prompt` quotes and escapes labels when
> generating the `outcomes_block` for >26 options; `parser.parse_answer` has a
> round-trip unit test (label → letter → label) for `multiple_choice` with >26
> options; and logs and reports record letters and corresponding labels in
> parallel for manual review.

The ground-truth reverse lookup, used for display or logging, is:

```python
opts    = json.loads(row["options"])
letters = [t.strip() for t in row["answer"].split(",")]
labels  = [opts[ord(L) - ord('A')] for L in letters]
```

---

## 5. The forecasting system $`F_M`$

The ReAct loop in `react.run_react` (react.py:L248–L632) is the heart of
$`F_M`$. It interleaves LLM turns with Tavily calls under a deterministic
priority chain that governs how the harness intervenes as the budget
approaches its limits.

### 5.1 Loop skeleton

The skeleton below preserves all v5.1 wiring while staying short enough to
read in one screen. Inline comments mark the four contracts the loop must
honour. Numbered references in the comments point to the full Python at
`react.py`.

```python
async def run_react(q: Question, model: str, sample_idx: int, settings: Settings):
    # ① χ_i is invisible to the LLM; it is computed here and threaded through Tavily.
    end_date = (date.fromisoformat(q.end_time)
                + timedelta(days=settings.TAVILY_END_DATE_OFFSET_DAYS)).isoformat()

    # ② m_0 = R(q^in): one user message with all enabled protocols appended.
    user_prompt = prompts.render_user_prompt(q, settings.PROMPT_TEMPLATES,
                  budget_awareness=BUDGET_AWARENESS_TEXT_OR_NONE,
                  reflection_protocol=REFLECTION_TEXT_OR_NONE,
                  belief_protocol=BELIEF_TEXT_OR_NONE)
    messages = [{"role": "user", "content": user_prompt}]

    for step in range(settings.REACT_MAX_STEPS):
        # ③ Pre-step injection (priority chain in §5.2): at most one of four fires.
        injection = pick_injection(step, search_calls, pending_continuation, settings)
        if injection is not None:
            messages.append({"role": "user", "content": injection})

        # Tool-schema decision: tools=[] when last-step cutoff or budget-dropped.
        tools = [] if force_final_hard_cutoff or budget_dropped else [WEB_SEARCH_SCHEMA]

        resp = await llm.chat(model=model, messages=messages, tools=tools, ...)
        msg = resp.choices[0].message
        messages.append(msg.model_dump())

        if settings.BELIEF_PROTOCOL:
            beliefs_per_step.append(parser.parse_belief(msg.content or "", q))

        if not msg.tool_calls:
            # No tool call: maybe nudge (soft floor), or break on \boxed{}, else continuation.
            if soft_floor_unmet_and_have_nudges():
                inject_nudge(); continue
            if "\\boxed{" not in (msg.content or ""):
                pending_continuation = True; continue
            final_raw = msg.content or ""
            break

        # Tool calls: validate, then dispatch one at a time.
        for tc in msg.tool_calls:
            err = _validate_tool_call(tc, settings, searches_done=len(search_calls))
            if err is not None:
                messages.append(prompts.tool_error_message(tc, err))
                continue
            # ④ χ_i is injected here, not by the LLM. The detector audits the result.
            result = await search.tavily_search(query=extract_query(tc), end_date=end_date,
                                                settings=settings)
            search_calls.append(result.to_search_call_record())
            messages.append(prompts.tool_result_message(tc, result.to_llm_payload()))

    # v5.1 D1 backstop: mop up an empty final_raw with a tools=[] retry.
    if final_raw == "" and settings.REACT_FINAL_ANSWER_RETRY:
        messages.append({"role": "user",
                         "content": "Time to commit. Output your final \\boxed{...} now."})
        resp = await llm.chat(model=model, messages=messages, tools=[], ...)
        final_raw = resp.choices[0].message.content or ""
        final_answer_retry_used = 1

    parsed = parser.parse_answer(final_raw, q)        # frozenset[str] | None
    correct = parser.is_correct(parsed, parser.parse_gt(q.answer))
    return SampleResult(...)
```

The four contracts are: $`\chi_i`$ stays invisible to the LLM (①, ④); the
user message is exactly the renderer output plus appended protocols (②); only
one harness injection fires per step, by priority (③); and the post-loop
backstop produces a final answer when the loop exits empty.

### 5.2 The harness-resilience priority chain

When the loop is near its budget limit, the harness injects user-side
guidance to push the model towards committing an answer. Four mechanisms
participate, and they fire in a strict priority order so their behaviour is
deterministic and test-pinned. The implementation lives at react.py:L266 and
uses the comment "Priority is (1) > (2) > (3) > (4)" verbatim.

| Priority | Trigger                                                              | Effect on `tools`     | Injection builder (`prompts.py`)                  |
| -------- | -------------------------------------------------------------------- | --------------------- | ------------------------------------------------- |
| 1        | Last step inside `LOOKAHEAD` window: `T - step == 1`                  | `tools=[]`            | `build_last_step_force_finalisation` (L294)        |
| 2        | Penultimate window: `1 < T - step <= LOOKAHEAD`                       | `tools=[WEB_SEARCH]`  | `build_penultimate_step_warning` (L254)            |
| 3        | Search budget hit: `len(search_calls) >= C` (fires once per run)      | `tools=[]`            | `build_search_budget_exhausted_commit` (L325)      |
| 4        | Previous turn produced no `\boxed{}` (continuation flag)              | `tools=[WEB_SEARCH]`  | `build_continuation_after_unboxed_content` (L354)  |

All four share the status-header builder `_build_status_header`
(prompts.py:L128–L164), which prepends a uniform line of the form
`[Harness status] step k/N (R remaining) · web_search s/C used (M left).`

The seven knobs that govern this chain are summarised below. The `LOOKAHEAD`
parameter is clamped to $`[1, T]`$ at startup (config.py:L696–L707).

| Knob                                  | Default | Scope                  | Effect                                                                                       |
| ------------------------------------- | ------- | ---------------------- | -------------------------------------------------------------------------------------------- |
| `REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT` | True    | In-loop graded transition | Soft warning in the penultimate window, hard `tools=[]` cutoff on the last step.              |
| `REACT_FORCE_FINAL_ANSWER_LOOKAHEAD`  | 2       | Soft window              | Steps before the limit at which intervention begins.                                          |
| `REACT_BUDGET_AWARENESS_PROTOCOL`     | True    | Prompt layer             | Appends `T` and `C` to the user prompt so the model can plan holistically.                    |
| `REACT_BUDGET_EXCEEDED_DROP_TOOLS`    | True    | In-loop budget gate      | Once cumulative `web_search >= C`, drops tools for all subsequent rounds.                     |
| `REACT_FINAL_ANSWER_RETRY`            | False   | Post-loop backstop       | When loop ends with empty `final_raw`, calls LLM once more with `tools=[]` to force `\boxed`. |
| `REACT_MIN_SEARCH_CALLS`              | 0       | Soft floor               | If the model tries to commit before reaching `MIN`, injects a nudge.                          |
| `REACT_MAX_NUDGES`                    | 2       | Soft floor cap           | Per-sample nudge budget.                                                                       |

### 5.3 Per-step belief processing

When `BELIEF_PROTOCOL=True`, every assistant turn (including the post-loop
final-answer retry) is parsed by `parser.parse_belief` (parser.py:L117–L213).
Per-step results land in `beliefs_per_step` and are then aggregated into
three persisted fields:

* `belief_final` is the JSON of last-step probabilities when the last-step
  belief parses; otherwise NULL.
* `belief_trace` is a JSON array of every step's belief summary, with a `null`
  entry for steps whose belief did not parse.
* `belief_parse_ok` is `1` iff the last-step belief parsed; it is independent
  of `parse_ok`.

The belief JSON schema is strict (prompts.py:L66–L105):

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

The `probabilities` keys must exactly match the expected letter set
(parser.py:L150). Single-choice answers must sum to $`1.0 \pm 10^{-3}`$
(parser.py:L167); multi-select answers leave each entry independently in
$`[0, 1]`$. Confidence must be one of `low`, `medium`, `high`
(parser.py:L173). A failed parse does not affect `parse_ok`, since the
`\boxed{}` path is the sole correctness signal.

### 5.4 The reflection protocol

`prompts.REFLECTION_PROTOCOL` (prompts.py:L31–L53) is a six-step reasoning
scaffold appended at runtime to the user message:

1. **Decompose** into sub-questions whose joint answer settles the prediction.
2. **Plan distinct angles**: at least three different investigation angles
   before any `web_search`.
3. **Search iteratively, reflect after every result**: paraphrase, tag
   relevance, identify contradictions, and pick the next query to fill the
   largest gap.
4. **Cross-validate** with at least two independent sources before committing.
5. **Stress-test the opposite** by articulating the strongest case for the
   opposite outcome.
6. **Calibrate, then commit** by stating confidence, failure mode, and
   decisive evidence; only then `\boxed{...}`.

The full text is hashed to `reflection_protocol_hash` (sha256[:16]) and stored
verbatim in `run_meta.reflection_protocol_text`. This text is not included in
`prompt_templates_hash`, so toggling reflection on or off keeps the template
hash invariant (§6.3).

---

## 6. Persistence

Each combination of run and model corresponds to one independent SQLite file
under `runs/{run_id}/db/<model_slug>.db`. Every file self-contains a copy of
`questions` and `prompt_templates`, so a single file can be replayed without
touching the source. Aggregations and statistics are not persisted; the
`forecast_eval.analysis` package writes them post-hoc into `analysis/`.

### 6.1 Schema (current = v5)

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
    filters_snapshot          TEXT NOT NULL,
    source_db_hash            TEXT NOT NULL,
    metadata_hash             TEXT NOT NULL,
    prompt_templates_hash     TEXT NOT NULL,
    reflection_protocol_text  TEXT,            -- v3+
    reflection_protocol_hash  TEXT,            -- v3+
    belief_protocol_text      TEXT,            -- v4+
    belief_protocol_hash      TEXT,            -- v4+
    training_cutoff           TEXT,            -- κ_M (YYYY-MM-DD)
    started_at                TEXT NOT NULL,
    finished_at               TEXT
);

-- ④ wide table: one row per question, one s{i}_* column group per sample.
-- 24 fields × SAMPLING_N columns generated dynamically by db.init_schema.
CREATE TABLE run_results (
    question_id TEXT PRIMARY KEY,
    user_prompt TEXT,                          -- COALESCE; first sample wins

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
    s0_step_metrics         TEXT,
    s0_response_id          TEXT,
    s0_system_fingerprint   TEXT,
    s0_service_tier         TEXT,
    -- v4 belief (3 columns)
    s0_belief_final         TEXT,
    s0_belief_trace         TEXT,
    s0_belief_parse_ok      INTEGER,
    -- v5 harness-resilience (1 column)
    s0_final_answer_retry_used INTEGER,

    -- ...same s1_* / s2_* / ... groups...

    FOREIGN KEY (question_id) REFERENCES questions(id)
);
CREATE INDEX idx_run_results_question ON run_results(question_id);
```

Schema migrations (db.py:L222–L345) are performed via `ALTER TABLE … ADD
COLUMN`, which SQLite executes as metadata-only and therefore O(1):

| Version | Change                                                              | Migration function          |
| ------- | ------------------------------------------------------------------- | --------------------------- |
| v2      | Base 14 per-sample columns; bare `run_meta`                          | `_init_v2_schema`            |
| v2 → v3 | Adds 6 per-sample observability fields and 2 reflection columns      | `_migrate_v2_to_v3` (L222)   |
| v3 → v4 | Adds 3 per-sample belief columns and 2 belief columns in `run_meta`   | `_migrate_v3_to_v4` (L269)   |
| v4 → v5 | Adds 1 per-sample column `final_answer_retry_used`                    | `_migrate_v4_to_v5` (L312)   |

When `Settings.BELIEF_PROTOCOL=False`, all belief columns write NULL and the
analysis pipeline early-exits the probabilistic family. The first time an old
DB is opened on the resume path, it is auto-migrated.

The connection-init PRAGMA, executed on every `sqlite3.connect`:

```sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;
```

### 6.2 Per-sample field write conventions

| Field                             | Source                                                                                                                                                          |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `s{i}_final_answer_letters`       | The `frozenset[str]` from `parser.parse_answer(final_raw, q)`, written as `json.dumps(sorted(...))`.                                                              |
| `s{i}_final_answer_raw`           | The full `content` text of the LLM's last assistant message.                                                                                                      |
| `s{i}_correct`                    | `frozenset == frozenset` cast to `int`; NULL when parse fails or the sample is not in $`\mathcal{S}`$.                                                              |
| `s{i}_parse_ok`                   | `final_answer_letters is not None`, equivalent to the validity flag $`v_{i,M}`$.                                                                                       |
| `user_prompt`                     | Return of `prompts.render_user_prompt(q, templates, …)`, rendered once per question and retained via COALESCE.                                                      |
| `s{i}_messages_trace`             | The full `messages` list as JSON, or NULL when `WRITE_MESSAGES_TRACE=False`.                                                                                       |
| `s{i}_search_calls`               | Per-call metadata: `query`, `end_date`, `n_results`, `published_dates`. With the leak filter on, also `n_results_raw / n_results_kept / detector_verdicts / detector_latency_ms / detector_error_kind`. |
| `s{i}_error`                      | Error classification after retries; NULL on normal completion, including refusal or parse failure.                                                                  |
| `s{i}_created_at`                 | UTC ISO-8601 at write time; the unique signal that "this slot has been filled".                                                                                     |
| `s{i}_finish_reason`              | Last round's `ChatCompletion.choices[0].finish_reason`; NULL for error rows.                                                                                          |
| `s{i}_nudges_used`                | Count of "strict floor not met → reminder injected" within this sample, capped by `REACT_MAX_NUDGES`.                                                                |
| `s{i}_step_metrics`               | JSON array of per-round snapshots: `step / prompt / completion / reasoning / latency_ms / finish_reason / n_tool_calls`.                                            |
| `s{i}_response_id`                | Last round's `ChatCompletion.id`.                                                                                                                                  |
| `s{i}_system_fingerprint`         | Last round's `ChatCompletion.system_fingerprint` when the provider supplies it; useful for detecting provider-side model-routing changes.                            |
| `s{i}_service_tier`               | Last round's `ChatCompletion.service_tier`.                                                                                                                         |
| `s{i}_belief_final`               | v4. The JSON-serialised `Belief.probabilities` from the final step; NULL when parsing fails or `BELIEF_PROTOCOL=False`.                                              |
| `s{i}_belief_trace`               | v4. JSON array of belief summaries for every loop step.                                                                                                              |
| `s{i}_belief_parse_ok`            | v4. Whether the last-step belief parsed legally (0 or 1); independent of `parse_ok`.                                                                                  |
| `s{i}_final_answer_retry_used`    | v5. 0 or 1, set when `REACT_FINAL_ANSWER_RETRY` mopped up an empty `final_raw` (§5.1).                                                                                |

### 6.3 Three independent protocol fingerprints

Three independent SHA-256 prefixes describe what the LLM actually saw in this
run, and they decouple ablation axes that would otherwise collide on a single
hash (design rationale in DESIGN.md §5.6).

* `prompt_templates_hash` covers the renderer $`R`$, hashed over the eight
  template keys of §2.3.
* `reflection_protocol_hash` covers the search-behaviour prior, hashed over
  `prompts.REFLECTION_PROTOCOL` text only. Toggling reflection or editing the
  text changes the hash.
* `belief_protocol_hash` covers the probabilistic-family populator, hashed
  over `prompts.BELIEF_PROTOCOL` text only.

All three live both in `run_meta` and at the top level of `manifest.json`
(evaluation.py:L171–L178), so "grep the protocol fingerprint without opening
the DB" covers every protocol axis.

### 6.4 Resume queries

The resume contract is defined in §3.2; the implementation iterates over each
sample slot:

```sql
-- For i ∈ 0..SAMPLING_N-1:
SELECT question_id FROM run_results
 WHERE s{i}_created_at IS NOT NULL
   AND (s{i}_error IS NULL OR s{i}_error = 'skipped_training_cutoff');
```

Results merge into `set[(question_id, sample_idx)]` and are removed from the
task queue. Since each model's own DB contains exactly one run, `run_id` does
not enter the filter; the single row in `run_meta` decides it.

### 6.5 Concurrent-write strategy

Every DB connection executes the four PRAGMAs from §6.1 at startup. There is
**one async writer task per model**: `runner.run` opens one
`db.AsyncWriter` per model DB (runner.py:L362), and every worker's result is
enqueued to that model's writer. The writer flushes either every
`DB_COMMIT_BATCH` entries or every second
(`AsyncWriter.FLUSH_INTERVAL_S = 1.0`); transactions are short, and SQLite
writes go through `await asyncio.to_thread(...)` so the event loop never
blocks. A single-model DB consequently has one writer and many readers, which
is safe under WAL.

`asyncio.Queue` is not cross-thread; for cross-thread consumption a
`queue.Queue` or `janus.Queue` would be needed. The current design stays
fully async on a single thread.

---

## 7. Configuration

A condensed view of the most load-bearing knobs follows; the full annotated
block lives in `.env.example`.

### 7.1 The `.env` contract

```ini
# -------- LLM Endpoint (OpenAI-compatible) --------
LLM_API_KEY=REPLACE_ME
LLM_BASE_URL=https://openrouter.ai/api/v1
MODELS=openai/gpt-5,anthropic/claude-sonnet-4.5,google/gemini-2.5-pro,deepseek/deepseek-r1
# κ_M per model: declare for every evaluated model
MODEL_TRAINING_CUTOFFS=openai/gpt-5=2024-10-01,anthropic/claude-sonnet-4.5=2025-03-01,...

# LLM call parameters
LLM_MAX_TOKENS=12000
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
TAVILY_API_KEY=tvly-REPLACE_ME           # single value or CSV (multi-key pool)
TAVILY_KEY_COOLDOWN_S=60                 # 429 cooldown for a single key
TAVILY_MAX_RESULTS=5                     # R_tav axis; multi-value triggers grid
TAVILY_SEARCH_DEPTH=basic                # basic (1 credit) | advanced (2 credits)
TAVILY_INCLUDE_RAW_CONTENT=markdown      # false | markdown (default) | text
TAVILY_RAW_CONTENT_MAX_CHARS=8000        # per-result raw_content truncation
TAVILY_INCLUDE_ANSWER=false              # off (avoids second-LLM contamination)
TAVILY_END_DATE_OFFSET_DAYS=-1           # δ; project default -1 (strict)
SEARCH_MAX_CONCURRENCY=5
SEARCH_RETRY_MAX=3
SEARCH_BACKOFF_S=2,5,15

# -------- ReAct Loop --------
REACT_MAX_STEPS=12                       # T (max ReAct steps per sample)
REACT_MAX_SEARCH_CALLS=8                 # C axis; multi-value triggers grid
REACT_REFLECTION_PROTOCOL=true
REACT_BUDGET_AWARENESS_PROTOCOL=true
REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=true
REACT_FORCE_FINAL_ANSWER_LOOKAHEAD=2
REACT_MIN_SEARCH_CALLS=0                 # soft floor; opt-in
REACT_MAX_NUDGES=2

# v5.1 harness-resilience
REACT_FINAL_ANSWER_RETRY=false           # mop up empty final_raw with a tools=[] retry
REACT_BUDGET_EXCEEDED_DROP_TOOLS=true    # drop tool schema once C is hit

# -------- Search Leak Filter (Stage-2 detector) --------
ENABLE_SEARCH_LEAK_FILTER=true
LEAK_DETECTOR_API_KEY=REPLACE_ME
LEAK_DETECTOR_BASE_URL=                  # empty → falls back to LLM_BASE_URL
LEAK_DETECTOR_MODEL=anthropic/claude-sonnet-4.6
LEAK_DETECTOR_TIMEOUT_S=60
LEAK_DETECTOR_TEMPERATURE=0.0
LEAK_DETECTOR_MAX_TOKENS=512
LEAK_DETECTOR_RETRY_MAX=3
LEAK_DETECTOR_BACKOFF_S=2,5,15
LEAK_DETECTOR_FAIL_ACTION=drop           # drop (fail-closed, default) | keep (A/B escape hatch)
LEAK_DETECTOR_CONCURRENCY=5
LEAK_DETECTOR_PROMPT_VERSION=v1

# -------- Composite score weights --------
COMPOSITE_WEIGHTS_QTYPE=yes_no=0.15,binary_named=0.15,multiple_choice=0.70
COMPOSITE_WEIGHTS_CTYPE=single=0.40,multi=0.60
COMPOSITE_WEIGHT_OVERRIDES_QTYPE=
COMPOSITE_WEIGHT_OVERRIDES_CTYPE=

# -------- Sampling --------
SAMPLING_N=5

# -------- Run / Resume --------
RUN_ID=
RESUME=true

# -------- Database --------
SOURCE_DB=./forecast_eval_set_example.db
SOURCE_TABLE=forecast_eval_set_example
RUNS_ROOT=./runs
DB_COMMIT_BATCH=10
WRITE_MESSAGES_TRACE=true

# -------- Logging --------
LOG_LEVEL=INFO
LOG_DIR=./logs

# -------- Belief protocol (v4 probabilistic family, off by default) --------
BELIEF_PROTOCOL=false

# -------- Grid search anchors (optional; only when R / C are multi-valued) --------
GRID_DEFAULT_R=
GRID_DEFAULT_C=
```

### 7.2 Startup validation (fail-fast)

Before any LLM or Tavily call, `Settings()` enforces the following checks
(config.py:L577–L851). A failing check raises `ValueError` and aborts the
run before a single API call goes out.

| Check                                                              | Where (line)        | Failure mode                                       |
| ------------------------------------------------------------------ | ------------------- | -------------------------------------------------- |
| `RUN_ID` matches `^\d{8}-\d{6}-[0-9a-f]{4}$` when non-empty        | L577–L584           | ValueError                                         |
| `SOURCE_TABLE` matches `^[A-Za-z_][A-Za-z0-9_]*$`                  | L586–L595           | ValueError, with SQL-injection rationale            |
| `MODELS` non-empty; no `:online`; no `::`                          | L599–L614           | ValueError                                         |
| `LLM_API_KEY` non-empty; no placeholder tokens                     | L617–L622           | ValueError                                         |
| `TAVILY_API_KEY` non-empty when `ENABLE_WEB_SEARCH=True`           | L623–L636           | ValueError per key                                 |
| `LLM_MAX_CONCURRENCY` ≥ 1; `SAMPLING_N` ≥ 1; `REACT_MAX_STEPS` ≥ 1 | L641–L646           | ValueError                                         |
| `REACT_MAX_SEARCH_CALLS` items > 0; `TAVILY_MAX_RESULTS` items > 0 | L455–L460           | ValueError per cell                                |
| `REACT_MIN_SEARCH_CALLS` ≤ min($`C`$)                                | L661–L671           | ValueError                                         |
| `REACT_FORCE_FINAL_ANSWER_LOOKAHEAD` ∈ $`[1, T]`$                    | L696–L707           | ValueError                                         |
| `GRID_DEFAULT_R` ∈ `TAVILY_MAX_RESULTS` when set                    | L711–L715           | ValueError                                         |
| `GRID_DEFAULT_C` ∈ `REACT_MAX_SEARCH_CALLS` when set                | L716–L720           | ValueError                                         |
| `LEAK_DETECTOR_API_KEY` and `_MODEL` non-empty when filter enabled  | L758–L777           | ValueError; no `:online` on detector slug          |
| `COMPOSITE_WEIGHTS_*` buckets in known set; weights ≥ 0; ≥1 > 0    | L781–L851           | ValueError                                         |
| `COMPOSITE_WEIGHT_OVERRIDES_*` metric names in allowlist            | L515–L535 + composite.py:L77–L127 | ValueError on typo                  |

`test_config.py` covers roughly 155 lines of boundary cases.

### 7.3 Secret redaction

Before writing `run_meta.config_snapshot`, `db.compute_redacted_config_snapshot`
redacts every sensitive field. The redaction format is the first 4 characters,
followed by length and `sha256[:12]`. `TAVILY_API_KEY` is `list[str]` and is
persisted as `[{prefix, sha256_12, length, provider}, ...]` so that "which
keys this run used" is auditable. Sensitive plaintext is never persisted.

---

## 8. Errors and observability

All exceptions are routed through `errors.py`, which classifies them into
seven tiers and selects a backoff sequence per tier.

### 8.1 Error tiers

| Tier                              | Identification                                                                  | Handling                                                                              |
| --------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Network / Timeout**             | `httpx.ConnectError`, `httpx.ReadTimeout`, `asyncio.TimeoutError`, `RemoteProtocolError`, `WriteError`, `WriteTimeout`, `PoolTimeout` | Backoff per `LLM_BACKOFF_NETWORK_S`; on exhaustion → `error="network"`                |
| **Rate Limit (429)**              | HTTP 429                                                                        | Honour `Retry-After` header first; otherwise `LLM_BACKOFF_RATE_LIMIT_S`                 |
| **Server 5xx**                    | HTTP 500/502/503/504                                                            | Backoff per `LLM_BACKOFF_SERVER_5XX_S`; on exhaustion → `error="server_5xx"`            |
| **Auth (401/403)**                | HTTP 401/403                                                                    | Fail immediately; abort the entire run via `AuthError`                                  |
| **Bad Request (400)**             | HTTP 400 + `model_not_found` or `invalid_request`                               | Skip immediately, `error="bad_request"`                                                 |
| **Content Policy**                | HTTP 400 with body matching `errors.CONTENT_POLICY_NEEDLES`                     | No retry; `error="content_policy"`, `parse_ok=0`, `correct=NULL`                       |
| **LLM soft refusal**              | Normal return but `\boxed{...}` not found or parsed `frozenset` empty           | Not an error; `parse_ok=0`, `correct=NULL`                                              |
| **Exceeds `REACT_MAX_STEPS`**     | ReAct loop exhausted without a final answer                                      | Not an error; `parse_ok=0`, `correct=NULL` unless `REACT_FINAL_ANSWER_RETRY` mops up    |
| **Tool-arguments JSON parse fail** | LLM `arguments` not legal JSON                                                  | Tell the LLM the error and continue the loop (non-fatal)                                |
| **Tavily error itself**           | Independent retry via `SEARCH_BACKOFF_S`; on exhaustion fed to LLM as `tool_result` | LLM may retry the query or abandon it                                                  |
| **Detector error (Stage 2)**      | Retry via `LEAK_DETECTOR_BACKOFF_S`; AUTH errors immediate-fail-closed         | On `LEAK_DETECTOR_FAIL_ACTION=drop` (default) → drop the item; `keep` → pass through    |
| **Training-data contamination filter** | Detected during task-plan: `q.end_time <= κ_M` (§4.2)                       | Does not invoke the LLM; writes `error="skipped_training_cutoff"` directly             |

Six boundaries deserve emphasis. (i) Auth errors stop the entire run because
continuing to burn budget on a wrong key is meaningless;
`runner._run_task_with_retry` re-raises `AuthError` (runner.py:L245), and the
outer loop cancels all tasks, flushes the writer, and exits. (ii) Content
policy is not retried, since re-sending the same question yields the same
result; reports tally how many rejections each model accumulated. (iii)
Refusal is not an error: a legal LLM response that fails to commit a boxed
answer is part of model capability and is counted in statistics, not in
`error`. (iv) Tavily failure degrades to a `tool_result` error, so the LLM
decides whether to retry the query or give up without interrupting the whole
sample. (v) Detector failure is fail-closed by default because detector
hiccups are uncorrelated with item content. (vi) `skipped_training_cutoff`
does not count toward error rate, since it is active data cleansing rather
than model failure.

The eight content-policy needles are (errors.py:L39–L48):

```python
CONTENT_POLICY_NEEDLES = (
    "content_policy", "content filter", "content_filter", "safety",
    "content_policy_violation", "data_inspection_failed",
    "inappropriate content", "sensitive",
)
```

The list covers OpenAI-style and Anthropic-style English bodies, plus
Aliyun DashScope's `data_inspection_failed` and `inappropriate content`.
`_body_matches` performs case-insensitive substring matching, so all needles
must be lowercase ASCII.

### 8.2 Error and parsing coupling rules

This matrix is the contract between `react.py`'s output and the analysis
denominators.

| State                                       | `parse_ok` | `correct` | Counted in $`\mathcal{S}`$? | Counted in $`\mathcal{D}^{\mathrm{eval}}`$? |
| ------------------------------------------- | ---------- | --------- | ------------------------- | ----------------------------------------- |
| Cutoff-excluded                              | 0          | NULL      | No                         | No (excluded)                              |
| Non-cutoff call error (network/5xx/policy)   | 0          | NULL      | No                         | Yes (denominator), No (numerator)          |
| Parse failure or soft refusal                | 0          | 0         | Yes                        | Yes                                       |
| Strict equality match                        | 1          | 1         | Yes                        | Yes                                       |
| Strict equality miss                         | 1          | 0         | Yes                        | Yes                                       |

### 8.3 Logging

Logging uses `loguru` with a stderr sink at `LOG_LEVEL` (default `INFO`) and a
DEBUG-level rotating file sink under `LOG_DIR`:

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

Progress prints one line per sample completion:

```text
12:03:44 | INFO    | [run=20260424-120344-a7k3] [5/1610] q=69566c13 qt=binary_named ct=single model=openai/gpt-5 sample=2/5 correct=True steps=4 tool_calls=3 latency=8421ms
```

The denominator `[5/1610]` equals
`len(questions_after_filter) × len(MODELS) × SAMPLING_N` minus completed
resume tasks. On error the line shifts to `ERROR` level and reads
`[x/xx] q=.. model=.. error=rate_limit retry_exhausted`.

---

## 9. Metrics ($`\Gamma`$)

Metrics are computed entirely by `forecast_eval.analysis` after the run
finishes and are never stored in the DB. Artefacts land in
`runs/{run_id}/analysis/`. Each definition below ties one mathematical object
to the function that computes it.

### 9.0 Reading guide

A reader who arrives with a specific question can navigate the metric stack
through the table below. The columns indicate which subsections to consult
first; the §X.Y references point to subsections of this section.

| Question                                                                 | Read first                | Then                                |
| ------------------------------------------------------------------------ | ------------------------- | ----------------------------------- |
| "Which model is most accurate overall?"                                   | §9.6 Composite Accuracy   | §9.5 FSS, §9.10 paired bootstrap    |
| "Which model is most cost-effective?"                                    | §9.7 Per-correct cost      | §9.6 Composite Accuracy             |
| "How robust is the ranking under repeated sampling?"                     | §9.4 Multi-trial consistency | §9.10 paired bootstrap, §9.3 pass@k |
| "How often does a model commit a parseable answer at all?"               | §9.1 Validity              | §9.7 Per-correct cost               |
| "Did the leakage barrier hold?"                                          | §4.4 Detector audit        | §4.6 residual surface               |
| "Where is the model spending its tokens and tool calls?"                  | §9.11 Behavioural diagnostics | §9.12 output artefacts          |

### 9.1 Validity ($`\mathcal{E}^{\mathrm{valid}}`$)

The validity flag $`v_{i,M} = \mathbb{1}[\Psi_i(o_{i,M}) \ne \bot]`$ records
whether the model's raw output yields a parseable letter set. The DB columns
derived from it are:

| Metric                          | Definition                                                                              | DB column source              |
| ------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------ |
| `parse_failure_rate`            | $`1 - \mathbb{E}[v_{i,M}]`$ over the scorable set $`\mathcal{S}`$                            | `s{i}_parse_ok = 0`            |
| `final_answer_retry_rate`       | Share of samples where the v5.1 backstop mopped up an empty `final_raw`                  | `s{i}_final_answer_retry_used = 1` |
| `error_rate`                    | Share of samples with non-cutoff `s{i}_error`                                            | `s{i}_error NOT IN (NULL, 'skipped_training_cutoff')` |
| `cutoff_skip_rate` per model    | `count(error='skipped_training_cutoff') / count(*)` per model                            | `s{i}_error = 'skipped_training_cutoff'` |
| `error_breakdown` (CSV)         | `Counter[error]` across all samples, including cutoff                                    | `s{i}_error`                   |
| `finish_reason_breakdown` (CSV) | `Counter[finish_reason]` over eligible samples; spot abnormal `length` or `content_filter` | `s{i}_finish_reason`            |

### 9.2 Item-level scoring ($`\mathcal{E}^{\mathrm{item}}`$)

A `(question_id, model)` has $`n`$ samples where $`n`$ is `SAMPLING_N`. The tally
excludes rows with `s{i}_error="skipped_training_cutoff"`, since those are
excluded questions rather than the model getting them wrong.

**Strict equality**: $`r_{i,M} = \mathbb{1}[\widehat{G}_{i,M} = G_i]`$
corresponds to `s{i}_correct` in the DB.

**Exam-style partial credit** is the project's headline per-sample score:

$$
\text{exam-score}(\hat S, G) = \begin{cases}
\dfrac{|\hat S \cap G|}{|G|}, & \hat S \setminus G = \varnothing \\\\
0, & \hat S \setminus G \ne \varnothing
\end{cases}
$$

Single-answer questions degenerate to strict $`0/1`$. The implementation lives
in `analysis.exam_score.exam_score` (exam_score.py:L62), with this decision
tree (exam_score.py:L78–L91):

```
is_cutoff           → None  (excluded)
error is not None   → None  (excluded)
parse_ok != 1       → 0.0   (parse failure counts as 0)
FP > 0              → 0.0   (any false positive vetoes)
otherwise           → |TP| / |G|
```

**Tversky similarity**, used for FSS:

$$
T(\hat S, G) = \frac{|\hat S \cap G|}{|\hat S \cap G| + \alpha\,|\hat S \setminus G| + \beta\,|G \setminus \hat S|}
$$

The project default $`(\alpha, \beta) = (2.0, 0.5)`$ makes FP penalty four times
the FN penalty. The implementation is `analysis.accuracy.tversky_score`
(accuracy.py:L286).

**Hamming score** is multi-only and symmetric in missing versus wrong:

$$
\text{hamming}(\hat S, G, \mathcal{O}) = 1 - \frac{1}{k}\sum_{\ell\in\mathcal{O}}|\mathbb{1}[\ell\in\hat S] - \mathbb{1}[\ell\in G]|
$$

Single-choice degenerates to $`0/1`$.

### 9.3 Question-level aggregation ($`\mathcal{E}^{\mathrm{question}}`$)

| Metric                          | Definition                                                                                            | Implementation                              |
| ------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `pass_at_1_avg` ($`\text{pass@1}`$) | Per-question intra-mean of strict hits, then equal-weight cross-question                              | `accuracy._aggregate` (accuracy.py:L124)     |
| `pass_any_at_n` ($`\text{pass-any@n}`$) | $`\mathbb{1}[\exists s: c_{q,s}=1]`$ averaged across questions; this is the standard `pass@k`        | `accuracy._aggregate` (L134)                 |
| `at_least_all_at_n` ($`\text{pass-all@n}`$) | $`\prod_s c_{q,s}`$ averaged; a repeated-consistency lower bound                                | `accuracy._aggregate` (L141)                 |
| `at_least_majority_at_n`        | $`\mathbb{1}[\sum_s c_{q,s} \ge \lceil n/2 \rceil]`$ averaged                                            | `accuracy._aggregate`                        |
| `majority_vote_accuracy`        | Counter-based letter-set vote, single winner, then strict equality vs $`G_q`$                            | `accuracy._aggregate` (L150–L164)             |
| `exam_score_at_n_avg`           | Two-step (intra-question mean → inter-question mean) over the scored index $`\mathcal{J}_q^{\mathrm{cnt}}`$ | `exam_score.exam_score_at_n_avg` (L94–L129) |
| `cohen_kappa`                   | $`(\text{acc} - p_e)/(1 - p_e)`$ with $`p_e = 1/k_q`$ for single or $`0.5`$ per-label for multi              | `accuracy.cohen_kappa` (L493–L532)           |
| `hamming_score`                 | Cross-question mean of per-question Hamming (multi only)                                                | `accuracy.hamming_score_per_question` (L535–L574) |

### 9.4 Multi-trial consistency ($`n \ge 2`$)

| Metric          | Definition                                                                                                              | Implementation                              |
| --------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `fleiss_kappa`  | $`(\bar{P} - \bar{P}_e)/(1 - \bar{P}_e)`$ on the $`K_q^{\mathrm{eff}}`$-trial vote matrix; stratified by $`k_q`$ for single, per-label for multi  | `consistency.fleiss_kappa` (L257–L297)       |
| `mean_entropy`  | Per-question mean Shannon entropy of the vote distribution; per-label binary mean for multi                              | `consistency.prediction_entropy_*` (L305–L399) |
| `vci`           | $`\text{VCI}_q = \max_\ell n_{q,\ell}/K_q^{\mathrm{eff}}`$, cross-question mean                                            | `consistency.mean_vci` (L401–L425)            |
| `mvg`           | $`\text{MV-Acc} - \text{pass@1}`$; positive values indicate self-consistency gain                                          | `consistency.mvg` (L427–L450)                 |

### 9.5 Format Skill Score (FSS)

The headline chance-corrected skill metric. For the $`j`$-th trial of question
$`q`$:

$$
\bar{T}_q = \frac{1}{K_q^{\mathrm{eff}}}\sum_{j\in\mathcal{J}_q^{\mathrm{ok}}} T(P_{q,j}, G_q),
\qquad
\text{fss}_q = \frac{\bar{T}_q - T_q^{\mathrm{chance}}}{1 - T_q^{\mathrm{chance}}}
$$

The chance-baseline closed form is

$$
T_q^{\mathrm{chance}} = \begin{cases}
\dfrac{1}{k_q}, & \text{single-answer} \\\\[6pt]
2^{-k_q}\sum_{tp=1}^{m_q}\sum_{fp=0}^{k_q-m_q}\binom{m_q}{tp}\binom{k_q-m_q}{fp}\cdot\dfrac{tp}{tp+\alpha\,fp+\beta(m_q-tp)}, & \text{multi-answer}
\end{cases}
$$

The dataset-level value is
$`\text{fss} = \frac{1}{|\mathcal{D}^{\mathrm{ok}}|}\sum_q \text{fss}_q`$ where
$`\mathcal{D}^{\mathrm{ok}} = \{q : \bar{T}_q \ne \text{None}\}`$.

The implementation is `accuracy.fss` (accuracy.py:L386–L479), with the
closed-form chance via `accuracy.tversky_baseline` (L316–L350). It returns
`{"fss", "n_valid", "mean_pe", "per_question"}` so downstream callers can
decompose by question. Pinned by `test_fss.py` (528 LOC) for correctness
against analytical baselines, and by `test_fss_sensitivity.py` for the
$`(\alpha, \beta)`$ sweep.

### 9.6 Composite Accuracy (the headline)

Composite Accuracy is the model-level summary metric. Substituting
$`\text{exam}_{avg}^{(b)}`$ as the per-bucket value:

$$
\text{Composite Accuracy}_m = \frac{\sum_{b\in B_{\mathrm{valid}}(m)} w_b \cdot \text{exam}_{avg}^{(b),m}}{\sum_{b\in B_{\mathrm{valid}}(m)} w_b}
$$

where $`B_{\mathrm{valid}}(m) = \{b\in B : v_{m,b}\ne\text{None} \wedge w_b > 0\}`$.
Missing buckets are dropped and the remaining weights are renormalised. If
$`B_{\mathrm{valid}}(m) = \varnothing`$ the composite is `None`.

The default weights (config.py:L365–L368) are:

```text
yes_no          = 0.15
binary_named    = 0.15
multiple_choice = 0.70
```

with choice-type weights:

```text
single = 0.40
multi  = 0.60
```

Per-metric overrides flow through `COMPOSITE_WEIGHT_OVERRIDES_QTYPE` and
`COMPOSITE_WEIGHT_OVERRIDES_CTYPE`, both CSV with shape
`metric=bucket=w,bucket=w;metric=...`. Misspelled metric names raise at
runtime via the known-metrics allowlist (composite.py:L77–L127). The
implementation is `composite.compute_composite` (composite.py:L18) plus
`composite.slice_v5_metrics_by_bucket` (L151–L198).

### 9.7 Per-correct cost

The cost-effectiveness scalar amortises the OpenRouter invoice across the
difficulty-weighted notional correct count:

$$
C^{\mathrm{per\text{-}correct}}_m = \frac{C^{\mathrm{total}}_m}{|\mathcal{D}^{\mathrm{eval}}| \cdot n \cdot \text{Composite Accuracy}_m}
$$

The denominator
$`|\mathcal{D}^{\mathrm{eval}}| \cdot n \cdot \text{Composite Accuracy}_m`$ is
the difficulty-weighted notional correct-sample count: when bucket weights
coincide with empirical question-type prevalence it equals the raw correct
count, and otherwise it acts as a discrimination-aware reference count that
up-weights harder buckets.

$`C^{\mathrm{total}}_m`$ is read directly from OpenRouter's billing endpoint.
The platform invoice is the single financial fact verifiable by third parties,
which avoids divergences from "published unit price × token usage" calculations
caused by reasoning-token billing, prompt-cache discounts, tool-call billing,
and provider routing.

### 9.8 Probabilistic family (v4 companion, demoted under K=5)

`forecast_eval/analysis/proper_score.py` and `probabilistic.py` are active
only when `BELIEF_PROTOCOL=True`.

| Metric                    | Formula                                                                                  | Applicable      |
| ------------------------- | ---------------------------------------------------------------------------------------- | --------------- |
| **Brier Index (BI)**      | $`100(1 - \sqrt{\overline{\text{BS}^{\mathrm{lab}}}})`$, mean-then-square-root              | All qtypes      |
| **BI_dec**                | Decision-wise Brier index                                                                  | Single only     |
| **NLL**                   | Single: $`-\log p_{q,l^*}`$; multi: per-label BCE; clip $`\epsilon = 10^{-3}`$                | All qtypes      |
| **MBS**                   | $`100(\log_2 p_{q,l^*} + 1)`$, clip same                                                     | Single only     |
| **ABI (crowd / uniform)** | Sign-aware $`100(1 \mp \sqrt{|\overline{\text{ABS}}|})`$ vs LOO crowd or uniform baselines  | Crowd: multi-model |
| **fallback share**        | Share of questions through the §9.8.1 fallback                                              | All runs        |

> **K=5 disclaimer.** When `SAMPLING_N` is small, the empirical probability
> $`\hat p = n/K`$ takes only six discrete values, which makes Reliability
> Diagram, Murphy three-decomposition, and Platt LOO calibration statistically
> meaningless. v5 deletes `calibration.py` and its five artefacts; the
> probabilistic columns retain a `†` footnote in `per_model_summary.md`.
> Reintroducing calibration requires raising $`K`$ to at least 30.

#### 9.8.1 Belief fallback when `belief_final IS NULL` but `parse_ok = 1`

Legacy v3 runs and v4 belief-parse-failures still benefit from a degenerate
probability vector for proper scoring:

$$
p_l = \begin{cases} 1 - \epsilon, & \ell \in \widehat{G}_{i,M} \\\\ \dfrac{\epsilon}{k - |\widehat{G}_{i,M}|}, & \text{otherwise} \end{cases},\quad \epsilon = 0.05
$$

The sample is recorded with `belief_parse_ok=0`. Samples with full failure
(`parse_ok=0`) are not allowed into probabilistic averaging; the pollution
defence lives in flatten.py:L126–L152.

### 9.9 Aggregation strategies (`aggregation.py`)

For probability vectors across $`K`$ samples per question:

| Strategy             | Formula                                                                                  | Use                                            |
| -------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------- |
| Arithmetic mean      | $`\hat p_l = (1/K)\sum_k p_{k,l}`$                                                          | Phase 1 default                                |
| Logit-space mean     | Single: softmax of mean log-prob; multi: per-label sigmoid of mean logit                 | Bayesian model average                          |
| LOO shrinkage        | Scan $`\alpha \in \{0, 0.1, ..., 1.0\}`$; blend toward uniform prior on logit              | Adaptive smoothing (`aggregation.loo_shrinkage`, L145–L199) |

### 9.10 Statistical inference (`inference.py`)

| Function                                  | Algorithm                                                                                | Output                                |
| ----------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------- |
| `paired_bootstrap(bs_a, bs_b)`            | $`B=5000`$ paired resampling; the same indices index both A and B                           | `delta_mean / ci_low / ci_high / p_two_sided` |
| `holm_bonferroni(p_values)`               | $`(n-i) \cdot p_{(i)}`$ then cumulative max                                                | Adjusted p-values                     |
| `difficulty_tertile(gammas)`              | Sort per-question $`\gamma_q`$, cut into tertiles                                          | `low / mid / high` buckets             |
| `posterior_a_better_than_b(bs_a, bs_b)`   | Monte-Carlo $`\Pr(\overline{BS}_A < \overline{BS}_B)`$ on paired bootstrap                  | $`\Pr(\mathrm{BI}_A > \mathrm{BI}_B) \in [0,1]`$ |
| `metric_paired_bootstrap(metric_fn, ...)` | Generic paired bootstrap on any metric (FSS, Acc, MV-Acc, Fleiss, EBI)                   | `delta_mean / ci_low / ci_high / p_two_sided / cohens_d` |
| `pairwise_paired_bootstrap(...)`          | All-pairs application of `paired_bootstrap` over models                                   | `list[ModelPairResult]`               |

The multi-comparison control is Holm-Bonferroni at the FWER level. The paired
bootstrap is **same-indexed**: the same bootstrap draws the same question
into both A and B's arrays, which controls the question-level variance that
typically dominates total variance in this evaluation.

### 9.11 Behavioural diagnostics (`behavior.py`)

Active when `BELIEF_PROTOCOL=True`. Four diagnostic groups:

| Group                          | Metrics                                                                                   | Output                            |
| ------------------------------ | ----------------------------------------------------------------------------------------- | --------------------------------- |
| Belief evolution               | Per-trial volatility $`V`$, inter-trial variance $`\sigma`$, convergence step, evidence efficiency $`\eta`$, counter-evidence engagement | `belief_evolution.csv`             |
| Reflection A/B                 | Paired-bootstrap 95% CI of $`\Delta\text{BI}`$, $`\Delta\sigma`$, $`\Delta C`$, $`\Delta\eta`$ under matched `reflection_protocol_hash` | `reflection_ab.csv`                |
| Tool-usage PDP                 | Logistic and linear regression of `Pr(correct \| x)` and `E[NLL \| x]` on `tool_calls_count / react_steps / latency_ms / prompt_tokens / completion_tokens` | `tool_usage_pdp.csv`               |
| Confidence calibration         | Subjective 3-bin (low/medium/high) and numeric max-$`p`$ binned hit-rate; conflict flag      | `confidence_calibration_*.csv`     |

### 9.12 Output artefacts (`writers.py`)

A run's `analysis/` directory contains:

| File                                            | Schema                                          | Contents                                |
| ----------------------------------------------- | ----------------------------------------------- | --------------------------------------- |
| `per_model_summary.csv` and `.md`               | 24 v3 + 4 FSS + 4 consistency + 7 prob = 39 cols | One row per model                        |
| `per_model_by_question_type.csv`                | sliced summary                                   | Bucketed by `question_type`             |
| `per_model_by_choice_type.csv`                  | sliced summary                                   | Bucketed by `choice_type`               |
| `per_model_composite_by_question_type.csv`      | composite weights + per-bucket metrics           | Composite Accuracy with subtype weights  |
| `per_model_composite_by_choice_type.csv`        | composite weights + per-bucket metrics           | Composite Accuracy with choice weights   |
| `error_breakdown.csv`                           | `Counter[error]`                                 | All samples (incl. cutoff)               |
| `finish_reason_breakdown.csv`                   | `Counter[finish_reason]`                         | Eligible samples only                    |
| `paired_delta_bi.csv`                           | `ModelPairResult`                                | Paired-bootstrap deltas (BI units)      |
| `paired_delta_bi_by_difficulty.csv`             | per-tertile result                               | Difficulty-stratified pair tests        |
| `metric_pairwise_bootstrap.csv`                 | per-metric × per-pair result                     | v5 multi-metric pairwise                 |
| `belief_evolution.csv`                          | `BeliefEvolutionRow`                             | Volatility, variance, convergence        |
| `reflection_ab.csv`                             | `ReflectionABRow`                                | Reflection A/B paired CIs                |
| `tool_usage_pdp.csv`                            | `PDPRow`                                         | Feature importance                      |
| `confidence_calibration_subjective.csv`         | `ConfidenceCalibrationRow`                       | 3-bin calibration                        |
| `confidence_calibration_numeric.csv`            | `NumericConfidenceCalibrationRow`                | max-$`p`$ binned                           |
| `entropy_accuracy_bins.csv`                     | per-bucket entropy/acc/Fleiss                    | Per-tertile diagnostic                   |
| `overall.json`                                  | aggregated metrics + metadata                    | Single JSON for downstream tooling       |
| `grid_summary.csv` (when grid enabled)          | per `(real_model, R, C)` 17-col main             | Grid main table                          |
| `grid_marginal_C.csv`, `grid_marginal_R.csv`    | scan along axis with the other anchored          | Saturation curves                        |
| `grid_pareto.csv`                               | one row per cell + `dominated_by`                | Pareto frontier                          |
| `grid_winrate.csv`                              | per real-model pair × cross-(R,C) cell wins/ties + significance count | Winrate matrix |

The default rounding is 4 decimals (writers.py:L113–L116); `avg_react_steps`
uses 2 decimals and `avg_latency_ms` uses 1 decimal.

---

## 10. Grid search

`Settings.TAVILY_MAX_RESULTS` (the $`R_{\mathrm{tav}}`$ axis) and
`REACT_MAX_SEARCH_CALLS` (the $`C`$ axis) accept CSV lists. When either has
length greater than 1, the run becomes a Cartesian grid over $`R \times C
\times M`$ cells, each producing its own DB file via a *virtual slug*:

```text
{real_model}::r{R}::c{C}
```

The composition is `db.compose_virtual_slug(real_model, R, C)`
(db.py:L477–L516); parsing is `db.parse_virtual_slug(slug)`, which returns
`(real_model, R, C)` or `None` for legacy single-cell runs. The `::`
delimiter is chosen to avoid collision with provider slugs, which is further
enforced by config validation that rejects `::` inside `MODELS`.

`runner._resolve_settings(slug)` (runner.py:L160) reads the slug, clones
`Settings` via `model_copy(update=...)` with the cell's $`R`$ and $`C`$
overrides, and hands each cell its own settings view.

The `grid_summary.csv` artefact (§9.12) emits, per cell, `real_model`, $`R`$,
$`C`$, `n_eligible`, `n_total`, `acc_mean`, `acc_ci_lo` / `acc_ci_hi`,
`bi_mean`, `bi_ci_lo` / `bi_ci_hi`, `nll_mean`, `ece`, `mean_search_calls`,
`mean_latency_ms`, `parse_ok_rate`, `belief_parse_ok_rate`. Bootstrap CIs at
the cell level are computed by `grid._bi_ci_from_bs_array` and
`grid._acc_ci_for_samples` (grid.py:L122–L142, $`B=5000`$, seed=42).

`GRID_DEFAULT_R` and `GRID_DEFAULT_C` (config.py:L319–L322) pin the marginal
slices when the grid is multi-axis; if unset, `r_list[0]` and `c_list[0]`
apply (validated to belong to the lists, see §7.2).

Pinned by `test_grid_slug.py`, `test_grid_dispatcher.py`,
`test_grid_analysis.py`, `test_grid_settings_view.py`, and
`test_runner_grid_model.py`.

---

## 11. Tests

The repository ships 33 test files covering roughly 560 individual test
cases; all run offline. Tavily and OpenRouter exist as fixtures or mocked
stand-ins, since a single end-to-end evaluation is costly and getting tests
stable saves significant API spend.

### 11.1 CI redlines

Five tests map one-to-one to components of $`\mathcal{R}`$ that, if broken,
invalidate the entire run unit. They must stay green on every commit:

1. `test_prompts.py` guards the renderer $`R`$.
2. `test_parser.py` guards $`\Psi`$ and $`\phi`$.
3. `test_training_cutoff.py` guards $`\kappa_M`$ admissibility.
4. `test_llm_no_browsing.py` guards the information barrier.
5. `test_analysis.py` guards $`\Gamma`$.

If any of these fails, the run unit's contract is broken and no downstream
result can be trusted.

### 11.2 Test-to-invariant map

The tests act as proofs that each component of $`\mathcal{R}`$ is implemented as
advertised.

| Component / claim                                                                          | Pin tests                                                                                                                                       |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| $`\mathcal{D}`$: dataset shape, templates contract, hashes deterministic                     | `test_db.py`, `test_evaluation.py`                                                                                                              |
| $`M`$: per-model DBs, virtual slugs, resume per model                                        | `test_runner_grid_model.py`, `test_runner_resume.py`                                                                                            |
| $`\kappa_M`$: admissibility filter, cutoff-row write contract                                | `test_training_cutoff.py`                                                                                                                       |
| $`\delta`$: tool-layer injection, LLM never sees `end_date`                                  | `test_search.py`, `test_react.py`                                                                                                               |
| $`T`$, $`C`$: ReAct loop bounded, budget gates, harness priority chain, v5.1 switches          | `test_react.py` (1432 LOC), `test_react_reflection.py`                                                                                          |
| $`R`$: renderer correct for all three qtypes; protocol additions outside `prompt_templates_hash` | `test_prompts.py`                                                                                                                              |
| $`\Psi`$ and $`\phi`$: parser correctness, strict equality, >26-option round-trip              | `test_parser.py`, `test_parser_belief.py`                                                                                                       |
| $`\Gamma`$: aggregation correctness end-to-end                                               | `test_analysis.py` (670 LOC), `test_aggregation.py`, `test_consistency.py`, `test_inference.py`, `test_proper_score.py`                          |
| $`H_{\mathrm{aux}}`$: detector whitelist, fail-closed, AUTH-immediate-drop                    | `test_leak_filter.py`                                                                                                                           |
| Composite: weights validation, per-metric overrides, allowlist                              | `test_composite_score.py`                                                                                                                       |
| FSS: closed-form chance baselines, $`(\alpha,\beta)`$ sensitivity                            | `test_fss.py`, `test_fss_sensitivity.py`                                                                                                        |
| Exam-score: corner cases (FP-veto, parse-fail = 0, cutoff = None)                           | `test_exam_score.py`                                                                                                                            |
| Behavioural metrics: belief evolution, reflection A/B, tool PDP, confidence calibration     | `test_behavior.py`                                                                                                                              |
| Grid: virtual-slug encoding, per-cell settings view, analysis pipeline                     | `test_grid_slug.py`, `test_grid_dispatcher.py`, `test_grid_analysis.py`, `test_grid_settings_view.py`, `test_plot_analysis_grid.py`             |
| DB schema migration: v2→v5 forward path                                                     | `test_db_v4_migration.py`, `test_db_v5_migration.py`                                                                                            |
| Information barrier: provider-native browsing forbidden                                    | `test_llm_no_browsing.py`                                                                                                                       |
| Error tiering: classification + backoff lookup                                              | `test_errors.py`                                                                                                                                |
| Configuration: every validator, every env-var contract                                     | `test_config.py`                                                                                                                                |
| End-to-end: dry-run replaces all transports with stubs                                     | `test_smoke_dry_run.py`                                                                                                                         |

### 11.3 Heaviest test files

The complexity of the implementation is reflected in test file sizes. The
longest tests are:

| File                           | LOC   | What it covers                                                               |
| ------------------------------ | ----- | ---------------------------------------------------------------------------- |
| `test_react.py`                | 1432  | Full ReAct loop: every harness branch, priority chain, finalisation, v5.1     |
| `test_search.py`               |  830  | Tavily wrapper, key rotation, raw-content truncation, audit metadata          |
| `test_behavior.py`             |  762  | Belief evolution, reflection A/B, tool PDP, confidence calibration            |
| `test_analysis.py`             |  670  | Phase 0–6 of the analysis orchestrator                                        |
| `test_db.py`                   |  630  | Schema, migrations, AsyncWriter, hashes, redaction                            |
| `test_inference.py`            |  630  | Paired bootstrap, Holm, posterior, multi-metric                               |
| `test_grid_analysis.py`        |  605  | Virtual-slug grid analysis end-to-end                                         |
| `test_consistency.py`          |  595  | Fleiss κ stratification, entropy, VCI, MVG                                    |
| `test_leak_filter.py`          |  550  | Whitelist, fail-closed, audit fields                                          |
| `test_fss.py`                  |  528  | FSS Tversky, chance baselines, edge cases                                     |
| `test_composite_score.py`      |  509  | Composite weights, allowlist, override parsing                                |
| `test_prompts.py`              |  447  | Renderer rules across all three qtypes plus protocol toggles                   |
| `test_exam_score.py`           |  426  | exam-score corner cases (FP-veto, parse-fail, cutoff exclusion)                |

Run all tests with:

```bash
pytest tests/ -q
```

---

## 12. Setup, operation, and reimplementation

### 12.1 From-scratch implementation order

For a from-scratch reimplementation, this order keeps each step locally
verifiable and lists the test that should pass before moving on:

| Step | Module                          | Pin test                          |
| ---: | ------------------------------- | --------------------------------- |
|  1   | `environment.yml` + `.env.example` + `.gitignore` | smoke: `python -c 'import forecast_eval'` |
|  2   | `forecast_eval/config.py`       | `test_config.py`                   |
|  3   | `forecast_eval/db.py`           | `test_db.py`, `test_db_v5_migration.py` |
|  4   | `forecast_eval/loader.py`       | covered by `test_db.py`            |
|  5   | `forecast_eval/prompts.py`      | `test_prompts.py`                  |
|  6   | `forecast_eval/parser.py`       | `test_parser.py`, `test_parser_belief.py` |
|  7   | `forecast_eval/errors.py`       | `test_errors.py`                   |
|  8   | `forecast_eval/search.py`       | `test_search.py`                   |
|  9   | `forecast_eval/leak_filter.py`  | `test_leak_filter.py`              |
| 10   | `forecast_eval/tools.py`        | covered by `test_search.py`        |
| 11   | `forecast_eval/llm.py`          | `test_llm_no_browsing.py`          |
| 12   | `forecast_eval/react.py`        | `test_react.py`, `test_react_reflection.py` |
| 13   | `forecast_eval/runner.py`       | `test_runner_resume.py`, `test_runner_grid_model.py`, `test_training_cutoff.py` |
| 14   | `forecast_eval/analysis/*`      | `test_analysis.py` plus per-metric tests |
| 15   | `evaluation.py` (main entry)    | `test_evaluation.py`, `test_smoke_dry_run.py` |

Get a smoke pass first with `--question-type yes_no`, `MODELS=openai/gpt-4o-mini`,
and `SAMPLING_N=1`; verify the renderer output and parser normalisation, and
only then open up to a full evaluation.

### 12.2 Conda environment

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
      - tenacity>=9.0
      - pytest>=8.0
      - pytest-asyncio>=0.23
      - respx>=0.21
```

To create the environment:

```bash
conda env create -f environment.yml
conda activate forecast
cp .env.example .env
# Edit .env: LLM_API_KEY, TAVILY_API_KEY, LEAK_DETECTOR_API_KEY, MODELS, MODEL_TRAINING_CUTOFFS
python evaluation.py --question-type yes_no
```

`matplotlib` is intentionally absent from `environment.yml` because the
analysis pipeline stays dependency-light. Install it locally only to render
the on-demand plot family in `scripts/plot_analysis.py`.

### 12.3 CLI

The main entry point is `evaluation.py`. Three flags control input:

```bash
# Run the entire dataset
python evaluation.py

# Filter by question_type (repeatable)
python evaluation.py --question-type yes_no --question-type binary_named

# Filter by choice_type (repeatable)
python evaluation.py --choice-type single

# Combined filter (AND): only multi-select multiple_choice
python evaluation.py --question-type multiple_choice --choice-type multi

# Skip analysis at run end (raw DBs still land in db/)
python evaluation.py --skip-analysis

# Refresh analysis/ independently (does not modify the DB)
python -m forecast_eval.analysis runs/{run_id}
```

`--question-type` accepts `yes_no`, `binary_named`, or `multiple_choice` and
is repeatable; `--choice-type` accepts `single` or `multi` and is repeatable.
Omitting either flag means no restriction. Every other tunable lives in
`.env`.

The detailed step-by-step run flow inside `evaluation.py` is:

1. `argparse` parses the three flags, assembling a `QFilter`.
2. `Settings()` loads and validates `.env` per §7.2.
3. The `run_id` is generated or reused, and `run_dir = RUNS_ROOT/{run_id}`
   is created with `db/`, `analysis/`, and `logs/`.
4. The four reproducibility hashes are computed (evaluation.py:L46–L75).
5. For each model (or virtual slug under grid):
   * Open `conn = RUNS_ROOT/{run_id}/db/{safe_slug(model)}.db`, where the
     model-slug alphabet is `[A-Za-z0-9._-]` and illegal characters are
     replaced.
   * `db.init_schema(conn, SAMPLING_N)` creates `s{i}_*` columns dynamically
     and applies the v2→v5 migrations as needed.
   * Sync `prompt_templates` and `questions` from the source DB.
   * `db.register_run_meta(conn, run_id, model, hashes, training_cutoff, ...)`.
6. `_write_manifest()` writes `manifest.json` (evaluation.py:L123–L192) with
   `run_id`, `schema_version`, `analysis_schema`, `sampling_n`, `models`,
   `model_files`, `model_training_cutoffs`, `filters`, `hashes`,
   `reflection_protocol_hash`, `belief_protocol_hash`, `grid`, `started_at`,
   and `finished_at: null`.
7. `runner.run(...)` starts the asyncio event loop, runs the resume baseline,
   applies the $`\kappa_M`$ filter, dispatches tasks under three semaphores,
   and writes per-completion log lines.
8. `db.finish_run_meta(conn, run_id)` and `_finalise_manifest()` write
   `finished_at` per model.
9. Unless `--skip-analysis`, `forecast_eval.analysis.run_analysis(run_dir)`
   runs the full metric stack and writes the artefacts of §9.12.

---

## Appendix A. Module catalogue

The package layout mirrors the run-unit decomposition: each module owns one
contract and exposes the symbols listed below.

### A.1 Directory layout

```text
Forecast/
├── .env                           # gitignored, user-filled
├── .env.example                   # template, git-managed
├── .gitignore
├── environment.yml                # conda env definition
├── README.md                      # user-facing entry
├── DESIGN.md                      # rationale (this implements design)
├── FRAME.md                       # this document
├── evaluation.py                  # main entry: CLI → runner.run → analysis.run_analysis
├── forecast_eval_set_example.db   # source data (read-only, checked into Git)
├── runs/                          # all evaluation outputs (gitignored)
│   └── {run_id}/
│       ├── manifest.json
│       ├── db/{model_slug}.db     # one sqlite per model; self-contained replay
│       ├── analysis/
│       └── logs/{run_id}.log
├── forecast_eval/
│   ├── __init__.py
│   ├── config.py                  # pydantic-settings; Settings + grid axes + composite weights
│   ├── db.py                      # per-model wide-table schema + AsyncWriter + hashes
│   ├── loader.py                  # syncs questions + prompt_templates from SOURCE_DB
│   ├── prompts.py                 # renderer R + reflection / budget-awareness / belief / harness
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
│       ├── __init__.py            # run_analysis(run_dir) orchestrator
│       ├── accuracy.py            # strict-equality + pass@k family + FSS / Cohen κ / Hamming
│       ├── exam_score.py          # exam-style partial credit
│       ├── composite.py           # subtype-weighted composite accuracy
│       ├── consistency.py         # Fleiss κ, mean entropy, VCI, MVG (K-trial)
│       ├── proper_score.py        # BI / NLL / MBS / ABI (probabilistic companion)
│       ├── aggregation.py         # arithmetic / logit-space mean / LOO shrinkage
│       ├── inference.py           # paired bootstrap, Holm-Bonferroni, posterior, multi-metric
│       ├── grid.py                # grid-search analysis (virtual slug decode, marginal / pareto / winrate)
│       ├── behavior.py            # reflection A/B, tool-usage PDP, confidence calibration, belief evolution
│       ├── probabilistic.py       # probabilistic family report builder
│       ├── flatten.py             # wide-table → SampleRow + per-question grouping
│       └── writers.py             # CSV / MD / JSON serialisers; column rounding rules
├── scripts/                       # operator scripts
│   ├── build_forecast_eval_set.py # dataset construction (stratified sample + topic cap)
│   ├── smoke_leak_filter.py       # smoke test of Stage-2 detector pipeline
│   ├── verify_leak_filter_e2e.py  # end-to-end leak-filter audit reproducer
│   ├── fss_sensitivity.py         # FSS α/β sensitivity sweep
│   ├── plot_analysis.py           # matplotlib renders for analysis/
│   └── migrate_split_mc_output_format.py  # one-shot dataset-metadata migration
└── tests/                         # 33 unit/integration tests (~13K LOC), all offline (§11)
```

### A.2 Module responsibilities

| Module                  | Implements                                                                                                  | Key interfaces                                                                                                 | Pin tests                              |
| ----------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `config.py`             | Reads `.env` via pydantic-settings; validates types; parses CSV lists; runs all §7.2 startup checks         | `Settings` class (singleton); `_parse_csv`, `_parse_int_list`, `_parse_cutoffs`                                | `test_config.py`                        |
| `loader.py`             | Syncs `<SOURCE_TABLE>` → `questions`; `dataset_metadata.features_json.prompt_reconstruction` → `prompt_templates` | `sync_questions(source_db, conn, filters, table=...) -> list[Question]`; `sync_prompt_templates(...)`         | covered by `test_db.py` + `test_evaluation.py` |
| `prompts.py`            | Renderer $`R`$; reflection / belief / budget-awareness protocol bodies; harness status injection builders      | `render_user_prompt`, `REFLECTION_PROTOCOL`, `BELIEF_PROTOCOL`, `build_budget_awareness_protocol`, `build_*_warning`, `_build_status_header` | `test_prompts.py`                       |
| `tools.py`              | Defines the `web_search` OpenAI schema; LLM-visible part has no date                                         | `WEB_SEARCH_SCHEMA`, `parse_tool_arguments`, `extract_query`, `tool_error_message`, `tool_result_message`        | `test_search.py`                        |
| `search.py`             | Tavily wrapper; injects `end_date = q.end_time + δ`; truncates raw_content; dispatches Stage-2 detector       | `tavily_search(query, end_date, settings) -> SearchResult`                                                      | `test_search.py`                        |
| `leak_filter.py`        | Detector $`H_{\mathrm{aux}}`$: per-result `keep` or `drop`; whitelist; fail-closed                              | `filter_search_result(result, cutoff_date, settings)`                                                           | `test_leak_filter.py`                   |
| `tavily_keys.py`        | Multi-key pool: least-used + 401/403 blacklist + 429 cooldown                                                  | `TavilyKeyPool.acquire / report_failure`; `get_pool(keys, cooldown_s)`                                          | covered by `test_search.py`              |
| `llm.py`                | OpenAI-compatible client; tiered retry by error kind; rejects `:online`, `plugins`, non-whitelist tools        | `chat(model, messages, tools, ...) -> ChatResponse`; `_assert_no_browsing`                                      | `test_llm_no_browsing.py`               |
| `react.py`              | Forecasting system $`F_M`$: ReAct loop with 4-knob harness resilience; per-step belief parsing                  | `run_react(q, model, sample_idx, settings) -> SampleResult`                                                     | `test_react.py` (1432 LOC), `test_react_reflection.py` |
| `parser.py`             | Parser $`\Psi`$ + normalisation $`\phi`$: `\boxed{}` extraction → letter `frozenset[str]`; belief JSON validator    | `parse_answer(text, q)`, `parse_gt(answer)`, `is_correct(pred, gt)`, `parse_belief(text, q)`                    | `test_parser.py`, `test_parser_belief.py` |
| `errors.py`             | Error classification + backoff lookup + AuthError                                                              | `ErrorKind`, `classify(exc)`, `should_retry(kind)`, `backoff_seconds(kind, attempt, settings, retry_after)`     | `test_errors.py`                        |
| `db.py`                 | Schema + AsyncWriter + hashes + redaction; v2→v5 migrations; resume queries; model-slug safety                | `init_schema`, `AsyncWriter.enqueue_result`, `load_completed_samples`, `register_run_meta`, `compute_*_hash`, `model_slug_safe` | `test_db.py`, `test_db_v4_migration.py`, `test_db_v5_migration.py` |
| `runner.py`             | Task orchestration: cartesian dedup → $`\kappa_M`$ filter → asyncio concurrency → progress log → `finish_run_meta` | `run(settings, filters, questions, templates, run_id, conns) -> RunStats`; `build_task_plan`                  | `test_runner_resume.py`, `test_runner_grid_model.py`, `test_training_cutoff.py` |
| `analysis/__init__.py`  | Aggregation $`\Gamma`$ orchestrator: walks DBs → runs metric stack → writes CSV/MD/JSON; auto-invoked or `python -m forecast_eval.analysis runs/{run_id}` | `run_analysis(run_dir: Path) -> list[Path]`                                                                      | `test_analysis.py`                      |

`QFilter` (types.py:L26–L51) is a dataclass with `question_types: frozenset[str] | None`
and `choice_types: frozenset[str] | None`; `None` means no filtering.
`apply_sql()` returns `(WHERE clause, params)` for SQLite parameterised
execution; `snapshot()` returns a dict for `manifest.filters_snapshot`.

### A.3 `prompts.render_user_prompt` reference

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
    # Order: budget-awareness → reflection → belief (matches react.py wiring).
    for protocol in (budget_awareness, reflection_protocol, belief_protocol):
        if protocol:
            body += "\n\n" + protocol
    return body
```

### A.4 `parser.parse_answer` reference

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

## Appendix B. Symbol index

| Symbol                     | Meaning                                            | Defined in                            |
| -------------------------- | -------------------------------------------------- | ------------------------------------- |
| $`\mathcal{R}`$              | The run unit                                       | §1.1                                   |
| $`\mathcal{D}`$              | The discrete forecasting dataset                   | §1.1, §2                                |
| $`\mathcal{D}^{\mathrm{eval}}`$ | Evaluable subset after $`\kappa_M`$ filter           | §4.2, §9.1                              |
| $`\mathcal{D}^{\mathrm{pred}}_M`$ | Per-model evaluable subset                        | §4.2                                   |
| $`\mathcal{S}`$              | Scorable sample set                                | §8.2, §9.1                              |
| $`M`$                        | Evaluated model slug                                | §1.1                                   |
| $`\kappa_M`$                 | Knowledge cutoff for $`M`$                            | §1.1, §4.2                              |
| $`\delta`$                   | Temporal masking offset (days)                      | §1.1, §4.3                              |
| $`\chi_i`$                   | Per-question search cutoff $`\tau_i + \delta`$        | §4.3                                   |
| $`\tau_i`$                   | Question resolution time                            | §2.1                                   |
| $`T`$                        | Max ReAct steps                                    | §1.1, §5                                |
| $`C`$                        | Max search calls per sample                         | §1.1                                   |
| $`R`$                        | Input renderer                                     | §1.1, §4.7                              |
| $`R_{\mathrm{tav}}`$         | Tavily results-per-call (grid axis)                 | §1.1                                   |
| $`\Psi`$                     | Output parser                                      | §1.1, §4.8                              |
| $`\phi`$                     | Letter normalisation map                            | §1.1, §4.8                              |
| $`\Gamma`$                   | Aggregation rule                                   | §1.1, §9                                |
| $`H_{\mathrm{aux}}`$         | Stage-2 leakage detector                            | §1.1, §4.4                              |
| $`\hat{p}_{q,j}`$            | Belief vector for question $`q`$, trial $`j`$            | §1.1, §5.3                              |
| $`G_q`$, $`\hat{S}_{q,j}`$     | Gold and prediction letter sets                     | §9.2                                   |
| $`k_q`$, $`m_q`$               | Option count and gold-answer count for $`q`$           | §9.2, §9.5                              |
| $`\text{exam}_{avg}^{(b)}`$           | Bucket-$`b`$ exam-score mean                          | §9.6                                   |

---

> **One sentence.** This codebase realises the run unit
> $`\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)`$,
> together with the auxiliary detector $`H_{\mathrm{aux}}`$, as a Python module
> per symbol, a SQLite column per observation, a CSV column per metric, and a
> unit test per invariant; every number that ever appears in a report can be
> traced back to a row in a wide table, a hash in `run_meta`, an audit verdict
> in `search_calls`, or a green test in `tests/`.
