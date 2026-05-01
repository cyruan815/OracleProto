# OracleProto — Technical Framework

> This document is the engineering specification of the OracleProto reference implementation.
> It maps every formal object of the paper's run unit
> $\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)$ to a concrete
> Python module, database column, configuration knob, or invariant guarded by a unit test.
> Read this alongside `paper/main.tex` (formal framework) and `DESIGN.md` (rationale behind
> each trade-off).

---

## 1. Project goal

This codebase is the reference implementation of **OracleProto**: a reproducible framework for
benchmarking the *native forecasting capability* of LLMs via knowledge cutoffs and temporal
masking. The goal is one paragraph long:

> Materialise a single run unit $\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi,
> \phi, \Gamma)$ such that the same configuration always produces byte-equivalent
> intermediate artefacts and stochastic-only differences in final-answer text — and bind the
> auxiliary leakage detector $H_{\mathrm{aux}}$ via a SHA-256 fingerprint to the run metadata
> so the leakage barrier itself is byte-reproducible.

Every section below answers the question "how does this realise some part of $\mathcal{R}$?".

### 1.1 Run unit ↔ implementation map

| Symbol         | Object                              | Where in code                                                                |
| -------------- | ----------------------------------- | ---------------------------------------------------------------------------- |
| $\mathcal{D}$  | Discrete forecasting dataset        | `forecast_eval/loader.py` reads `SOURCE_DB`/`SOURCE_TABLE`                   |
| $M$            | Evaluated model                     | `MODELS` entry; one DB file per $M$ under `runs/{run_id}/db/`                |
| $\kappa_M$     | Knowledge cutoff                    | `MODEL_TRAINING_CUTOFFS[M]`; `runner.py::build_task_plan` admissibility filter |
| $\delta$       | Temporal masking offset             | `TAVILY_END_DATE_OFFSET_DAYS`; injected at `search.py::tavily_search`         |
| $T$            | Max ReAct steps                     | `REACT_MAX_STEPS`; `react.py` loop bound                                      |
| $C$            | Max search calls                    | `REACT_MAX_SEARCH_CALLS`; `react.py` budget gate                              |
| $R$            | Input renderer                      | `prompts.py::render_user_prompt`                                              |
| $\Psi$         | Output parser & validity            | `parser.py::parse_answer`                                                     |
| $\phi$         | Answer normalization map            | letter encoding (`A` / `A,B`); `parser.py::parse_gt`                          |
| $\Gamma$       | Aggregation rule                    | `forecast_eval/analysis/*`                                                    |
| $H_{\mathrm{aux}}$ | Leakage detector                | `forecast_eval/leak_filter.py`; logged in `run_meta.config_snapshot`          |

### 1.2 Hard constraints derived from $\mathcal{R}$

1. **The LLM never sees $\chi_i$.** The `web_search` tool schema exposed to the LLM has only a
   `query` parameter; $\chi_i = \tau_i + \delta$ is hard-coded by the tool implementation.
2. **Provider-native browsing is forbidden.** No `:online` slug, no `plugins` field, no
   provider-specific web tool. `tests/test_llm_no_browsing.py` enforces this on the wire.
3. **Sample admission is upstream of LLM calls.** The $\kappa_M \le \chi_i$ check happens at
   task generation; admissibility violations write `error="skipped_training_cutoff"` rows
   without consuming any API budget.
4. **Strict frozenset equality scores answers.** `parser.is_correct(pred, gt)` is one line:
   `pred == gt`. All three question types reduce to this.
5. **DBs store raw observations only.** No aggregates, no derived metrics — those live in
   `analysis/`.

---

## 2. Data source

### 2.1 Source database `forecast_eval_set_example.db` (read-only)

The example dataset shipped with the repo is named `forecast_eval_set_example.db`, and its
main table is named `forecast_eval_set_example`. Both are configurable via `.env`'s `SOURCE_DB`
/ `SOURCE_TABLE` parameters; with a custom dataset, just keep the 7-column schema and
`dataset_metadata` structure consistent. `SOURCE_TABLE` only accepts SQLite-legal identifiers
`[A-Za-z_][A-Za-z0-9_]*` and is validated at startup.

Main table `<SOURCE_TABLE>`, **N rows × 7 columns**:

| Field           | Type    | Description                                                                                                                          |
| --------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `id`            | TEXT PK | Unique question ID (sourced from HuggingFace)                                                                                  |
| `choice_type`   | TEXT    | `single` \| `multi`, based on the letter count in `answer` (1 → `single`, >1 → `multi`)                                                  |
| `question_type` | TEXT    | `yes_no` \| `binary_named` \| `multiple_choice`, determining which prompt template set to use                                                       |
| `event`         | TEXT    | Event description $x_i$ (**without** options, **without** role-setting, **without** format requirements)                                                                  |
| `options`       | TEXT    | $\mathcal{A}_i$ as a JSON array of strings. `yes_no`=`["Yes","No"]`; `binary_named`=two entity names; `multiple_choice`=labels in A/B/C... order            |
| `answer`        | TEXT    | $Y_i$ encoded as letters: single-select `'A'`; multi-select `'A, B'` (comma + space separated). Letter ↔ option-index rule in §3.7 |
| `end_time`      | TEXT    | Resolution time $\tau_i$ (Asia/Shanghai), `YYYY-MM-DD` format                                                                              |

Indexes (shipped with the example dataset; for custom datasets, follow the
`idx_<table>_<column>` naming for consistency):
`idx_<table>_choice_type` / `idx_<table>_question_type` / `idx_<table>_end_time`.

Auxiliary table `dataset_metadata` (single row), contains `features_json`, recording all
prompt templates, column descriptions, and conversion logs. The renderer $R$ reads templates
from this table at runtime; **do not hard-code them in source**.

### 2.2 The example dataset

The bundled `forecast_eval_set_example.db` contains 319 questions spanning 2026-01-15 to
2026-04-14, distributed across question types as follows:

| question_type / choice_type | single | multi | total |
| --------------------------- | -----: | ----: | ---: |
| `yes_no`                    |     93 |     0 |   93 |
| `binary_named`              |     11 |     0 |   11 |
| `multiple_choice`           |    181 |    34 |  215 |
| **total**                   |  **285** | **34** | **319** |

`multiple_choice` option count range: 3 ~ 35 (when > 26, letters enter the ASCII continuation,
see §3.7).

The paper's main experimental run uses a **curated 80-question subset** of FutureX-Past with
the same schema, to constrain the leakage audit cost; the framework itself is dataset-agnostic
once the 7-column contract is met.

### 2.3 Examples

`yes_no`:

```yaml
event:    "2026 a dream year for trump?"
options:  ["Yes","No"]
answer:   "B"           # B = No
end_time: "2026-01-31"
```

`binary_named`:

```yaml
event:    "Golden Knights vs. Kings"
options:  ["Golden Knights","Kings"]
answer:   "A"           # A = Golden Knights
end_time: "2026-01-15"
```

`multiple_choice` (single):

```yaml
event:    "Bank of Brazil decision in January?"
options:  ["No change in the Selic rate ...",
           "the Bank of Brazil raise ...",
           "the Bank of Brazil lower ..."]
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

> Key convention: the `event` field **does not contain** options or format requirements; those
> are spliced in at call time by the template (§3.6).
> `dataset_metadata.features_json.prompt_reconstruction` stores all templates and concatenation
> rules.

---

## 3. The information boundary in code

### 3.1 The LLM never sees $\chi_i$

The schema of `web_search` exposed to the LLM has **only the `query` parameter**:

```python
{
  "type": "function",
  "function": {
    "name": "web_search",
    "description": "Search the web for information relevant to the question.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "Search query"}
      },
      "required": ["query"]
    }
  }
}
```

When Tavily is actually called, the `end_date` parameter $\chi_i = \tau_i + \delta$ is
**hard-coded and injected by the tool implementation layer**:

```python
def web_search(query: str, question_end_time: str) -> dict:
    end_date = (date.fromisoformat(question_end_time)
                + timedelta(days=TAVILY_END_DATE_OFFSET_DAYS)).isoformat()
    return tavily_client.search(query=query, end_date=end_date, ...)
```

### 3.2 Sample admission: $\kappa_M \le \chi_i$

**Motivation.** A question whose resolution time precedes the model's training cutoff is
likely already in the training corpus — the model "remembers" the answer rather than
forecasting it. Such samples cannot reflect native forecasting capability and must be removed
from that model's evaluation set $\mathcal{D}^{\mathrm{pred}}_M$ (paper Eq. 4).

**Mechanism.**

* Configure `MODEL_TRAINING_CUTOFFS` in `.env`: `<model_slug>=YYYY-MM-DD`, comma-separated.
* During task-queue generation, filter each `(question, model)`:
  ```python
  cutoff = MODEL_TRAINING_CUTOFFS.get(model)   # None = not declared, no filtering
  if cutoff is not None and question.end_time <= cutoff:
      # equivalent at day-rounded timestamps to χ_i < κ_M for δ = -1 day
      # skip all sample_idx for this model
  ```
* Filtered `(question, model, sample_idx)` **still records a row** into `run_results`, with
  `error = "skipped_training_cutoff"`, `parse_ok = 0`, `correct = NULL`,
  `final_answer_raw = NULL`, `messages_trace = NULL`, `search_calls = NULL`, numeric fields
  set to 0.
* Reports can clearly show "how many questions were filtered out per model and how many remain
  comparable" (paper Table 2's "Excluded by Cutoff" column); resume will not retry the row.

**Resume semantics for cutoff rows** (see §5.3):
* `error IS NULL` → completed normally; not retried.
* `error = "skipped_training_cutoff"` → actively excluded; **not retried**.
* other `error` values (`network` / `server_5xx` / `bad_request` / `content_policy`) →
  retried per §9.

### 3.3 Strict temporal cutoff: $\delta = -1$ day

`end_time` is already at `YYYY-MM-DD` granularity. To avoid same-day information leakage, the
default is `TAVILY_END_DATE_OFFSET_DAYS=-1`:

```text
question.end_time (τ_i) = 2026-01-18
→ Tavily end_date (χ_i) = 2026-01-17
```

This can be changed in `.env` to `0` (same day visible, more lenient) or `-2`, `-3` (more
conservative). The project uses `-1` uniformly as the baseline, and all reports default to
comparison under `-1`. Numbers under different offsets are not directly comparable.

### 3.4 Leakage barrier and threat model

The four controllable layers and the residual surface are documented in `paper/main.tex` §3.5
and §4.4. Code-layer enforcement:

| Leakage source                              | Controllable?     | Mitigation in code                                                                    |
| ------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------- |
| Tool search content (Tavily returned content) | ✅              | `end_date = end_time + TAVILY_END_DATE_OFFSET_DAYS` injected by tool layer (§3.1, §3.3) |
| Provider-native browsing / web tool          | ✅                | `llm.chat` attaches only `WEB_SEARCH_SCHEMA`; no `:online`, no `plugins`              |
| Model parametric memory                      | ⚠️ Partial        | $\kappa_M$ admissibility filter (§3.2)                                                |
| Page bodies that mention post-$\chi_i$ events | ⚠️ Partial      | Stage-2 LLM detector (`leak_filter.py`); audited residual ≈ 1.1%                      |
| Time clues in the question text              | ❌                | inherent to the data; accepted as evaluation bias                                      |
| External knowledge backflow after training   | ❌                | accepted as evaluation bias                                                            |

Code-layer hard constraints:
* In `llm.chat` calls, `tools=[WEB_SEARCH_SCHEMA]` is the only allowed tool schema; **no**
  provider-native browsing/online switch may be added.
* Slugs containing `:online` cause an immediate startup error.
* If a provider forcibly attaches a built-in tool that cannot be disabled, explicitly mark
  "this model is unsuitable for strict evaluation" in the README and reports.

### 3.5 Stage-2 LLM detector (`search-leak-filter-v1`)

`forecast_eval/leak_filter.py` runs an independent OpenAI-compatible client (`_detector_client`,
distinct from the main LLM `_client`) over each `SearchResultItem` returned by Tavily,
producing a verdict ∈ {`keep`, `drop`, `failed:network` | `failed:bad_request` |
`failed:auth` | …}. Items with verdict=`drop` are removed entirely before any field reaches
the main LLM context.

Key wiring points:

* **Cut point.** End of the 200 path in `search.py:tavily_search`, before `return`.
* **Input fields whitelist.** `title / url / published_date / content / raw_content /
  cutoff_date`. **Must not contain** any field of `Question` — to prevent the detector from
  morphing into an answer auditor.
* **Failure mode.** `LEAK_DETECTOR_FAIL_ACTION=drop` (default) → fail-closed after
  `LEAK_DETECTOR_RETRY_MAX` retries with `LEAK_DETECTOR_BACKOFF_S` backoff; AUTH errors
  (401/403) are caught locally and immediately drop (no propagation to the main run).
* **Observability.** Per-call audit fields land in `search_calls` JSON entries:
  `n_results_raw / n_results_kept / detector_verdicts / detector_latency_ms /
  detector_error_kind`. Three-key fingerprint
  (`leak_detector_enabled / leak_detector_model / leak_detector_prompt_hash`) lands in
  `run_meta.config_snapshot`.

When `ENABLE_SEARCH_LEAK_FILTER=false` the detector path is byte-level rolled back; behaviour
is identical to v5.1.

### 3.6 Prompt assembly: the renderer $R$

The source DB **stores only raw material** (`event` / `options` / `question_type` / `end_time`).
When the system spawns a sample, `prompts.py::render_user_prompt` reads the templates from
`dataset_metadata.features_json.prompt_reconstruction` and assembles a complete user message
per `question_type`:

```text
{agent_role} The event to be predicted: "{event} (resolved around {end_time} (GMT+8)).{outcomes_block}"

IMPORTANT: Your final answer MUST end with this exact format:
{output_format}
{guidance}
```

Per-slot rendering rules:

| slot              | Rendering logic                                                                                                       |
| ----------------- | -------------------------------------------------------------------------------------------------------------- |
| `agent_role`      | constant `"You are an agent that can predict future events."`, inserted as-is                                            |
| `event`           | `<SOURCE_TABLE>.event` original text                                                                                    |
| `end_time`        | `<SOURCE_TABLE>.end_time` original text (`YYYY-MM-DD`)                                                                 |
| `outcomes_block`  | `yes_no` / `binary_named` → **empty string** (the options are already implicit in `output_format`)<br>`multiple_choice` → `"\n" + "A. <options[0]>\nB. <options[1]>\n..."`, with letters generated via the §3.7 index→letter rule |
| `output_format`   | one of three (per `question_type`): `yes_no_output_format` / `binary_named_output_format` / `multiple_choice_output_format`. **The `binary_named` template contains `<options[0]>` / `<options[1]>` placeholders, which must be replaced with the actual two entity names during assembly** |
| `guidance`        | constant `"Do not use any other format. Do not refuse to make a prediction. ..."`, inserted as-is                       |

Shape of the three `output_format`s:
* `yes_no` — requires `\boxed{Yes}` or `\boxed{No}`
* `binary_named` — template contains placeholders; after rendering looks like
  `\boxed{Golden Knights} or \boxed{Kings}`
* `multiple_choice` — requires `\boxed{A}` or `\boxed{B, C}`, with an example attached

The reflection / budget-awareness / forced-finalisation protocol additions are appended at
runtime (when their respective switches are on); they do **not** enter `dataset_metadata`, so
`prompt_templates_hash` is unaffected, but the rendered full user message lands in each
sample's `user_prompt` field. The protocol-text fingerprints are persisted independently in
`run_meta.reflection_protocol_hash` and `run_meta.belief_protocol_hash` (see §5.2).

### 3.7 Answer encoding ↔ decoding (the map $\phi$)

The DB uniformly uses **letters** as the canonical answer, but the LLM's output form varies by
`question_type`:

| question_type      | LLM output (inside `\boxed{}`)                       | Parser normalisation target ($\phi$)                                   |
| ------------------ | ------------------------------------------------ | ------------------------------------------------------------------ |
| `yes_no`           | `Yes` / `No` (case-insensitive)                    | `frozenset({"A"})` / `frozenset({"B"})` — `Yes`=A, `No`=B          |
| `binary_named`     | one of the entries in `options` (exact match, trim + case-insensitive)| look up the index in the `options` list → letter → frozenset                     |
| `multiple_choice`  | one or more letters, comma- or space-separated (`A` / `B, C` / `B,C`) | split directly → frozenset[str]                                        |

Letter ↔ index rule (supports up to 35 options for multiple_choice):

```text
index = ord(letter) - ord('A')
A=0, B=1, ..., Z=25
[ =26, \ =27, ] =28, ^ =29, _ =30, ` =31, a =32, b =33, c =34, ...
```

Reverse (when assembling the prompt, index → letter):
`letter = chr(ord('A') + index)`.

> ⚠️ **Source-data compatibility-mode warning.** The DB has 4 `multiple_choice` questions with
> > 26 options, of which 3 have ground-truth answers landing on non-letter symbols like
> `[ \ ] ^ _ `` ` ` a b c …`. These ASCII-continuation labels are extremely unfriendly to LLMs
> (backticks and underscores get swallowed by markdown/code blocks; lowercase `a` and uppercase
> `A` coexist and are easily confused). **We keep this scheme only to preserve a one-to-one
> mapping with the source-data letter encoding** for letter-set scoring.
>
> Mandatory defences:
> 1. `prompts.render_user_prompt` must explicitly quote or escape labels when generating the
>    `outcomes_block` for > 26 options (e.g. wrap `` `[` ``, `` `\` `` in backticks/quotes), to
>    avoid being lost in markdown rendering.
> 2. `parser.parse_answer` must have a round-trip unit test (label→letter→label) for
>    `multiple_choice` with > 26 options.
> 3. Logs/reports record letters and corresponding labels in parallel for manual review.

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
│                   OracleProto Evaluation Pipeline                      │
└────────────────────────────────────────────────────────────────────────┘

[.env]  →  [python evaluation.py [--question-type ...] [--choice-type ...]]
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │  1. Load Settings      │
                          │  & Init run_id         │
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
                          └──────────────────────────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │  3. Resume Check       │
                          │  skip rows where       │
                          │  s{i}_created_at NOT   │
                          │  NULL & error in       │
                          │  {NULL,                │
                          │   skipped_training_    │
                          │   cutoff}              │
                          └────────────────────────┘
                                      │
                                      ▼
                 ┌────────────────────────────────────────┐
                 │  4. Task Queue (D × M × N)             │
                 │  - apply κ_M admissibility filter      │
                 │    (§3.2): write skipped_training_     │
                 │    cutoff rows directly                │
                 │  - asyncio.Semaphore for concurrency   │
                 └────────────────────────────────────────┘
                                      │
                      ┌───────────────┼───────────────┐
                      ▼               ▼               ▼
                ┌──────────┐    ┌──────────┐    ┌──────────┐
                │ Worker 1 │    │ Worker 2 │    │ Worker N │
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
                      │     llm.chat(messages,     │
                      │               tools=...)   │
                      │     if no tool_call:       │
                      │       break                │
                      │     for each tool_call:    │
                      │       u_t = tavily.search( │
                      │              query, χ_i)   │
                      │       ũ_t = AuxLeakFilter( │
                      │              u_t, χ_i)     │
                      │   ↓                        │
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
                      │  Single writer per     │
                      │  model; WAL + batch    │
                      │  commit                │
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

Each run × model corresponds to **one independent SQLite file**. The file self-contains copies
of `questions` / `prompt_templates`, so a single file can be replayed independently.
Aggregations / statistics are **not persisted**; after the run finishes,
`forecast_eval.analysis` writes them separately into the `analysis/` directory.

### 5.1 Schema

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
    answer        TEXT NOT NULL,             -- comma-separated letters: 'A' / 'A, B'
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

-- ③ unique (run, model) metadata for this DB; single row
CREATE TABLE run_meta (
    run_id                TEXT PRIMARY KEY,
    model                 TEXT NOT NULL,
    sampling_n            INTEGER NOT NULL,
    config_snapshot       TEXT NOT NULL,   -- redacted .env JSON
    filters_snapshot      TEXT NOT NULL,   -- {"question_types":..., "choice_types":..., "question_ids":[...], "question_count":N}
    source_db_hash        TEXT NOT NULL,
    metadata_hash         TEXT NOT NULL,
    prompt_templates_hash TEXT NOT NULL,
    reflection_protocol_text TEXT,         -- full text of prompts.REFLECTION_PROTOCOL; NULL when REACT_REFLECTION_PROTOCOL=false
    reflection_protocol_hash TEXT,         -- sha256[:16]; same as above, NULL when off
    belief_protocol_text   TEXT,           -- v4. full text of prompts.BELIEF_PROTOCOL; NULL when BELIEF_PROTOCOL=false
    belief_protocol_hash   TEXT,           -- v4. sha256[:16]; same as above, NULL when off
    training_cutoff       TEXT,            -- κ_M (YYYY-MM-DD), NULL when not declared
    started_at            TEXT NOT NULL,
    finished_at           TEXT
);

-- ④ wide table: one row per question, one s{i}_* column group per sample
-- dynamically generates ~23 × SAMPLING_N columns; the shape below is for SAMPLING_N=3 only
CREATE TABLE run_results (
    question_id TEXT PRIMARY KEY,
    user_prompt TEXT,                      -- shared across all samples (COALESCE-written, first sample wins)

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
    s0_step_metrics         TEXT,          -- JSON array, one step snapshot per element, see §5.2
    s0_response_id          TEXT,          -- ChatCompletion.id (last round)
    s0_system_fingerprint   TEXT,          -- ChatCompletion.system_fingerprint (last round)
    s0_service_tier         TEXT,          -- ChatCompletion.service_tier (last round)
    -- v4 belief (3 columns)
    s0_belief_final         TEXT,          -- final-step Belief.probabilities as JSON ({letter: float}); NULL when parse fails
    s0_belief_trace         TEXT,          -- per-step belief summary JSON array
    s0_belief_parse_ok      INTEGER,       -- whether the final-step belief parsed legally (0/1); independent of parse_ok
    -- v5 harness-resilience (1 column)
    s0_final_answer_retry_used INTEGER,    -- 0/1 — see §10.3

    -- ...same s1_* / s2_* field groups...

    FOREIGN KEY (question_id) REFERENCES questions(id)
);
CREATE INDEX idx_run_results_question ON run_results(question_id);
```

> **Schema migration notes.** v2 → v3 → v4 → v5 are performed by `forecast_eval.db._migrate_*`
> via `ALTER TABLE … ADD COLUMN` (SQLite's ADD COLUMN only writes table metadata, completing
> in O(1); new columns on old rows default to NULL). On the resume path, the first time an old
> DB is opened it is auto-migrated. When `Settings.BELIEF_PROTOCOL=false`, all belief columns
> write NULL and the existing accuracy metrics output zero changes.

Connection-init PRAGMA (executed on every sqlite3 connection):

```sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;     -- safe enough under WAL and faster
PRAGMA busy_timeout = 5000;      -- avoid SQLITE_BUSY in multi-reader scenarios
```

### 5.2 Field write conventions

| Field                                | Source                                                                                                    |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `s{i}_final_answer_letters`          | `frozenset[str]` returned by `parser.parse_answer(final_raw, q)`, written after `sorted()` + `json.dumps`           |
| `s{i}_final_answer_raw`              | the full `content` text of the LLM's last assistant message                                                        |
| `s{i}_correct`                       | `frozenset == frozenset` → `int`; `NULL` when parse fails or scoring is impossible (not in $\mathcal{S}$) |
| `s{i}_parse_ok`                      | `final_answer_letters is not None` (i.e. $v_{i,M}$ from paper §3.4)                                      |
| `user_prompt`                        | the return value of `prompts.render_user_prompt(q, templates, …)`; rendered once per question, retained via COALESCE after the first sample writes       |
| `s{i}_messages_trace`                | full `messages` list as JSON; NULL when `WRITE_MESSAGES_TRACE=false`                                         |
| `s{i}_search_calls`                  | metadata list for each `web_search` call (`query / end_date / n_results / published_dates`; when leak filter is enabled, additionally `n_results_raw / n_results_kept / detector_verdicts / detector_latency_ms / detector_error_kind` — see `search-leak-filter-v1`) |
| `s{i}_error`                         | error classification code after retries are exhausted; NULL on normal completion (including refusal / parse fail)                                    |
| `s{i}_created_at`                    | UTC ISO-8601 at write time; the unique signal for "whether this sample slot has been filled"                                         |
| `s{i}_finish_reason`                 | the last round's `ChatCompletion.choices[0].finish_reason` (`stop` / `tool_calls` / `length` / `content_filter` …); NULL for error rows |
| `s{i}_nudges_used`                   | count of "strict floor not met → reminder injected" within this sample; capped by `REACT_MAX_NUDGES`                                  |
| `s{i}_step_metrics`                  | JSON array of each ReAct round; element keys `step / prompt / completion / reasoning / latency_ms / finish_reason / n_tool_calls`     |
| `s{i}_response_id`                   | last round's `ChatCompletion.id`                                                                                                       |
| `s{i}_system_fingerprint`            | last round's `ChatCompletion.system_fingerprint` (when the provider supplies it; used to detect provider-side model-routing changes)        |
| `s{i}_service_tier`                  | last round's `ChatCompletion.service_tier`                                                                                                |
| `s{i}_belief_final`                  | v4. JSON-serialised `Belief.probabilities` (`{letter: float}`) returned by `parser.parse_belief(content, q)` at the final step; NULL when parsing fails or `BELIEF_PROTOCOL=false` |
| `s{i}_belief_trace`                  | v4. JSON array of belief summaries for every loop step                                                                                  |
| `s{i}_belief_parse_ok`               | v4. whether the final-step belief parses legally (0/1); **independent** of `parse_ok`                                                  |
| `s{i}_final_answer_retry_used`       | v5. 0/1 — set when `REACT_FINAL_ANSWER_RETRY` mopped up an empty `final_raw`                                                            |

Three independent protocol fingerprints (paper §3.5; see DESIGN.md §5.6):

* `prompt_templates_hash` — main template (renderer $R$).
* `reflection_protocol_hash` — switch on the search-behaviour prior; varies along {on/off,
  text edits, version}.
* `belief_protocol_hash` — switch on whether the probabilistic-family metrics are populated.

Manifest top-level mirrors all three so "grep the protocol fingerprint without opening the DB"
covers every protocol.

### 5.3 Resume

Each sample slot is judged independently:

```sql
-- execute once per i ∈ 0..N-1:
SELECT question_id FROM run_results
 WHERE s{i}_created_at IS NOT NULL
   AND (s{i}_error IS NULL OR s{i}_error = 'skipped_training_cutoff');
```

Results are merged into `set[(question_id, sample_idx)]` and removed from the task queue.
Since each model's own DB contains only one run, `run_id` no longer enters the filter (the
single row in `run_meta` decides it).

State classification:

| `error` value                    | Meaning                            | Retry on next resume?                              |
| -------------------------------- | ---------------------------------- | -------------------------------------------------- |
| `NULL`                           | completed normally                 | no                                                 |
| `'skipped_training_cutoff'`      | actively excluded by §3.2         | no                                                 |
| `'network'` / `'server_5xx'`     | still failing after exhausted backoff | yes                                             |
| `'bad_request'`                  | model_not_found, etc.              | yes (after config change)                          |
| `'content_policy'`               | provider refusal                    | optional: default retry once and overwrite the original row |

Rules:
* Re-running with the same `run_id` = resume; writes into the existing
  `runs/{run_id}/db/<slug>.db`.
* Changing `run_id` = a fresh run; creates a new `runs/{new_run_id}/` directory.
* Overwrite semantics are backed by `INSERT ... ON CONFLICT(question_id) DO UPDATE SET s{i}_*
  = excluded.s{i}_*`; `user_prompt` is preserved with `COALESCE` to keep the first sample's
  value.

### 5.4 Concurrent-write strategy

* Every DB connection executes PRAGMA `journal_mode=WAL / foreign_keys=ON /
  synchronous=NORMAL / busy_timeout=5000` at startup.
* **One async writer task per model**: the runner opens a `forecast_eval.db.AsyncWriter` for
  each model DB; every worker's result is enqueued via the writer for that model.
* The writer task flushes every `DB_COMMIT_BATCH` entries or every 1 second, with short
  transactions; SQLite writes go through `await asyncio.to_thread(...)` to avoid blocking the
  event loop.
* A single-model DB has only one writer and multiple readers; under WAL, concurrency is
  sufficient.
* If switched to cross-thread consumption, `queue.Queue` / `janus.Queue` is required;
  `asyncio.Queue` is not cross-thread safe.

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
├── evaluation.py                  # main entry: parse CLI flags -> runner.run -> analysis.run_analysis
├── forecast_eval_set_example.db   # source data (read-only, **checked into Git**)
├── runs/                          # root for all evaluation outputs (gitignored)
│   └── {run_id}/
│       ├── manifest.json          # run-level metadata + model_files mapping + grid block
│       ├── db/
│       │   └── {model_slug}.db    # one sqlite per model; self-contains questions + prompt_templates copies
│       ├── analysis/              # statistical artefacts generated after the run
│       └── logs/{run_id}.log
├── forecast_eval/
│   ├── __init__.py
│   ├── config.py                  # pydantic-settings; Settings + RUNS_ROOT + MODEL_TRAINING_CUTOFFS parsing
│   ├── db.py                      # per-model wide-table schema + AsyncWriter + hash / redaction
│   ├── loader.py                  # syncs questions + prompt_templates from SOURCE_DB into each DB
│   ├── prompts.py                 # renderer R (renders user message per question_type)
│   ├── llm.py                     # OpenAI-compatible client + tiered retry (provider-native browsing forbidden)
│   ├── search.py                  # Tavily + end_date injection + Stage-2 detector dispatch
│   ├── leak_filter.py             # Stage-2 LLM detector H_aux (independent client, fail-closed)
│   ├── tavily_keys.py             # multi-key TavilyKeyPool (least-used + 401/403 blacklist + 429 cooldown)
│   ├── tools.py                   # web_search schema (LLM-visible part, no date)
│   ├── react.py                   # ReAct loop F_M (single sample)
│   ├── parser.py                  # parser Ψ + label normalisation φ + strict matching
│   ├── errors.py                  # error classification + backoff strategy (includes skipped_training_cutoff)
│   ├── runner.py                  # task orchestration + multi-model writer + κ_M filtering
│   ├── types.py                   # dataclass definitions (Question / SampleResult / etc.)
│   └── analysis/                  # post-hoc statistics (Γ); read DB → CSV / MD / JSON
│       ├── __init__.py            #   `run_analysis(run_dir)` entry
│       ├── accuracy.py            #   strict-equality accuracy + pass@k family
│       ├── exam_score.py          #   exam-style partial credit
│       ├── composite.py           #   subtype-weighted composite accuracy
│       ├── consistency.py         #   Cohen κ, Fleiss κ, mean entropy, VCI, MVG
│       ├── proper_score.py        #   BI / NLL / MBS / ABI (probabilistic companion)
│       ├── aggregation.py         #   arithmetic / logit-space mean / LOO shrinkage
│       ├── inference.py           #   paired bootstrap, Holm-Bonferroni, posterior
│       ├── grid.py                #   grid-search analysis (virtual slug decode)
│       ├── behavior.py            #   reflection A/B, tool-usage PDP, confidence joint
│       ├── probabilistic.py       #   probabilistic family report builder
│       └── …
└── tests/                         # unit tests (§14)
    ├── test_prompts.py
    ├── test_parser.py
    ├── test_search.py
    ├── test_leak_filter.py
    ├── test_db.py
    ├── test_errors.py
    ├── test_llm_no_browsing.py
    ├── test_runner_resume.py
    ├── test_training_cutoff.py
    ├── test_analysis.py
    └── test_smoke_dry_run.py
```

---

## 7. `.env` configuration reference

A condensed view of the most load-bearing knobs; for the full annotated block see
`.env.example`.

```ini
# -------- LLM Endpoint (OpenAI-compatible) --------
LLM_API_KEY=REPLACE_ME
LLM_BASE_URL=https://openrouter.ai/api/v1                 # any OpenAI-compatible endpoint
MODELS=openai/gpt-5,anthropic/claude-sonnet-4.5,google/gemini-2.5-pro,deepseek/deepseek-r1
# κ_M per model — declare for every evaluated model
MODEL_TRAINING_CUTOFFS=openai/gpt-5=2024-10-01,anthropic/claude-sonnet-4.5=2025-03-01,...

# LLM call parameters
LLM_MAX_TOKENS=12000           # covers reasoning + completion (3-8k reasoning tokens common)
LLM_TIMEOUT_S=240
LLM_TEMPERATURE=0.7
LLM_TOP_P=1.0
# Reasoning-model substring list: matched models call **without** temperature / top_p
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
TAVILY_API_KEY=tvly-REPLACE_ME      # single value or CSV multi-value (multi-key pool)
TAVILY_KEY_COOLDOWN_S=60             # 429 cooldown for a single key
TAVILY_MAX_RESULTS=5                 # R axis (paper main: R_tav = 5; multi-value → grid search)
TAVILY_SEARCH_DEPTH=basic            # basic (1 credit) | advanced (2 credits, higher recall)
TAVILY_INCLUDE_RAW_CONTENT=markdown  # false | markdown (default) | text
TAVILY_RAW_CONTENT_MAX_CHARS=8000    # per-result raw_content truncation
TAVILY_INCLUDE_ANSWER=false          # off by default (avoid second-LLM contamination)
TAVILY_END_DATE_OFFSET_DAYS=-1       # δ; project default -1 (strict)
SEARCH_MAX_CONCURRENCY=5
SEARCH_RETRY_MAX=3
SEARCH_BACKOFF_S=2,5,15

# -------- ReAct Loop --------
REACT_MAX_STEPS=12                   # T (paper main: 12)
REACT_MAX_SEARCH_CALLS=8             # C axis (paper main: C = 4; multi-value → grid search)
REACT_REFLECTION_PROTOCOL=true       # decompose → ≥3 angles → reflect → cross-validate → opposite check → confidence
REACT_BUDGET_AWARENESS_PROTOCOL=true # front-load total step / search count
REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=true
REACT_FORCE_FINAL_ANSWER_LOOKAHEAD=2 # soft reminder at second-to-last + hard switch on last
REACT_MIN_SEARCH_CALLS=0             # soft floor (off by default; rules are last resort)
REACT_MAX_NUDGES=2

# v5.1 harness-resilience
REACT_FINAL_ANSWER_RETRY=false       # mop up empty final_raw with a tools=[] retry
REACT_BUDGET_EXCEEDED_DROP_TOOLS=true # drop tool schema once C is hit

# -------- Search Leak Filter (Stage-2 detector) --------
ENABLE_SEARCH_LEAK_FILTER=true       # pair with LEAK_DETECTOR_API_KEY/MODEL
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

# -------- Composite score weights (composite-score-by-subtype) --------
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

### 7.1 Redaction

Before writing `run_meta.config_snapshot`, `config.py` MUST redact sensitive fields. The
redaction format keeps the first 4 chars + length + `sha256[:12]` for each secret;
`TAVILY_API_KEY` is `list[str]` and is persisted as `[{prefix, sha256_12, length, provider},
…]`, for later auditing of "which keys this run used". Sensitive plaintext is **never**
persisted.

---

## 8. Core module responsibilities

| Module                  | Responsibility                                                                                                  | Key interfaces                                                                                                 |
| ----------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `config.py`             | reads `.env` via `pydantic-settings`, validates types, parses comma-separated lists, validates `MODELS` for `:online`, validates `COMPOSITE_WEIGHTS_*` | `Settings` class (singleton)                                                                                          |
| `loader.py`             | syncs two tables from `SOURCE_DB` into the per-model DB: ① `<SOURCE_TABLE>` → `questions` (filtered by filters); ② `dataset_metadata.features_json.prompt_reconstruction` → `prompt_templates` (key/value flat) | `sync_questions(source_db, conn, filters, table=...) -> list[Question]`, `sync_prompt_templates(source_db, conn) -> dict[str,str]` |
| `prompts.py`            | implements renderer $R$ per `question_type`: ① generates `outcomes_block` (multiple_choice enumerates options via the §3.7 letter rule); ② selects one of the three `output_format`s; ③ assembles the final text; ④ optional protocol additions (reflection / budget / belief) | `render_user_prompt(q, templates, reflection_protocol=None, budget_awareness=None, belief_protocol=None) -> str`     |
| `tools.py`              | defines the `web_search` OpenAI-schema; **the LLM-visible part contains no date**                                                       | `WEB_SEARCH_SCHEMA`, `execute_tool_call(tc, q, cfg)`                                                           |
| `search.py`             | wraps Tavily `/search`, injects `end_date = q.end_time + δ`; controls page-body form per `TAVILY_INCLUDE_RAW_CONTENT` and truncates per `TAVILY_RAW_CONTENT_MAX_CHARS`; dispatches Stage-2 detector | `tavily_search(query, end_date, settings) -> SearchResult`                                                     |
| `leak_filter.py`        | runs the auxiliary detector $H_{\mathrm{aux}}$: per-result `keep` / `drop` verdict; whitelisted input fields; fail-closed by default | `audit_results(items, cutoff_date, settings) -> list[Verdict]`                                                  |
| `tavily_keys.py`        | multi-key pool with least-used scheduling, 401/403 permanent blacklist, 429 cooldown                            | `TavilyKeyPool.acquire / release / mark_failure`                                                                |
| `llm.py`                | OpenAI-compatible client; tiered retry by error kind; **forces no provider-native browsing** (no `plugins`, no `:online` suffix, no provider-private web-tool fields) | `chat(model, messages, tools, ...) -> ChatResponse`                                                            |
| `react.py`              | implements forecasting system $F_M$: one ReAct inference = one sample; loops until no tool_call or limits hit                                                        | `run_react(q, model, sample_idx, cfg) -> SampleResult`                                                         |
| `parser.py`             | implements parser $\Psi$ and normalisation $\phi$: parse `\boxed{...}` per `question_type` → letter `frozenset[str]`; strict frozenset equality against the letter set parsed from `q.answer` | `parse_answer(text, q) -> frozenset[str] \| None`, `parse_gt(answer) -> frozenset[str]`, `is_correct(pred, gt) -> bool` |
| `errors.py`             | maps httpx/openai exceptions to error classifications; gives wait-seconds                                                                | `classify(exc) -> ErrorKind`, `backoff_seconds(kind, attempt)`, `CONTENT_POLICY_NEEDLES` constant               |
| `db.py`                 | connection management, WAL + PRAGMA, **per-model wide-table schema dynamic generation** (`init_schema(conn, sampling_n)` creates `s{i}_*` columns), `register_run_meta` / `finish_run_meta`, `AsyncWriter` UPSERT by `(question_id, sample_idx)`, `load_completed_samples`, source/metadata/templates hash computation, config redaction, model-slug safe-ification | `init_schema`, `AsyncWriter.enqueue_result`, `load_completed_samples`, `register_run_meta`, `model_slug_safe`, `compute_*_hash` |
| `runner.py`             | task orchestration: Cartesian product → dedup (per-model completed set) → **filter via `MODEL_TRAINING_CUTOFFS` and write skipped_training_cutoff rows into the corresponding model DB** → asyncio concurrency → progress log → cleanup `finish_run_meta` | `run(settings, filters, questions, templates, run_id, conns: dict[model, sqlite3.Connection]) -> RunStats`, `build_task_plan(...)` |
| `analysis/__init__.py`  | implements aggregation rule $\Gamma$: scans `runs/{run_id}/db/*.db` → computes all metrics in §11 → writes `analysis/` CSV / MD / JSON. **Does not modify the DB.** Auto-invoked by `evaluation.py`, or invokable independently via `python -m forecast_eval.analysis runs/{run_id}` | `run_analysis(run_dir: Path) -> list[Path]` |

`QFilter` is a dataclass containing `question_types: set[str] | None` and
`choice_types: set[str] | None`, where `None` means no filtering.

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
        lines = [f"{chr(ord('A') + i)}. {label}" for i, label in enumerate(options)]
        outcomes_block = "\n" + "\n".join(lines)
        output_format = templates["multiple_choice_output_format"]

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
    for protocol in (reflection_protocol, budget_awareness, belief_protocol):
        if protocol:
            body += "\n\n" + protocol
    return body
```

### 8.2 `parser.parse_answer` reference implementation

```python
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")

def _index_to_letter(i: int) -> str:
    return chr(ord("A") + i)

def _letter_to_index(L: str) -> int:
    return ord(L) - ord("A")

def parse_answer(text: str, q: Question) -> frozenset[str] | None:
    matches = BOXED_RE.findall(text or "")
    if not matches:
        return None
    payload = matches[-1].strip()                       # take the last \boxed{...}

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
                return frozenset({_index_to_letter(i)})
        return None

    if q.question_type == "multiple_choice":
        # split on comma or whitespace, drop empties
        tokens = [t.strip() for t in re.split(r"[,\s]+", payload) if t.strip()]
        opts_n = len(json.loads(q.options))
        letters = set()
        for t in tokens:
            if len(t) != 1:
                return None
            idx = _letter_to_index(t)
            if not (0 <= idx < opts_n):
                return None
            letters.add(t)
        return frozenset(letters) if letters else None

    return None

def parse_gt(answer: str) -> frozenset[str]:
    return frozenset(t.strip() for t in answer.split(",") if t.strip())

def is_correct(pred: frozenset[str], gt: frozenset[str]) -> bool:
    return pred == gt
```

---

## 9. Error tiering & backoff strategy

All exceptions are routed by the table below:

| Error type                              | Identification                                                                  | Handling strategy                                                                  |
| --------------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Network / Timeout**                   | `httpx.ConnectError`, `httpx.ReadTimeout`, `asyncio.TimeoutError`, `RemoteProtocolError`, `WriteError`, `WriteTimeout`, `PoolTimeout` | use `LLM_BACKOFF_NETWORK_S`; on still-failure → `error="network"` skip               |
| **Rate Limit (429)**                    | HTTP 429                                                                        | prefer the `Retry-After` header; otherwise `LLM_BACKOFF_RATE_LIMIT_S`               |
| **Server 5xx**                          | HTTP 500/502/503/504                                                            | use `LLM_BACKOFF_SERVER_5XX_S`; on exhaustion → `error="server_5xx"` skip            |
| **Auth (401/403)**                      | HTTP 401/403                                                                    | **fail immediately, stop the whole run**                                            |
| **Bad Request (400)**                   | HTTP 400 + `model_not_found` / `invalid_request`                                | skip immediately, `error="bad_request"`                                              |
| **Content Policy**                      | HTTP 400 + match against `errors.CONTENT_POLICY_NEEDLES` (`content_policy / content_filter / safety / content_policy_violation / data_inspection_failed / inappropriate content / sensitive`) | **no retry**, `error="content_policy"`, `parse_ok=0`, `correct=NULL`               |
| **LLM soft refusal**                    | normal return but `\boxed{...}` not found or parsed `frozenset` empty           | not an error, `parse_ok=0`, `correct=NULL`                                          |
| **Exceed `REACT_MAX_STEPS`**            | ReAct loop exhausted without producing a final answer                            | not an error, `parse_ok=0`, `correct=NULL`                                          |
| **Tool arguments JSON parse fails**     | the LLM's arguments are not legal JSON                                          | tell the LLM the error and continue the loop (non-fatal)                             |
| **Tavily error itself**                 | retry independently via `SEARCH_BACKOFF_S`; on exhaustion, feed the error to the LLM as tool_result | the LLM can choose to retry or give up                                              |
| **Detector error (Stage-2)**            | retry via `LEAK_DETECTOR_BACKOFF_S`; AUTH errors immediate-fail-closed         | on `LEAK_DETECTOR_FAIL_ACTION=drop` (default) → drop the item; `keep` → pass through |
| **Training-data contamination filter** | detected during task generation: `q.end_time <= κ_M` (see §3.2)               | **does not invoke the LLM**, directly writes `error="skipped_training_cutoff"`     |

### 9.1 Key boundaries

1. **Auth errors stop the whole run.** Continuing to burn budget on a wrong key is meaningless;
   early-stop saves money.
2. **Content policy is not retried.** Re-sending the same question yields the same result.
   Mark it directly, and tally how many each model was rejected on at the end.
3. **Refusal ≠ error.** The LLM returned a legal response but did not answer (missing boxed /
   letter outside the option range) — this is part of model capability, counted in statistics
   but not the error field.
4. **Tavily failure degrades to a tool_result error.** Let the LLM decide whether to retry the
   query or give up, without interrupting the whole sample.
5. **Detector failure is fail-closed by default.** Detector hiccups (timeout, network) are
   uncorrelated with item content; biasing the residual towards "drop on uncertainty" is the
   conservative choice.
6. **`skipped_training_cutoff` does not count toward error rate.** This is active data
   cleansing, not a model failure; reports tally "questions excluded / ratio" separately.

---

## 10. ReAct loop pseudocode (the forecasting system $F_M$)

```python
async def run_react(q: Question, model: str, sample_idx: int, cfg: Settings) -> SampleResult:
    # ① compute χ_i: invisible to the LLM
    end_date = (date.fromisoformat(q.end_time)
                + timedelta(days=cfg.TAVILY_END_DATE_OFFSET_DAYS)).isoformat()

    # ② render m_0 = R(q^in): single user message with all protocol additions
    user_prompt = prompts.render_user_prompt(
        q,
        cfg.PROMPT_TEMPLATES,
        reflection_protocol=prompts.REFLECTION_PROTOCOL if cfg.REACT_REFLECTION_PROTOCOL else None,
        budget_awareness=prompts.BUDGET_AWARENESS_PROTOCOL if cfg.REACT_BUDGET_AWARENESS_PROTOCOL else None,
        belief_protocol=prompts.BELIEF_PROTOCOL if cfg.BELIEF_PROTOCOL else None,
    )

    messages = [{"role": "user", "content": user_prompt}]
    search_calls: list[dict] = []
    final_raw = ""
    step_metrics: list[dict] = []
    nudges_used = 0
    final_answer_retry_used = 0
    t0 = time.monotonic()
    tokens = {"prompt": 0, "completion": 0, "reasoning": 0}
    step = 0

    for step in range(cfg.REACT_MAX_STEPS):
        # v5.1 D2 / force-final-answer-near-limit: drop tools as we approach the limit
        tools = [WEB_SEARCH_SCHEMA]
        if cfg.REACT_BUDGET_EXCEEDED_DROP_TOOLS and len(search_calls) >= cfg.REACT_MAX_SEARCH_CALLS:
            tools = []
        if cfg.REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT and step >= cfg.REACT_MAX_STEPS - 1:
            tools = []

        # Forced-finalisation soft reminder / hard cutover (see DESIGN.md §7.3)
        if cfg.REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT:
            messages = _maybe_inject_finalization_reminder(messages, step, cfg)

        resp = await llm.chat(
            model=model,
            messages=messages,
            tools=tools,
            temperature=cfg.LLM_TEMPERATURE,
            top_p=cfg.LLM_TOP_P,
            max_tokens=cfg.LLM_MAX_TOKENS,
            timeout=cfg.LLM_TIMEOUT_S,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_unset=True))
        _accumulate_tokens(tokens, resp.usage)
        step_metrics.append(_capture_step_metrics(step, resp))

        # No tool_call → LLM wants to give a final answer.
        if not msg.tool_calls:
            # Optional fallback: nudge if minimum search count not met.
            if _should_nudge(cfg, search_calls, nudges_used, step):
                messages.append(_build_nudge_message(cfg, search_calls))
                nudges_used += 1
                continue
            final_raw = msg.content or ""
            break

        # Handle every tool_call (OpenAI supports parallel).
        for tc in msg.tool_calls:
            if tc.function.name != "web_search":
                messages.append(_tool_error(tc, "unknown tool"))
                continue
            if len(search_calls) >= cfg.REACT_MAX_SEARCH_CALLS:
                messages.append(_tool_error(tc, "search budget exceeded"))
                continue
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                messages.append(_tool_error(tc, f"invalid arguments JSON: {e}"))
                continue

            # ③ inject χ_i (invisible to the LLM); Stage-2 detector audits results.
            result = await search.tavily_search(query=args["query"], end_date=end_date)
            search_calls.append(result.to_search_call_record())  # includes detector_verdicts
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result.to_llm_payload()),
            })
    # exceeded the step count; final_raw stays empty → parser will mark parse_ok=0

    # v5.1 D1 final-answer retry: mop up empty final_raw with tools=[]
    if cfg.REACT_FINAL_ANSWER_RETRY and not final_raw:
        messages.append({
            "role": "user",
            "content": "Time to commit. Output your final \\boxed{...} answer now without "
                       "further searches or tool calls.",
        })
        resp = await llm.chat(model=model, messages=messages, tools=[], **call_kwargs)
        final_raw = resp.choices[0].message.content or ""
        step_metrics.append(_capture_step_metrics(step + 1, resp))
        final_answer_retry_used = 1

    # ④ Ψ ∘ φ: parsing and scoring at the letter-set level
    parsed = parser.parse_answer(final_raw, q)              # frozenset[str] | None
    gt = parser.parse_gt(q.answer)                          # frozenset[str]
    correct = parser.is_correct(parsed, gt) if parsed is not None else None

    return SampleResult(
        run_id=cfg.RUN_ID,
        question_id=q.id,
        model=model,
        sample_idx=sample_idx,
        final_answer_letters=json.dumps(sorted(parsed)) if parsed is not None else None,
        final_answer_raw=final_raw,
        correct=int(correct) if correct is not None else None,
        parse_ok=1 if parsed is not None else 0,
        tool_calls_count=len(search_calls),
        react_steps=step + 1 + final_answer_retry_used,
        prompt_tokens=tokens["prompt"],
        completion_tokens=tokens["completion"],
        reasoning_tokens=tokens["reasoning"],
        latency_ms=int((time.monotonic() - t0) * 1000),
        user_prompt=user_prompt,
        messages_trace=json.dumps(messages) if cfg.WRITE_MESSAGES_TRACE else None,
        search_calls=json.dumps(search_calls),
        nudges_used=nudges_used,
        step_metrics=json.dumps(step_metrics),
        finish_reason=resp.choices[0].finish_reason,
        response_id=resp.id,
        system_fingerprint=resp.system_fingerprint,
        service_tier=resp.service_tier,
        final_answer_retry_used=final_answer_retry_used,
        error=None,
        created_at=utcnow_iso(),
    )
```

---

## 11. Evaluation metrics (the aggregation rule $\Gamma$)

Metrics are **computed entirely by `forecast_eval.analysis` after the run finishes**, not
stored in the DB. Artefacts land in `runs/{run_id}/analysis/` (CSV / MD / JSON). The
definitions below match the source implementation and the paper's notation.

### 11.1 Validity ($\mathcal{E}^{\mathrm{valid}}$, paper §3.2)

$v_{i,M} = \mathbb{1}[\Psi_i(o_{i,M}) \ne \bot]$ — whether the model's raw output yields a
parseable letter set (`parse_ok` in the DB).

| Metric                          | Definition                                                                              | Notes                    |
| ------------------------------- | --------------------------------------------------------------------------------------- | ----------------------- |
| **parse_failure_rate**          | $1 - \mathbb{E}[v_{i,M}]$ over the scorable set                                          | reflects format adherence / refusal rate |
| **final_answer_retry_rate**     | share of samples where v5.1 D1 mopped up an empty `final_raw`                            | how much the fallback caught       |

### 11.2 Item-level ($\mathcal{E}^{\mathrm{item}}$, paper §3.2.4)

A `(question_id, model)` has $n$ samples ($n=$ `SAMPLING_N`). When tallying, **first exclude**
rows with `s{i}_error="skipped_training_cutoff"` (these are excluded questions, not the model
getting them wrong).

* **Strict equality** $r_{i,M} = \mathbb{1}[\widehat{G}_{i,M} = G_i]$ — `s{i}_correct` in the
  DB.
* **Exam-style partial credit** — the project's *headline* per-sample score for the composite.
  $\text{exam-score}(\hat S, G) = \lvert\hat S \cap G\rvert / \lvert G\rvert$ when $\hat S
  \setminus G = \varnothing$, else 0. Single-answer questions degenerate to strict 0/1.

### 11.3 Question-level ($\mathcal{E}^{\mathrm{question}}$, paper §3.2.5–3.2.6)

| Metric                          | Definition                                                                              | Notes                    |
| ------------------------------- | --------------------------------------------------------------------------------------- | ----------------------- |
| **pass@1 avg** ($\passone$)     | `mean(correct over n samples)`                                                          | reflects stable capability        |
| **pass_any@N**                  | $\mathbb{1}[\exists s: c_{q,s}=1]$                                                      | best-of-N upper bound (standard `pass@k`) |
| **pass_all@N** ($\passall$)     | $\prod_s c_{q,s}$ — all correct                                                          | repeated-consistency lower bound  |
| **at_least_k_correct@N**        | $\mathbb{1}[\sum_s c_{q,s} \ge k]$                                                      | threshold analysis        |
| **majority vote correct**       | majority-vote on N `final_answer_letters` (as frozensets), then compared with `q.answer` | self-consistency metric   |
| **exam_score_at_n_avg**         | $e_q = \frac{1}{|S_q|}\sum_s \text{exam-score}$ (scored samples only; parse failures count as 0) | exam-style partial credit per question |

### 11.4 Model-level ($\mathcal{E}^{\mathrm{model}}$, paper §3.3)

| Metric                          | Definition                                                                              | Notes                    |
| ------------------------------- | --------------------------------------------------------------------------------------- | ----------------------- |
| **Composite Accuracy**          | subtype-weighted average of $\examavg^{(b)}$ (paper Eq. 18)                             | headline score; default weights yes_no=0.15 / binary_named=0.15 / multiple_choice=0.70 |
| **avg tool_calls / react_steps / latency_ms / tokens** | mean over scored samples                                                  | cost & strategy axes      |
| **error rate by kind**          | percentages by `error` classification (excluding `skipped_training_cutoff`)             | reflects stability        |
| **training_cutoff_skip rate**   | `count(error='skipped_training_cutoff') / count(*)` per model                           | how many questions excluded |
| **avg_nudges_used**             | `mean(nudges_used)` over eligible samples                                                | "strict-floor trigger rate" |
| **finish_reason_breakdown**     | per-model `Counter[finish_reason]` over eligible samples                                  | spot abnormal `length` / `content_filter` proportions |
| **per-correct cost** ($C^{\text{per-correct}}_m$) | $C^{\text{total}}_m / (\lvert\mathcal{D}^{\text{eval}}\rvert \cdot n \cdot \text{Composite\,Acc}_m)$ | difficulty-weighted USD per correct prediction |

### 11.5 Discrete-native family (v5 main line)

`forecast_eval/analysis/accuracy.py` + `consistency.py`:

| Metric                          | Formula                                                                                  | Notes                    |
| ------------------------------- | ---------------------------------------------------------------------------------------- | ----------------------- |
| **FSS** (Format Skill Score)    | Tversky $(\alpha=2, \beta=0.5)$ → per-question mean → chance correction → cross-question mean | primary metric; multi-select wrong cost = 4× missed; single-select degenerates to strict 0/1 |
| **Cohen's κ**                   | $(\mathrm{acc} - p_e)/(1 - p_e)$, single $p_e = 1/k_q$ / multi $p_e = 0.5$              | chance correction for strict 0/1                          |
| **Hamming Score**               | $1 - \tfrac{1}{k_q}\sum_l\lvert\hat{y}_l - o_l\rvert$                                   | partial credit for multi-select (NULL on pure single-select runs) |
| **Fleiss' κ**                   | $(\bar{P}-\bar{P}_e)/(1-\bar{P}_e)$ on the $K$-trial vote matrix                         | inter-trial consistency (K-trial exclusive) |
| **Predictive entropy** $H_q$    | single: $-\sum_l \hat{p}_l \log_2 \hat{p}_l$; multi: per-label binary-entropy mean       | per-question uncertainty |
| **Entropy-accuracy joint**      | per-model tertile buckets → per-bucket Acc / MV Acc / Fleiss κ                            | "how does the model perform on high-entropy questions vs low-entropy ones" |
| **VCI**                         | $\max_l n_{q,l}/K$ cross-question mean                                                   | vote concentration       |
| **MVG**                         | MV_Acc - pass@1 avg                                                                      | majority-vote signal gain (K-trial exclusive) |

### 11.6 Probabilistic family (v4 companion; demoted under K=5)

`forecast_eval/analysis/proper_score.py`:

| Metric                          | Formula                                                                                  | Applicable     |
| ------------------------------- | ---------------------------------------------------------------------------------------- | -------------- |
| **Brier Index (BI)**            | $100(1 - \sqrt{\overline{\text{BS}^{\text{lab}}}})$, **average first then square root**    | all qtypes     |
| **BI_dec**                      | decision-wise Brier index                                                                  | single only    |
| **NLL**                         | single: $-\log p_{q,l^*}$; multi: label-wise BCE; clip $\epsilon = 10^{-3}$              | all qtypes     |
| **MBS**                         | $100(\log_2 p_{q,l^*} + 1)$, clip same as NLL                                            | single only    |
| **ABI (crowd / uniform)**       | sign-aware $100(1 \mp \sqrt{\lvert\overline{\text{ABS}}\rvert})$                          | crowd: multi-model run |
| **fallback share**              | share of questions that went through the §11.6.1 fallback                                  | all runs       |

> **K=5 disclaimer.** When SAMPLING_N is small (e.g. 5), the empirical probability $\hat p =
> n/K$ takes only 6 discrete values, and Reliability Diagram / Murphy three-decomposition /
> Platt LOO calibration become statistically meaningless. v5 deletes `calibration.py` and its
> 5 artefacts; the probabilistic columns retain the `†` footnote in `per_model_summary.md`. To
> reintroduce calibration, increase $K$ to ≥ 30.

#### 11.6.1 §2.4 fallback

When `s{i}_belief_final IS NULL` but `s{i}_parse_ok = 1` (legacy v3 runs, or v4 belief parse
failed but boxed parse succeeded), $p_l = 1-\epsilon$ (matched boxed letter) /
$\epsilon/(k-|\text{boxed}|)$ (others), $\epsilon = 0.05$. The sample goes through fallback
with `belief_parse_ok=0`. Samples with full failure (`parse_ok=0`) MUST NOT enter
probabilistic-metric averaging, to avoid pollution.

### 11.7 Statistical inference

`forecast_eval/analysis/inference.py`:

| Function                                  | Algorithm                                                                                | Output                                |
| ----------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------- |
| `paired_bootstrap(bs_a, bs_b)`            | $B=5000$ paired resampling (same indices index A and B)                                   | `delta_mean / ci_low / ci_high / p_two_sided` |
| `holm_bonferroni(p_values)`               | $(n-i) \cdot p_{(i)}$ then cumulative max                                                | adjusted p-values                     |
| `difficulty_tertile(gammas)`              | sort per-question $\gamma_q$, cut into tertiles                                          | `low / mid / high` buckets             |
| `paired_bootstrap_by_difficulty`          | independent paired bootstrap per tier                                                     | `{tier: PairedBootstrapResult}`         |
| `posterior_a_better_than_b(bs_a, bs_b)`   | Monte-Carlo $\Pr(\overline{BS}_A < \overline{BS}_B)$ on paired bootstrap                  | $\Pr(\mathrm{BI}_A > \mathrm{BI}_B) \in [0,1]$ |

Multi-comparison control: Holm-Bonferroni at the FWER level. The paired-bootstrap is
*same-indexed* — the same bootstrap draws the same question id to index both A's and B's
arrays — to control the question-level variance the paper §G.2 quantifies at 62% of total.

### 11.8 Composite-score-by-subtype (`composite-score-by-subtype`)

Module: `forecast_eval/analysis/composite.py`; writes:
`per_model_composite_by_question_type.csv` / `per_model_composite_by_choice_type.csv` /
`composite_meta.json`, and embeds a `composite` section in `overall.json`.

**Input collection** — three sources unified into `{model: {metric: {bucket: value}}}`:

| Source                                                                                  | Data shape                                | Column coverage                                             |
| --------------------------------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------- |
| `analysis._slice_by(samples, key_fn=question_type/choice_type, ...)`                     | `{model: {bucket: Aggregate}}`            | v3 accuracy + final_answer_retry_rate (23 columns total)     |
| `composite.slice_v5_metrics_by_bucket(...)`                                              | `{model: {bucket: V5SliceResult}}`        | v5 discrete family 8 columns                                |
| `probabilistic.build_probabilistic_report(...).per_model_by_qtype/_by_ctype`              | `{model: {bucket: ModelProbabilisticAggregate}}` | v4 probabilistic family 7 columns                       |

**Weighting formula** (paper Eq. 18):
$$\text{composite}_m = \frac{\sum_{b \in B_{\text{valid}}} w_{m,b} \cdot v_{m,b}}{\sum_{b \in B_{\text{valid}}} w_{m,b}}.$$

Missing buckets are dropped and renormalised; all None → composite = None; weights are not
required to be normalised. Misspelled metric names raise from `compute_composite` during the
analysis phase rather than "silently falling back to default".

### 11.9 Grid-search analysis (`react-tavily-grid-search`)

`Settings.TAVILY_MAX_RESULTS` (R) and `REACT_MAX_SEARCH_CALLS` (C) support multi-value lists.
`forecast_eval/analysis/grid.py` decodes the `{real}::r{R}::c{C}` virtual slug, re-aggregates,
and emits paper long tables:

| File                           | Content                                                                                       |
| ------------------------------ | --------------------------------------------------------------------------------------------- |
| `grid_summary.csv`             | per `(real_model, R, C)` 17-column main table: accuracy/BI/NLL + 95% CI + cost columns         |
| `grid_marginal_C.csv`          | scan along C with `R = default_r` fixed                                                       |
| `grid_marginal_R.csv`          | scan along R with `C = default_c` fixed                                                       |
| `grid_pareto.csv`              | one row per cell; `dominated_by` empty for Pareto-frontier cells, else lex-smallest dominator |
| `grid_winrate.csv`             | per `(real_model_a, real_model_b)` cross-(R,C) cell wins/ties + paired-bootstrap significant-cell count |

For legacy v4 runs (manifest without a `grid` block), `run_grid_analysis` early-exits and
writes no `grid_*.csv`; the plot flow also skips the grid figure family.

---

## 12. CLI and how to run

### 12.1 Commands

```bash
# Run the entire dataset
python evaluation.py

# Filter by question_type (repeatable)
python evaluation.py --question-type yes_no --question-type binary_named

# Filter by choice_type (repeatable)
python evaluation.py --choice-type single

# Combined filter (AND): only multi-select multiple_choice questions
python evaluation.py --question-type multiple_choice --choice-type multi

# Do not generate analysis/ at run end (raw DBs still land in db/)
python evaluation.py --skip-analysis

# Refresh analysis/ independently (does not modify the DB)
python -m forecast_eval.analysis runs/{run_id}
```

`--question-type` values: `yes_no` / `binary_named` / `multiple_choice`, repeatable; if not
passed = no restriction. `--choice-type` values: `single` / `multi`, repeatable; if not passed
= no restriction. All tunables other than `--skip-analysis` go through `.env`.

### 12.2 Flow

```text
1. argparse parses --question-type / --choice-type / --skip-analysis, assembling a QFilter
2. Settings() loads and validates .env (including MODEL_TRAINING_CUTOFFS + RUNS_ROOT)
3. Generate or reuse run_id → determine run_dir = RUNS_ROOT/{run_id}; create db/ /
   analysis/ / logs/
4. Compute source_db_hash / metadata_hash / prompt_templates_hash (and protocol hashes when
   the corresponding switches are on)
5. For each MODELS[i]:
   a. open conn = RUNS_ROOT/{run_id}/db/{safe_slug(model)}.db
   b. db.init_schema(conn, SAMPLING_N)  # dynamically create s{i}_* columns
   c. loader.sync_prompt_templates(src, conn) / loader.sync_questions(src, conn, filter)
   d. db.register_run_meta(conn, run_id=..., model=..., hashes=..., training_cutoff=...)
6. Write manifest.json (run_id, models, model_files, sampling_n, filters, hashes, started_at,
   reflection_protocol_hash, belief_protocol_hash, optional grid block)
7. runner.run(..., conns={model: conn, ...}) starts the asyncio event loop:
   a. For each model, db.load_completed_samples(conn, SAMPLING_N) becomes the resume baseline.
   b. Generate the Cartesian product: questions × MODELS × range(SAMPLING_N); subtract the
      resume set.
   c. §3.2 filter: (q, model, idx) with `q.end_time <= cutoff` are written directly as
      skipped_training_cutoff rows to the corresponding model's writer, not entering the LLM
      task queue.
   d. Remaining tasks: Semaphore-limited (one each for LLM / Search / Detector) concurrency.
   e. Each completion → routed to that model's writer → batch UPSERT into s{i}_* columns.
   f. One log line per completion: [x/xx] q=.. qt=.. ct=.. model=.. idx=.. correct=..
8. For each model, db.finish_run_meta(conn, run_id); finalise manifest.finished_at.
9. Unless --skip-analysis: call forecast_eval.analysis.run_analysis(run_dir), write analysis/.
```

---

## 13. Logging (`loguru`)

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

### 13.1 Progress printing

Format:

```text
12:03:44 | INFO    | [run=20260424-120344-a7k3] [5/1610] q=69566c13 qt=binary_named ct=single model=openai/gpt-5 sample=2/5 correct=True steps=4 tool_calls=3 latency=8421ms
```

* `[5/1610]` denominator = `len(questions_after_filter) × len(MODELS) × SAMPLING_N` (minus
  completed resume tasks).
* One line printed per sample completion.
* On error, print at `ERROR` level: `[x/xx] q=.. model=.. error=rate_limit retry_exhausted`.

---

## 14. Test plan (`tests/`)

A single evaluation is costly (entire dataset × number of models × N samples), so getting tests
stable first saves a lot of API spend. All tests run **offline** and **do not burn the API**:
Tavily / OpenRouter exist as fixtures or mocked stand-ins.

| Test file                   | Subject               | Key cases                                                                                                                                                                                                                |
| --------------------------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_prompts.py`           | `prompts.py` ($R$)    | ① snapshots of all three template renderings for `yes_no` / `binary_named` / `multiple_choice` (≤26 options); ② correct `binary_named` placeholder substitution; ③ accurate `outcomes_block` labels for `multiple_choice` > 26 options (using fixtures from real questions in the DB); ④ protocol additions appear when their switches are on, and `prompt_templates_hash` is invariant under protocol toggles                  |
| `test_parser.py`            | `parser.py` ($\Psi$, $\phi$) | ① positive `\boxed{}` paths for all three question types; ② multiple `\boxed{}` → take the last one; ③ mixed case, spaces, comma/space separators; ④ illegal letter out-of-range; ⑤ > 26 options label↔letter round-trip; ⑥ `parse_gt` parses `"A, B"`; ⑦ soft refusal → None without raising               |
| `test_search.py`            | `search.py` + `tools.py` | ① `web_search` schema LLM-visible fields **do not contain** `end_date`; ② `tavily_search` injects `end_date = q.end_time + δ`; ③ Tavily errors retry via `SEARCH_BACKOFF_S`; ④ after retry exhaustion, returns an error payload instead of throwing; ⑤ `_build_request_payload` maps the three `TAVILY_INCLUDE_RAW_CONTENT={false,markdown,text}` / `TAVILY_SEARCH_DEPTH` / `TAVILY_INCLUDE_ANSWER` enum values to the Tavily protocol form; ⑥ overly long `raw_content` is truncated to `TAVILY_RAW_CONTENT_MAX_CHARS` at `_truncate_raw_content` with an ellipsis marker; ⑦ `to_llm_payload` does not output `null` placeholders for missing fields |
| `test_leak_filter.py`       | `leak_filter.py` ($H_{\mathrm{aux}}$) | ① detector input fields whitelist enforced (no `Question` fields leak in); ② fail-closed on retry exhaustion; ③ AUTH errors immediate-fail-closed without propagation; ④ `search_calls` JSON entry contains `n_results_raw / n_results_kept / detector_verdicts / detector_latency_ms / detector_error_kind`; ⑤ disable path is byte-equivalent to v5.1 |
| `test_db.py`                | `db.py`               | ① per-model schema dynamically creates `s{i}_*` columns per `sampling_n` + PRAGMA; ② fail-fast on schema `N` mismatch; ③ `model_slug_safe` rules; ④ hash computation stable; ⑤ `config_snapshot` redaction; ⑥ UPSERT overrides by `(qid, sample_idx)`; ⑦ `AsyncWriter` bucket-wise batched commits |
| `test_runner_resume.py`     | `runner.py`           | ① `load_completed_samples` excludes retryable errors; ② `build_task_plan` deduplicates by per-model completed; ③ models not declared in `completed` default to empty set (all enqueued)                                                                    |
| `test_training_cutoff.py`   | §3.2 admissibility    | ① every N samples for `q.end_time <= cutoff` writes skipped_training_cutoff; ② models without declared cutoff are not filtered; ③ resume takes precedence over cutoff; ④ after writing, `load_completed_samples` matches                                                       |
| `test_llm_no_browsing.py`   | `llm.py`              | mock client asserts the request payload **does not contain** `plugins`, `tools` does not contain provider-native web_search, and the model name does not end in `:online`                                                                                              |
| `test_errors.py`            | `errors.py`           | various `httpx` / OpenAI exceptions → correct `ErrorKind`; `Retry-After` header takes precedence over default backoff; v5.1 `CONTENT_POLICY_NEEDLES` widening covers Aliyun `data_inspection_failed`; v5.1 network family covers `RemoteProtocolError / WriteError / WriteTimeout / PoolTimeout` |
| `test_analysis.py`          | `analysis/__init__.py` ($\Gamma$) | ① hand-crafted wide-table fixture; ② pass@1 / pass_any@N / pass_all@N / majority_vote / parse_failure / error_rate / cutoff_skip values are correct; ③ `overall.json` aligns with the CSVs; ④ `error_breakdown.csv` aggregation; ⑤ exam_score / FSS / Cohen κ / Fleiss κ values |
| `test_smoke_dry_run.py`     | end-to-end dry-run    | replace OpenRouter + Tavily + detector with httpx stubs, run 3 questions × 1 model × 1 sample, verify the wide-table `s0_*` fields are complete, `messages_trace` is legal JSON, and `search_calls` records `end_date` + detector verdicts                                          |

Run:

```bash
pytest tests/ -q
```

CI minimum: `test_prompts.py` / `test_parser.py` / `test_training_cutoff.py` /
`test_llm_no_browsing.py` / `test_analysis.py` — these five must stay green. They map
one-to-one to the four components of $\mathcal{R}$ that, if broken, invalidate the entire run
unit: $R$ (`test_prompts`), $\Psi$ + $\phi$ (`test_parser`), the $\kappa_M$ admissibility
filter (`test_training_cutoff`), the information barrier (`test_llm_no_browsing`), and $\Gamma$
(`test_analysis`).

---

## 15. Conda environment (`environment.yml`)

```yaml
name: forecast
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - pip:
      - openai>=1.50            # OpenAI-compatible SDK (used by main LLM + detector)
      - tavily-python>=0.5
      - pydantic>=2.6
      - pydantic-settings>=2.2
      - python-dotenv>=1.0
      - loguru>=0.7
      - httpx>=0.27
      - tenacity>=9.0           # retry decorator
      - pytest>=8.0
      - pytest-asyncio>=0.23
      - respx>=0.21             # httpx mocking, for LLM / Tavily / detector dry-run
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
dependency-light; install it locally only if you want to render the on-demand plot family
(see README §10).

---

## 16. Final-premise summary (for last review)

1. **Source data has 7 fields**: `id / choice_type / question_type / event / options / answer
   / end_time`, with letter-encoded answers throughout.
2. **Source DB is checked into Git** (read-only example dataset, ensures `source_db_hash`
   reproducibility; `SOURCE_DB` / `SOURCE_TABLE` can point to a custom dataset).
3. **The LLM does not see $\chi_i$**; injection happens at the tool implementation layer.
4. **Tavily `end_date = end_time + δ`**; the project uses $\delta=-1$ as the strict default
   baseline (all reports default to comparison under $\delta=-1$).
5. **Sample admission** via $\kappa_M$: samples with `q.end_time ≤ cutoff` are written as
   `error="skipped_training_cutoff"`, do not call the LLM, and are not retried on resume.
6. **Three-layer leakage barrier**: tool-level `end_date` filter (algorithmic) + Stage-2 LLM
   detector (semantic, fail-closed) + manual curation (upstream). Audited residual ≈ 1.1%.
7. **Provider-native browsing forbidden**: no `:online`, no `plugins`, no provider-private web
   tool — code-layer enforced and test-layer guarded.
8. **Prompt assembly** is performed by `prompts.py`: pull templates from `dataset_metadata` →
   render `outcomes_block` and `output_format` per `question_type`. Protocol additions
   (reflection / budget / belief) live as runtime slots; `prompt_templates_hash` is invariant
   under protocol toggles, but the rendered full user message lands in each sample's
   `user_prompt` field.
9. **Three independent protocol fingerprints** (`prompt_templates_hash` /
   `reflection_protocol_hash` / `belief_protocol_hash`) enable three-axis ablation studies.
10. **Evaluation = letter-set frozenset strict equality**; missed and extra selections are all
    wrong. Soft-penalty companions (exam-score, FSS) coexist for analysis.
11. **Parse failure ≠ error**; refusal / format_failure rate are tallied separately.
12. **Multi-model single-run Cartesian product**, with resume via `run_id`; one DB per model;
    `run_meta` records `filters_snapshot` + four hashes + `training_cutoff` + redacted
    `config_snapshot`.
13. **Auth errors stop the entire run**; other errors are retried with tiered backoff.
14. **Content policy violations are not retried**, just marked.
15. **All flexible parameters live in `.env`**; CLI exposes only `--question-type` /
    `--choice-type` / `--skip-analysis`.
16. **Main entrypoint `evaluation.py`**: creates `RUNS_ROOT/{run_id}/`, runs the runner, runs
    analysis (unless `--skip-analysis`).
17. **Conda + Python 3.12 + loguru**, with progress `[x/xx]` logged.
18. **SQLite WAL + `PRAGMA foreign_keys=ON` + one async writer task per model**, avoiding
    concurrent-write lock contention.
19. **Each model DB is self-contained**: built-in `questions` + `prompt_templates` copies +
    `run_meta`, independently distributable and replayable.
20. **Metric naming**: the standard `pass@k` corresponds to this project's `pass_any@N`; the
    legacy threshold-style metric is renamed `at_least_k_correct@N`.
21. **Recording and analysis are separated**: the DB stores only raw sample records; pass@1 /
    pass_any@N / FSS / Cohen κ / Fleiss κ / BI / per-correct cost etc. are computed post-hoc
    by `forecast_eval.analysis` and written to `analysis/` as CSV / MD / JSON.
22. **Grid search via virtual slug**: `(real_model, R, C)` encoded as
    `{real}::r{R}::c{C}`; runner / DB / main analysis pipeline are byte-unchanged.
23. **Composite accuracy** is the headline scoring; default subtype weights follow the
    "harder questions discriminate better" principle (yes_no / binary_named / multiple_choice
    = 0.15 / 0.15 / 0.70; single / multi = 0.40 / 0.60); per-metric overrides supported.
24. **Per-correct cost** ($C^{\text{per-correct}}_m$) is the official cost-effectiveness axis;
    invoice-based amortisation across difficulty-weighted notional correct count places
    "expensive but accurate" and "cheap but reckless" on the same scale.

---

## 17. Suggested module landing order

For a from-scratch reimplementation, this order keeps each step locally verifiable:

1. `environment.yml` + `.env.example` + `.gitignore`.
2. `forecast_eval/config.py` (Settings class, with `RUNS_ROOT` and all subtype-weight
   validators).
3. `forecast_eval/db.py` (per-model wide-table schema + `AsyncWriter` + resume queries +
   prompt_templates table + `model_slug_safe`).
4. `forecast_eval/loader.py` (sync questions + prompt_templates).
5. `forecast_eval/prompts.py` ($R$ — render user message per question_type, **with unit tests
   covering all three types and protocol toggles**).
6. `forecast_eval/parser.py` ($\Psi$, $\phi$ — `\boxed{}` parsing + letter-set normalisation +
   strict matching, **with unit tests covering all three types + edge cases**).
7. `forecast_eval/errors.py` (error classification + backoff).
8. `forecast_eval/search.py` (Tavily + `end_date` injection).
9. `forecast_eval/leak_filter.py` ($H_{\mathrm{aux}}$ — Stage-2 detector with whitelist + fail-closed).
10. `forecast_eval/tools.py` (schema + `execute_tool_call`).
11. `forecast_eval/llm.py` (OpenAI-compatible client + retry; `:online` ban).
12. `forecast_eval/react.py` ($F_M$ — single-sample ReAct loop).
13. `forecast_eval/runner.py` (orchestration + concurrency + multi-model writer + $\kappa_M$
    filtering + progress).
14. `forecast_eval/analysis/*` ($\Gamma$ — post-hoc statistics, read DB → CSV / MD / JSON).
15. `evaluation.py` (main, create directories + register_run_meta + runner + analysis).

Get a smoke test passing first via `--question-type yes_no` + `MODELS=openai/gpt-4o-mini` +
`SAMPLING_N=1`, verify that `prompts.render_user_prompt` output and `parser.parse_answer`
normalisation are correct, then open up to full evaluation.

---

> **One sentence.** This codebase is the contract that turns the paper's run unit
> $\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)$ into a Python
> module per symbol, a SQLite column per observation, and a CSV column per metric — so that
> every number that ever appears in the report can be traced back to a row in a wide table, a
> hash in `run_meta`, or an audit verdict in `search_calls`.
