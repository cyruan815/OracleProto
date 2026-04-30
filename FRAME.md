# Forecast Evaluation — Project Framework

## 1. Project goal

Evaluate the LLM's ability on **forecasting-style single/multi-choice
questions** based on the `forecast_eval_set_example.db` dataset.

Core distinguishing feature: the in-house `web_search` tool restricts the
LLM's information-acquisition boundary — **the LLM is only allowed to find
information published before each question's `end_time` (the event
resolution date)** — simulating the "predict the future at the question's
point in time" scenario and preventing information leakage.

> Important caveat: tool-level time cutoff constrains only the **tool-search**
> information channel; leakage sources like the model's parametric memory,
> provider built-in browsing, and search-result snippets/caches cannot be
> blocked by the tool layer. The complete threat model and mitigations are
> in §3.8.

- 319 questions (`yes_no` 93 + `binary_named` 11 + `multiple_choice` 215),
  including 285 single-select + 34 multi-select
- Evaluates multiple models concurrently via OpenRouter's OpenAI-compatible
  API
- The LLM interacts with the `web_search` tool in ReAct + Tool Use mode
- Evaluation results are written into independent `results.db` files, with
  analysis performed independently afterwards

---

## 2. Data source

### 2.1 Source database `forecast_eval_set_example.db` (read-only)

> Note: the example dataset file shipped with the repo is named
> `forecast_eval_set_example.db`, and its main table is named
> `forecast_eval_set_example`. Both are configurable via `.env`'s
> `SOURCE_DB` / `SOURCE_TABLE` parameters; with a custom dataset, just keep
> the 7-column schema and `dataset_metadata` structure consistent.
> `SOURCE_TABLE` only accepts SQLite-legal identifiers
> `[A-Za-z_][A-Za-z0-9_]*` and is validated at startup.

Main table `forecast_eval_set_example`, **319 rows × 7 columns**:

| Field           | Type    | Description                                                                                                                          |
| --------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `id`            | TEXT PK | Unique question ID (sourced from HuggingFace)                                                                            |
| `choice_type`   | TEXT    | `single` \| `multi`, based on the letter count in `answer` (1 → `single`, >1 → `multi`)                                                  |
| `question_type` | TEXT    | `yes_no` \| `binary_named` \| `multiple_choice`, determining which prompt template set to use                                                       |
| `event`         | TEXT    | Event description (**without** options, **without** role-setting, **without** format requirements)                                                                  |
| `options`       | TEXT    | JSON array of strings. `yes_no`=`["Yes","No"]`; `binary_named`=two entity names; `multiple_choice`=labels in A/B/C... order            |
| `answer`        | TEXT    | Letter encoding: single-select `'A'`; multi-select `'A, B'` (comma + space separated). Letter ↔ option-index rule in §3.7                                            |
| `end_time`      | TEXT    | Event resolution date (Asia/Shanghai), `YYYY-MM-DD` format                                                                              |

Indexes (shipped with the example dataset; for custom datasets, follow the
`idx_<table>_<column>` naming for consistency):
`idx_forecast_eval_set_example_choice_type` /
`idx_forecast_eval_set_example_question_type` /
`idx_forecast_eval_set_example_end_time`.

Auxiliary table `dataset_metadata` (single row), contains `features_json`,
recording all prompt templates, column descriptions, and conversion logs.

### 2.2 Question count distribution

| question_type / choice_type | single | multi | total |
| --------------------------- | -----: | ----: | ---: |
| `yes_no`                    |     93 |     0 |   93 |
| `binary_named`              |     11 |     0 |   11 |
| `multiple_choice`           |    181 |    34 |  215 |
| **total**                   |  **285** | **34** | **319** |

Time range: `2026-01-15` ~ `2026-04-14`.
`multiple_choice` option count range: 3 ~ 35 (when > 26, letters enter the
ASCII continuation, see §3.7).

### 2.3 Examples

`yes_no`：
```
event:    "2026 a dream year for trump?"
options:  ["Yes","No"]
answer:   "B"           # B = No
end_time: "2026-01-31"
```

`binary_named`：
```
event:    "Golden Knights vs. Kings"
options:  ["Golden Knights","Kings"]
answer:   "A"           # A = Golden Knights
end_time: "2026-01-15"
```

`multiple_choice`（single）：
```
event:    "Bank of Brazil decision in January?"
options:  ["No change in the Selic rate ...",
           "the Bank of Brazil raise ...",
           "the Bank of Brazil lower ..."]
answer:   "A"
end_time: "2026-01-27"
```

`multiple_choice` (multi):
```
event:    "Oscars 2026: Achievement in Casting Nominations"
options:  [<12 nominee list entries>]
answer:   "A, B, D, E"
end_time: "2026-01-22"
```

> Key convention: the `event` field **does not contain** options or format
> requirements; those are spliced in at call time by the template (§3.6).
> `dataset_metadata.features_json.prompt_reconstruction` already stores all
> templates and concatenation rules — the loader reads them out and uses
> them, **do not hard-code them in source**.

---

## 3. Core design principles

### 3.1 The LLM does not see `end_date` (the most important safety boundary)

The schema the `web_search` tool exposes to the LLM has **only the
`query` parameter**:

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

When Tavily is actually called, the `end_date` parameter **is hard-coded
and injected by the tool implementation layer from the current question's
`end_time`** — the LLM cannot perceive it and cannot bypass it.

```python
def web_search(query: str, question_end_time: str) -> dict:
    end_date = (date.fromisoformat(question_end_time)        # already YYYY-MM-DD
                + timedelta(days=TAVILY_END_DATE_OFFSET_DAYS)).isoformat()
    return tavily_client.search(query=query, end_date=end_date, ...)
```

### 3.2 Strict time cutoff: `end_date = end_time - 1 day`

`end_time` is already at `YYYY-MM-DD` granularity. To avoid "same-day
information leakage" (many events/news are resolved the same day), the
default is `TAVILY_END_DATE_OFFSET_DAYS=-1` (the recommended strict
default; smaller values are more conservative):

```
question.end_time = 2026-01-18
→ Tavily end_date = 2026-01-17
```

This can be changed in `.env` to `0` (same day visible, more lenient) or
`-2`, `-3` (more conservative). The project uses `-1` uniformly as the
baseline, and all reports default to comparison under `-1`.

### 3.3 Source data is read-only; each run gets its own directory; each model gets its own DB

`forecast_eval_set_example.db` is never touched. Every invocation of
`python evaluation.py` creates an independent `{run_id}/` subdirectory
under `RUNS_ROOT` (default `./runs`), with the following structure:

```
{run_id}/
  manifest.json     # run-level metadata (run_id, sampling_n, models, filters, hashes...)
  db/<model_slug>.db  # one sqlite per evaluated model, self-contains questions + prompt_templates copies
  analysis/         # CSV / MD / JSON generated by forecast_eval.analysis after the run
  logs/{run_id}.log
```

The DB layer stores **raw records only** and performs no aggregation /
statistics. Metrics like pass@1, pass_any@N, majority, etc. are computed
separately by the post-hoc `analysis/` process and written back to disk
(see §5 / §11).

### 3.4 Strict-match scoring (at the letter-set level)

Scoring happens **entirely at the letter-set level**, independent of each
question type's output form:

- The DB `answer` field is a comma-separated letter string (`'A'` / `'A,
  B'`), split into `frozenset({'A'})` / `frozenset({'A','B'})`
- The LLM's `\boxed{...}` output is normalised by the parser per
  `question_type` into the same `frozenset[str]` (see §3.7)
- `frozenset == frozenset` is correct; missed selections / extra
  selections / ordering are all scored as "strict equality"

### 3.5 Parse failure is not an error

When the LLM does not output `\boxed{...}` or outputs a soft refusal like
"I cannot predict the future", **the retry path is not taken**; instead
`parse_ok=0`, `correct=NULL` is recorded, and the refusal rate is
separately accumulated as one dimension of model capability.

### 3.6 Prompt assembly (user message assembly)

The source DB **stores only raw material** (`event` / `options` /
`question_type` / `end_time`). When the system spawns a sample, it reads
the template from `dataset_metadata.features_json.prompt_reconstruction`
and assembles a complete user message per `question_type` to feed to the
LLM.

Template (`prompt_template`, stored in metadata):
```
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
- `yes_no` — requires `\boxed{Yes}` or `\boxed{No}`
- `binary_named` — template contains placeholders; after rendering looks like `\boxed{Golden Knights} or \boxed{Kings}`
- `multiple_choice` — requires `\boxed{A}` or `\boxed{B, C}`, with an example attached

How the `system` / `user` roles are split is decided by the runner (see
§10's simplified approach: as a single user message in its entirety —
the most faithful to the template).

### 3.7 Answer encoding and decoding (letter ↔ label)

The DB uniformly uses **letters** as the canonical answer, but the LLM's
output form varies by `question_type`:

| question_type      | LLM output (inside `\boxed{}`)                       | Parser normalisation target                                                    |
| ------------------ | ------------------------------------------------ | ------------------------------------------------------------------ |
| `yes_no`           | `Yes` / `No` (case-insensitive)                    | `frozenset({"A"})` / `frozenset({"B"})` — `Yes`=A, `No`=B          |
| `binary_named`     | one of the entries in `options` (exact match, trim + case-insensitive)| look up the index in the `options` list → letter → frozenset                     |
| `multiple_choice`  | one or more letters, comma- or space-separated (`A` / `B, C` / `B,C`) | split directly → frozenset[str]                                        |

Letter ↔ index rule (supports up to 35 options for multiple_choice):
```
index = ord(letter) - ord('A')
A=0, B=1, ..., Z=25
[ =26, \ =27, ] =28, ^ =29, _ =30, ` =31, a =32, b =33, c =34, ...
```

Reverse (when assembling the prompt, index → letter):
`letter = chr(ord('A') + index)`.

> ⚠️ **Source-data compatibility-mode warning**: the DB has 4
> `multiple_choice` questions with > 26 options, of which 3 have
> ground-truth answers landing on non-letter symbols like `[ \ ] ^ _ `
> ` ` a b c ...`. These ASCII-continuation labels are extremely
> unfriendly to LLMs (backticks and underscores get swallowed by
> markdown/code blocks; lowercase `a` and uppercase `A` coexist and are
> easily confused). **We keep this scheme only to preserve a one-to-one
> mapping with the source-data letter encoding**, for letter-set
> scoring.
>
> Mandatory defences:
> 1. `prompts.render_user_prompt` must explicitly quote or escape labels
>    when generating the `outcomes_block` for > 26 options (e.g. wrap
>    `` `[` ``, `` `\` `` in backticks/quotes), to avoid being lost in
>    markdown rendering.
> 2. `parser.parse_answer` must have a round-trip unit test
>    (label→letter→label) for `multiple_choice` with > 26 options.
> 3. Logs/reports record letters and corresponding labels in parallel
>    for manual review.
>
> If we later confirm LLM performance is dragged down by the labelling
> scheme, evaluate migration to a stable labelling scheme like `AA/AB`
> or `A01/A02`.

Ground-truth reverse lookup (`answer` letters → labels, for display or
logging):
```python
opts    = json.loads(row["options"])
letters = [t.strip() for t in row["answer"].split(",")]
labels  = [opts[ord(L) - ord('A')] for L in letters]
```

### 3.8 Leak boundary and threat model

This project can strictly control only the **tool-search** information
channel. The full leakage surface and the project's mitigation strategy:

| Leakage source                            | Controllable? | Mitigation                                                                                          |
| --------------------------------- | -------- | ------------------------------------------------------------------------------------------------- |
| Tool search content (Tavily returned content)  | ✅ Controllable   | `end_date = end_time + TAVILY_END_DATE_OFFSET_DAYS` injected by the tool implementation layer, invisible to the LLM (§3.1 / §3.2) |
| Provider built-in browsing / web tool | ✅ Controllable   | **Forcibly forbidden**: `llm.chat` attaches only `WEB_SEARCH_SCHEMA`, with no provider-native browsing / retrieval plugin enabled; OpenRouter routing does not pass the `:online` suffix or the `plugins` field |
| Model parametric memory (training data)          | ⚠️ Partially controllable | See §3.9: filter questions earlier than the model's training cutoff                                                  |
| "Future leakage" in search-result snippets   | ⚠️ Partially controllable | Tavily's `end_date` filter already cuts off at the publish-date layer; v5.2 onwards adds a detector LLM auditing each item (`search-leak-filter-v1`), with verdict=drop removing the entry entirely |
| Time clues in the question text itself (e.g. year)  | ❌ Uncontrollable | inherent to the question text, no intervention                                                                          |
| External knowledge backflow appearing after LLM training      | ❌ Uncontrollable | accept this bias                                                                                        |

Code-layer hard constraints:
- In `llm.chat` calls, `tools=[WEB_SEARCH_SCHEMA]` is the only allowed
  tool schema; **no** provider-native browsing/online switch may be
  added.
- If a provider forcibly attaches a built-in tool that cannot be
  disabled, explicitly mark "this model is unsuitable for strict
  evaluation" in the README and reports.

### 3.9 Filtering questions by model training cutoff date

**Motivation**: if a question's `end_time` is earlier than a model's
training cutoff date, the model very likely has "seen the answer" in its
training corpus; such samples cannot reflect the "predict the future"
ability and must be removed from that model's evaluation set.

**Mechanism**:
- Configure `MODEL_TRAINING_CUTOFFS` in `.env`, declaring a training
  cutoff date (`YYYY-MM-DD`) for each model
- During task-queue generation, filter each `(question, model)`:
  ```
  cutoff = MODEL_TRAINING_CUTOFFS.get(model)   # None = not declared, no filtering
  if cutoff is not None and question.end_time <= cutoff:
      # skip all sample_idx for this model
  ```
- Filtered `(question, model, sample_idx)` **still records a row** into
  `run_results`, with fields:
  - `error = "skipped_training_cutoff"`
  - `parse_ok = 0`, `correct = NULL`
  - `final_answer_raw = NULL`, `messages_trace = NULL`,
    `search_calls = NULL`
  - numeric fields set to 0
- Purpose: reports can clearly show "how many questions were filtered
  out per model, and how many remain comparable", and resume will not
  retry the row

**Resume semantics refinement** (see §5.3):
- `error IS NULL` → completed normally
- `error = "skipped_training_cutoff"` → actively excluded, **no retry**
- other `error` values (`network` / `server_5xx` / `bad_request` /
  `content_policy`) → handled per §9

When a user does not declare a cutoff for a model, that model is not
filtered. We recommend explicitly giving a cutoff for every model under
evaluation in `.env` to ensure fairness.

---

## 4. End-to-end pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Evaluation Pipeline                             │
└────────────────────────────────────────────────────────────────────────┘

[.env]  →  [python evaluation.py [--question-type ...] [--choice-type ...]]
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │  1. Load Config (.env) │
                          │  & Init run_id         │
                          └────────────────────────┘
                                      │
                                      ▼
                          ┌──────────────────────────────────┐
                          │  2. Sync Source                  │
                          │  forecast_eval_set_example.db            │
                          │    → results.db.questions        │
                          │    → results.db.prompt_templates │
                          │  (filtered by filters)           │
                          └──────────────────────────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │  3. Resume Check       │
                          │  skip completed        │
                          │  (run_id,              │
                          │  question_id, model,   │
                          │  sample_idx)           │
                          └────────────────────────┘
                                      │
                                      ▼
                 ┌────────────────────────────────────────┐
                 │  4. Task Queue                         │
                 │  Cartesian product: questions ×        │
                 │  models × N                            │
                 │  asyncio.Semaphore for concurrency     │
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
                      │  ReAct Loop (per sample)   │
                      │                            │
                      │  ┌──────────────────────┐  │
                      │  │ prompts.render(q)    │  │
                      │  │ → user_message       │  │
                      │  └──────────┬───────────┘  │
                      │             ▼              │
                      │  ┌──────────────────────┐  │
                      │  │ LLM.chat(            │  │
                      │  │   model, messages,   │  │
                      │  │   tools=[web_search])│  │
                      │  └──────────┬───────────┘  │
                      │             │              │
                      │  tool_call? ┼── No → break │
                      │             │ Yes          │
                      │             ▼              │
                      │  ┌──────────────────────┐  │
                      │  │ web_search(query,    │  │
                      │  │   end_date = inject  │  │
                      │  │     from q.end_time) │  │
                      │  │ → Tavily API         │  │
                      │  └──────────────────────┘  │
                      │                            │
                      │  loop ≤ REACT_MAX_STEPS    │
                      │  and ≤ REACT_MAX_SEARCH_CALLS│
                      └──────────┬─────────────────┘
                                 │
                                 ▼
                      ┌────────────────────────┐
                      │  5. Parse \boxed{...}  │
                      │  Normalise per         │
                      │  question_type         │
                      │  → frozenset[str]      │
                      └──────────┬─────────────┘
                                 │
                                 ▼
                      ┌────────────────────────┐
                      │  6. Score (frozenset   │
                      │  letter-set strict     │
                      │  equality)             │
                      └──────────┬─────────────┘
                                 │
                                 ▼
                      ┌────────────────────────┐
                      │  7. Enqueue → writer   │
                      │  Single writer thread  │
                      │  WAL + batch commit    │
                      └──────────┬─────────────┘
                                 │
                                 ▼
                      ┌────────────────────────┐
                      │  8. Done (results.db)  │
                      │  Subsequent analysis   │
                      │  runs independently    │
                      └────────────────────────┘
```

---

## 5. Database design (`runs/{run_id}/db/<model_slug>.db`)

Each run × model corresponds to **one independent sqlite file**. The file
self-contains copies of `questions` / `prompt_templates`, so a single
file can be replayed independently. Aggregations/statistics are **not
persisted**; after the run finishes, `forecast_eval.analysis` writes them
separately into the `analysis/` directory.

### 5.1 schema

```sql
-- ⓪ schema version table
CREATE TABLE schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- ① source-question copy (each model DB stores its own copy, for self-contained distribution)
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
    reflection_protocol_hash TEXT,         -- sha256(reflection_protocol_text)[:16]; same as above, NULL when off
    belief_protocol_text   TEXT,           -- v4. full text of prompts.BELIEF_PROTOCOL; NULL when BELIEF_PROTOCOL=false
    belief_protocol_hash   TEXT,           -- v4. sha256(belief_protocol_text)[:16]; same as above, NULL when off
    training_cutoff       TEXT,            -- this model's cutoff (YYYY-MM-DD), NULL when not declared
    started_at            TEXT NOT NULL,
    finished_at           TEXT
);

-- ④ wide table: one row per question, one s{i}_* column group per sample
-- dynamically generates 14 × SAMPLING_N columns; the shape below is for SAMPLING_N=3 only
CREATE TABLE run_results (
    question_id TEXT PRIMARY KEY,
    user_prompt TEXT,                      -- shared across all samples (COALESCE-written, first sample wins)

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
    -- v3 newly added observation columns (schema_version=3): per-step metrics + final-state envelope
    s0_finish_reason        TEXT,
    s0_nudges_used          INTEGER,
    s0_step_metrics         TEXT,          -- JSON array, one step snapshot per element, see §5.2
    s0_response_id          TEXT,          -- ChatCompletion.id (last round)
    s0_system_fingerprint   TEXT,          -- ChatCompletion.system_fingerprint (last round)
    s0_service_tier         TEXT,          -- ChatCompletion.service_tier (last round)
    -- v4 newly added observation columns (schema_version=4): structured belief-protocol output
    s0_belief_final         TEXT,          -- final-step Belief.probabilities as JSON ({letter: float}); NULL when parse fails
    s0_belief_trace         TEXT,          -- per-step belief summary JSON array [{step, p, confidence, delta_reason}|null, ...]
    s0_belief_parse_ok      INTEGER,       -- whether the final-step belief parsed legally (0/1); independent of parse_ok

    -- ...same s1_* / s2_* field groups...

    FOREIGN KEY (question_id) REFERENCES questions(id)
);
CREATE INDEX idx_run_results_question ON run_results(question_id);
```

> **schema_version 3 upgrade notes**: v2 → v3 is performed by
> `forecast_eval.db._migrate_v2_to_v3` via `ALTER TABLE … ADD COLUMN`
> (`run_results` adds 6 × N `s{i}_*` columns, `run_meta` adds 2 columns,
> and INSERT `(3, utcnow_iso())` into `schema_version`). SQLite's ADD
> COLUMN only writes table metadata, completing in O(1); new columns on
> old rows default to NULL. On the resume path, the first time an old
> DB is opened it is auto-migrated.
>
> **schema_version 4 upgrade notes**: v3 → v4 is performed by
> `forecast_eval.db._migrate_v3_to_v4` via `ALTER TABLE … ADD COLUMN`
> (`run_results` adds 3 × N `s{i}_*` belief columns, `run_meta` adds 2
> columns, and INSERT `(4, utcnow_iso())` into `schema_version`).
> `init_schema` calls v2→v3→v4 in chain, idempotent. When
> `Settings.BELIEF_PROTOCOL=false`, all belief columns write NULL and
> the existing accuracy metrics output zero changes. Full design in
> `ANALYSIS_DESIGN_v4.md`.

Connection-init PRAGMA (executed on every sqlite3 connection):
```sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;     -- safe enough under WAL and faster
PRAGMA busy_timeout = 5000;      -- avoid SQLITE_BUSY in multi-reader scenarios
```

### 5.2 Field write conventions

| Field                          | Source                                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------------------- |
| `s{i}_final_answer_letters`   | `frozenset[str]` returned by `parser.parse_answer(final_raw, q)`, written after `sorted()` + `json.dumps`           |
| `s{i}_final_answer_raw`       | the full `content` text of the LLM's last assistant message                                                        |
| `s{i}_correct`                | `frozenset == frozenset` → `int`; `NULL` when parse fails or scoring is impossible                                         |
| `s{i}_parse_ok`               | `final_answer_letters is not None`                                                                      |
| `user_prompt`                 | the return value of `prompts.render_user_prompt(q, templates)`; rendered once per question, retained via COALESCE after the first sample writes       |
| `s{i}_messages_trace`         | full `messages` list as JSON; NULL when `WRITE_MESSAGES_TRACE=false`                                         |
| `s{i}_search_calls`           | metadata list for each `web_search` call (query / end_date / n_results / published_dates; when leak filter is enabled, additionally `n_results_raw / n_results_kept / detector_verdicts / detector_latency_ms / detector_error_kind` — see `search-leak-filter-v1`) |
| `s{i}_error`                  | error classification code after retries are exhausted; NULL on normal completion (including refusal / parse fail)                                    |
| `s{i}_created_at`             | UTC ISO-8601 at write time; the unique signal for "whether this sample slot has been filled"                                         |
| `s{i}_finish_reason`          | the last round's `ChatCompletion.choices[0].finish_reason` (`stop` / `tool_calls` / `length` / `content_filter` …); NULL for error rows (never reached LLM) |
| `s{i}_nudges_used`            | count of "strict floor not met → reminder injected" within this sample; capped by `REACT_MAX_NUDGES`; 0 for error rows    |
| `s{i}_step_metrics`           | a JSON array of each ReAct round; element keys `step / prompt / completion / reasoning / latency_ms / finish_reason / n_tool_calls`; `latency_ms` is `time.monotonic()` wall time around that round's `llm.chat` (LLM call only, not search) |
| `s{i}_response_id`            | last round's `ChatCompletion.id` (provider-unique ID, useful for tracing / appeals)                                        |
| `s{i}_system_fingerprint`     | last round's `ChatCompletion.system_fingerprint` (when the provider supplies it; used to detect provider-side model-routing changes)        |
| `s{i}_service_tier`           | last round's `ChatCompletion.service_tier` (the actual tier returned by OpenAI etc., e.g. `default` / `scale` / `flex`)     |
| `s{i}_belief_final`           | v4. JSON-serialised `Belief.probabilities` (`{letter: float}`) returned by `parser.parse_belief(content, q)` at the final step; NULL when parsing fails or `BELIEF_PROTOCOL=false` |
| `s{i}_belief_trace`           | v4. JSON array of belief summaries for every loop step; element keys `step / p / confidence / delta_reason`; `null` for elements where intermediate-step parsing fails; whole column NULL when every step fails |
| `s{i}_belief_parse_ok`        | v4. whether the final-step belief parses legally (0/1); **independent** of `parse_ok` — belief failure MUST NOT affect the boxed path's `parse_ok` / `correct`; written 0 for error / cutoff rows |

> The 5 newly added fields (`finish_reason` / `response_id` /
> `system_fingerprint` / `service_tier` / `step_metrics`) reflect only
> the **last** `llm.chat` envelope; the finish_reason of intermediate
> steps goes into `step_metrics`; envelopes (response_id etc.) are
> per-round per OpenAI ChatCompletion top-level semantics and are
> currently not all persisted, to control wide-table column blow-up.
>
> `run_meta.reflection_protocol_text` / `reflection_protocol_hash` are
> **separated independently** from `prompt_templates_hash`: the former
> only fingerprints the content of `prompts.REFLECTION_PROTOCOL`
> (sensitive to on/off + text changes), enabling cross-run distinction
> of "is the reflection protocol enabled / has it been revised" without
> polluting the main template's content fingerprint.
>
> v4 adds `run_meta.belief_protocol_text` / `belief_protocol_hash`:
> **completely parallel** to the reflection-protocol fields,
> fingerprinting the content of `prompts.BELIEF_PROTOCOL`; same as
> above, no pollution of `prompt_templates_hash` or
> `reflection_protocol_hash` — the three fingerprints are mutually
> independent. v4 also writes `belief_protocol_hash` at the top level
> of `manifest.json` (alongside `reflection_protocol_hash`), so the
> "grep the protocol fingerprint without opening the DB" path covers
> both protocols; and adds `analysis_schema: "v4"` as a top-level
> field, so the analysis module can dispatch probabilistic-family
> metrics / accuracy-only fallback as needed.

### 5.3 Resume

Each sample slot is judged independently:
```sql
-- execute once per i ∈ 0..N-1:
SELECT question_id FROM run_results
 WHERE s{i}_created_at IS NOT NULL
   AND (s{i}_error IS NULL OR s{i}_error = 'skipped_training_cutoff');
```
Results are merged into `set[(question_id, sample_idx)]` and removed from
the task queue. Since each model's own DB contains only one run, `run_id`
no longer enters the filter (the single row in `run_meta` decides it).

State classification:
| `error` value                    | Meaning             | Retry on next resume? |
| -------------------------------- | ------------------- | ---------------- |
| `NULL`                           | completed normally  | no               |
| `'skipped_training_cutoff'`      | actively excluded by §3.9       | no               |
| `'network'` / `'server_5xx'`     | still failing after exhausted backoff      | yes              |
| `'bad_request'`                  | model_not_found, etc.  | yes (after config change) |
| `'content_policy'`               | provider refusal       | optional: default retry once and overwrite the original row |

Rules:
- Re-running with the same `run_id` = resume; writes into the existing
  `runs/{run_id}/db/<slug>.db`
- Changing `run_id` = a fresh run; creates a new
  `runs/{new_run_id}/` directory
- Overwrite semantics are backed by `INSERT ... ON CONFLICT(question_id)
  DO UPDATE SET s{i}_* = excluded.s{i}_*`; `user_prompt` is preserved
  with `COALESCE` to keep the first sample's value

### 5.4 Concurrent-write strategy

- Every DB connection executes PRAGMA `journal_mode=WAL /
  foreign_keys=ON / synchronous=NORMAL / busy_timeout=5000` at startup
- **One async writer task per model**: the runner opens a
  `forecast_eval.db.AsyncWriter` for each model DB; every worker's
  result is enqueued via the writer for that model
- The writer task flushes every `DB_COMMIT_BATCH` entries or every 1
  second, with short transactions; sqlite writes go through `await
  asyncio.to_thread(...)` to avoid blocking the event loop
- A single-model DB has only one writer and multiple readers; under WAL,
  concurrency is sufficient
- If switched to cross-thread consumption, `queue.Queue` /
  `janus.Queue` is required; `asyncio.Queue` is not cross-thread safe

---

## 6. Directory layout

```
Forecast/
├── .env                           # gitignored, user-filled
├── .env.example                   # template, git-managed
├── .gitignore
├── environment.yml                # conda env definition
├── README.md
├── FRAME.md                       # this document
├── evaluation.py                  # main entry: parse CLI flags -> runner.run -> analysis.run_analysis
├── forecast_eval_set_example.db           # source data, read-only, **checked into Git** (so source_db_hash is reproducible)
├── runs/                          # root for all evaluation outputs (gitignored)
│   └── {run_id}/
│       ├── manifest.json          # run-level metadata + model_files mapping
│       ├── db/
│       │   └── {model_slug}.db    # one sqlite per model; self-contains questions + prompt_templates copies
│       ├── analysis/              # statistical artefacts generated after the run
│       │   ├── per_model_summary.csv / .md
│       │   ├── per_model_by_question_type.csv         # includes v5 columns (composite-score-by-subtype)
│       │   ├── per_model_by_choice_type.csv           # includes v5 columns (composite-score-by-subtype)
│       │   ├── per_model_composite_by_question_type.csv  # composite-score table (composite-score-by-subtype)
│       │   ├── per_model_composite_by_choice_type.csv    # composite-score table (composite-score-by-subtype)
│       │   ├── composite_meta.json                    # composite-score audit trail
│       │   ├── error_breakdown.csv
│       │   └── overall.json                            # embeds the composite section
│       └── logs/{run_id}.log
├── forecast_eval/
│   ├── __init__.py
│   ├── config.py                 # pydantic-settings; RUNS_ROOT + MODEL_TRAINING_CUTOFFS parsing
│   ├── db.py                     # per-model wide-table schema + AsyncWriter + hash / redaction
│   ├── loader.py                 # syncs questions + prompt_templates from forecast_eval_set_example.db into each DB
│   ├── prompts.py                # renders user message per question_type
│   ├── llm.py                    # OpenAI-compatible client + tiered retry (provider-native browsing explicitly disabled)
│   ├── search.py                 # Tavily + end_date injection + retry
│   ├── tools.py                  # web_search schema (LLM-visible part, no date)
│   ├── react.py                  # ReAct loop (single sample)
│   ├── parser.py                 # \boxed{} parsing + letter-set normalisation + strict matching
│   ├── errors.py                 # error classification + backoff strategy (includes skipped_training_cutoff)
│   ├── runner.py                 # task orchestration + multi-model writer + training-cutoff filtering
│   └── analysis.py               # post-hoc statistics (read DB -> CSV / MD / JSON), invokable independently via `python -m`
└── tests/                        # unit tests (§17)
    ├── test_prompts.py
    ├── test_parser.py
    ├── test_search.py
    ├── test_db.py
    ├── test_errors.py
    ├── test_llm_no_browsing.py
    ├── test_runner_resume.py
    ├── test_training_cutoff.py
    ├── test_analysis.py
    └── test_smoke_dry_run.py
```

---

## 7. Full `.env.example` configuration

```ini
# =============================================================
#  Forecast Evaluation — environment-variable configuration
#  Copy to .env, fill in API keys, then run: python evaluation.py
# =============================================================

# -------- LLM Endpoint (OpenAI-compatible) --------
# LLM_BASE_URL examples: OpenRouter / Aliyun Bailian / OpenAI / DeepSeek / SiliconFlow / local vLLM
# See .env.example comments for details
LLM_API_KEY=REPLACE_ME
LLM_BASE_URL=https://openrouter.ai/api/v1

# Comma-separated list of models to evaluate (Cartesian product: every model runs every question × every sample)
# ⚠️ Do not append ":online" to a model slug, and do not enable any provider-native browsing (see §3.8)
MODELS=openai/gpt-5,anthropic/claude-sonnet-4.5,google/gemini-2.5-pro,deepseek/deepseek-r1

# Model training cutoff dates (§3.9): (q, model) pairs where end_time <= cutoff are skipped and marked skipped_training_cutoff
# Format: "<model_slug>=YYYY-MM-DD", multiple groups comma-separated. Models not declared are not filtered
# Recommended to declare a cutoff explicitly for every evaluated model, to ensure fairness
MODEL_TRAINING_CUTOFFS=openai/gpt-5=2024-10-01,anthropic/claude-sonnet-4.5=2025-03-01,google/gemini-2.5-pro=2025-01-01,deepseek/deepseek-r1=2024-07-01

# LLM call parameters (max_tokens already gives ample reasoning + output budget)
LLM_MAX_TOKENS=12000
LLM_TIMEOUT_S=240
LLM_TEMPERATURE=0.7
LLM_TOP_P=1.0

# Reasoning-model slug substring list: when matched, **do not pass** temperature / top_p
# (reasoning models like o-series / deepseek-r1 / qwq return 400 directly on custom sampling parameters)
LLM_REASONING_MODEL_PATTERNS=o1,o3,o4,r1,qwq

# LLM concurrency & retry
LLM_MAX_CONCURRENCY=5
LLM_RETRY_MAX=5
# Backoff sequences (seconds) by error kind; if exhausted with failure, skip the sample and record error
LLM_BACKOFF_NETWORK_S=2,5,15,30,60
LLM_BACKOFF_RATE_LIMIT_S=10,30,60,120,300
LLM_BACKOFF_SERVER_5XX_S=5,15,30,60,120

# -------- Tavily Search --------
# Single key or CSV multiple keys (`tvly-aaa,tvly-bbb`); multi-key handled by TavilyKeyPool with least-used
# scheduling + 401/403 permanent blacklist + 429 transient cooldown — see .env.example.
TAVILY_API_KEY=tvly-REPLACE_ME
# Cooldown seconds when a single key hits 429 (default 60); 401/403 permanent blacklisting is unaffected by this parameter.
TAVILY_KEY_COOLDOWN_S=60
TAVILY_MAX_RESULTS=5
# search_depth: basic (1 credit/call, default) | advanced (2 credits/call)
TAVILY_SEARCH_DEPTH=basic
# include_raw_content: false | markdown | text (legacy true is mapped to markdown for compatibility)
# Large in size — must be paired with TAVILY_RAW_CONTENT_MAX_CHARS truncation
TAVILY_INCLUDE_RAW_CONTENT=markdown
# Per-result raw_content truncation length. 0 = no truncation (caution: a single result can exceed 40k chars)
TAVILY_RAW_CONTENT_MAX_CHARS=8000
# include_answer: false | basic | advanced. Default false to avoid Tavily's internal LLM quick answer polluting evaluation purity
TAVILY_INCLUDE_ANSWER=false
# end_date = question.end_time + offset. Project default -1 (one day before, to avoid same-day information leakage).
# Smaller is more conservative: -2/-3 stricter; 0 = same-day visible (debug only, do not use for formal evaluation)
TAVILY_END_DATE_OFFSET_DAYS=-1

# Tavily concurrency & retry (paired with Tavily)
SEARCH_MAX_CONCURRENCY=5
SEARCH_RETRY_MAX=3
SEARCH_BACKOFF_S=2,5,15

# -------- ReAct Loop --------
REACT_MAX_STEPS=12
REACT_MAX_SEARCH_CALLS=8
# Reflection protocol: when enabled, append a multi-step reasoning scaffold to the end of the user message,
# significantly raising tool-call count and thinking depth. Not written into dataset_metadata
# (prompt_templates_hash unchanged); the protocol text is persisted per sample via the user_prompt field,
# and the configuration toggle is recorded in config_snapshot.
REACT_REFLECTION_PROTOCOL=true
# Soft minimum search count (default 0=off). When > 0, if the LLM tries to give a final answer with
# fewer than this many searches, a user nudge is injected to keep retrieving. Bounded by REACT_MAX_SEARCH_CALLS.
REACT_MIN_SEARCH_CALLS=0
# Per-sample upper bound on nudge injections (default 2), to prevent infinite nudge loops.
REACT_MAX_NUDGES=2

# -------- Sampling --------
# How many samples per question per model (pass@1 avg / pass_any@N / majority vote are all based on these N)
SAMPLING_N=5

# -------- Run / Resume --------
# Empty → auto-generate YYYYMMDD-HHMMSS-{4-char short uuid}. Same run_id resumes
RUN_ID=
RESUME=true

# -------- Database --------
SOURCE_DB=./forecast_eval_set_example.db
# Question table name (inside SOURCE_DB). For a custom dataset, change to your own table name; only [A-Za-z_][A-Za-z0-9_]*.
SOURCE_TABLE=forecast_eval_set_example
# Each evaluation creates a separate {run_id}/ directory under RUNS_ROOT (db/, analysis/, logs/)
RUNS_ROOT=./runs
DB_COMMIT_BATCH=10
# false skips the full messages trace, can shrink DB by 80%
WRITE_MESSAGES_TRACE=true

# -------- Logging --------
LOG_LEVEL=INFO
LOG_DIR=./logs
```

### 7.1 Key parameter notes

- **`MODELS`**: comma-separated, Cartesian product. To run a single model just leave one. Empty → error and exit. **Do not** append `:online` to the slug or enable provider built-in browsing (see §3.8).
- **`MODEL_TRAINING_CUTOFFS`**: list of `model=YYYY-MM-DD`, comma-separated. `config.py` parses it as `dict[str, date]`. Models not declared are not filtered. Filtering happens during the runner's task-generation phase; skipped samples write a row of `error="skipped_training_cutoff"` into `run_results`.
- **`LLM_MAX_CONCURRENCY` vs `SEARCH_MAX_CONCURRENCY`**: separately controlled, because Tavily's rate limit is generally tighter than the LLM's.
- **The three `LLM_BACKOFF_*` sequences**: correspond to different error types (see §9); the sequence length determines the max retry count.
- **`TAVILY_SEARCH_DEPTH`**: `basic` (default, 1 credit) / `advanced` (2 credits, higher recall). A single prediction averages 3-5 searches; `basic` controls cost.
- **`TAVILY_INCLUDE_RAW_CONTENT`**: `false` / `markdown` (default) / `text`. Controls the page-body form the LLM sees. When the volume is large, also set `TAVILY_RAW_CONTENT_MAX_CHARS`. The legacy `bool` value is still compatible (`true → markdown`).
- **`TAVILY_RAW_CONTENT_MAX_CHARS`**: per-result `raw_content` truncation threshold (chars), default `8000` ≈ 2k tokens. `0` = no truncation (caution: 5 results combined can exceed 200k chars, easily blowing the LLM context).
- **`TAVILY_INCLUDE_ANSWER`**: `false` (default) / `basic` / `advanced`. Default off to avoid introducing a "second LLM judgement" that pollutes evaluation purity (when enabled, differences between strong and weak models compress).
- **`TAVILY_END_DATE_OFFSET_DAYS`**: project default `-1` (one day before, the recommended strict default). Smaller is more conservative; `0` is for debug only. All reports default to comparison under `-1`.
- **`RUN_ID` auto-generation format**: `YYYYMMDD-HHMMSS-xxxx`, e.g. `20260424-120344-a7k3`; `ls` naturally sorts by time, and this is also the directory name under `RUNS_ROOT/{run_id}/`.
- **`RUNS_ROOT`**: root directory for evaluation outputs (default `./runs`); each run takes one subdirectory.
- **`WRITE_MESSAGES_TRACE`**: `true` stores the full messages JSON (handy for debugging, increases DB size); `false` stores only key fields.
- **`REACT_REFLECTION_PROTOCOL`**: `true` (default) appends a multi-step reasoning scaffold to the end of each sample's user message (decompose / ≥3 retrieval angles / reflect after each search / cross-validate / opposite-direction self-check / confidence statement). The protocol text is not entered into `dataset_metadata`, so `prompt_templates_hash` is unaffected, but the rendered full user message goes into each sample's `user_prompt` field; the toggle is also recorded in `run_meta.config_snapshot`, allowing post-hoc behaviour comparison between protocol on / off.
- **`REACT_MIN_SEARCH_CALLS` / `REACT_MAX_NUDGES`**: optional fallback mechanism. When the LLM tries to give a final answer with fewer `web_search` calls than `REACT_MIN_SEARCH_CALLS`, the system injects a user nudge into the message sequence asking it to retrieve from another angle; the same sample is nudged at most `REACT_MAX_NUDGES` times, with the overall flow still bounded by `REACT_MAX_STEPS` / `REACT_MAX_SEARCH_CALLS` ceilings. `REACT_MIN_SEARCH_CALLS=0` (default) is equivalent to disabling the fallback, relying solely on the reflection protocol; when `ENABLE_WEB_SEARCH=false`, the nudge is automatically disabled (no search to do). Settings validation rejects `min > max`.
- **Redaction**: before writing `run_meta.config_snapshot`, `config.py` MUST redact sensitive fields like `LLM_API_KEY` / `TAVILY_API_KEY` (keep only the first 4 chars + length + `sha256[:12]`); sensitive plaintext is never persisted. `TAVILY_API_KEY` is now `list[str]`, each key redacted independently and persisted as `[{prefix, sha256_12, length, provider}, ...]`, for later auditing of "which keys this run used".

---

## 8. Core module responsibilities

| Module       | Responsibility                                                                                                  | Key interfaces                                                                                                 |
| ------------ | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `config.py`  | reads `.env` via `pydantic-settings`, validates types, parses comma-separated lists                                                  | `Settings` class (singleton)                                                                                          |
| `loader.py`  | syncs two tables from `SOURCE_DB` (default `forecast_eval_set_example.db`) into `results.db`: ① `<SOURCE_TABLE>` (default `forecast_eval_set_example`) → `questions` (filtered by filters); ② `dataset_metadata.features_json.prompt_reconstruction` → `prompt_templates` (key/value flat) | `sync_questions(source_db, conn, filters, table=...) -> list[Question]`, `sync_prompt_templates(source_db, conn) -> dict[str,str]` |
| `prompts.py` | renders the user message per `question_type`: ① generates `outcomes_block` (multiple_choice enumerates options via the §3.7 letter rule); ② selects one of the three `output_format`s; for binary_named, replaces `<options[i]>` placeholders with the actual entity names; ③ assembles the final text via `prompt_template` | `render_user_prompt(q: Question, templates: dict[str,str]) -> str`                                             |
| `tools.py`   | defines the `web_search` OpenAI-schema; **the LLM-visible part contains no date**                                                       | `WEB_SEARCH_SCHEMA`, `execute_tool_call(tc, q, cfg)`                                                           |
| `search.py`  | wraps Tavily `/search`, injects `end_date = q.end_time + OFFSET`; controls page-body form per `TAVILY_INCLUDE_RAW_CONTENT` and truncates per `TAVILY_RAW_CONTENT_MAX_CHARS`; retry | `tavily_search(query, end_date, settings) -> SearchResult`                                                     |
| `llm.py`     | OpenAI-compatible client (OpenRouter), tiered retry by error kind; **forces no provider-native browsing** (no `plugins`, no `:online` suffix, no provider-private web-tool fields) | `chat(model, messages, tools, ...) -> ChatResponse`                                                            |
| `react.py`   | one ReAct inference = one sample; loops until no tool_call or limits hit                                                        | `run_react(q, model, sample_idx, cfg) -> SampleResult`                                                         |
| `parser.py`  | parses `\boxed{...}` per `question_type` → letter `frozenset[str]` (yes_no: Yes/No→A/B; binary_named: label→letter; mc: split letters); strict frozenset equality against the letter set parsed from `q.answer` | `parse_answer(text: str, q: Question) -> frozenset[str] \| None`, `parse_gt(answer: str) -> frozenset[str]`, `is_correct(pred, gt) -> bool` |
| `errors.py`  | maps httpx/openai exceptions to error classifications; gives wait-seconds                                                                | `classify(exc) -> ErrorKind`, `backoff_seconds(kind, attempt)`                                                 |
| `db.py`      | connection management, WAL + PRAGMA, **per-model wide-table schema dynamic generation** (`init_schema(conn, sampling_n)` creates `s{i}_*` columns), `register_run_meta` / `finish_run_meta`, `AsyncWriter` UPSERT by `(question_id, sample_idx)`, `load_completed_samples`, source/metadata/templates hash computation, config redaction, model-slug safe-ification | `init_schema(conn, sampling_n)`, `AsyncWriter.enqueue_result`, `load_completed_samples`, `register_run_meta`, `upsert_sample_sync`, `model_slug_safe`, `compute_*_hash` |
| `runner.py`  | task orchestration: Cartesian product → dedup (per-model completed set) → **filter via `MODEL_TRAINING_CUTOFFS` and write skipped_training_cutoff rows into the corresponding model DB** → asyncio concurrency → progress log → cleanup `finish_run_meta` | `run(settings, filters, questions, templates, run_id, conns: dict[model, sqlite3.Connection]) -> RunStats`, `build_task_plan(...)` |
| `analysis.py`| post-hoc statistics: scans `runs/{run_id}/db/*.db` → computes all metrics in §11 → writes `analysis/` CSV / MD / JSON. **Does not modify the DB.** Auto-invoked by `evaluation.py`, or invokable independently via `python -m forecast_eval.analysis runs/{run_id}` to refresh | `run_analysis(run_dir: Path) -> list[Path]` |

`QFilter` is a dataclass containing `question_types: set[str] | None` and
`choice_types: set[str] | None`, where `None` means no filtering.

### 8.1 `prompts.render_user_prompt` reference implementation

```python
def render_user_prompt(q: Question, templates: dict[str, str]) -> str:
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

    return templates["prompt_template"].format(
        agent_role=templates["agent_role"],
        event=q.event,
        end_time=q.end_time,
        outcomes_block=outcomes_block,
        output_format=output_format,
        guidance=templates["guidance"],
    )
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

| Error type                         | Identification                                                              | Handling strategy                                                                 |
| -------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **Network / Timeout**            | `httpx.ConnectError`, `httpx.ReadTimeout`, `asyncio.TimeoutError`     | use the `LLM_BACKOFF_NETWORK_S` backoff sequence; on still-failure → `error="network"` skip |
| **Rate Limit (429)**             | HTTP 429                                                              | prefer the `Retry-After` header; otherwise use `LLM_BACKOFF_RATE_LIMIT_S`           |
| **Server 5xx**                   | HTTP 500/502/503/504                                                  | use `LLM_BACKOFF_SERVER_5XX_S`; on exhaustion → `error="server_5xx"` skip          |
| **Auth (401/403)**               | HTTP 401/403                                                          | **fail immediately, stop the whole run** (continuing is pointless when the key is wrong)                      |
| **Bad Request (400)**            | HTTP 400 + `model_not_found` / `invalid_request`                      | skip immediately, `error="bad_request"`                                          |
| **Content Policy**               | HTTP 400 + `content_policy_violation` / provider rejection code               | **no retry**, `error="content_policy"`, `parse_ok=0`, `correct=NULL`       |
| **LLM soft refusal**             | normal return but `\boxed{...}` not found or parsed `frozenset` empty              | not an error, `parse_ok=0`, `correct=NULL` (counted into refusal rate)            |
| **Exceed `REACT_MAX_STEPS`**     | ReAct loop exhausted without producing a final answer                                            | not an error, `parse_ok=0`, `correct=NULL`                                 |
| **Tool arguments JSON parse fails** | the LLM's arguments are not legal JSON                                      | tell the LLM the error and continue the loop (non-fatal)                                        |
| **Tavily error itself**          | retry independently via `SEARCH_BACKOFF_S`; on exhaustion, feed the error to the LLM as tool_result | the LLM can choose to retry or give up                                                   |
| **Training-data contamination filter** | detected during task generation: `q.end_time <= MODEL_TRAINING_CUTOFFS[model]` (see §3.9) | **does not invoke the LLM**, directly writes `error="skipped_training_cutoff"`, `parse_ok=0`, `correct=NULL`; resume does not retry |

### 9.1 Key boundaries

1. **Auth errors stop the whole run**: continuing to burn budget on a wrong key is meaningless; early-stop saves money.
2. **Content policy is not retried**: re-sending the same question yields the same result. Mark it directly, and tally how many each model was rejected on at the end.
3. **Refusal ≠ error**: the LLM returned a legal response but did not answer (missing boxed / letter outside the option range) — this is part of model capability, counted in statistics but not the error field.
4. **Tavily failure degrades to a tool_result error**: let the LLM decide whether to retry the query or give up, without interrupting the whole sample.
5. **`skipped_training_cutoff` does not count toward error rate**: this is active data cleansing, not a model failure; the report tallies "questions excluded / ratio" separately and does not include it in `error rate by kind`.

---

## 10. ReAct Loop pseudocode

```python
async def run_react(q: Question, model: str, sample_idx: int, cfg: Settings) -> SampleResult:
    # ① inject end_date: the LLM never sees it
    end_date = (date.fromisoformat(q.end_time)
                + timedelta(days=cfg.TAVILY_END_DATE_OFFSET_DAYS)).isoformat()

    # ② assemble the user message: agent_role + event + outcomes_block + output_format + guidance
    #    all read from prompt_templates, decoupled from the source data; the reflection protocol
    #    is appended as an addendum when REACT_REFLECTION_PROTOCOL=true, still a single user message.
    user_prompt = prompts.render_user_prompt(
        q,
        cfg.PROMPT_TEMPLATES,
        reflection_protocol=prompts.REFLECTION_PROTOCOL if cfg.REACT_REFLECTION_PROTOCOL else None,
    )

    # ③ as a single user message in its entirety (most faithful to the template; no system/user split)
    messages = [{"role": "user", "content": user_prompt}]
    search_calls: list[dict] = []
    final_raw = ""
    t0 = time.monotonic()
    tokens = {"prompt": 0, "completion": 0, "reasoning": 0}
    nudges_used = 0
    step = 0

    for step in range(cfg.REACT_MAX_STEPS):
        resp = await llm.chat(
            model=model,
            messages=messages,
            tools=[WEB_SEARCH_SCHEMA],
            temperature=cfg.LLM_TEMPERATURE,
            top_p=cfg.LLM_TOP_P,
            max_tokens=cfg.LLM_MAX_TOKENS,
            timeout=cfg.LLM_TIMEOUT_S,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_unset=True))
        _accumulate_tokens(tokens, resp.usage)

        # no tool_call = LLM wants to give a final answer. Soft-minimum-search-count fallback:
        # if insufficient, nudge once to keep retrieving (jointly protected by REACT_MAX_NUDGES and REACT_MAX_STEPS).
        if not msg.tool_calls:
            nudge_enabled = (
                cfg.ENABLE_WEB_SEARCH
                and cfg.REACT_MIN_SEARCH_CALLS > 0
                and cfg.REACT_MAX_NUDGES > 0
            )
            if (
                nudge_enabled
                and len(search_calls) < cfg.REACT_MIN_SEARCH_CALLS
                and nudges_used < cfg.REACT_MAX_NUDGES
                and step < cfg.REACT_MAX_STEPS - 1
            ):
                messages.append({
                    "role": "user",
                    "content": prompts._build_nudge_message(
                        searches_done=len(search_calls),
                        min_required=cfg.REACT_MIN_SEARCH_CALLS,
                    ),
                })
                nudges_used += 1
                continue
            final_raw = msg.content or ""
            break

        # handle every tool_call (OpenAI supports parallel)
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

            # inject end_date (invisible to the LLM)
            result = await search.tavily_search(query=args["query"], end_date=end_date)
            search_calls.append({
                "query": args["query"],
                "end_date": end_date,
                "n_results": len(result.results),
                "published_dates": [r.published_date for r in result.results],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result.to_llm_payload()),
            })
    # exceeded the step count; final_raw stays empty → parser will mark parse_ok=0

    # ④ parsing and scoring: all at the letter-set level
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
        react_steps=step + 1,
        prompt_tokens=tokens["prompt"],
        completion_tokens=tokens["completion"],
        reasoning_tokens=tokens["reasoning"],
        latency_ms=int((time.monotonic() - t0) * 1000),
        user_prompt=user_prompt,
        messages_trace=json.dumps(messages) if cfg.WRITE_MESSAGES_TRACE else None,
        search_calls=json.dumps(search_calls),
        error=None,
        created_at=utcnow_iso(),
    )
```

---

## 11. Evaluation metric definitions

Metrics are **computed entirely by `forecast_eval.analysis` after the run
finishes**, not stored in the DB. Artefacts land in
`runs/{run_id}/analysis/` (CSV / MD / JSON). The definitions below match
the source implementation.

A `(question_id, model)` has N samples (`N = SAMPLING_N`). When tallying,
**first exclude** rows with `s{i}_error="skipped_training_cutoff"` (these
are excluded questions, not the model getting them wrong):

| Metric                          | Definition                                                                              | Notes                    |
| ------------------------------- | --------------------------------------------------------------------------------- | ----------------------- |
| **pass@1 avg**                  | `mean(correct over N samples)`                                                    | reflects stable capability        |
| **pass_any@N** (was `pass@3`)   | `1 if any(correct) across N samples else 0`                                       | best-of-N potential (the standard pass@k meaning) |
| **at_least_k_correct@N**        | `1 if sum(correct) ≥ k else 0`                                                    | repeated-consistency correctness, suited for threshold analysis |
| **majority vote correct**       | majority-vote on N `final_answer_letters` (as frozensets), then compared with `q.answer`     | self-consistency metric   |
| **parse failure rate**          | `mean(1 - parse_ok)`                                                              | reflects format adherence / refusal rate |
| **avg tool_calls**              | `mean(tool_calls_count)`                                                          | reflects search-usage strategy        |
| **avg react_steps**             | `mean(react_steps)`                                                               | reflects reasoning depth            |
| **avg latency_ms / avg tokens** | average of the same-named fields                                                                      | reflects cost                |
| **error rate by kind**          | percentages by `error` classification (excluding `skipped_training_cutoff`)                         | reflects stability              |
| **training_cutoff_skip rate**   | `count(error='skipped_training_cutoff') / count(*)` per model                     | how many questions this model was excluded on      |
| **avg_nudges_used**             | `mean(nudges_used)` over eligible samples (since v3)                                 | reflects "strict-floor trigger rate" — larger means the model triggers reminders more often; 0 means almost all searches were spontaneous |
| **finish_reason_breakdown**     | per-model `Counter[finish_reason]` over eligible samples (since v3)             | NULL counted in the `<missing>` bucket; used to identify abnormal proportions like `length` (output truncation) / `content_filter` (refused) |

> Naming change: the legacy doc's `pass@3 = sum(correct)≥3` is inconsistent
> with the standard `pass@k` semantics ("any correct in k") and is easily
> misread. We now use `pass_any@N` (= any) and `at_least_k_correct@N` (=
> threshold) as two distinct names.

Report slicing dimensions: `model × question_type × choice_type`. Output
tables:

| File                              | Content                                                                              |
| --------------------------------- | --------------------------------------------------------------------------------- |
| `per_model_summary.csv` / `.md`   | one row per model, with all metrics from the table above                                                        |
| `per_model_by_question_type.csv`  | `model × question_type` slice, same metric set                                            |
| `per_model_by_choice_type.csv`    | `model × choice_type` slice, same metric set                                              |
| `error_breakdown.csv`             | `model × error_kind` count + sample share (includes `<ok>` and `skipped_training_cutoff`)    |
| `overall.json`                    | structured aggregation of every slice, for further processing                                                |

### 11.5 Discrete-native metric family (v5 main line) + probabilistic family (companion)

**v5 reorientation**: this project samples `K=5` in parallel; the
empirical probability $\hat{p} = n/K$ for each (question, label) takes
only 6 discrete values $\{0, 0.2, 0.4, 0.6, 0.8, 1.0\}$. This pushes
v4's Reliability Diagram / Murphy three-decomposition / Platt scaling
LOO into the "mathematically correct, statistically meaningless"
position (see archived `2026-04-26-probabilistic-analysis-v4` and the
`discrete-native-analysis-v5` proposal). v5 redirects the analysis stack
to the **discrete-native** metric family suited for K=5; BS / NLL / MBS
/ BI / ABI are demoted to auxiliary columns.

**v5 first-class citizens** (`forecast_eval/analysis/accuracy.py` +
`consistency.py`):

| Metric | Formula | Interpretation |
| --- | --- | --- |
| **FSS** | Tversky α=2 / β=0.5 per-sample → per-question mean → chance correction $s_q = (c_q - p_e)/(1 - p_e)$ → cross-question mean | primary metric. Multi-select wrong cost = 4× missed; single-select degenerates to strict 0/1 |
| Tversky baseline | multi-select $p_e$ exact enumeration $O(m \times (k-m))$; single-select $p_e = 1/k$ | the chance-correction term in the FSS chain |
| Cohen's κ | $(\mathrm{acc} - p_e)/(1 - p_e)$, single-select $p_e = 1/k$ / multi-select $p_e = 0.5$ | chance correction for strict 0/1 acc |
| Hamming Score | $1 - \tfrac{1}{k}\sum_l|\hat{y}_l - o_l|$ | partial credit for multi-select question types; pure single-select runs return NULL |
| **Fleiss' κ** | $(\bar{P}-\bar{P}_e)/(1-\bar{P}_e)$ on the $K$-trial vote matrix; single uses letter argmax / multi uses the mean of per-label binary Fleiss | multi-rater agreement, K-trial exclusive |
| Predictive entropy $H_q$ | single: $-\sum_l \hat{p}_l \log_2 \hat{p}_l$; multi: per-label binary-entropy mean | per-question uncertainty |
| **Entropy-accuracy joint** | per-model tertile buckets → per-bucket Acc / MV Acc / Fleiss κ | "how does the model perform on high-entropy questions vs low-entropy ones" — v5's most academically original diagnostic dimension |
| VCI | $\max_l n_{q,l}/K$ cross-question mean | vote concentration |
| MVG | MV_Acc - Pass@1_Acc | majority-vote signal gain (K-trial exclusive) |

**Multi-metric paired bootstrap**
(`forecast_eval/analysis/inference.py`):

`metric_paired_bootstrap(metric_fn, samples_a_by_q, samples_b_by_q, gt_map, ...)`
parameterised paired bootstrap, runs 5000 resamples in parallel for FSS
/ Acc / MV_Acc / Fleiss κ / EBI, outputting 95% CI / p-value / Cohen's
d. `pairwise_bootstrap.csv` is the model-vs-model long table; the paper's
Figure 2 ΔFSS forest plot is from this.

The v4 BS-paired bootstrap (`paired_bootstrap` /
`pairwise_paired_bootstrap`) is retained — `grid.py`'s per-cell BI CI
and `paired_delta_bi.csv` / 4 v4 artefacts depend on it.

**Companion probabilistic family** (kept as auxiliary columns, with a
K=5 disclaimer):

v4 added first-order proper scoring rules and difficulty-adjusted metrics
beyond accuracy. Phase 0 collected the LLM's implicit probability signals
into structured fields (`s{i}_belief_final` / `belief_trace` /
`belief_parse_ok`); Phase 1 used these fields plus the §2.4 fallback in
the `analysis/` package to compute BS / NLL / MBS / BI / ABI and write
into `per_model_summary.csv` etc. v5 keeps these columns as benchmark
anchors against ForecastBench / BLF papers; the markdown table adds the
`†` footnote: "Probabilistic metrics are computed from empirical vote
frequencies over K=5 parallel trials, yielding only 6 discrete
probability levels per label. These values serve as ordinal companions
to the primary discrete metrics."

**Unified representation (per-option Bernoulli label vector)**: the
ground truth $\mathbf{o}_q \in \{0,1\}^{k_q}$ and prediction
$\mathbf{p}_q \in [0,1]^{k_q}$ for question $q$ are both arranged in
letter order; single-select question types require $\sum_l p_l = 1$
(tolerance $10^{-3}$); for multi-select question types, each $p_l$ is
the independent Bernoulli probability of that option being in the answer
set.

**First-order metrics
(`forecast_eval/analysis/proper_score.py`)**:

| Metric | Formula | Applicable | CSV column |
| --- | --- | --- | --- |
| Label-wise Brier | $\mathrm{BS}_q^{\text{lab}} = \tfrac{1}{k_q}\sum_l (p_{q,l}-o_{q,l})^2$ | all question types | `bi` (after aggregation) |
| Decision-wise Brier | $\mathrm{BS}_q^{\text{dec}} = \sum_l (p_{q,l}-o_{q,l})^2 = k_q\cdot\mathrm{BS}_q^{\text{lab}}$ | single only | `bi_dec` |
| Brier Index | $\mathrm{BI} = 100(1 - \sqrt{\overline{\mathrm{BS}^{\text{lab}}}})$, **average first then square root** | all question types | `bi` |
| NLL | single: $-\log p_{q,l^*}$; multi: label-wise BCE; clip $\epsilon = 10^{-3}$ | all question types | `nll` |
| MBS | $100(\log_2 p_{q,l^*} + 1)$, clip same as NLL | single only; multi writes NULL | `mbs` |
| ABI (crowd) | $\mathrm{ABI}^{(m_0)} = $ sign-aware $100(1\mp\sqrt{|\overline{\mathrm{ABS}^{(m_0)}}|})$, $\overline{\mathbf{p}}$ excludes $m_0$ | multi-model run | `abi_crowd` |
| ABI (uniform) | same as above, but baseline is $\mathbf{p}=(1/k,\dots,1/k)$ | all runs; for single-model runs `abi_crowd` degenerates to equal this column | `abi_uniform` |
| fallback share | the question count that went through §2.4 fallback / the model's scoreable question count | all runs | `fallback_share` |

**ABI sign convention**: $\overline{\mathrm{ABS}} \ge 0$ (model not better
than baseline) → $100(1 - \sqrt{\overline{\mathrm{ABS}}})$, lands in $[0,
100]$; $\overline{\mathrm{ABS}} < 0$ (model better than baseline) →
$100(1 + \sqrt{|\cdot|})$, exceeds 100, preserving the "better is higher"
monotonicity.

**§2.4 fallback**: when `s{i}_belief_final IS NULL` but `s{i}_parse_ok =
1` (legacy v3 runs, or v4 belief parse failed but boxed parse
succeeded), $p_l = 1-\epsilon$ (matched boxed letter) /
$\epsilon/(k-|\text{boxed}|)$ (others), $\epsilon = 0.05$. The sample
goes through fallback with `belief_parse_ok=0`. Samples with full
failure (`parse_ok=0`) MUST NOT enter probabilistic-metric averaging, to
avoid pollution.

**Multi-trial aggregation
(`forecast_eval/analysis/aggregation.py`)**:

Phase 1's default is the arithmetic mean of K sample probability vectors
per (model, question); Phase 2 adds two paper §C.9-style alternative
aggregators and one diagnostic scan:

| Function | Formula | Use |
| --- | --- | --- |
| `arithmetic_mean(predictions)` | $\hat{\mathbf{p}} = \tfrac{1}{K}\sum_k \mathbf{p}^{(k)}$ | Phase 1 default |
| `logit_space_mean(predictions, ctype)` | single: $\mathrm{softmax}(\overline{\log p})$; multi: $\sigma(\overline{\mathrm{logit}\,p})$ | paper default; same as arithmetic mean when K is consistent |
| `loo_shrinkage(...)` | computes the BS of $\mathrm{softmax}(\alpha\overline{\log p})$ on the $\alpha \in \{0, 0.1, \dots, 1.0\}$ grid; returns $\alpha^*$ + the full curve | diagnoses whether the dataset needs shrinkage toward the prior |
| `majority_vote_accuracy_v4(...)` | argmax after logit-space mean; K floating-point logits almost never tie | recovers the ~10% tie-unresolved cases of v3 majority\_vote in one shot |

`majority_vote_accuracy_v4` is the upgraded version of v3's letter-set
vote; currently wired as a unit-testable function and has not replaced
the v3 `majority_vote_accuracy` column to avoid breaking byte
regression. The $\alpha$-grid BI of `loo_shrinkage` lands in
`analysis/shrinkage_alpha_curve.csv`.

**Stratified calibration (deleted in v5 due to K=5 discrete-resolution constraints)**:

v4's `calibration.py` implemented per-(question_type, choice_type) cell
Platt / Temperature scaling LOO + ECE / Murphy three-decomposition /
Reliability bins. At the K=5 working point:

* Fitting Platt scaling sigmoid on 6 unique probability values is textbook overfitting;
* Temperature scaling's single parameter is barely stable, but the "temperature" is semantically dubious on 6-level discrete data;
* ECE with 15 bins always has 9+ empty bins; weighted-average has high variance and is not comparable;
* Murphy three-decomposition's CAL/RES terms have their variance swallowed by empty bins.

v5 deletes `calibration.py` entirely and discontinues 5 artefacts:
`calibration_params.json` / `per_model_summary_calibrated.csv` /
`reliability_data*.json` / `brier_decomposition.csv`. `per_model_summary.md`
removes the `BI_cal / NLL_cal / ECE_uncal / ECE_cal` columns and the
`cal*` sentinel. If K is increased to ≥30 in the future, calibration can
be reintroduced in a new change.

**Statistical inference
(`forecast_eval/analysis/inference.py`)**:

| Function | Algorithm | Output |
| --- | --- | --- |
| `paired_bootstrap(bs_a, bs_b)` | $B=5000$ paired resampling (the same indices index A and B simultaneously) | `delta_mean / ci_low / ci_high / p_two_sided` |
| `holm_bonferroni(p_values)` | $(n - i) \cdot p_{(i)}$ then cumulative max | adjusted p-values, returned in original order |
| `difficulty_tertile(gammas)` | sort per-question $\gamma_q$ then cut into tertiles | `low / mid / high` buckets |
| `paired_bootstrap_by_difficulty(...)` | independent paired bootstrap per tier | `{tier: PairedBootstrapResult}` |
| `posterior_a_better_than_b(bs_a, bs_b)` | Monte-Carlo $\Pr(\overline{BS}_A < \overline{BS}_B)$ on paired bootstrap | $\Pr(\mathrm{BI}_A > \mathrm{BI}_B) \in [0, 1]$ |
| `posterior_normal_fit(...)` | normal closed-form $\Phi(-\bar\Delta / SE)$ | sanity-check channel for the above |

The paired bootstrap is the same-indexed version — the same bootstrap
draws the same question id to index both A's and B's BS arrays — to
control question-level variance (quantified at 62% of total variance in
paper §G.2). Multiple comparisons are controlled via Holm-Bonferroni at
the FWER level.

**v5 artefact list**:

| File | Content | Status |
| --- | --- | --- |
| `per_model_summary.csv` | v3 + v5 discrete (FSS / Cohen κ / Hamming / Fleiss κ / mean entropy / VCI / MVG) + v4 probabilistic family (companion) | v5 revised column order |
| `per_model_summary.md` | same as above markdown; v5 columns in the main area, v4 probabilistic columns get a `†` footnote | v5 revised |
| `inter_trial_consistency.csv` | per-model Fleiss κ / mean entropy / VCI / MVG | **v5 new** |
| `entropy_accuracy_bins.csv` | per-model × tertile (Acc / MV Acc / Fleiss κ); per-model bucket boundaries differ | **v5 new** |
| `pairwise_bootstrap.csv` | multi-metric paired bootstrap: FSS / Acc / MV_Acc / Fleiss κ / EBI × pairs × ΔMean / 95% CI / p / Cohen's d | **v5 new** |
| `shrinkage_alpha_curve.csv` | per-(model, ctype) $\alpha$ grid mean BS / BI | v4 retained |
| `paired_delta_bi.csv` | BS-paired model-vs-model ΔBS + 95% CI + Holm + posterior | v4 retained (grid.py depends on it) |
| `pairwise_significance.csv` | $\alpha = 0.05$ significance flag (raw + Holm) | v4 retained |
| `posterior_pairwise.csv` | $\Pr(\mathrm{BI}_A > \mathrm{BI}_B)$ | v4 retained |
| `per_model_by_difficulty.csv` | BI / NLL / ABI stratified by difficulty tertile | v4 retained |
| `paired_delta_bi_by_difficulty.csv` | independent paired bootstrap per tier | v4 retained |
| ~~`calibration_params.json`~~ | ~~per-cell Platt / temperature~~ | **v5 deleted** |
| ~~`per_model_summary_calibrated.csv`~~ | ~~calibrated metrics~~ | **v5 deleted** |
| ~~`reliability_data.json` / `_calibrated.json`~~ | ~~per-(model, qtype) bins~~ | **v5 deleted** |
| ~~`brier_decomposition.csv`~~ | ~~Murphy three-decomposition~~ | **v5 deleted** |

Byte-regression protection of `error_breakdown.csv` /
`finish_reason_breakdown.csv` continues from Phase 1 — Phase 2 does not
modify these two files.

**Behavioural analysis
(`forecast_eval/analysis/behavior.py`)**:

Phase 3 turns the `belief_trace` JSON time series into 5 first-class
metrics, plus three groups of diagnostics — reflection
protocol A/B, tool-usage PDP, confidence joint diagnosis:

| Metric | Formula | Interpretation |
| --- | --- | --- |
| Trial-internal volatility | $V_{q,k} = \tfrac{1}{T-1}\sum_t \|b_t-b_{t-1}\|_2$ | total magnitude of belief change within this trial |
| Inter-trial variance | $\sigma_q = \mathrm{std}_k\,b^{(q,k)}_T$ | matches paper §4 Figure 2 |
| Convergence step | $C_{q,k} = \min\{t : \|b_T-b_t\|_2<0.05\}$ | how many steps to reach the final belief |
| Evidence efficiency | $\eta_{q,k} = (\mathrm{NLL}(b_0) - \mathrm{NLL}(b_T))/\max(1, \text{search\_calls})$ | information gain per search |
| Counterevidence engagement | at least one counterevidence string contains a letter that is not the final choice (letter match, no NLP) | whether opposite-direction self-check was performed |

Reflection protocol A/B (`find_paired_runs` + `reflection_ab_report`)
scans every run and pairs them by "differing
`reflection_protocol_hash`, every other hash equal"; for each pair it
computes paired-bootstrap 95% CI of ΔBI / Δσ / ΔC / Δη, reported
stratified by question_type. Inconsistent fingerprints MUST NOT pair —
that is the hard constraint of spec 26.5.

Tool-usage PDP (`tool_usage_pdp`) fits with pure-Python IRLS the
relationship of $\Pr(\text{correct}\mid\mathbf{x})$ (logistic) and
$\mathbb{E}[\mathrm{NLL}\mid\mathbf{x}]$ (ridge linear) over 5 features —
`tool_calls_count / react_steps / latency_ms / prompt_tokens /
completion_tokens` — and computes a quantile-grid partial dependence per
feature. L2 regularisation + step clipping keep IRLS stable (a lesson
learnt on the saturated sigmoid in Phase 2).

Confidence-calibration joint diagnosis (`confidence_calibration` /
`numeric_confidence_calibration`): treat the final-step `confidence ∈
{low, medium, high}` from `belief_trace` as subjective confidence, and
`max_l p_l` as numeric confidence; compare each against hit rate.
`confidence_conflict_models` sentinel:
(a) `low` bucket `mean_max_p > 0.70` — verbally conservative +
numerically overconfident;
(b) `high` bucket `mean_max_p < 0.55` — verbally confident +
numerically underconfident. On either, append `conflict*` after the
model name in `per_model_summary.md`. This is a diagnostic dimension
**absent** from the paper — the paper only has binary $p$ and cannot
decouple *language* and *numeric* confidence.

**Phase 3 artefact list**:

| File | Content |
| --- | --- |
| `belief_evolution.csv` | per-(model, q, k) 5-metric rows |
| `reflection_ab.csv` | paired-run ΔBI / Δσ / ΔC / Δη paired-bootstrap CI (with per-qtype slice) |
| `tool_usage_pdp.csv` | per-(model, feature, value) PDP rows |
| `confidence_calibration.csv` | per-(model, low/medium/high) subjective confidence vs hit rate |
| `numeric_confidence_calibration.csv` | per-(model, max_p bin) numeric confidence vs hit rate |
| `per_model_summary.md` | append `conflict*` sentinel (alongside the existing `cal*`) |

**Visualisation**: `scripts/plot_analysis.py` is an on-demand CLI
(matplotlib is installed only on the user's local machine); reads
`analysis/*.csv` to produce figures:

* **v5 main figures**: `fss_bar_with_ci.png` / `delta_fss_forest.png` /
  `entropy_accuracy_grid_<model>.png` (per-model 3 buckets × 3 metrics);
* **Companion / appendix**: `bi_bar_with_ci.png` (BLF benchmark anchor) /
  `delta_bi_forest.png` / `difficulty_grid.png` / `belief_trajectory_*.png`
  / `tool_pdp_*.png`.

v5 removed three figures
(`reliability_diagram_per_model.png` / `_calibrated.png` /
`brier_decomp_stacked.png`) since their input data was deleted. All
figures land in `analysis/figs/` (gitignored). matplotlib is not in
`environment.yml` and does not affect CI.

**FSS sensitivity (on-demand CLI)**: `scripts/fss_sensitivity.py` is a
one-shot script running 4 (α, β) tiers and writing
`fss_sensitivity.csv`; not part of the `run_analysis` main flow
(Decision 12). The paper appendix uses it to answer "why (2, 0.5) and
not (1, 1)?".

When `Settings.BELIEF_PROTOCOL=false`, the legacy accuracy columns
output zero changes; the new probabilistic columns are still computed
via fallback (although the calibration signal is weakened); behavioural
analysis degrades to empty deliverables (legacy v3 runs lack
belief_trace, so `belief_evolution.csv` is not written). This
backward-compatibility ensures the v3→v4 unidirectional migration does
not require re-running historical runs.

<!-- exam-score-metric: removable section ↓ — delete down to the next `### 11.6` heading -->

#### Exam-style partial credit (exam_score)

`forecast_eval/analysis/exam_score.py` adds an "explanatory ruler for
the public" alongside §11.5's discrete-native metric family. Formula:
$\text{exam\_score}(\hat S, G) = (|\hat S \cap G|/|G|) \cdot \mathbb{1}(\hat S \setminus G = \emptyset)$
— any wrong selection → 0 directly, otherwise score by partial recall;
orthogonal to FSS's chance correction / Tversky soft penalty.

Aggregation uses the per-question mean → cross-question mean two-step
($e_q = \frac{1}{|S_q|}\sum_s \text{exam\_score}$, global $=
\frac{1}{|Q|}\sum_q e_q$); the per-question denominator is the **actual
sample count entering the denominator** (cutoff / error excluded, parse
failure counted as 0).

Written into the `exam_score_at_n_avg` column of
`per_model_summary.csv`, immediately following `at_least_all_at_n`;
automatically follows all `_slice_by` slices via the `Aggregate` field.

SAMPLING_N is reinterpreted under this metric's view as "the number of
independent trials" (each score independent [0,1], final arithmetic
mean), coexisting with the best-of-N "take the highest" framework
(pass_any_at_n / majority_vote_accuracy).

Constraint: this metric is **fully removable** — delete `exam_score.py`
+ `tests/test_exam_score.py` + the hook points marked in code/docs (the
marker literal and catalogue are in spec
`openspec/changes/add-exam-score-metric/specs/exam-score-metric/spec.md`
§"removability equivalence"); the repo returns to byte-level identical
state.

### 11.5.5 Composite score weighted by sub-question type (composite-score-by-subtype)

Module: `forecast_eval/analysis/composite.py`; writes:
`per_model_composite_by_question_type.csv` /
`per_model_composite_by_choice_type.csv` / `composite_meta.json`，
and embeds a `composite` section in `overall.json`.

**Input collection**:

| Source | Data shape | Column coverage |
| --- | --- | --- |
| `analysis._slice_by(samples, key_fn=question_type/choice_type, ...)` | `{model: {bucket: Aggregate}}` | v3 accuracy + final_answer_retry_rate (23 columns total) |
| `composite.slice_v5_metrics_by_bucket(...)` | `{model: {bucket: V5SliceResult}}` | v5 discrete family 8 columns (FSS / Cohen κ / Hamming / Fleiss κ / mean entropy / VCI / MVG / fss_pe_mean) |
| `probabilistic.build_probabilistic_report(...).per_model_by_qtype/_by_ctype` | `{model: {bucket: ModelProbabilisticAggregate}}` | v4 probabilistic family 7 columns (BI / BI_dec / NLL / MBS / ABI_crowd / ABI_uniform / fallback_share) |

`composite.collect_bucket_values(...)` aggregates the three sources into
the unified shape `{model: {metric: {bucket: value}}}`, which is then
processed by `compute_composite(...)` to produce a `CompositeReport`.

**Weighting formula**: see `DESIGN.md` §3.5. Missing buckets are dropped
and proportionally renormalised; all None → composite returns None;
weights are not required to be normalised.

**Configuration** (`Settings`):

| Field | Default | Notes |
| --- | --- | --- |
| `COMPOSITE_WEIGHTS_QTYPE` | `yes_no=0.15,binary_named=0.15,multiple_choice=0.70` | global default weights for the qtype dimension |
| `COMPOSITE_WEIGHTS_CTYPE` | `single=0.40,multi=0.60` | global default weights for the ctype dimension |
| `COMPOSITE_WEIGHT_OVERRIDES_QTYPE` | `{}` | per-metric overrides for the qtype dimension; form: `"fss=yes_no=0.05,multiple_choice=0.95"`, semicolon-separated for multiple metrics |
| `COMPOSITE_WEIGHT_OVERRIDES_CTYPE` | `{}` | same as above for the ctype dimension |

Startup validation (`Settings.model_validator`): bucket name must ∈ the
legal set, weight ≥ 0, at least one > 0. **Metric-name validation** is
placed at the `compute_composite` entrypoint (raising at runtime) — to
avoid a reverse-import cycle (`config.py` importing `analysis.composite`);
the cost of a misspelled metric name is failing during the analysis
phase rather than startup, with equally clear error messages.

**Output contract**:

| File | Column order |
| --- | --- |
| `per_model_composite_by_question_type.csv` | `model + sampling_n + weights_kind + (_SUMMARY_FIELDS data columns)` |
| `per_model_composite_by_choice_type.csv` | same as above |

`weights_kind` ∈ {`default`, `overridden`}: if any metric in this
(model) row hits an override, the row is marked `overridden`.

`composite_meta.json` is the audit trail: per (model, metric), records
`value / buckets_used / weights_used_normalized / bucket_values /
weights_kind`. The `composite` section embedded in `overall.json` is a
condensed version with the same structure.

**Alignment with existing slice tables**:
`per_model_by_question_type.csv` / `per_model_by_choice_type.csv` now
also include the v5 discrete-family columns (previously NULL
placeholders) — a by-product of `slice_v5_metrics_by_bucket`, with
column order matching `per_model_summary.csv`.

**Wiring with evaluation**: `evaluation.py` passes the 4 `COMPOSITE_*`
fields on `Settings` through to the keyword arguments of
`analysis.run_analysis(...)`; the `analysis` module itself does not read
`.env`. The CLI entrypoint `python -m forecast_eval.analysis ...`
calls `load_settings()` best-effort; on failure (e.g. no `.env`), it
falls back to `composite.DEFAULT_WEIGHTS_*` + empty overrides — unit
tests can produce composite files with zero configuration.

### 11.6 Grid-search analysis (`react-tavily-grid-search`)

`Settings.TAVILY_MAX_RESULTS` (R) and `REACT_MAX_SEARCH_CALLS` (C)
support comma-separated multi-value lists; the evaluation entrypoint
performs Cartesian expansion over `MODELS × R_list × C_list`, encoding
each `(real_model, R, C)` cell as the virtual slug
`{real}::r{R}::c{C}`. The runner / DB schema / existing analysis main
pipeline are **unchanged byte-for-byte**.
`forecast_eval/analysis/grid.py` decodes the triple, re-aggregates, and
emits paper long tables. For detailed decisions see `DESIGN.md` "grid
search via virtual slug (option C)".

| File | Content |
| --- | --- |
| `grid_summary.csv` | per `(real_model, R, C)` 17-column main table: accuracy/BI/NLL + 95% CI + cost columns like `mean_search_calls / mean_latency_ms` |
| `grid_marginal_C.csv` / `grid_marginal_R.csv` | scan along C with `R = default_r` fixed / scan along R with `C = default_c` fixed |
| `grid_pareto.csv` | one row per cell; for frontier cells the `dominated_by` column is empty, otherwise records the lex-smallest dominator's virtual slug |
| `grid_winrate.csv` | per `(real_model_a, real_model_b)` pair: cross-(R, C) cell wins/ties + paired-bootstrap significant-cell count |

All CIs go through `inference.paired_bootstrap` (5000 resamples,
seed=42); BI-domain CIs are obtained via "BS-domain paired bootstrap +
monotone transform $\mathrm{BI}=100(1-\sqrt{\mathrm{BS}})$" with no new
statistical code (see `DESIGN.md` D8).

Visualisation is produced on demand by `scripts/plot_analysis.py` when
the main flow detects the `manifest.grid` block:

| Figure | Content |
| --- | --- |
| `grid_pareto_C.png` | Fig 1 main: with `R = default_r` fixed, one `BI vs mean_search_calls` curve per real_model + 95% CI band, Pareto cells starred |
| `grid_pareto_C_R{R}.png` | Appendix: same-format figure for each non-default R |
| `grid_heatmap_RC_<real_model>.png` | Fig 2 per real_model: (R, C) plane BI heatmap; cells whose CI overlaps the best cell are hatched |
| `grid_curve_C.png` / `grid_curve_R.png` | Fig 3: 3 rows (BI / NLL / Acc) × M columns panel, with CI shading + saturation point (first-order difference < 0.01) marked with a dashed vertical line |
| `grid_winrate_matrix.png` | Fig 4: `M × M` matrix of "row beats column" share, with cells where sig\_cells_* ≥ 1 marked `*` |

For legacy v4 runs (manifest without a `grid` block),
`run_grid_analysis` early-exits and writes no `grid_*.csv`; the plot
flow also skips the grid figure family — a single-value .env, parsed
under the new code as a length-1 list → Cartesian product producing a
single virtual slug, behaves byte-equivalently to the pre-change
behaviour (except for the `__r{R}__c{C}` suffix on .db filenames).

---

## 12. CLI and how to run

### 12.1 Commands

```bash
# Run all 319 questions
python evaluation.py

# Filter by question_type (repeatable)
python evaluation.py --question-type yes_no --question-type binary_named

# Filter by choice_type (repeatable)
python evaluation.py --choice-type single

# Combined filter (AND): only multi-select multiple_choice questions (34)
python evaluation.py --question-type multiple_choice --choice-type multi

# Do not generate analysis/ at run end (raw DBs still land in db/)
python evaluation.py --skip-analysis

# Refresh analysis/ independently (does not modify the DB)
python -m forecast_eval.analysis runs/{run_id}
```

`--question-type` values: `yes_no` / `binary_named` /
`multiple_choice`, repeatable; if not passed = no restriction.
`--choice-type` values: `single` / `multi`, repeatable; if not passed =
no restriction. All tunables other than `--skip-analysis` still go
through `.env`.

### 12.2 Flow

```
1. argparse parses --question-type / --choice-type / --skip-analysis, assembling a QFilter
2. Settings() loads and validates .env (including MODEL_TRAINING_CUTOFFS + RUNS_ROOT)
3. Generate or reuse run_id -> determine run_dir = RUNS_ROOT/{run_id}; create db/ / analysis/ / logs/
4. Compute source_db_hash / metadata_hash / prompt_templates_hash
5. For each MODELS[i]:
   a. open conn = RUNS_ROOT/{run_id}/db/{safe_slug(model)}.db
   b. db.init_schema(conn, SAMPLING_N)  # dynamically create s{i}_* columns
   c. loader.sync_prompt_templates(src, conn) / loader.sync_questions(src, conn, filter)
   d. db.register_run_meta(conn, run_id=..., model=..., hashes=..., training_cutoff=...)
6. Write manifest.json (run_id, models, model_files, sampling_n, filters, hashes, started_at)
7. runner.run(..., conns={model: conn, ...}) starts the asyncio event loop
   a. For each model, db.load_completed_samples(conn, SAMPLING_N) becomes the resume baseline
   b. Generate the Cartesian product: questions × MODELS × range(SAMPLING_N); subtract the resume set
   c. §3.9 filter: rows with q.end_time <= cutoff for (q, model, idx) are written directly as
      skipped_training_cutoff rows to the corresponding model's writer, not entering the LLM task queue
   d. Remaining tasks: Semaphore-limited (one each for LLM / Search) concurrency
   e. Each completion → routed to that model's writer → batch UPSERT into s{i}_* columns
   f. One log line per completion: [x/xx] q=.. qt=.. ct=.. model=.. idx=.. correct=..
8. For each model, db.finish_run_meta(conn, run_id); finalise manifest.finished_at
9. Unless --skip-analysis: call forecast_eval.analysis.run_analysis(run_dir), write analysis/
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
```
12:03:44 | INFO    | [run=20260424-120344-a7k3] [5/1610] q=69566c13 qt=binary_named ct=single model=openai/gpt-5 sample=2/5 correct=True steps=4 tool_calls=3 latency=8421ms
```

- `[5/1610]` denominator = `len(questions_after_filter) × len(MODELS) × SAMPLING_N` (minus completed resume tasks)
- One line printed per sample completion
- On error, print at `ERROR` level: `[x/xx] q=.. model=.. error=rate_limit retry_exhausted`

---

## 14. Conda environment (`environment.yml`)

```yaml
name: forecast
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - pip:
      - openai>=1.50            # OpenRouter uses the OpenAI-compatible SDK
      - tavily-python>=0.5
      - pydantic>=2.6
      - pydantic-settings>=2.2
      - python-dotenv>=1.0
      - loguru>=0.7
      - httpx>=0.27
      - tenacity>=9.0           # retry decorator, implements tiered backoff
      - pytest>=8.0             # §17 tests
      - pytest-asyncio>=0.23    # async test support
      - respx>=0.21             # httpx mocking, for LLM / Tavily dry-run
```

Create the environment:
```bash
conda env create -f environment.yml
conda activate forecast
cp .env.example .env
# Edit .env to fill in LLM_API_KEY and TAVILY_API_KEY (LLM_BASE_URL can point to any OpenAI-compatible endpoint)
python evaluation.py --question-type yes_no
```

---

## 15. Final-premise summary (for last review)

1. **Source data has 7 fields**: `id / choice_type / question_type / event / options / answer / end_time`, with letter-encoded answers throughout
2. **Source DB `forecast_eval_set_example.db` is checked into Git** (read-only example dataset, ships with the repo, ensures `source_db_hash` reproducibility; `SOURCE_DB` / `SOURCE_TABLE` can point to a custom dataset)
3. **The LLM does not see `end_date`**; injection happens at the tool implementation layer
4. **Tavily `end_date = end_time + TAVILY_END_DATE_OFFSET_DAYS`**; the project uses `-1` as the strict default baseline (all reports default to comparison under `-1`)
5. **Leak boundary and threat model** (§3.8): the tool layer can only constrain tool-search; provider-native browsing / `:online` is forcibly disabled; parametric memory is partially mitigated by §3.9's training-cutoff filtering
6. **Filter questions by model training cutoff** (§3.9): `.env`'s `MODEL_TRAINING_CUTOFFS` specifies each model's cutoff; samples with `q.end_time ≤ cutoff` are written as `error="skipped_training_cutoff"`, do not call the LLM, and are not retried on resume
7. **Prompt assembly is performed by `prompts.py`**: pull templates from `dataset_metadata` → render `outcomes_block` and `output_format` per `question_type` (substitute `<options[i]>` placeholders for binary_named); > 26 options use the source-data ASCII-continuation compatibility mode (§3.7 warning)
8. **Evaluation = letter-set frozenset strict equality**; missed and extra selections are all wrong
9. **Parse failure ≠ error**; refusal / format_failure rate are tallied separately
10. **Multi-model single-run Cartesian product**, with resume via `run_id`; one DB per model, the single row in `run_meta`
    records `filters_snapshot` + `source_db_hash` + `metadata_hash` + `prompt_templates_hash`
    + `training_cutoff` + **redacted** `config_snapshot` (no API-key plaintext is persisted)
11. **Auth errors stop the entire run**; other errors are retried with tiered backoff; on retry exhaustion, skip + record `error`
12. **Content policy violations are not retried**, just marked
13. **All flexible parameters live in `.env`**; CLI exposes only `--question-type` / `--choice-type` / `--skip-analysis`
14. **Main entrypoint `evaluation.py`**: creates `RUNS_ROOT/{run_id}/`, runs the runner, runs analysis (unless `--skip-analysis`)
15. **Conda + Python 3.12 + loguru**, with progress `[x/xx]` logged
16. **SQLite WAL + `PRAGMA foreign_keys=ON` + one async writer task per model**, avoiding concurrent-write lock contention
17. **Each model DB is self-contained**: built-in `questions` + `prompt_templates` copies + `run_meta`, independently distributable and replayable
18. **Metric naming**: the standard `pass@k` corresponds to this project's `pass_any@N`; the legacy threshold-style metric is renamed `at_least_k_correct@N`
19. **Recording and analysis are separated**: the DB stores only raw sample records; pass@1 / pass_any@N / majority / parse_failure / cutoff_skip etc. are computed post-hoc by `analysis.py` and written to `analysis/` as CSV / MD / JSON

---

## 16. Suggested module landing order

1. `environment.yml` + `.env.example` + `.gitignore`
2. `forecast_eval/config.py` (Settings class, with `RUNS_ROOT`)
3. `forecast_eval/db.py` (per-model wide-table schema + `AsyncWriter` + resume queries + prompt_templates table + model_slug_safe)
4. `forecast_eval/loader.py` (sync questions + prompt_templates)
5. `forecast_eval/prompts.py` (render user message per question_type, **with unit tests covering all three types**)
6. `forecast_eval/parser.py` (`\boxed{}` parsing + letter-set normalisation + strict matching, **with unit tests covering all three types + edge cases**)
7. `forecast_eval/errors.py` (error classification + backoff)
8. `forecast_eval/search.py` (Tavily + end_date injection)
9. `forecast_eval/tools.py` (schema + execute_tool_call)
10. `forecast_eval/llm.py` (OpenRouter client + retry)
11. `forecast_eval/react.py` (single-sample ReAct loop)
12. `forecast_eval/runner.py` (orchestration + concurrency + multi-model writer + progress)
13. `forecast_eval/analysis.py` (post-hoc statistics, read DB -> CSV / MD / JSON)
14. `evaluation.py` (main, create directories + register_run_meta + runner + analysis)

Get a smoke test passing first via `--question-type yes_no` +
`MODELS=openai/gpt-4o-mini` + `SAMPLING_N=1` (93 questions, the cheapest
question type), verify that `prompts.render_user_prompt` output and
`parser.parse_answer` normalisation are correct, then open up to full
evaluation.

---

## 17. Test plan (`tests/`)

A single evaluation is costly (319 questions × number of models × N
samples), so getting tests stable first saves a lot of API spend. All
tests run **offline** and **do not burn the API**: Tavily / OpenRouter
exist as fixtures or mocked stand-ins.

| Test file                   | Subject               | Key cases                                                                                                                                                                                                                |
| --------------------------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_prompts.py`           | `prompts.py`          | ① snapshots of all three template renderings for `yes_no` / `binary_named` / `multiple_choice` (≤26 options); ② correct `binary_named` placeholder substitution; ③ accurate `outcomes_block` labels for `multiple_choice` > 26 options (using fixtures from the 4 real questions in the DB)                  |
| `test_parser.py`            | `parser.py`           | ① positive `\boxed{}` paths for all three question types; ② multiple `\boxed{}` → take the last one; ③ mixed case, spaces, comma/space separators; ④ illegal letter out-of-range; ⑤ > 26 options label↔letter round-trip; ⑥ `parse_gt` parses `"A, B"`; ⑦ soft refusal → None without raising               |
| `test_search.py`            | `search.py` + `tools.py` | ① `web_search` schema LLM-visible fields **do not contain** `end_date`; ② `tavily_search` injects `end_date = q.end_time + OFFSET`; ③ Tavily errors retry via `SEARCH_BACKOFF_S`; ④ after retry exhaustion, returns an error payload instead of throwing; ⑤ `_build_request_payload` maps the three `TAVILY_INCLUDE_RAW_CONTENT={false,markdown,text}` / `TAVILY_SEARCH_DEPTH` / `TAVILY_INCLUDE_ANSWER` enum values to the Tavily protocol form; ⑥ overly long `raw_content` is truncated to `TAVILY_RAW_CONTENT_MAX_CHARS` at `_truncate_raw_content` with an ellipsis marker; ⑦ `to_llm_payload` does not output `null` placeholders for missing fields (`score` / `raw_content` / `published_date` / `answer`) |
| `test_db.py`                | `db.py`               | ① per-model schema dynamically creates `s{i}_*` columns per `sampling_n` + PRAGMA; ② fail-fast on schema `N` mismatch; ③ `model_slug_safe` rules; ④ hash computation stable; ⑤ `config_snapshot` redaction; ⑥ UPSERT overrides by `(qid, sample_idx)`; ⑦ `AsyncWriter` bucket-wise batched commits |
| `test_runner_resume.py`     | `runner.py`           | ① `load_completed_samples` excludes retryable errors; ② `build_task_plan` deduplicates by per-model completed; ③ models not declared in `completed` default to empty set (all enqueued)                                                                    |
| `test_training_cutoff.py`   | §3.9 filtering logic | ① every N samples for `q.end_time <= cutoff` writes skipped_training_cutoff; ② models without declared cutoff are not filtered; ③ resume takes precedence over cutoff; ④ after writing, `load_completed_samples` matches                                                       |
| `test_llm_no_browsing.py`   | `llm.py`              | mock client asserts the request payload **does not contain** `plugins`, `tools` does not contain provider-native web_search, and the model name does not end in `:online`                                                                                              |
| `test_errors.py`            | `errors.py`           | various `httpx` / OpenAI exceptions → correct `ErrorKind`; `Retry-After` header takes precedence over default backoff                                                                                                                                      |
| `test_analysis.py`          | `analysis.py`         | ① hand-crafted wide-table fixture; ② pass@1 / pass_any@N / ≥majority / ≥all / majority_vote / parse_failure / error_rate / cutoff_skip values are correct; ③ `overall.json` aligns with the CSVs; ④ `error_breakdown.csv` aggregation                           |
| `test_smoke_dry_run.py`     | end-to-end dry-run    | replace OpenRouter + Tavily with httpx stubs, run 3 questions × 1 model × 1 sample, verify the wide-table `s0_*` fields are complete, `messages_trace` is legal JSON, and `search_calls` records `end_date`                                                               |

Run:
```bash
pytest tests/ -q
```
CI minimum: `test_prompts.py` / `test_parser.py` / `test_training_cutoff.py`
/ `test_llm_no_browsing.py` / `test_analysis.py` — these five must stay
green (core semantics + safety boundary + statistical correctness).
