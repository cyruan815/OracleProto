# Forecast Evaluation — Design Rationale

> This document is for readers new to the project who want to understand "why
> we did it this way" rather than "how exactly to do it". For concrete
> interfaces, field definitions, and parameter lists, read this alongside
> `FRAME.md`; here we focus on **the motivation behind each design trade-off**.

---

## 0. Foreword: the question this project tries to answer

> **If we never let the LLM see information published after an event has been
> resolved, how strong is its ability across 319 real-world forecasting
> questions?**

Around that single question, the project imposes three almost "religious"
hard constraints on itself:

1. **The information boundary must be strict**: the model can only obtain
   external information through our `web_search` tool, and the tool may only
   surface content from before the event resolution date.
2. **Results must be reproducible**: the same dataset + same config + same set
   of models → anyone can re-run and obtain comparable numbers.
3. **The process must be auditable**: every model call, every search, every
   parse must be traceable down to the field level after the fact.

The whole design — from prompt assembly, tool schema, DB schema to resume
semantics — is the concrete embodiment of these three constraints. Once you
internalise this, every "seemingly over-strict" design below feels natural.

---

## 1. Information isolation: the project's "first principle"

### 1.1 The LLM never sees `end_date`

The schema the `web_search` tool exposes to the model has only one `query`
parameter. When Tavily is actually called, `end_date` is **hard-coded and
injected by the tool implementation layer** from the current question's
`end_time` — the model can neither perceive nor bypass it.

There are two design philosophies underneath:

* **Aligning capability boundaries with tool boundaries**: capability
  ("knowing the world up to a particular day") is determined by system
  configuration, and should not be something prompt engineering or model
  behaviour can affect. By making the model unable to even see "which day I
  am cut off at", we prevent it from inferring or working around that
  boundary via prompt construction, parameter injection, etc.
* **Single, controllable failure mode**: if we exposed `end_date` as an LLM
  tool argument, we'd have to assume the model could "forget to fill it in"
  or "deliberately fill in a future date". Holding the decision inside the
  tool implementation collapses the failure mode from "the model might make
  a mistake" to "our code might make a mistake" — the latter is testable,
  auditable, and unit-testable.

### 1.2 Default leans strict: `end_date = end_time - 1 day`

`TAVILY_END_DATE_OFFSET_DAYS=-1` is the project default. The reason is
straightforward: many questions (sports events, central bank decisions, Oscar
nominations) get resolved on the same day, and using the question's
`end_time` as the search cutoff would likely surface news summaries that
already contain the answer. Pushing the search time forward by one day
**trades a little information granularity for strictness**.

Reports also default to comparison at `-1` — this is itself a design
constraint: "numbers under different offsets are not directly comparable".

### 1.3 Provider-native browsing is forcibly disabled

OpenRouter / OpenAI / Anthropic each have their own web tool or `:online`
suffix. The moment we go down that path, the time cutoff is completely out
of control. The project enforces this on two layers:

* **Code layer**: `llm.chat` only attaches our own `WEB_SEARCH_SCHEMA`; any
  `plugins` / `:online` / provider-native retrieval is forbidden.
* **Test layer**: `test_llm_no_browsing.py` directly mocks the client and
  asserts the outbound payload contains none of those fields.

Design philosophy: **the "temptation" of external tools must be rejected at
the earliest possible stage**. If even one release "for convenience" turned
this on once, the comparability of the entire dataset would be ruined.

### 1.4 Training-data contamination: filtering, not lying

The tool cutoff cannot constrain facts the model has already memorised in
its parameters. The project takes a very plain strategy:

> Declare each model's **training cutoff date**; if a question's `end_time ≤
> cutoff`, that question is simply skipped for that model.

Skipped samples still write a row into the DB with
`error="skipped_training_cutoff"`:

* Reports can clearly show "how many questions were filtered out per model
  and how many remain comparable".
* `resume` will not retry that row (distinguishing it from transient errors
  like `network`).
* It is not counted in `error rate by kind` — it is not a failure, but
  **active data cleansing**.

Behind this is a design principle the project repeatedly invokes: **"filtered
out" and "failed" are two different semantics and must be separated at the
data layer**. If we only used a boolean `skipped` field, future stratified
reporting by cutoff would lose information.

### 1.5 What we can and cannot control

`FRAME.md §3.8` contains a "threat model" table that is in essence an
**honest confession**:

| Leakage source                          | Controllable? |
| --------------------------------------- | ------------- |
| Tavily returned content                 | ✅           |
| Provider built-in browsing              | ✅           |
| Model parameter memory                  | ⚠️ Partial (mitigated by §1.4) |
| Future leakage in Tavily snippets       | ⚠️ Partial (since v5.2 backed by detector, see §1.6) |
| Time clues in the question text itself (e.g. year)  | ❌  |
| External knowledge backflow after training | ❌        |

Design philosophy: **acknowledging what we cannot control matters more than
pretending we can**. The uncontrollable parts are accepted as part of the
evaluation bias; the controllable parts are locked down by code + tests.

### 1.6 Stage 2 LLM content audit: covering the semantic leakage Tavily's protocol layer cannot catch (v5.2)

§1.1 - §1.4 are protocol-layer (schema / `end_date` injection / `:online`
disabling / cutoff skipping) defences. The class of leakage this layer
cannot cover is **the body of a Tavily-returned page describing events that
happened after `end_time`**: Tavily's `end_date` filter operates on a page's
*crawl/index* time, not on the event time *described in the page content*. A
wiki / aggregator page / long article indexed before `end_date` can perfectly
well reference future events in its body.

The `search-leak-filter-v1` solution is to add an independent LLM audit layer
(the "detector") at the end of `tavily_search`, before the main LLM sees
`tool_result`: each `SearchResultItem` is sent to the detector individually,
with verdict ∈ {keep, drop, failed:*}. Items with verdict=drop are removed
entirely — the main LLM never sees any field of a dropped item (including
title / url / content / raw_content).

| Dimension      | Implementation                          |
| -------------- | --------------------------------------- |
| Cut point      | end of the 200 path in `forecast_eval/search.py:tavily_search`, before `return` |
| Client         | `_detector_client: AsyncOpenAI`, independent singleton, not shared with the main LLM `_client` |
| input fields   | whitelist: title / url / published_date / content / raw_content / cutoff_date; MUST NOT contain any field of `Question` (to prevent the detector from morphing into an "answer auditor") |
| prompt strictness | 6 principles: cutoff_date placeholder, treat specific/scheduled/speculative future events equally, "ambiguous → drop", forbid parametric knowledge, strict JSON output, no awareness of the question |
| Failure mode   | FAIL-RETRY → CLOSED: by default, K retries still failing → drop; AUTH errors are caught locally and immediately drop (no propagation, no aborting the whole run) |
| Observability  | `search_calls.detector_*` five fields + `run_meta.config_snapshot` detector three-key fingerprint |
| Master switch  | `ENABLE_SEARCH_LEAK_FILTER`, default True; when off, byte-level rollback |

The detector's reference date is *the question's* `end_time +
TAVILY_END_DATE_OFFSET_DAYS` (sharing the same source as Tavily's
`end_date`), independent of §1.4's `MODEL_TRAINING_CUTOFFS` (the model
training cutoff, indexed *per model*). Even after a question passes through
the cutoff filter and enters execution, the detector still audits the search
results for *that question* — the two do not substitute for each other.

Reference: BLF paper ([arXiv:2604.18576](https://arxiv.org/abs/2604.18576)
§B.1 Stage 2): their LLM-based leak classifier adds a layer of LLM audit
after Brave's date filter; post-hoc audit shows the runtime filter catches
320/341 = 93.8% of actual leakage, and the residual leakage rate the agent
ultimately sees is only 1.5%. Stage 2 is an empirically validated
"algorithmic-layer + semantic-layer double insurance" engineering practice.

For the full spec see `openspec/changes/search-leak-filter-v1/specs/`
(capabilities `search-leak-filter` / `search-tool` / `information-barrier` /
`results-persistence`).

---

## 2. Reproducibility: every run is its own independent space-time

### 2.1 The source database is checked into Git

`forecast_eval_set_example.db` goes straight into the repo. It is the
evaluation's "gold-standard" example dataset and must ship with the repo;
anyone can `git clone` and obtain the exact same 319 questions. The filename
(`SOURCE_DB`) and the internal question table name (`SOURCE_TABLE`, default
`forecast_eval_set_example`) are both exposed as `.env` parameters; with a
custom dataset, just change these two variables and the loader splices
`<SOURCE_TABLE>` into the SQL `FROM` clause at runtime. The table name is
whitelist-validated (`[A-Za-z_][A-Za-z0-9_]*`) at the Settings stage to
foreclose injection.

Each run also computes `source_db_hash` and writes it to `run_meta`; together
with `metadata_hash` and `prompt_templates_hash`, this forms a three-part
fingerprint of "exactly which inputs this run is based on".

### 2.2 Each run gets its own directory

```
runs/{run_id}/
  manifest.json          # run-level metadata
  db/{model_slug}.db     # one sqlite per model
  analysis/              # post-hoc statistical artefacts
  logs/{run_id}.log
```

A few details of the design choice:

* **Directory per run, not single DB**: the early "single `results.db`" was
  replaced. The reason: with a single DB, the boundary between runs depended
  entirely on the `run_id` column, which made independent distribution hard
  and made it easy to mix data from other runs into analysis.
* **`run_id` defaults to `YYYYMMDD-HHMMSS-xxxx`**: `ls` naturally sorts by
  time, and since this is also the directory name you can tell "when it ran"
  at a glance.
* **`RUN_ID` empty → new run; same value → resume**. One variable handles
  both, no extra `--resume` CLI flag needed.

### 2.3 One SQLite per model

Why not "one big DB per run"? Three reasons, in order of importance:

1. **Independently distributable**: hand `runs/{run_id}/db/openai__gpt-5.db`
   to someone else and they can replay just this one model, with no need to
   obtain the other models' results.
2. **Non-interfering write paths**: one async writer task per model, with
   single-writer-multi-reader WAL mode providing ample concurrency, and one
   model's stall cannot block another's.
3. **Easy schema-evolution isolation**: if some model needs to store a
   special field (e.g. reasoning trace), its schema can be extended
   independently without affecting others.

The cost is that the analysis layer must scan multiple files, but
`analysis.py` already encapsulates that.

### 2.4 Each DB self-contains `questions` + `prompt_templates` copies

Each model DB embeds copies of the source question set and prompt templates.
This looks redundant at first glance, but it serves "independent replay":

> Whoever receives `openai__gpt-5.db` does not need to track down
> `forecast_eval_set_example.db`, nor hunt for which metadata version was in
> use at the time — every input the evaluation needed is inside this single
> DB.

Consistency between copies is guaranteed by hash verification: three fields
— `run_meta.source_db_hash` / `metadata_hash` / `prompt_templates_hash` —
pin down "the source data at the time".

### 2.5 The config is redacted before being written into the DB

`run_meta.config_snapshot` stores the redacted `.env` as JSON. Sensitive
fields like `LLM_API_KEY` only retain the first 4 characters + length +
`sha256[:12]`.

Design philosophy:

* **Want to know which parameters (temperature, concurrency, retry sequence)
  this run used?** — stored.
* **Want to know the plaintext of the key that was used?** — never stored.
* "Auditable" and "non-leakable" coexist within the same field.

---

## 3. Scoring system: as plain as it gets

### 3.1 Strict frozenset equality on letter sets

The entire scoring logic is one line:

```python
predicted_letters == ground_truth_letters  # both are frozenset[str]
```

Missed selections, extra selections, ordering — all are scored as wrong by
"strict equality". This is the project's most "fastidious" design.

#### Why not partial credit (Jaccard / F1)?

* **Explanation cost is too high**: `pass@1=0.62` is far more intuitive than
  `mean Jaccard=0.74`; one number is enough for a paper.
* **Avoid half-credit masking the real problem**: if a model misses one or
  two selections every time, its average score can still be 70+, but
  fundamentally it has **not really mastered** that question. Strict
  matching scores this behaviour as 0, forcing the report to be honest.
* **Unifies the scoring interface across three question types**: `yes_no` /
  `binary_named` / `multiple_choice` are all frozensets at the scoring
  stage; write the code once, all three types work.

#### Letter encoding is the canonical answer

The source DB's `answer` field uniformly uses letters (`'A'` or `'A, B'`)
rather than option text:

* yes_no: `Yes=A, No=B`
* binary_named: the first entity = A, the second = B
* multiple_choice: A/B/C/... follow the order of the options array

The model's output form varies by question_type (`Yes`, entity name, letter
list are all possible), but the parser uniformly normalises to
`frozenset[str]`. This design completely decouples "how the model says it"
from "how the system scores it".

### 3.2 `parse_ok=0` is not an error

The LLM didn't output `\boxed{...}`, or wrote "I cannot predict the future"
— **this is not a system failure, it is part of the model's capability**.

Concretely:

* **`parse_ok=0`, `correct=NULL`**: parse failures / refusals are
  accumulated separately into "refusal rate" and surfaced in reports.
* **No retry**: when the model itself says it cannot answer, asking again
  yields the same answer — backoff retries only waste tokens.
* **`error` field stays NULL**: the `error rate by kind` report is not
  polluted by such "soft failures".

Design philosophy: **every behaviour must have its own cell in the
report**. "System error rate" and "model refusal rate" are two different
things and cannot be lumped under a single total error rate.

### 3.3 ASCII continuation: imperfect, but keeps the mapping stable

The DB contains 4 `multiple_choice` questions with > 26 options, of which 3
have ground-truth answers landing on ASCII continuation characters like
`[ \ ] ^ _ ` ` ` a b c ...`.

These characters are extremely unfriendly to LLMs (backticks are eaten by
markdown, lower/upper-case a/A are easily confused), but the project still
keeps them — in order to **preserve a one-to-one letter ↔ index mapping**.

The cost is mitigated by several defences:

1. `prompts.render_user_prompt` explicitly quotes or escapes labels when
   generating an `outcomes_block` for > 26 options.
2. `parser.parse_answer` MUST have a round-trip unit test for > 26 options.
3. Logs/reports record letters and labels in parallel for manual review.

If we later confirm LLM performance is meaningfully dragged down by the
labelling scheme, we'll migrate to a stable scheme like `AA/AB` or
`A01/A02`. **This is a documented "debt", not an ignored bug**.

<!-- exam-score-metric: removable section ↓ — delete down to the next `---` separator -->

### 3.4 Exam-style partial credit (exam_score)

`forecast_eval/analysis/exam_score.py` provides an "explanatory ruler for
the public" alongside the primary metric FSS (Tversky α=2/β=0.5 + chance
correction). The formula is one line:

$$
\text{exam\_score}(\hat S, G) = \begin{cases}
|\hat S \cap G| / |G| & \hat S \setminus G = \emptyset \\
0 & \hat S \setminus G \ne \emptyset
\end{cases}
$$

— any wrong selection → 0 directly; otherwise score by the proportion
correct. Formally equivalent to **Recall under zero-FP gate**: Recall plus a
hard "any FP vetoes" gate.

#### Denominator rule (intentionally different from `pass_at_1_avg`)

| sample state | `exam_score` | enters denominator? |
| --- | --- | --- |
| `is_cutoff` (question later than training cutoff) | `None` | no |
| non-cutoff `error` (sensitive word, API timeout, etc.) | `None` | no |
| `error=None` and `parse_ok != 1` | `0.0` | yes ("completed but wrong") |
| `error=None` and `parse_ok=1`, FP > 0 | `0.0` | yes |
| `error=None` and `parse_ok=1`, FP = 0 | $|\hat S \cap G|/|G|$ | yes |

Semantic difference between excluding vs counting as 0: cutoff / error means
"the process did not complete" (infrastructure failure / information
barrier — should not count against the model); parse failure means "the
process completed but no one can read the output" (under exam-grading logic
= wrong answer). This is **intentionally different** from `pass_at_1_avg`
which also excludes parse failures.

#### Aggregation: per-question mean → cross-question mean

$$e_q = \frac{1}{|S_q|}\sum_{s \in S_q}\text{exam\_score}(s, G_q), \quad \text{global} = \frac{1}{|Q|}\sum_q e_q$$

The per-question denominator $|S_q|$ is **the count of samples actually
entering the denominator** (not a fixed SAMPLING_N — for a question with 3
samples where 1 fails on a sensitive word and 2 enter the denominator,
average over 2). A question with empty denominator has $e_q = \text{None}$
and does not participate in the global cross-question denominator.

#### Differences vs existing scoring metrics

| Metric | Wrong-selection penalty | Missed-selection penalty | chance correction | Single-select degeneration |
| --- | --- | --- | --- | --- |
| `parser.is_correct` (strict) | all-or-nothing | all-or-nothing | no | 0/1 |
| `fss` (Tversky α=2/β=0.5) | soft (α multiplier) | soft (β multiplier) | yes (`p_e`) | 0/1 |
| `hamming_score` | symmetric with missed | symmetric with wrong | no | 0/1 |
| **`exam_score`** | veto (hard gate) | proportional deduction | no (intuitive) | 0/1 |

#### Dual semantics of SAMPLING_N

* **Best-of-N view** (`pass_any_at_n` / `at_least_majority_at_n` /
  `at_least_all_at_n` / `majority_vote_accuracy`): N is "the probability of
  getting the answer right at least once across multiple random samples for
  one question".
* **Independent-trials view** (`exam_score_at_n_avg`): N is reinterpreted
  as "the number of independent trials", with each score independently in
  [0, 1] and the final result is the arithmetic mean.

Two semantic frames coexist over the same data; analysts choose: report
`exam_score_at_n_avg` for non-author audiences, and `fss` for papers.

#### Removability equivalence

This metric promises "removability equivalence": delete
`forecast_eval/analysis/exam_score.py` + `tests/test_exam_score.py` + the
small set of hook points marked in code/docs (markers and the catalogue are
in `openspec/changes/add-exam-score-metric/design.md` §D8), and the repo
returns to byte-level identical state with the pre-introduction commit.

### 3.5 Composite score weighted by sub-question type (composite-score-by-subtype)

`per_model_summary.csv` mixes every question type into a single mean; this
is not necessarily ideal for "discriminating model capability" — easy
questions (e.g. `yes_no`, k=2) generally have models near 100% and
discriminate poorly, while multi-select questions are answered poorly by
almost every model and discriminate strongly. Mixing both classes together
effectively means "low-information-density buckets" and
"high-information-density buckets" contribute equally to the final score.

`forecast_eval/analysis/composite.py` provides the "compute by sub-question
type first → compose with user-configured weights → drop missing buckets
and renormalise" pipeline. **Two dimensions are computed independently**
(decoupled from each other, since `multiple_choice` itself contains both
single and multi — the two are not orthogonal):

* `question_type` dimension (buckets: `yes_no` / `binary_named` /
  `multiple_choice`) → writes
  `per_model_composite_by_question_type.csv`;
* `choice_type` dimension (buckets: `single` / `multi`) → writes
  `per_model_composite_by_choice_type.csv`.

#### Weighting formula

For each (model, dimension, metric):

$$\text{composite}_m = \frac{\sum_{b \in B_{\text{valid}}} w_{m,b} \cdot v_{m,b}}{\sum_{b \in B_{\text{valid}}} w_{m,b}}$$

* $B_{\text{valid}}$ = the bucket set under that (model, metric) where the
  sub-question-type measurement is non-None **and** the bucket weight > 0
* Missing buckets (slice unavailable or weight 0) are dropped, and the
  remaining weights are renormalised (they are not treated as 0);
* All None → composite = None;
* Weights are not required to sum to 1; the denominator is normalised
  automatically.

#### "Harder questions discriminate better" rationale for default weights

| Dimension | Bucket | Default weight | Difficulty rationale |
|---|---|---|---|
| `question_type` | `yes_no` | 0.15 | k=2, blind guess 50%, almost no inter-model discrimination |
| `question_type` | `binary_named` | 0.15 | k=2 as above, adds name recognition |
| `question_type` | `multiple_choice` | 0.70 | k=2..N wide range, includes multi-select, highest discrimination |
| `choice_type` | `single` | 0.40 | overall easier (includes yes_no / binary_named) |
| `choice_type` | `multi` | 0.60 | true multi-select, almost every model performs poorly, high discrimination |

One sentence: **let buckets that discriminate model capability contribute
more**. To switch to "I care more about easy questions", just flip the
numbers. This is an opinionated default, not a "neutral" one — we believe
the orientation above is more reasonable for the vast majority of
evaluation scenarios; users with a different view solve it by overriding
one line of `.env`.

#### Configuration entrypoint (`Settings` / `.env`)

* `COMPOSITE_WEIGHTS_QTYPE` / `COMPOSITE_WEIGHTS_CTYPE`: global default
  weights, shared by all metrics that are not explicitly overridden.
* `COMPOSITE_WEIGHT_OVERRIDES_QTYPE` /
  `COMPOSITE_WEIGHT_OVERRIDES_CTYPE`: per-metric independent overrides.
  Form: `"fss=yes_no=0.05,multiple_choice=0.95"`, semicolon-separated for
  multiple metrics. Misspelled metric names (not in `_SUMMARY_FIELDS` data
  columns) raise from `compute_composite` during the analysis phase rather
  than "silently falling back to default" — this is intentional: when you
  misconfigure, we make sure you know.
* Startup-time validation: bucket name must ∈ the legal set, weight ≥ 0,
  at least one > 0.

#### Relationship to existing CSVs

* `per_model_summary.csv` semantics are unchanged (still the mixed mean) —
  no impact on downstream;
* `per_model_by_question_type.csv` / `per_model_by_choice_type.csv` now
  also include the v5 discrete-family columns (FSS / Cohen κ / Hamming /
  Fleiss κ / mean entropy / VCI / MVG), with column order matching
  `per_model_summary.csv`;
* `per_model_composite_*.csv` column order is also aligned with
  `per_model_summary.csv` (with one extra `weights_kind` column marking
  `default` / `overridden`); downstream scripts only need to swap file
  paths to read the "weighted-by-sub-question-type total table";
* `composite_meta.json` is the audit trail: which buckets each composite
  value actually used, the normalised weights, and the raw slice value per
  bucket are all there, fully reproducible one-to-one.

#### "Per-bucket slicing" of the v5 discrete family

`fss` / `cohen_kappa` / `hamming_score` / `fleiss_kappa` / `mean_entropy` /
`vci` / `mvg` are computed model-globally on the v5 main path (written into
`per_model_summary.csv`); for weighted composition we must first slice by
bucket and then compute, so `composite.slice_v5_metrics_by_bucket` takes
this over. Its results are **also** routed back into `per_model_by_*.csv`
— users reading the sub-question-type detail tables can now see these
metrics too, instead of NULL placeholders.

---

## 4. ReAct + Tool Use: "unfolding" the model's reasoning process

### 4.1 The whole prompt as a single user message

The template is one entire prompt block (agent_role + event + outcomes +
format + guidance); the project chooses to feed it in **as a single user
message** rather than splitting into system / user.

Reasoning:

* **Most faithfully reproduces the source metadata template**: the source
  data `dataset_metadata.features_json.prompt_reconstruction` is a single
  string; forcibly extracting a system part would lose the semantics of
  the original concatenation.
* **Cross-model consistency**: different providers handle system messages
  differently (OpenAI hard-caches them, Anthropic uses an independent
  field). Going uniformly through a user message gives the most stable
  comparability.
* **Easy to hash and diff**: the entire prompt content is written directly
  into the `user_prompt` field, so any future template change is visible
  through a hash at a glance.

### 4.2 Hard ceilings on the ReAct loop

Each sample has two gates:

* `REACT_MAX_STEPS=12`: the LLM may interact with the system at most 12
  rounds in total (enabling the reflection protocol or nudges adds 2-4
  rounds beyond a single-step direct answer, so the default is slightly
  higher than the historical value).
* `REACT_MAX_SEARCH_CALLS=8`: after 8 cumulative `web_search` calls, the
  tool returns `search budget exceeded` directly to the LLM.

The design philosophy is to **define an upper bound on "the model's
autonomous searching" via a budget**:

* Without a cap, malicious/degenerate models could call indefinitely and
  burn through the API bill.
* Capping while returning an error rather than throwing lets the LLM still
  provide a "best-effort answer" based on existing information, and
  separates "out of budget" from "system crashed".

Exceeding step count without producing a boxed answer → `parse_ok=0`,
treated the same as a refusal (§3.2).

### 4.2.1 Reflection protocol: pulling the model off "one-shot direct answer" with prompt rather than rules

We observed that some models give a final answer after only ~1.6 searches
on average — this "confident one-shot" behaviour drastically lowers
`pass@1` on long-tail events. The project responds in two layers:

* **Preferred: reflection protocol (`REACT_REFLECTION_PROTOCOL=true`,
  default on).** Append a *Forecasting Protocol* to the end of each
  sample's user message: decompose the question → list ≥3 different
  retrieval angles → reflect after each search → cross-validate → check
  the opposite direction → state confidence. This addendum **is not
  written into `dataset_metadata`**, so `prompt_templates_hash` does not
  change; its existence is persisted through `run_meta.config_snapshot`
  alongside each sample's `user_prompt` field, enabling per-question
  diffing after the fact.
* **Fallback: soft minimum search count (`REACT_MIN_SEARCH_CALLS`, default
  `0`=off).** When the LLM tries to give a final answer with insufficient
  searches, the system injects a user nudge asking it to try a different
  angle and search again; the nudge count per sample is capped by
  `REACT_MAX_NUDGES`, and the overall step count remains bounded by
  `REACT_MAX_STEPS`. When `ENABLE_WEB_SEARCH=false` the nudge is
  automatically disabled (no search to do).

Design philosophy:

* **Prompt first, rules second.** The reflection protocol is "guidance",
  the nudge is "restriction". First try better guidance to make the model
  walk a few more steps spontaneously, only impose a soft floor when the
  model still insists — to avoid mixing "the capability under
  evaluation" with "the system's enforcement".
* **Toggleable, comparable.** Both switches have clear defaults, and
  turning them off degrades to the old behaviour (the same code can run
  "protocol on vs off" controlled experiments).
* **Auditable.** The protocol text and nudges both appear in
  `messages_trace`; the on/off state is anchored by `config_snapshot`, so
  this is **not implicit behaviour**.

### 4.3 Graceful degradation for tool-call errors

Within the ReAct loop, several tool-related errors do not interrupt the
whole sample:

| Situation                  | Handling                                                       |
| -------------------------- | -------------------------------------------------------------- |
| Unknown tool name          | return `unknown tool` to the LLM and let it change tack        |
| `arguments` JSON parse fails | send the error back as tool_result; the LLM can retry         |
| Search budget exhausted    | tool_result returns `search budget exceeded`                   |
| Tavily itself errors       | go through `SEARCH_BACKOFF_S` retries; if still failing → stuff the error into tool_result |

Design philosophy: **let the LLM "see" its own failures from the system's
perspective rather than papering over for it**. The capability numbers
this produces are closer to reality — a model that cannot handle tool
failures should naturally score lower.

### 4.4 "Forbidden words" for reasoning models

Some reasoning models (o1 / o3 / r1 / qwq…) directly return 400 on custom
sampling parameters like `temperature` / `top_p`.

The project maintains the substring list
`LLM_REASONING_MODEL_PATTERNS=o1,o3,o4,r1,qwq` in `.env`; for matching
models, those two parameters are **not passed** at call time.

This is a typical "shift maintenance cost forward" design: rather than
identifying 400 inside retry / error handling, handle it at request
construction time.

---

## 5. Data storage: wide table + single writer + post-hoc analysis

### 5.1 Why a wide table

One row per question, with an `s{i}_*` group of columns per sample (since
v3, 20 fields: original 14 + 6 newly added observation columns). Compared
to a "long table + (question_id, sample_idx) composite primary key", the
wide table's advantages:

* **Resume queries are naturally simple**: `SELECT question_id WHERE
  s{i}_created_at IS NOT NULL` simply scans one column, no group by needed.
* **Atomic single-row read**: the analysis script reads one row and has
  every sample; no join or aggregation needed.
* **Schema fixes N**: `SAMPLING_N` is pinned at table-creation time, so
  whenever the DB is reopened in the future, the structure matches what it
  was then.

The cost: `SAMPLING_N` must be determined before the run starts and cannot
be expanded mid-run; the schema also needs to "dynamically generate 20 × N
columns". This cost **is acceptable in evaluation scenarios** —
`SAMPLING_N` is by nature part of the run config and should not change
mid-run.

#### Why `step_metrics` uses a JSON column instead of a separate long table

ReAct's per-round step metrics are naturally 1-to-N (one sample → multiple
steps), and at first glance you'd want to factor them out into a long
table like `run_step_metrics(question_id, sample_idx, step, prompt,
completion, ...)`. The project ultimately compresses it into
`s{i}_step_metrics TEXT` (a JSON array) for three reasons:

* **No cross-step query need**: the analysis layer always "fetches the
  whole trajectory by sample and then processes it"; it never does
  row-level aggregation like `SELECT * FROM steps WHERE
  finish_reason='length'` — every filter happens at sample granularity.
  Normalising this data into a table would mean paying index/JOIN cost
  for queries that don't exist.
* **Preserves the simplicity of one writer per model**: v3 expanding from
  14 to 20 columns was just `ALTER TABLE ADD COLUMN`, zero complexity;
  switching to a long table would require a second table + a second
  foreign key + a second INSERT path, and the writer boundary would
  jump from "one-row upsert" to "multi-row transaction", conflicting
  with §5.2's "eliminate races via orchestration" principle.
* **JSON size is controllable**: the step count per sample is bounded by
  `REACT_MAX_STEPS` (default 6); a single JSON is typically < 1 KB; on
  v3 schema with SAMPLING_N=3 / ~100 questions, the DB delta is on the
  order of KB — WAL handles it easily.

The cost: "step-level aggregation" that a long table could do has to be
done by reload + parse in Python here. Given that the analysis script is
a one-shot script anyway (`python -m forecast_eval.analysis`), this cost
is acceptable.

#### Why `reflection_protocol_hash` is independent of `prompt_templates_hash`

Intuitively, the reflection protocol is part of the prompt, so it seems
natural to merge it into `prompt_templates_hash`. But the project
deliberately pulls it out, because **their change cadence and
explanatory dimensions differ**:

* `prompt_templates_hash` reflects "how question content is rendered to
  the model" — the **templates** for the stem, options, instructions,
  question-type description, and so on. Once a template changes, every
  question text changes — this is a coarse-grained run-distinguishing
  key.
* `reflection_protocol_hash` reflects "which meta-cognitive instruction
  was injected into the model in the ReAct main loop", essentially **a
  switch on a search-behaviour prior**. Its variation has only three
  axes: on/off, whether the text was modified, version number. Merging
  it into the main template hash would make a small change like "I just
  turned reflection off" appear equivalent to "I rewrote every question
  template" — losing the explanatory power for controlled experiments.

The benefit of separating the two hashes: when running A/B comparisons
across runs, you can choose "only `reflection_protocol_hash` differs,
everything else equal" — this is exactly what ablation studies want.
Similarly, `reflection_protocol_text` (full text) coexists in `run_meta`,
to enable post-hoc diffs without depending on the prompts.py source code
(e.g. when releasing a report, the recipient receives a redacted DB
rather than the git repo).

#### Why `belief_protocol_hash` is also independent (v4)

The `BELIEF_PROTOCOL` introduced in v4 is also a tail addendum at the end
of the user message (appended after `REFLECTION_PROTOCOL`), making the
LLM emit a strict-JSON `<belief>{...}</belief>` segment before
`\boxed{...}`, carrying the raw material for the probabilistic family of
metrics. Treated **completely in parallel** with the reflection protocol:

* Protocol body **not entered** into
  `dataset_metadata.prompt_reconstruction`, **not entered** into
  `prompt_templates_hash`;
* Two new columns `belief_protocol_text` / `belief_protocol_hash` are
  added to `run_meta`, with the same source and semantics as
  `reflection_protocol_*`;
* `manifest.json` top-level contains both `reflection_protocol_hash` and
  `belief_protocol_hash`, so the "grep the protocol fingerprint without
  opening the DB" retrieval path covers both protocols;
* The three fingerprints (`prompt_templates_hash` /
  `reflection_protocol_hash` / `belief_protocol_hash`) are mutually
  **independent**: swapping one protocol does not affect the main
  template hash, toggling one protocol does not affect the other
  protocol's fingerprint — enabling three-way independent ablation
  studies for belief A/B, reflection A/B, and template A/B.

The belief protocol's parsing path (`parser.parse_belief`) is also fully
independent of `parse_answer`: belief parse failures MUST NOT affect
`parse_ok` / `correct` / `final_answer_letters`, keeping the v3 boxed
path stable. v4 is **layered on**, not a replacement.

#### Phase 2 calibration uses LOO instead of holdout (v4)

Both Platt scaling and temperature scaling in
`forecast_eval/analysis/calibration.py` are forced to use leave-one-out
(the calibration parameters for question $q$ are learnt from
$\mathcal{Q}_t \setminus \{q\}$). Reasoning:

* **N is small**: this dataset has 319 questions, and after stratifying
  by cell each cell holds 50-150 questions. At this scale, holdout
  splits make calibration parameters high-variance — different splits of
  the same dataset can yield (a, b) differences of ±0.3, enough to
  render ECE comparisons meaningless. LOO uses every question and
  prevents that question from polluting its own calibration parameters
  — this is the core anti-overfit mechanism in the paper §C.11, and the
  project copies it.
* **Compute is cheap enough**: Platt with IRLS is ~10 Newton iterations
  per fit, each O(N) for the Hessian. LOO is N refits; a naive total
  cost of O(N²) ≈ 100k float ops, < 1s for 319 questions.
  Temperature scaling uses golden-section search with 30 evaluations,
  each O(N), also sub-second.
* **Overfit sentinels are in place**: LOO can still overfit on small
  cells (especially edge cells in the multi class).
  `ModelCalibrationReport.overfit_warning` returns True when `cal BI -
  uncal BI > 5`, and `per_model_summary.md` flags the model name with
  `cal*` — reviewers can tell at a glance that this row's calibration
  result is not trustworthy.

`scipy` was not introduced: both IRLS and golden-section search are
written in ~30 lines of pure Python with no dependency bloat. If a
future Phase 2.x adds variants like Dirichlet calibration that need
more sophisticated numerical methods, scipy will be introduced as
needed per design.md Open Q1.

#### Phase 3 confidence-calibration joint diagnosis (v4)

`forecast_eval/analysis/behavior.py::confidence_calibration` treats the
model's self-reported `confidence ∈ {low, medium, high}` as "subjective
confidence" and $\max_l p_l$ as "numeric confidence", comparing both
against hit rates. `confidence_conflict_models` returns a model set
across two kinds of dislocations:

* **Verbally conservative + numerically overconfident**: `low` bucket
  with `mean_max_p > 0.70` — the model speaks conservatively but the
  numbers expose its real confidence;
* **Verbally confident + numerically underconfident**: `high` bucket
  with `mean_max_p < 0.55` — the model boasts verbally but max_p
  cannot back up that posture.

Hitting either → `per_model_summary.md` appends a `conflict*` sentinel
after the model name (alongside Phase 2's `cal*`). This is a diagnostic
dimension **absent** from the paper — the BLF paper (arXiv 2604.18576v2)
covers only binary prediction, where *language confidence* and *numeric
max_p* degenerate into the same signal under binary $p$; only after
this project unifies yes_no / multiple_choice to per-option Bernoulli
do the two quantities decouple for the first time: language confidence
is a discrete token directly generated by the LLM, numeric max_p is a
statistic of the first-order probability vector — the two have totally
different sources. When they are systematically inconsistent, either
the prompt failed to anchor "low/medium/high" to the numerical
calibration (calibration prompting failure), or the model hedges in
prose while writing numbers from the actual token logits
(hedging-vs-revealing divergence). The former is solvable by rewriting
the prompt; the latter is a model-intrinsic property reflecting that
**the model's self-report is unreliable** — pointing to the same
underlying problem as chain-of-thought faithfulness research.
`conflict*` is therefore not just an "overfit" sentinel, but an
exposure of "verbal/numeric inconsistency" as a new quality dimension
to reviewers, providing empirical grounding for prompt-engineering and
reflection protocol design.

### 5.2 One writer per model + WAL

Concurrent writes to SQLite are a classic pitfall. The project's
strategy:

* **One async writer task per model DB**: every worker's results are
  sent via `asyncio.Queue` to the writer for that model.
* **`PRAGMA journal_mode=WAL` + `synchronous=NORMAL` +
  `busy_timeout=5000`**: ample throughput under single-writer
  multi-reader, with crash recovery still safe.
* **Batched commits**: flush every `DB_COMMIT_BATCH=10` entries or
  every 1 second.

The core idea behind this design: **eliminate races via orchestration,
don't solve races with locks**. Once we pin "one writer per DB", the
concurrency problem degenerates into ordinary single-threaded batch
inserts.

### 5.3 The DB stores raw observations only; aggregation happens later in `analysis/`

```
DB:        raw observations only
├── correct (bool, NULL)
├── parse_ok (bool)
├── tool_calls_count
├── react_steps
├── tokens / latency
└── error / created_at

analysis/: aggregations
├── pass@1
├── pass_any@N
├── majority_vote
├── parse_failure_rate
└── error_breakdown
```

This is one of the project's most important architectural decisions.

#### Why no pre-aggregation in the DB?

* **Metric definitions evolve**: today `pass@3` is "1 of 3 counts as a
  pass", tomorrow it might change to "at least 3 correct". If aggregation
  lands in the DB, every redefinition requires a backfill. Deferring all
  metrics to the analysis layer makes them re-computable at any time.
* **`analysis.py` is a pure function**: input = `runs/{run_id}/db/*.db`,
  output = `analysis/*.csv|md|json`. Can be re-run independently via
  `python -m forecast_eval.analysis`.
* **DB and paper/report are decoupled**: raw records are an engineering
  artefact; statistics are a product/academic artefact — their cadences
  are completely different.

#### Recalibrating the `pass@k` naming

In the wider community `pass@k` generally means "at least one correct in
k". The project early on used `pass@3 = sum(correct)≥3` (a threshold
semantic), which caused ambiguity. Now made explicit:

* `pass_any@N` ≡ standard `pass@k`: at least one correct in N
* `at_least_k_correct@N`: at least k correct in N (threshold analysis)
* `pass@1 avg`: average accuracy across N (stable capability)
* `majority vote correct`: whether the majority-vote frozenset across N
  is correct (self-consistency)

Design philosophy: **a name must either be unambiguous or explicitly
declare its semantics**.

---

## 6. Error handling: slicing "failure" into 8 semantics

The error-handling table is `FRAME.md §9`, but the spirit can be condensed
into a few principles.

### 6.1 Not every error should be retried

| Error                | Retry?               | Reason                                    |
| -------------------- | -------------------- | ----------------------------------------- |
| Network/5xx          | Yes (per backoff sequence) | Mostly transient                       |
| Rate limit           | Yes (prefer Retry-After) | The provider has told you how long to wait |
| Auth 401/403         | **Stop the entire run** | The key is wrong; retrying is pointless and stopping early saves money |
| Bad request          | No                   | Things like model_not_found only run after a config change |
| Content policy       | No                   | The same prompt sent again returns the same result |
| Refusal / parse fail | No                   | Not an error — it is model behaviour      |
| Tavily itself        | Has its own retry sequence | Once exhausted, return the error to the LLM |
| Training-cutoff filter | Not invoked        | Write `skipped_training_cutoff` directly  |

### 6.2 Three independent backoff sequences

```
LLM_BACKOFF_NETWORK_S=2,5,15,30,60
LLM_BACKOFF_RATE_LIMIT_S=10,30,60,120,300
LLM_BACKOFF_SERVER_5XX_S=5,15,30,60,120
```

Different error types use different backoffs — rate limit is much slower
than network, because the former typically needs minute-level cooldowns
while the latter usually clears in a few seconds. The sequence length also
determines the "max retry count"; configuration is unified in `.env`.

### 6.3 Error classification codes are first-class citizens of the report

The `error` field is not "fill in a string when something errors" but a
fixed finite enum:
`network` / `server_5xx` / `bad_request` / `content_policy` /
`skipped_training_cutoff`

`error_breakdown.csv` slices directly by this classification. Design
philosophy: **every failure behaviour must be categorisable and aggregatable
in the report** — an `error="something went wrong"` is useless.

#### 6.3.1 v5.1 harness-resilience: classification boundary expansion

Two common misclassifications encountered during cross-provider evaluation:

- **Aliyun content moderation (`data_inspection_failed`) mis-bucketed as
  `bad_request`**: v5.0's `_body_matches` only recognised English needles
  like `content_policy / content_filter / safety`; the
  `code=data_inspection_failed` returned by DashScope
  (`https://dashscope.aliyuncs.com`) fell through to the catch-all
  `bad_request`. v5.1 unified the needle list under
  `errors.CONTENT_POLICY_NEEDLES`, adding `data_inspection_failed` /
  `inappropriate content` / `sensitive`; on match → classify as
  `content_policy`, preserving the `MUST NOT retry` semantics.
- **Remote disconnect `RemoteProtocolError` mis-bucketed as `unknown`**:
  v5.0's network exception tuple only listed `ConnectError` /
  `ReadTimeout` / `ConnectTimeout` / `WriteTimeout`;
  `httpx.RemoteProtocolError` ("Server disconnected without sending a
  response.") fell into `UNKNOWN`, and the entire sample failed without
  retry. v5.1 expanded the network exception family to align with httpx's
  existing `NetworkError` subset: `+RemoteProtocolError / +WriteError /
  +PoolTimeout`, with parallel expansion on the LLM side
  (`errors.classify`) and the Tavily side (`search._single_request`).

### 6.4 v5.1 ReAct loop fallback mechanism

For cross-model comparisons, **`parse_failure_rate` must reflect only the
model's own format failure, not upstream resource exhaustion in the
harness**. In v5.0, after `REACT_MAX_SEARCH_CALLS` was exhausted the
`web_search` schema was still exposed to the LLM; the model kept asking
for the tool and hit the `REACT_MAX_STEPS` ceiling, with `final_raw=""`
becoming parse_ok=0 directly. That disguised "tool starvation" as "format
failure".

v5.1 added two defences:

- **D1**: `REACT_BUDGET_EXCEEDED_DROP_TOOLS=True` (default on) — once
  cumulative searches reach the cap, the next `llm_chat` call passes
  `tools=[]`, leaving the model only able to emit content; the existing
  "no tool_calls → break" branch naturally takes over.
- **D2**: `REACT_FINAL_ANSWER_RETRY=True` (default on) — if the loop exits
  cleanly but `final_raw == ""` (typical scenario: still hammering
  tool_calls just before steps run out), append a user message — "Time to
  commit. Output your final \boxed{...} answer now without further
  searches or tool calls." — and call the LLM once more with `tools=[]`.
  This retry counts toward `react_steps` and `step_metrics` (every step
  remains replayable) but NOT toward `nudges_used` (different semantics);
  a new dedicated column `final_answer_retry_used` records 0/1.

The new analysis column `final_answer_retry_rate` lands in
`per_model_summary.csv`, letting analysts see "how much the fallback
caught" separately and decide, when necessary, whether to deduct it from
the `pass_at_1` denominator. Both switches default on; turn them off only
for A/B controlled experiments. Schema upgraded to v5: each sample slot
in `run_results` adds `s{i}_final_answer_retry_used INTEGER`; old v4 DBs
auto-ALTER ADD via `init_schema` (NULL-compatible).

---

## 7. Configuration as contract: `.env` is the single source of truth

### 7.1 Almost every tunable is in `.env`

The CLI exposes only three flags — `--question-type` / `--choice-type` /
`--skip-analysis`; everything else goes through `.env`.

Reasoning:

* **Easy to re-run**: a single `.env` is enough to reproduce the entire
  configuration; CLI flags scattered in shell history are easily lost.
* **CI/scheduler-friendly**: scripted execution generally prefers
  managing a file rather than a command line.
* **Config ↔ DB self-consistency**: `config_snapshot` is written into
  `run_meta`, so reviewing a run later tells you "what its `.env` looked
  like at the time" (after redaction).

### 7.2 OpenAI-compatible endpoint: horizontal compatibility

`LLM_BASE_URL` accepts any OpenAI-compatible endpoint: OpenRouter, Aliyun
Bailian, OpenAI, DeepSeek, SiliconFlow, local vLLM all work.

Design philosophy: **the integration surface should be small and
standard**. OpenAI's chat completion + function calling protocol has
become the de facto standard; this project does not build a
provider-adaptation layer, but pushes adaptation responsibility to the
endpoint.

### 7.3 Training-cutoff config is quality config

`MODEL_TRAINING_CUTOFFS=openai/gpt-5=2024-10-01,...` is not optional — it
is **part of evaluation fairness**. The docs explicitly recommend
declaring an explicit cutoff for every model under evaluation; an
unspecified model is not filtered (with a warning).

---

## 8. Testing: shifting expensive failures to cheap local runs

### 8.1 Tests must not hit the network or burn the API

A complete run of 319 questions × number of models × N samples is **a few
tens to a few hundred dollars**. Tripping over a prompt / parser / schema
bug at that scale just wastes the money.

Core constraints of the test design:

* `tavily-python` must not actually send requests → `respx` mocks httpx
* The OpenAI client must not actually send requests → fixture replacement
* SQLite uses a temporary directory → `tmp_path` fixture
* The dataset must be small yet "look real" → use a few real questions
  from the source DB as fixtures

### 8.2 Five CI red lines

```
test_prompts / test_parser / test_training_cutoff /
test_llm_no_browsing / test_analysis
```

These five must always be green. They cover the parts of the project
most likely to "silently break":

| Test                         | Invariant guarded                                  |
| ---------------------------- | -------------------------------------------------- |
| `test_prompts`               | prompt template rendering is correct for all three question_types |
| `test_parser`                | letter parsing and strict-equality scoring         |
| `test_training_cutoff`       | training-cutoff filtering semantics and resume priority |
| `test_llm_no_browsing`       | provider-native browsing is never silently turned on |
| `test_analysis`              | report numbers reconcile with the raw DB           |

Design philosophy: **pick the invariants whose breakage would be
expensive, and use unit tests as sentinels**.

### 8.3 dry-run smoke test

`test_smoke_dry_run.py` replaces OpenRouter + Tavily with httpx stubs and
runs an end-to-end pipeline of 3 questions × 1 model × 1 sample. It does
not validate logic details — it validates "is the pipe still flowing":
schema, wide table, `messages_trace` JSON, `search_calls` fields all
present.

This expresses the e2e/unit test split: unit tests validate "local
correctness", smoke tests validate "integration doesn't blow up".

---

## 9. Observability: every sample is traceable

### 9.1 Progress log

```
12:03:44 | INFO | [run=20260424-120344-a7k3] [5/1610] q=69566c13 qt=binary_named ct=single model=openai/gpt-5 sample=2/5 correct=True steps=4 tool_calls=3 latency=8421ms
```

Every log line carries: question id, question_type, choice_type, model,
sample_idx, correctness, step count, tool-call count, latency.

Design philosophy: **a single log line should fully describe what path a
sample took through the system**. Reading the log is reading the trace —
no DB join required.

### 9.2 `messages_trace` and `search_calls`

The DB stores two JSON blobs directly:

* `messages_trace`: the complete ReAct message sequence (LLM replies,
  tool_call, tool_result).
* `search_calls`: each `web_search` call's query, end_date, result count,
  and per-result published_date.

These are large (~80% of the DB), so the `WRITE_MESSAGES_TRACE=false`
switch is provided. But the default is on — reasoning: **the value of
debugging one failure far outweighs the few extra MB of disk**.

### 9.3 `loguru` + structured + dual output

stderr (for humans) + rotating file (for machines), two channels;
rotation 100 MB / retention 5. Design philosophy: **humans and machines
have different needs when reading logs — serve them separately**.

---

## 10. Evolution path: spec-first changes driven by openspec

The repo root contains `openspec/changes/`, where changes are recorded in
spec form. `bootstrap-forecast-eval` is the initial bootstrap record.

Design philosophy:

* **Write the spec before the code**: avoid "discovering the design is
  wrong only after the code is merged".
* **Change archive and code diff coexist**: when reviewing the
  architectural evolution later, you can see "why we changed it", not
  just "what was changed".

### 10.1 Grid search via virtual slug (option C)

`react-tavily-grid-search` extends the *(Q × M × N)* three-axis space to
*(Q × M × R × C × N)*, but **without** schema upgrade and **without**
touching the runner core loop. The method: at the evaluation entrypoint
encode each `(real_model, R, C)` triple as a **virtual model slug**
`{real}::r{R}::c{C}`; runner / DB / analysis main pipeline treats it as
an opaque string — existing artefacts naturally expand into multiple
rows by virtual slug, while the new module
`forecast_eval/analysis/grid.py` decodes the triple, re-aggregates, and
emits paper long tables and figures. Full decision archive in
`openspec/changes/react-tavily-grid-search/design.md`; the 10 key
decisions:

| ID | Decision |
| --- | --- |
| **D1** | Pick option C (virtual slug + per-task settings view); reject A (single run, multi-(R, C) DB — schema v5 rewrite) and B (one run_dir per cell — `runs/` bloat + complex cross-run aggregation) |
| **D2** | Virtual slug uses `::r{R}::c{C}` suffix; `db.model_slug_safe` replaces `::` with `_` to land an fs-safe filename `openai__gpt-5__r5__c3.db`; the regex `^(?P<real>.+?)::r(?P<R>\d+)::c(?P<C>\d+)$` non-greedy captures real_model |
| **D3** | `runner.Task` carries a cell-local `settings: Settings`; the dispatcher derives an immutable sub-view via `model_copy(update={...})`; `react.py` / `search.py` are byte-unchanged |
| **D4** | Only raise when `REACT_MIN_SEARCH_CALLS > min(C_list)`; for a cell with `C < MIN`, silent clamp `effective_min = min(MIN, C)` and record it under `run_meta.config_snapshot.grid_origin` for audit |
| **D5** | `run_meta.config_snapshot` writes **single-valued** R/C; add a `grid_origin = {real_model, R, C, effective_min_search_calls}` sub-key; manifest top-level adds a `grid` block (`r_list / c_list / default_r / default_c / real_models / n_cells`) so the analysis layer doesn't have to decode the triple per .db |
| **D6** | `manifest.models` / `manifest.model_files` field semantics remain "list of virtual slugs"; the new `grid.real_models` is a deduped real-slug convenience field — v4 analysis main path's contract of "read `manifest.models` as the db file list" is preserved |
| **D7** | `analysis/__init__.py::run_analysis` main path is **zero-intrusive**; append a `grid.run_grid_analysis(...)` at the end wrapped in `try/except` (same best-effort pattern as reflection A/B), failures do not interrupt the existing pipeline |
| **D8** | Grid CIs all go through `inference.paired_bootstrap` (5000 resamples, seed=42); BI-domain CIs are obtained via "BS-domain paired bootstrap + monotone transform $\mathrm{BI}=100(1-\sqrt{\mathrm{BS}})$" — **no** new statistical code introduced |
| **D9** | Pareto frontier's cost dimension defaults to `mean_search_calls` (actual mean search count, more honest than the C ceiling), with `mean_latency_ms / C` fallback allowed; y-axis defaults to `bi_mean`, with `nll_mean` (minimisation direction) as an option |
| **D10** | Fig 1 main figure pins `R = GRID_DEFAULT_R` with one curve per real_model; other R values each get a same-format appendix figure to avoid main-figure unreadability after stacking M·\|R\| curves |

The three PRs `Phase 0 / 1 / 2` ship sequentially — each phase passes
`pytest -q` and `openspec validate --strict`, and after deleting the
phase's own code the system is equivalent to the previous phase's
completed state (Rollback Strategy). Single-value `.env` parses under
the new code as a length-1 list → Cartesian product produces a single
virtual slug, with the **only** visible difference being the
`__r{R}__c{C}` suffix on the .db filename; for legacy v4 runs (manifest
without a `grid` block), grid analysis and the grid figure family
early-exit altogether — zero intrusion.

---

## 11. Summary of design-consistency principles

Condensing the full document's design philosophy into a single set of
principles, placed at the end for review reference:

1. **Isolation > trust**: any boundary that can be enforced at the tool
   layer must never be delegated to the prompt.
2. **Honesty > prettiness**: parts of the threat model we cannot control
   are written into the documentation in plain language.
3. **Skip ≠ fail**: actively excluded samples are categorised
   independently and do not pollute the error rate.
4. **Raw > aggregated**: the DB stores observations only; statistics are
   deferred to `analysis/`.
5. **Strict > generous**: scoring uses frozenset strict equality; the
   offset defaults to a strict `-1`.
6. **Reproducibility > convenience**: source data goes into Git, each DB
   is self-contained, hashes pin down the fingerprint.
7. **Observability > elegance**: full messages_trace is on by default;
   the progress log is one line per sample.
8. **Categorise failures**: errors use a finite enum; every kind has its
   own cell in the report.
9. **Config as contract**: `.env` alone decides everything; CLI flags
   are minimal.
10. **Tests guard the expensive**: five CI red lines + dry-run smoke,
    shifting expensive failures to local.

---

## 12. Reading roadmap

If you are new to the project, we suggest reading in this order:

1. `README.md` — figure out in 5 minutes what this is and how to run it.
2. This document (`DESIGN.md`) — understand the motivation behind each
   trade-off.
3. `FRAME.md` — the complete spec at field, interface, and pseudocode
   level.
4. `forecast_eval/prompts.py` + `forecast_eval/parser.py` — the scoring
   core; these two files are practically the "heart" of the project.
5. `forecast_eval/runner.py` + `forecast_eval/react.py` — orchestration
   and the ReAct loop.
6. `tests/` — read tests to reverse-engineer the contract.
7. `openspec/changes/archive/` — to find out why things became what they
   are today, come here.

---

> **Summed up in one sentence**:
> This is a project that "uses engineering discipline to safeguard
> scientific rigour" — every seemingly excessive constraint (information
> isolation, strict matching, canonical letter encoding, wide table +
> single writer, error classification, CI red lines) exists so that the
> number in the final report actually means something.
