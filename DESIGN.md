# OracleProto — Design Rationale

> *This document explains why the codebase looks the way it does. For the field-level
> mechanics (schemas, signatures, error codes, line numbers), pair it with `FRAME.md`. For
> the formal framework, read `paper/main.tex` §§1–3. The reading order is paper §§1–3 →
> DESIGN → FRAME → source. Almost every "weird" choice in this codebase is the unique fixed
> point where two paper-level constraints meet one empirical observation; reading the source
> without that context is the fastest way to misread an engineering decision as either
> over-engineering or laziness.*

## How to read this document

Each section opens with the constraint it discharges, walks through the engineering choice
that satisfies it, and ends with the alternatives we rejected and the file path that pins
the decision. Paper citations use the section names from `paper/main.tex` (e.g. paper §3.5
Reproducibility and Leakage Boundary) and the canonical equation numbers as compiled by
LaTeX (e.g. paper Eq. 40 for strict-equality scoring). File anchors use `module.py:Lnnn`
relative to `forecast_eval/`, and the line numbers track the current main branch.

---

## 1. The question and the framework

### 1.1 The question OracleProto exists to answer

> If a model is never allowed to look at information published after an event has been
> resolved, how strong is its native forecasting capability across a leakage-controlled
> dataset?

Paper §1 and §2.3 articulate why the question is non-trivial. Existing evaluation practice
sits on an unstable middle ground between two regimes that each fail in their own way.
**Prospective live evaluation** such as ForecastBench and FutureX admits only events whose
answers do not yet exist when the forecast is submitted; this is the gold standard for
contamination control, but the leaderboard is a one-way temporal stream whose entries
disappear once they resolve, so the resulting evaluation is impermanent and not reusable.
**Retrospective evaluation** such as FutureX-Past or any archive of resolved live questions
is auditable and comparable, but is highly prone to mistaking factual recall for
forecasting capability; the FutureX-Past dataset card itself warns that historical
outcomes may already have entered newer models' training data.

The diagnostic literature surveyed in paper §2.3 has empirically shown that *simulated
ignorance* and *true ignorance* are systematically different: reasoning-optimised models
are particularly bad at the simulation, and a 1–5% label-noise rate alone is enough to
break proper scoring rules (Paleka et al. 2025; Li et al. 2026). BLF (Murphy 2026) reaches
the same conclusion from the inference side, namely that a single-inference defence does
not generalise across runs and that the discipline must live one level deeper, inside the
dataset itself.

OracleProto's response is to push the discipline into the dataset schema and the run unit
$\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)$ defined in
paper §3.5. Three almost religious hard constraints fall out immediately, and every other
decision in this document is downstream of these three.

1. **The information boundary lives at the dataset and tool layer, not the prompt layer.**
   Sample admission ($\kappa_M \le \chi_i < \tau_i$, paper Eq. 4) is an upstream filter
   rather than instructions to the model; tool-level temporal masking is injected by the
   tool implementation rather than a parameter the model can fill in. The model can
   propose queries, but it cannot alter the cutoff.
2. **Results are byte-reproducible.** A `git clone` plus the `.env` is enough for a
   third party to re-run the evaluation and obtain comparable numbers. The source DB is
   checked into the repository, and a six-part hash fingerprint pins down which inputs
   each run was based on.
3. **Every leakage path that we can control is controlled, and every path we cannot is
   declared.** The threat model in paper §3.5 is honest about what we cannot fix.

Once these three constraints are internalised, every "seemingly over-strict" choice in
the following sections looks natural.

### 1.2 The run unit $\mathcal{R}$ as a contract

There is one mental model that ties every section of this document together: **$\mathcal{R}$
is a contract, not a configuration**. A run is the same evaluation as another run if and
only if every field of $\mathcal{R}$ matches and every fingerprint matches; otherwise it
is, strictly speaking, a *different* evaluation rather than a noisier estimate of the same
one.

Concretely, two runs that differ on $\delta$ are not comparable because the admissibility
frontier $\chi_i$ differs and the visible retrieval set $\mathcal{T}_{\le\chi_i}$ is
different. Two runs that differ on $\kappa_M$ for the same model slug are not comparable
because the admissible question subset $\mathcal{D}^{\mathrm{pred}}_M = \{q_i : \kappa_M
\le \chi_i < \tau_i\}$ differs. Two runs whose `prompt_templates_hash` differs are not
comparable as evaluations of the same renderer $R$; they evaluate two different objects
whose only shared label is the model under test.

Every fingerprint, audit field, and config snapshot in this codebase exists to make those
"different evaluations" observable, so reports cannot accidentally be averaged across
inequivalent contracts.

### 1.3 Mapping $\mathcal{R}$ to the codebase

Each component of $\mathcal{R}$ maps to exactly one object in the codebase, and once those
objects are fixed the entire pipeline (sample admission, input construction, tool masking,
output parsing, metric aggregation) has a single audit-replay path.

| Symbol             | Object                              | Implementation                                                                  | Pinning test                |
| ------------------ | ----------------------------------- | ------------------------------------------------------------------------------- | --------------------------- |
| $\mathcal{D}$      | Discrete forecasting dataset        | `SOURCE_DB` / `SOURCE_TABLE`; `loader.sync_questions`                           | `test_db.py`                |
| $M$                | Evaluated model                     | one entry of `MODELS`; one SQLite file per $M$ under `runs/{run_id}/db/`         | `test_runner_grid_model.py` |
| $\kappa_M$         | Knowledge cutoff                    | `MODEL_TRAINING_CUTOFFS[M]` (config.py:L224); `runner.build_task_plan`           | `test_training_cutoff.py`   |
| $\delta$           | Temporal masking offset             | `TAVILY_END_DATE_OFFSET_DAYS` (default $-1$, config.py:L273); `search.tavily_search` | `test_search.py`        |
| $T$                | Max ReAct steps                     | `REACT_MAX_STEPS` (default 12, config.py:L279); `react.run_react` loop bound      | `test_react.py`             |
| $C$                | Max search calls                    | `REACT_MAX_SEARCH_CALLS` (default `[8]`, config.py:L283); `react.py` budget gate  | `test_react.py`             |
| $R$                | Input renderer                      | `prompts.render_user_prompt`                                                    | `test_prompts.py`           |
| $\Psi$             | Output parser and validity          | `parser.parse_answer` (parser.py:L40)                                           | `test_parser.py`            |
| $\phi$             | Answer normalization map            | letter encoding `A` / `A,B` per question_type; `parser.parse_gt` (parser.py:L92) | `test_parser.py`            |
| $\Gamma$           | Aggregation rule                    | `analysis/*` (composite accuracy, FSS, $\kappa$, BI, …)                          | `test_analysis.py`          |
| $H_{\mathrm{aux}}$ | Auxiliary leakage detector          | `leak_filter.filter_search_result`; logged in `run_meta.config_snapshot`         | `test_leak_filter.py`       |

The paper deliberately keeps $H_{\mathrm{aux}}$ outside the formal tuple (paper §3.5) and
binds it via SHA-256 fingerprint to run metadata, because the detector is a replaceable
empirical engineering layer that supports the boundary rather than a primitive component
of the forecasting system itself. The codebase mirrors this distinction: `MODELS` /
`MODEL_TRAINING_CUTOFFS` / `REACT_MAX_*` enter `run_meta` directly, while `LEAK_DETECTOR_*`
enters via `run_meta.config_snapshot.detector_*` plus `run_meta.leak_detector_prompt_hash`.

The information visible to model $M$ on question $q_i$ is given by paper Eq. 16:

$$\mathcal{I}_{i,M}^{\mathrm{vis}} = \mathcal{K}^{M}_{\le\kappa_M} \cup \mathcal{T}_{\le\chi_i},$$

where $\mathcal{K}^{M}_{\le\kappa_M}$ is parametric knowledge before the model's training
cutoff and $\mathcal{T}_{\le\chi_i}$ is temporally masked external information. The
forecasting system $F_M$ then produces

$$\widehat{Y}_{i,M} = F_M(q_i^{\mathrm{in}}; \mathcal{I}_{i,M}^{\mathrm{vis}}), \quad \widehat{Y}_{i,M} \subseteq \mathcal{A}_i.$$

Everything in this codebase is an enforcer of this equation: the LLM is asked to choose
from a finite candidate set $\mathcal{A}_i$ under bounded information, and every
engineering decision is judged by whether it strengthens or weakens that boundary.

### 1.4 What the framework leaves underspecified

Several engineering choices are not mandated by $\mathcal{R}$ and could legitimately be
different in an alternative implementation.

| Engineering choice               | Mandated by $\mathcal{R}$? | What is left to the implementor                                |
| -------------------------------- | -------------------------- | -------------------------------------------------------------- |
| Storage backend                  | No                         | We chose SQLite (one file per model); a row store like Postgres would also work |
| Retrieval backend                | No                         | We chose Tavily; any time-filterable retrieval is permitted (paper §3.3) |
| Detector model                   | No                         | We default to `Qwen3.5-Flash` for the audit; any sufficiently strict model works |
| Concurrency model                | No                         | We chose `asyncio` with one writer task per model; threads or processes also work |
| Backoff sequences                | No                         | Three sequences in `LLM_BACKOFF_*`, tuned for OpenRouter rather than fixed by paper |
| Logging stack                    | No                         | We chose `loguru`; any structured logger works                  |

Decisions in this column would change *which implementation it is*, but not *which
evaluation it is*. The fingerprint of $R$, $\Psi$, $\phi$, and $\Gamma$ covers the
evaluation; the fingerprint of the detector covers the auxiliary leakage barrier;
everything else is engineering. We document this distinction explicitly because it tells
you exactly which knobs are safe to swap when porting OracleProto to a different stack.

---

## 2. The information boundary

Paper §3.5 organises the boundary along three controlled channels plus an uncontrollable
residual.

| Leakage source                                 | Controllable?              | Mitigation                                       | Audited?                       |
| ---------------------------------------------- | -------------------------- | ------------------------------------------------ | ------------------------------ |
| Tavily-returned content (date filter)          | Yes                        | $\chi_i$ injection at the tool layer (§2.2–2.3) | yes (§2.6)                     |
| Provider-native browsing                       | Yes                        | code and test ban (§2.4)                         | yes (`test_llm_no_browsing`)   |
| Model parametric memory                        | Partial                    | $\kappa_M$ admissibility filter (§2.5)           | partial (model-card disclosure) |
| Page bodies that mention post-$\chi_i$ events | Partial                    | Stage-2 LLM detector (§2.6); audited residual ≈ 1.1% | yes (§2.6 audit)               |
| Time clues in the question text itself         | No                         | accepted as evaluation bias                       | no                             |
| External knowledge backflow after training    | No                         | accepted as evaluation bias                       | no                             |

The two ❌-style rows above are the residual sources our claims do not cover. We do not
pretend otherwise; any attack on OracleProto's claims should attack one of these two
rather than the four rows above them. §2.2 through §2.6 mirror these channels in order
from "strictly enforceable" to "honestly declared".

### 2.2 The model never sees $\chi_i$

The `web_search` schema exposed to the LLM has a single parameter, `query` (tools.py:L7).
When Tavily is actually called, $\chi_i = \tau_i + \delta$ is hard-coded and injected by
the tool implementation: the offset $\delta$ comes from `TAVILY_END_DATE_OFFSET_DAYS`
(default $-1$ day, config.py:L273), the cutoff is computed by `react._compute_end_date`
(react.py:L39), and the request body is assembled by `search._build_request_payload`
(search.py:L133). The model can neither perceive nor bypass this.

Two design philosophies sit underneath. **Capability boundaries align with tool
boundaries.** The capability of "knowing the world up to a particular day" is determined
by system configuration and should not be something prompt engineering or model behaviour
can affect. By making the model unable even to see "which day I am cut off at", we
prevent it from inferring or working around that boundary via prompt construction or
parameter injection. **Failure is single and controllable.** Were `end_date` exposed as a
tool argument, we would have to assume the model could either forget to fill it in or
deliberately fill in a future date; holding the decision inside the tool implementation
collapses the failure mode from "the model might make a mistake" to "our code might make
a mistake", which is testable, auditable, and unit-testable.

A natural alternative is to expose `end_date` as a tool parameter so that the model can
reason about cutoffs. We rejected this because it requires trusting the model never to
widen the cutoff; the pin test would have to assert on every emitted tool call,
inflating the test surface from $O(1)$ to $O(N \cdot n)$. A second alternative is to
rewrite the query string to insert a date filter, but this pushes boundary enforcement
into a brittle string-manipulation pass that providers can ignore, since some search
engines silently drop inline `before:2026-04-01` operators.

Pinned by `test_search.py` (the request body always contains `end_date` derived from
`q.end_time`, never from any LLM-supplied field) and `test_react.py` (the schema the LLM
sees has no date parameter at any step).

### 2.3 Why the default is strict: $\delta = -1$ day

`TAVILY_END_DATE_OFFSET_DAYS = -1` is the project default. Many questions in the example
set (sports events, central-bank decisions, Oscar nominations) resolve on the same day,
and using the question's `end_time` as the search cutoff would surface news summaries
that already contain the answer. Pushing the search horizon back by one day trades a
little information granularity for strictness.

Reports also default to comparison at $\delta = -1$, which is itself a design constraint:
numbers under different offsets are not directly comparable, because $\chi_i$ defines a
different admissible information state for each value of $\delta$ (§1.2). The audit
formula in paper §4.3.4 (paper Eq. 61) anchors on $\chi_i$ rather than $\tau_i$ when
classifying "leak / no leak", precisely because the audit definition must match the
operational cutoff actually enforced at the tool layer; any fact that falls in the
half-open interval $(\chi_i, \tau_i]$ is therefore both system-filtered and
audit-classified as a leak, which eliminates the otherwise-ambiguous border zone.

Two alternatives we rejected. With $\delta = 0$, the search cutoff is the question's
resolution day; on the example DB this empirically catches roughly 30–50% of same-day
news leakage depending on time zones and event types, which is too lax for the strict
admissibility we want, so $\delta = +1$ exists only as an ablation knob. With a
per-question $\delta_i$ we could give "end-of-day" events a stricter offset than
"monthly-resolution" events. We rejected this for two reasons. First, introducing a
per-question knob breaks the contract that one $\delta$ defines one evaluation. Second,
the cost of getting one event wrong is asymmetric: a false-negative leak is much worse
than a false-positive over-strict cut, and a single conservative default dominates a
fragile per-question heuristic.

### 2.4 Provider-native browsing is forcibly disabled

OpenRouter, OpenAI, and Anthropic each expose their own web tool or `:online` suffix.
The moment we go down that path the time cutoff is completely out of control. The
project enforces this on three layers, none of which can be silently bypassed.

At the **startup layer**, `Settings._post_validate` (config.py:L599) rejects any model
slug containing `:online` or `::` and aborts before any LLM or Tavily call. At the
**per-call layer**, `llm.chat` only attaches our own `WEB_SEARCH_SCHEMA`; any `plugins` /
`:online` / provider-native retrieval keyword in kwargs is intercepted by
`_assert_no_browsing` (llm.py:L74), and the detector path duplicates this assertion via
`_assert_detector_safe` (leak_filter.py:L139). At the **test layer**,
`test_llm_no_browsing.py` directly mocks the client and asserts that the outbound
payload contains none of those fields, on both the main-LLM and detector paths.

The triple is intentional. Any one of the three suffices to prevent a regression today,
but only the trio survives a refactor that bypasses one of them. Were enforcement
restricted to startup, a test fixture or partial config drift via
`model_copy(update={...})` in dispatcher code could bypass startup-time validation;
re-checking at `llm.chat` send time defends against that exact failure mode. Were
enforcement reduced to a warning rather than a refusal, the warning would eventually
be filtered by log levels, config templates, or CI noise. Refusals stop the run and
cannot be silently ignored.

### 2.5 Parametric memory: filter, do not lie

The tool cutoff cannot constrain facts the model has already memorised in its parameters.
The project takes a very plain strategy following paper §4.1.2: declare each model's
training cutoff $\kappa_M$, and if a question's $\tau_i \le \kappa_M$ (equivalently
$\chi_i < \kappa_M$ at $\delta = -1$ day-rounded timestamps), the question is simply
skipped for that model.

Skipped samples still write a row into the DB with `error="skipped_training_cutoff"`,
generated by `_skipped_cutoff_row` (runner.py:L94) and persisted by the runner main loop
(runner.py:L181). Three properties follow from keeping the skip explicit. Reports can
clearly show how many questions were filtered out per model and how many remain
comparable, which is exactly how paper Table 2's "Excluded by Cutoff" column is built.
`resume` will not retry that row, distinguishing it from transient errors like `network`.
And the row is not counted in the error rate by kind, because it is not a failure but
*active data cleansing*.

Behind this is a principle the project repeatedly invokes: **"filtered out" and "failed"
are two different semantics and must be separated at the data layer.** Were we to use a
boolean `skipped` field, future stratified reporting by cutoff would lose information.
The filter is applied during task generation, before the LLM is ever called, so cutoff
exclusions consume zero API budget; `test_training_cutoff.py` enforces this by asserting
that no `llm.chat` mock is hit on a cutoff-skipped sample.

The paper takes the most conservative interpretation when a model card discloses cutoff
only at month-level granularity (paper §4.1.2 footnote): adopt the *last day* of the
disclosed month as $\kappa_M$. The codebase loads this date as a `datetime.date` in
`_parse_training_cutoffs` (config.py:L181), so the comparison `q_end <= cutoff` is a
strict day-rounded equality match aligned with the paper convention.

We considered two alternatives. *Weighted exclusion*, where samples close to the cutoff
are discounted rather than dropped, adds analytic complexity without removing the
underlying contamination concern; either a question is in the model's training horizon or
it is not, and the binary decision keeps the admissibility set
$\mathcal{D}^{\mathrm{pred}}_M$ a clean subset of $\mathcal{D}$. *Dataset-wide skipping*,
where a question is dropped if any model fails admissibility, would shrink $\mathcal{D}$
to $\mathcal{D}^{\mathrm{pred}}_{\bigcap_m M}$, the intersection over all models, and
discard 10–20% of the corpus on a heterogeneous panel. Per-model admissibility keeps each
comparison fair without burning the shared corpus.

### 2.6 Stage-2 LLM content audit (v5.2)

The defences in §2.2 through §2.5 are protocol-layer (schema, `end_date` injection,
`:online` ban, cutoff skipping). The class of leakage they cannot cover is **the body of
a Tavily-returned page describing events that happened after $\chi_i$**: Tavily's
`end_date` filter operates on a page's crawl/index time, not on the event time described
in the page content. A wiki, an aggregator page, or a long article indexed before
$\chi_i$ can perfectly well reference future events in its body. Paper Table 3 places
the residual leakage rate of the Tavily-only baseline in the 3–16% range, high enough
that the single-digit accuracy gaps in paper Table 5 would become statistically
meaningless without further filtering.

The `search-leak-filter-v1` solution adds an independent LLM audit layer (the
"detector") at the tail of `tavily_search`, before the main LLM ever sees `tool_result`.
Each `SearchResultItem` is sent to the detector individually via
`leak_filter.filter_search_result`, with verdict in `{keep, drop, failed:*}`. Items whose
verdict is `drop` are removed entirely; the main LLM never sees any field of a dropped
item, including `title`, `url`, `content`, or `raw_content`. The detector verdict
surfaces only through `SearchResult.audit`, consumed by `react._record_search_call`,
never through any LLM-visible payload.

| Dimension          | Implementation                                                                                                          |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Cut point          | end of `forecast_eval/search.py:tavily_search`, before `return`                                                          |
| Client             | `_detector_client: AsyncOpenAI` (leak_filter.py:L109), independent module-level singleton, not shared with the main LLM |
| Input fields       | whitelist: `title / url / published_date / content / raw_content / cutoff_date`; the detector must not see any field of `Question` |
| Prompt strictness  | 6 principles in `LEAK_DETECTOR_PROMPT_TEMPLATE` (leak_filter.py:L55): cutoff_date placeholder, treat specific/scheduled/speculative future events equally, "ambiguous → drop", forbid parametric knowledge, strict JSON output, no awareness of the question |
| Parameters         | temperature `0.0`, max_tokens `512`, timeout `60s`, concurrency `5` (config.py:L345)                                    |
| Failure mode       | FAIL-RETRY → CLOSED: $K$ retries still failing → drop; AUTH errors are caught locally and immediately drop, with no propagation and no aborting the whole run |
| Observability      | `search_calls.detector_*` five fields plus `run_meta.config_snapshot` detector three-key fingerprint                    |
| Master switch      | `ENABLE_SEARCH_LEAK_FILTER` (config.py:L337), default True; when off, byte-level rollback to v5.1                       |

Three design choices deserve emphasis.

**The detector does not see the question.** A detector that knows the question can morph
into an "answer auditor" that drops everything arguing against the model's tentative
answer, producing question-specific second-order leakage. The whitelist enforces the
detector's role: classify temporal leakage of facts (does this page mention an event
after $\chi_i$?) rather than relevance to the answer. This is paper §3.3 and §3.5's
design rationale encoded as an inviolable input contract; pinned by `test_leak_filter.py`
asserting that the detector's user message never contains question fields.

**Fail-closed by default.** Detector hiccups (timeout, network) are uncorrelated with
item content; biasing the residual towards "drop on uncertainty" is the conservative
choice. The keep-on-failure mode (`LEAK_DETECTOR_FAIL_ACTION=keep`) exists only as an
A/B escape hatch for the rare case of comparing against the unfiltered baseline.
Authentication failures (401/403) skip retries and immediately drop, since retrying an
auth failure is both pointless and a billing footgun.

**An independent client singleton.** Reusing `forecast_eval/llm.py:_client` would couple
two error-budget pools, so main-LLM retries would inflate detector quota and confuse log
triage. Two singletons, two backoff sequences, two log namespaces. The independent
client also allows the detector to use a *more capable* model than the model under test,
which is strictly preferable, since a more capable detector is the cheapest place to
spend extra capability in the pipeline.

The detector's reference date is the question's $\chi_i = \tau_i + \delta$, sharing the
same source as Tavily's `end_date`, and is independent of §2.5's `MODEL_TRAINING_CUTOFFS`
(the model training cutoff $\kappa_M$, indexed per model). Even after a question passes
through the admissibility filter and enters execution, the detector still audits the
search results for that question; the two filters do not substitute for each other.

The BLF paper (Murphy 2026, §B.1 Stage 2) used an LLM-based leak classifier on top of
Brave's date filter and reported the runtime filter catching 320/341 = 93.8% of actual
leakage, leaving residual leakage on the order of 1.5%. Our own audit (paper Table 8)
on $N=270$ items measures recall 98.7% (235 / (235 + 3)) and per-audit-item residual
rate 1.1% (3/270, with Wilson 95% upper bound 3.2%), comparable to the lower end of the
Tavily-only baseline at two orders of magnitude lower marginal cost. Stage 2 is an
empirically validated "algorithmic-layer plus semantic-layer double insurance"
engineering practice. Full spec at `openspec/changes/search-leak-filter-v1/specs/`
(capabilities `search-leak-filter`, `search-tool`, `information-barrier`,
`results-persistence`).

---

## 3. Dataset reconstruction

The paper's central conceptual move (paper §3.2) is to rewrite a resolved event $z_i$ as
a time-bounded prediction:

$$(x_i, \mathcal{A}_i, Y_i, \tau_i, \rho_i) \quad \Rightarrow \quad q_i^{\mathrm{in}} = (x_i, \mathcal{A}_i, \chi_i, \rho_i).$$

The ground truth $Y_i$ is retained only for scoring; the visible prompt is rendered
deterministically from the structured fields. This construction equips the dataset with
four properties that make the evaluation object dataset-level rather than event-level
(paper §3.2). *Temporal reproducibility* means the same source row plus the same $\delta$
always yields the same $\chi_i$. *Model-dependent admissibility* means admissible question
sets vary by $\kappa_M$ while the underlying corpus is shared. *Discrete scorability* means
$\widehat{G}_{i,M}, G_i \subseteq \mathcal{L}_i$ are finite-cardinality sets rather than
free text. *Audit-reproducibility across calendar years* means the same dataset can be
replayed against models with later cutoffs, with the admissibility set automatically
shifting.

### 3.1 Discrete answer space, not free generation

Each question must have a finite answer space $2 \le K_i < \infty$ and a verified answer
set $Y_i \subseteq \mathcal{A}_i$ (paper §3.2). This prevents open-ended generation from
bypassing the evaluation constraint. The three question types `yes_no`, `binary_named`,
and `multiple_choice` all collapse to letter sets at the scoring layer, with single-answer
questions satisfying $|Y_i|=1$ and multi-answer questions $|Y_i| \ge 1$. The structural
constraint $\rho_i$ (single vs multi) is recorded in `choice_type` and consumed by the
parser to validate that the model's output cardinality is legal.

We rejected two alternatives. Open-ended natural-language outputs scored by an LLM judge
would import a second LLM into the scoring path and re-introduce a contamination concern,
since the judge may have its own training data overlap; strict letter-set scoring keeps
the scoring path deterministic, audit-replayable, and judge-independent. Numerical
probability outputs scored by Brier or NLL alone would fail under the K=5 sampling
regime, where the empirical $\hat p$ takes only six discrete values and calibration
parameters become statistically meaningless; the v4 belief protocol does collect
probabilities, but as a companion to letter-set output rather than a substitute (§6.6).

### 3.2 Letter encoding as the canonical answer

The source DB `answer` field uniformly uses letters (`'A'` or `'A, B'`) rather than
option text. For `yes_no` questions Yes maps to A and No to B. For `binary_named`
questions the first entity maps to A and the second to B. For `multiple_choice` questions
A/B/C/... follow the order of the `options` JSON array.

The model's output form varies by question_type (a `Yes`, an entity name, or a letter
list are all possible), but `parser.parse_answer` (parser.py:L40) uniformly normalises to
`frozenset[str]`. This decouples *how the model says it* from *how the system scores it*:
the same scoring code covers all three question types because they all reduce to set
equality on letter sets. Pinned by `test_parser.py` round-trips on representative inputs
of each type, including case variations and whitespace tolerance.

### 3.3 Strict frozenset equality is the scoring primitive

The entire scoring logic is one line at `parser.py:L102`:

```python
predicted_letters == ground_truth_letters  # both are frozenset[str]
```

Missed selections, extra selections, ordering: all are scored as wrong by strict
equality. This is the project's most fastidious design and implements paper Eq. 40
verbatim.

We do not use Jaccard, F1, or partial credit at the strict level for three reasons.
First, *explanation cost is too high*: `pass@1=0.62` is far more intuitive than `mean
Jaccard=0.74`, and one number is enough for a paper. Second, *avoid half-credit masking
the real problem*: a model that misses one or two selections every time can still average
70+ on a soft scorer while fundamentally not having mastered the question; strict
matching scores this behaviour as 0 and forces the report to be honest. Third, *unify
the scoring interface across three question types*: all three reduce to frozenset
equality at the scoring stage, so the code is written once and works for everything.

For multi-answer questions the project adds two soft-penalty companions alongside strict
equality (paper §4.2.2 and §4.2.7), without ever replacing it.

* **Exam-style partial credit** (paper Eq. 37, `analysis/exam_score.py:L62`) follows the
  rule "any FP vetoes to 0; otherwise score $|TP|/|G|$", which is recall under a zero-FP
  gate. It makes the headline composite-accuracy more nuanced for multi-answer buckets
  where strict equality has near-zero variance.
* **Format Skill Score (FSS)** (paper Eq. 59, `analysis/accuracy.py:L386`) is a
  chance-corrected Tversky similarity at $(\alpha, \beta) = (2.0, 0.5)$, penalising
  false positives 4× more than false negatives. The intuition is that *claiming an
  event will happen* is more dangerous than *missing an event*. Single-select questions
  degenerate to strict 0/1; the asymmetry only matters in multi-answer buckets.

The two soft penalties coexist with strict equality precisely so that the choice of
metric becomes an analyst-side decision rather than a system-side bias. The audit trail
(`composite_meta.json`) records exactly which buckets each composite uses.

The Tversky asymmetry $(\alpha, \beta) = (2.0, 0.5)$ encodes a prediction-domain prior:
in forecasting, claiming an event will happen carries more downstream risk
(acted-upon false signal) than missing it (opportunity cost). A 4× FP-vs-FN penalty is
one octave on the log scale, strong enough to flip cross-model rankings on the
multi-answer bucket and conservative enough to leave $\alpha = \beta = 1$ (Jaccard)
recoverable as an ablation knob. The values are configurable through the `tversky_score`
and `tversky_baseline` `alpha` and `beta` keyword arguments (accuracy.py:L289 and L320),
but the analysis pipeline hard-codes the `(2.0, 0.5)` defaults; changing them requires a
code edit rather than a `.env` change, because the *interpretation* of FSS depends on
the asymmetry being held fixed across runs.

### 3.4 ASCII continuation labels: documented debt

The example DB contains 4 `multiple_choice` questions with > 26 options, of which 3 have
ground-truth answers landing on ASCII continuation characters such as `[`, `\`, `]`,
`^`, `_`, `` ` ``, `a`, `b`, `c`. These characters are extremely unfriendly to LLMs
(backticks are eaten by markdown, lower- and upper-case `a`/`A` are easily confused),
but the project still keeps them because doing so preserves a one-to-one letter ↔ index
mapping.

The cost is mitigated by several defences: `prompts.render_user_prompt` explicitly quotes
or escapes labels when generating an `outcomes_block` for > 26 options;
`parser.parse_answer` (parser.py:L74) iterates `tokens` of length 1 only and uses
`letter_to_index` round-trip validation, pinned by `test_parser.py` round-trip cases on
> 26 options; logs and reports record letters and labels in parallel for manual review.
If we later confirm LLM performance is meaningfully dragged down by the labelling
scheme, we will migrate to a stable scheme like `AA/AB` or `A01/A02`. **This is a
documented debt, not an ignored bug.** The alternative of skipping the > 26 questions
discards roughly 4 / 215 multi-choice questions on the example DB, which loses
dataset-level coverage; we prefer the round-trip test plus the option-stable encoding.

### 3.5 Parse failure is not an error

When the LLM does not output `\boxed{...}`, or writes "I cannot predict the future",
this is not a system failure but part of the model's capability (paper §4.2.5). Three
mechanical consequences follow.

`parse_ok=0` and `correct=NULL`: parse failures and refusals are accumulated separately
into "refusal rate" and surfaced in reports. *No retry*: when the model itself says it
cannot answer, asking again yields the same answer, and backoff retries only waste
tokens. `error` field stays NULL: the `error rate by kind` report is not polluted by
such soft failures.

The same coupling rule is encoded in paper §4.2.4 as a four-state matrix; `exam_score`
(exam_score.py:L62) implements the matrix in a 7-line decision tree, pinned by
`test_exam_score.py` and `test_aggregation.py`. The principle is that **every behaviour
must have its own cell in the report**: "system error rate" and "model refusal rate" are
two different things and cannot be lumped under a single total error rate.

The alternative of *retry on parse failure* fails twice over. First, capability masking:
a model that refuses to commit to an answer is a model that lacks forecasting capability,
and retrying papers over the gap. Second, cost: retries on a 10K-item evaluation
multiply API spend without changing the population mean.

---

## 4. Hierarchical evaluation

The paper's evaluation system (paper §3.4) is

$$\mathcal{E}_M = (\mathcal{E}^{\mathrm{valid}}_M, \mathcal{E}^{\mathrm{item}}_M, \mathcal{E}^{\mathrm{question}}_M, \mathcal{E}^{\mathrm{model}}_M).$$

Four levels, each with a distinct semantic, each computed from the same normalised
discrete answer space.

| Level                     | Object                                                                              | What lives here                                       | Code                                                |
| ------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------- | --------------------------------------------------- |
| **Validity**              | $v_{i,M} = \mathbb{1}[\Psi_i(o_{i,M}) \ne \bot]$                                    | parse_ok / parse_failure_rate                         | `parser.parse_answer` / `analysis/aggregation.py`   |
| **Item**                  | $r_{i,M} = \mathbb{1}[\widehat{G}_{i,M} = G_i]$ on a single trial                  | strict equality / exam-score                          | `parser.is_correct` / `analysis/exam_score.py`      |
| **Question**              | $\{\widehat{G}_{i,M}^{(s)}\}_{s=1}^{S}$ across $S$ trials                           | pass_any@N / pass_all@N / Fleiss $\kappa$ / VCI / MV  | `analysis/accuracy.py` / `analysis/consistency.py`  |
| **Model**                 | $\Gamma(\{\mathcal{E}^{\mathrm{question}}_{i,M} \mid q_i \in \mathcal{D}^{\mathrm{pred}}_M\})$ | composite accuracy / FSS / BI / per-correct cost | `analysis/composite.py` / `analysis/__init__.py`    |

Each level is captured by a separate column family in the analytics output, and the
choice of $\Gamma$ at the model level is the analyst's lever rather than the system's.
The same raw observations support flat means, weighted composites, paired-bootstrap CIs,
and posterior comparisons because nothing is pre-aggregated in the DB.

### 4.1 Why no pre-aggregation in the DB

Three consequences make this one of the project's most important architectural choices.
*Metric definitions evolve*: today `pass@3` means "1 of 3 counts as a pass", and
tomorrow it might change to "at least 3 correct"; if aggregation lands in the DB, every
redefinition requires a backfill, and deferring all metrics to the analysis layer makes
them re-computable at any time. *`analysis.py` is a pure function*, where the input is
`runs/{run_id}/db/*.db` and the output is `analysis/*.csv|md|json`, and it can be re-run
independently via `python -m forecast_eval.analysis`. *DB and paper/report are
decoupled*: raw records are an engineering artefact, statistics are an academic artefact,
and their cadences are completely different.

This is the operational embodiment of the paper's "metric-agnostic design" claim
(paper §3.4). Pinned by `test_analysis.py` constructing a hand-crafted DB fixture and
confirming that `run_analysis` neither writes back nor mutates.

### 4.2 Recalibrating the `pass@k` naming

In the wider community `pass@k` generally means "at least one correct in $k$". The
project historically used `pass@3 = sum(correct) ≥ 3` (a threshold semantic), which
caused ambiguity. Now made explicit (paper §4.2.5, paper Eq. 42–45):

* `pass_any@N` is the standard `pass@k`: at least one correct in $N$.
* `at_least_k_correct@N`: at least $k$ correct in $N$ (threshold analysis).
* `pass@1 avg`: average accuracy across $N$ (stable capability).
* `majority vote correct`: whether the majority-vote frozenset across $N$ is correct
  (self-consistency).

The four columns are emitted by `analysis/accuracy.py::Aggregate.as_ordered_dict`
(accuracy.py:L66); their definitional identities $\mathrm{pass\_all} \le
\mathrm{pass@1}_{\mathrm{avg}} \le \mathrm{pass\_any}$ hold by construction and are
asserted in `test_aggregation.py`. The principle is simple: a name must either be
unambiguous or must explicitly declare its semantics.

### 4.3 Question-level signals: stability is not correctness

Paper §4.3.3 reports an instructive divergence among the six tested models that motivated
the entire question-level signal column family. DeepSeek and Kimi tie on
$\mathrm{pass\_any}@N$ ($0.80$, the best-of-3 hit upper bound), but Qwen leads on
$\mathrm{pass\_all}@N$ ($0.39$) and Fleiss' $\kappa$ ($0.45$, consistently same-answer
behaviour), while Doubao ranks 3rd on Fleiss ($0.42$) but last on $\mathrm{pass}@1$,
which is *consistently giving wrong answers*.

This is exactly the diagnostic the question-level signals are designed to expose: high
consistency does not imply correctness, and a high best-of-N ceiling can come from
"three different answers, one of which happens to hit" rather than from "consistently
correct each time". The project therefore reports both axes side by side rather than
collapsing them.

The Fleiss' $\kappa$ implementation (consistency.py:L176) follows paper §4.2.6 (paper
Eq. 49–53) verbatim, including the per-stratum decomposition for single-answer questions
where each $k_q$ stratum is its own $\kappa$ weighted by question count, and the
per-label binary decomposition for multi-answer questions. `test_consistency.py` pins
both decompositions on hand-crafted vote tables.

### 4.4 Composite accuracy weighted by sub-question type

`per_model_summary.csv` reports a flat mixed mean for backwards compatibility. For
headline scoring the paper uses composite accuracy: per-bucket exam-score followed by
subtype-weighted average (paper Eq. 35, paper §4.2.3). Two dimensions are computed
independently, since `multiple_choice` itself contains both single and multi and the two
are not orthogonal.

The `question_type` dimension (yes_no / binary_named / multiple_choice) lands in
`per_model_composite_by_question_type.csv`, while the `choice_type` dimension (single /
multi) lands in `per_model_composite_by_choice_type.csv`. For each (model, dimension,
metric):

$$\text{composite}_m = \frac{\sum_{b \in B_{\text{valid}}} w_{m,b} \cdot v_{m,b}}{\sum_{b \in B_{\text{valid}}} w_{m,b}}.$$

Missing buckets (slice unavailable or weight 0) are dropped, and the remaining weights
are renormalised proportionally; they are *not* treated as 0. All None means composite =
None. The identical formula and renormalisation rule appear at `composite.py:L18`
(formula) and `composite.py:L77` (allowlist plus per-metric override resolution).

#### Default weights: harder questions discriminate better

| Dimension       | Bucket            | Default weight | Difficulty rationale                                        |
| --------------- | ----------------- | -------------- | ----------------------------------------------------------- |
| `question_type` | `yes_no`          | 0.15           | $k=2$, blind guess 50%, almost no inter-model discrimination |
| `question_type` | `binary_named`    | 0.15           | $k=2$ as above, adds entity recognition                     |
| `question_type` | `multiple_choice` | 0.70           | $k=2..N$ wide range, includes multi-select, highest discrimination |
| `choice_type`   | `single`          | 0.40           | overall easier (includes yes_no and binary_named)           |
| `choice_type`   | `multi`           | 0.60           | true multi-select, almost every model performs poorly, high discrimination |

Defaults at `config.py:L365`. The principle is one sentence: **let buckets that
discriminate model capability contribute more.** Switching to "I care more about easy
questions" requires only flipping the numbers via `COMPOSITE_WEIGHTS_QTYPE` or
`COMPOSITE_WEIGHTS_CTYPE` in `.env`. This is an opinionated default rather than a
neutral one; we believe the orientation above is more reasonable for the vast majority
of evaluation scenarios, and users who disagree solve it by overriding one line of
`.env`.

Two alternatives are tempting but wrong. *Equal weights across buckets* sounds neutral
but is not, since the empirical question-type prevalence on the paper's curated
80-question set is roughly $\{\text{yes\_no}: 37/80, \text{binary}: 3/80, \text{mc}:
40/80\}$; equal weights would double-count yes_no relative to mc and erase the most
discriminative bucket. *Empirical-prevalence weights* are equivalent to a flat
unweighted mean and fail for the same discrimination reason: when 50% of questions are
near-random-baseline, weighting the composite by prevalence drowns out the
signal-bearing buckets.

#### Why the chance baseline matters under the exam view

Under the exam view, the chance baseline on the multi-choice multi-answer bucket is
$T^{\text{chance}}_q = 2^{-(k_q - m_q + 1)}$ (paper Eq. 41), which lands in $[0.06,
0.25]$ for typical $(k_q, m_q)$. Compare this with the strict-equality baseline
$0.5^{k_q}$, which is essentially zero for $k_q \ge 5$. The exam view places the
multi-answer column on the same order of magnitude as single-answer buckets in absolute
terms, so the multi-answer signal, which is the one that actually discriminates models,
is no longer drowned out by its near-zero strict-view variance. This is the core gain
of the exam composite over a strict-equality composite, documented in paper §4.2.4 as
the explicit rationale for choosing `exam_score_at_n_avg` rather than strict
$\mathrm{pass@1}$ as the *headline* composite.

#### Configuration entrypoint

`COMPOSITE_WEIGHTS_QTYPE` and `COMPOSITE_WEIGHTS_CTYPE` (config.py:L365 and L372) hold
the global default weights, shared by all metrics that are not explicitly overridden.
`COMPOSITE_WEIGHT_OVERRIDES_QTYPE` and `COMPOSITE_WEIGHT_OVERRIDES_CTYPE` hold per-metric
independent overrides in the form `"fss=yes_no=0.05,multiple_choice=0.95"`,
semicolon-separated for multiple metrics. Misspelled metric names raise from
`compute_composite` during the analysis phase rather than silently falling back to the
default; this is intentional, because when you misconfigure we want to make sure you
know. Startup-time validation (config.py:L515) requires bucket name to be in the legal
set, weight $\ge 0$, and at least one weight $> 0$.

### 4.5 Per-correct cost as a Pareto axis

The paper proposes (paper Eq. 60)

$$C^{\text{per-correct}}_m = \frac{C^{\text{total}}_m}{|\mathcal{D}^{\text{eval}}| \cdot n \cdot \text{Composite\,Accuracy}_m}.$$

Adopting OpenRouter's actual billing instead of a "published unit price × token usage"
calculation circumvents grey areas like whether reasoning tokens are billed, how the
prompt-cache discount is accounted for, whether tool calls are billed, and whether
prices differ across provider routings. The platform invoice is the single financial
fact verifiable by third parties.

Dividing by the **difficulty-weighted notional correct count** rather than by the raw
correct count matters because it places "expensive but accurate" and "cheap but
reckless" models on the same scale of cost-effectiveness, avoiding the false low-cost
illusion produced by "low per-sample unit price but high error rate". Semantically,
$C^{\text{per-correct}}$ is the reciprocal of "how many difficulty-weighted correct
predictions does one USD buy".

The paper's experimental table (Table 5) demonstrates the value of this axis: at
composite accuracy within 1.2pp of DeepSeek's lead ($0.6016$ vs $0.5896$), Qwen costs
$1/8$ the total ($\$0.45$ vs $\$3.60$). The joint (accuracy, cost-per-correct) Pareto
frontier is the only meaningful comparison surface; ranking on accuracy alone or cost
alone is misleading. On this axis Qwen and DeepSeek jointly span the frontier, while the
other four models are Pareto-dominated by at least one endpoint.

---

## 5. ReAct loop and tool use

### 5.1 The whole prompt as a single user message

The template is one entire prompt block (`agent_role + event + outcomes + format +
guidance`); the project chooses to feed it as a single user message rather than
splitting into system / user. Three reasons drive this.

The choice **most faithfully reproduces the source metadata template**: the source data
`dataset_metadata.features_json.prompt_reconstruction` is a single string, and forcibly
extracting a system part would lose the semantics of the original concatenation. It
**preserves cross-model consistency**: different providers handle system messages
differently (OpenAI hard-caches them, Anthropic uses an independent field), and going
uniformly through a user message yields the most stable comparability, which is exactly
the property paper §3.5 demands of $R$ as a deterministic renderer. And it **stays easy to
hash and diff**, since the entire prompt content is written directly into the
`user_prompt` field, so any future template change is visible through a hash at a
glance.

The alternative of splitting system / user loses cross-provider consistency; the same
$R$ becomes effectively two different renderers depending on provider.

### 5.2 Hard ceilings on the loop

Each sample has two gates. `REACT_MAX_STEPS = 12` (config.py:L279) is the maximum number
of LLM-system interaction rounds per sample; enabling the reflection protocol or nudges
adds 2–4 rounds beyond a single-step direct answer, so the default is slightly higher
than the historical value. `REACT_MAX_SEARCH_CALLS = [8]` (config.py:L283) is the
cumulative `web_search` budget per sample; once spent, the tool returns `search budget
exceeded` directly to the LLM. The paper main-table runs use $C = 4$ as the headline
configuration, with the rationale $R_{\mathrm{tav}} \cdot C = 5 \cdot 4 = 20$,
approximately two pages of Google search results.

Defining an upper bound on the model's autonomous searching via a budget serves two
purposes. Without a cap, malicious or degenerate models could call indefinitely and
burn through the API bill. Capping while returning an error rather than throwing lets
the LLM still provide a "best-effort answer" based on existing information, and
separates "out of budget" from "system crashed".

Exceeding step count without producing a boxed answer is treated identically to a
refusal: `parse_ok=0` (§3.5).

The codebase default $C = 8$ is *deeper* than the paper main run $C = 4$, and this is
intentional. The paper's main-table configuration is a deliberately tight budget for
discrimination (paper §4.1.4, "two pages of Google"), while the example DB ships with a
wider budget for smoother behavioural analysis. To exactly reproduce the paper main run,
override `REACT_MAX_SEARCH_CALLS=4` in `.env` (FRAME §1.3 reconciliation table).

### 5.3 Reflection protocol pulls the model off "one-shot direct answer"

Some models give a final answer after only ~1.6 searches on average; this confident
one-shot behaviour drastically lowers `pass@1` on long-tail events. The project responds
with a three-part protocol family (paper §4.1.3 inference-protocol bullets).

The **reflection protocol** (`REACT_REFLECTION_PROTOCOL=true`, default on,
config.py:L288) appends a *Forecasting Protocol* to the end of each sample's user
message: decompose the question, list ≥3 different retrieval angles, reflect after each
search, cross-validate, check the opposite direction, and state confidence. The
**budget-awareness protocol** (`REACT_BUDGET_AWARENESS_PROTOCOL=true`, default on,
config.py:L313) front-loads "total step count + total search count" in the prompt so the
model can plan holistically and reserve the final step for emitting `\boxed{...}`. The
**forced finalisation near the limit** (`REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=true` with
`LOOKAHEAD=2`, default on, config.py:L314 and L315) actively injects user messages as
the loop approaches its limit: a soft reminder at the second-to-last step where tool
calls are still permitted, and a hard cutover at the final step with an empty tool list,
forcing content-only output of `\boxed{...}`.

These protocol additions are *not* written into `dataset_metadata`, so
`prompt_templates_hash` does not change; their existence is persisted through
`run_meta.config_snapshot` alongside each sample's `user_prompt` field, enabling
per-question diffing after the fact.

A complementary fallback `REACT_MIN_SEARCH_CALLS` (soft minimum search count, default
`0` meaning off, config.py:L292) exists for the rare case where prompt guidance alone
cannot pull a model off one-shot direct answers; when on, the system injects a user
nudge asking it to try a different angle and search again, with the per-sample nudge
count capped by `REACT_MAX_NUDGES`.

The design philosophy is **prompt first, rules second**. The protocol family is
guidance; the nudge is restriction. We first try better guidance to make the model walk
a few more steps spontaneously, and only impose a soft floor when the model still
insists, in order to avoid mixing "the capability under evaluation" with "the system's
enforcement". All switches have clear defaults, and turning them off degrades to the
historical behaviour, so the same code can run "protocol on vs off" controlled
experiments. The protocol text and nudges both appear in `messages_trace`, and the
on/off state is anchored by `config_snapshot`, so this is *not* implicit behaviour.

The alternative of a hard-floor `REACT_MIN_SEARCH_CALLS` by default would conflate
capability ("does the model search enough?") with enforcement ("we made it search"),
which is why the default 0 keeps the floor opt-in and the reflection protocol drives
natural search depth.

### 5.4 v5.1 four-knob priority chain

For cross-model comparisons, **`parse_failure_rate` must reflect only the model's own
format failure, not upstream resource exhaustion in the harness.** In v5.0, after
`REACT_MAX_SEARCH_CALLS` was exhausted the `web_search` schema was still exposed to the
LLM; the model kept asking for the tool and hit the `REACT_MAX_STEPS` ceiling, with
`final_raw=""` becoming `parse_ok=0` directly. That disguised "tool starvation" as
"format failure".

v5.1 added four orthogonal switches at `react.run_react`, with the priority decision
logic at react.py:L266, each defending a different failure mode.

| # | Switch                                    | Default | What it does                                                                |
| - | ----------------------------------------- | ------- | --------------------------------------------------------------------------- |
| 1 | `REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT`     | True    | Last step → `tools=[]` plus force-finalise text (graded by LOOKAHEAD)       |
| 2 | `REACT_BUDGET_EXCEEDED_DROP_TOOLS`        | True    | Once $\ge C$ searches done → subsequent rounds get `tools=[]` (in-loop)     |
| 3 | `REACT_FINAL_ANSWER_RETRY`                | False   | Loop ended cleanly but `final_raw=""` → one extra LLM call with `tools=[]` |
| 4 | `REACT_MIN_SEARCH_CALLS` / `MAX_NUDGES`   | 0 / 2   | Opt-in soft floor; nudge user message when below floor                      |

The four are evaluated each iteration in **strict priority order**, encoded verbatim at
react.py:L266 as `Priority is (1) > (2) > (3) > (4)`.

1. **Last-step hard cutoff.** Detected via `(REACT_MAX_STEPS - step) <=
   REACT_FORCE_FINAL_ANSWER_LOOKAHEAD` with `remaining == 1` (react.py:L272). Supersedes
   everything else; the model can ONLY emit content this turn.
2. **Penultimate soft warning.** Same `force_final_active` flag with `remaining ∈ [2,
   LOOKAHEAD]` (react.py:L297). Tools still exposed; the warning text branches
   internally on whether the search budget is already spent.
3. **Budget-exhausted commit notice.** Fires once when `searches_done_now >=
   REACT_MAX_SEARCH_CALLS` and `REACT_BUDGET_EXCEEDED_DROP_TOOLS=True` (react.py:L310).
   After this fires, every subsequent round gets `tools=[]` regardless of which other
   branch fires.
4. **Continuation reminder.** Lowest priority; previous turn was content without
   `\boxed{...}` and nothing else needs to fire (react.py:L320). Replaces the historical
   inline "Harness: step N complete" injection that could double-inject with later
   branches.

The new analysis column `final_answer_retry_rate` lands in `per_model_summary.csv`,
letting analysts see how much the fallback caught separately and decide, when necessary,
whether to deduct it from the `pass_at_1` denominator. Schema upgraded to v5: each
sample slot in `run_results` adds `s{i}_final_answer_retry_used INTEGER`; old v4 DBs
auto-ALTER ADD via `init_schema` (NULL-compatible).

The default for `REACT_FINAL_ANSWER_RETRY` is **False**, superseded by switch #1
(force-final-answer-near-limit, in-loop) and kept as an optional out-of-loop emergency
backstop. Enabling it costs one extra LLM step (`react_steps + 1`) but does NOT count
toward `nudges_used`, since the semantics differ: nudges are about search depth, this is
about format compliance.

The principle is **make every reason a sample fails to commit visible and separable**.
The four knobs are orthogonal because the failure modes they target are orthogonal;
pinning them to a priority chain prevents two switches from fighting each other. The
alternative of a single "harness rescue" switch would lose ablation discrimination: two
of the four switches are pure rescue (1 and 3), two are pure shaping (2 and 4), and
collapsing them would erase the ability to A/B test which intervention actually moved
`parse_failure_rate`.

### 5.5 Graceful degradation for tool-call errors

Within the ReAct loop, several tool-related errors do not interrupt the whole sample.

| Situation                       | Handling                                                                        |
| ------------------------------- | ------------------------------------------------------------------------------- |
| Unknown tool name               | return `unknown tool` to the LLM and let it change tack                         |
| `arguments` JSON parse fails    | send the error back as `tool_result`; the LLM can retry                         |
| Search budget exhausted         | `tool_result` returns `search budget exceeded`                                  |
| Tavily itself errors            | go through `SEARCH_BACKOFF_S` retries; if still failing, stuff the error into `tool_result` |

The principle is to **let the LLM see its own failures from the system's perspective
rather than papering over them**. The capability numbers this produces are closer to
reality, since a model that cannot handle tool failures should naturally score lower.

### 5.6 Reasoning-model parameter exclusions

Some reasoning models (`o1`, `o3`, `r1`, `qwq`) directly return 400 on custom sampling
parameters like `temperature` or `top_p`. The project maintains the substring list
`LLM_REASONING_MODEL_PATTERNS=o1,o3,o4,r1,qwq` in `.env` (default at config.py:L241);
for matching models, those two parameters are not passed at call time. This is a
"shift maintenance cost forward" design: rather than identifying 400 inside retry or
error handling, we handle it at request construction time.

---

## 6. Storage and writers

### 6.1 Per-run directory and per-model SQLite

```text
runs/{run_id}/
  manifest.json          # run-level metadata
  db/{model_slug}.db     # one sqlite per model (one virtual slug per file under grid)
  analysis/              # post-hoc statistical artefacts
  logs/{run_id}.log
```

The early "single `results.db`" was replaced. With a single DB the boundary between runs
depended entirely on the `run_id` column, making independent distribution hard and
making it easy to mix data from other runs into analysis. `run_id` defaults to
`YYYYMMDD-HHMMSS-xxxx`, so `ls` naturally sorts by time and the directory name tells you
"when it ran" at a glance. An empty `RUN_ID` starts a new run; the same value resumes
the existing one, so one variable handles both modes without an extra `--resume` CLI
flag.

We chose one SQLite file per model rather than one per run, for three reasons in order
of importance.

*Independently distributable.* Hand `runs/{run_id}/db/openai__gpt-5.db` to someone else
and they can replay just this one model, with no need to obtain the other models'
results. *Non-interfering write paths.* One async writer task per model, with
single-writer-multi-reader WAL mode providing ample concurrency, and one model's stall
cannot block another's. *Schema-evolution isolation.* If some model needs to store a
special field (e.g. a reasoning trace), its schema can be extended independently without
affecting others.

The cost is that the analysis layer must scan multiple files, but `analysis.py` already
encapsulates that. The alternative of "one DB with a `model` column" trips up all three
properties; specifically, single-writer contention turns one slow provider into a global
stall because each writer holds the same DB-level lock under WAL.

### 6.2 Each DB self-contains questions and prompt_templates

Each model DB embeds copies of the source question set and prompt templates. This looks
redundant at first glance, but it serves *independent replay*: whoever receives
`openai__gpt-5.db` does not need to track down `forecast_eval_set_example.db`, nor hunt
for which metadata version was in use at the time, since every input the evaluation
needed is inside this single DB.

Consistency between copies is guaranteed by hash verification: the three fields
`run_meta.source_db_hash`, `metadata_hash`, and `prompt_templates_hash` pin down "the
source data at the time". The alternative of storing paths instead of copies breaks the
self-contained property; once the path moves, the DB becomes unreplayable.

### 6.3 Wide table with N pinned at table-creation time

One row per question, with an `s{i}_*` group of columns per sample. As of v3 the schema
holds 20 fields per sample (the original 14 plus 6 newly added observation columns); v4
adds 3 belief columns; v5 adds 1 final-answer-retry column. Compared with a "long table
plus (question_id, sample_idx) composite primary key", the wide table has three
advantages.

*Resume queries are naturally simple.* `SELECT question_id WHERE s{i}_created_at IS NOT
NULL` simply scans one column, no group-by needed. *Atomic single-row read.* The
analysis script reads one row and has every sample, with no join or aggregation needed.
*Schema fixes $N$.* `SAMPLING_N` is pinned at table-creation time, so whenever the DB is
reopened in the future, the structure matches what it was then.

The cost is that `SAMPLING_N` must be determined before the run starts and cannot be
expanded mid-run; the schema also needs to dynamically generate `20 × N` columns. This
cost is acceptable in evaluation scenarios, since `SAMPLING_N` is by nature part of the
run config and should not change mid-run. The alternative of a long table plus composite
key wins on flexible $N$, loses on trivial resume queries, and loses on JOIN-free
analysis: long tables are better for production telemetry, wide tables are better for
evaluation artefacts.

### 6.4 `step_metrics` is a JSON column rather than a separate long table

ReAct's per-round step metrics are naturally 1-to-N (one sample yields multiple steps),
and at first glance you would want to factor them into a long table. The project
ultimately compresses them into `s{i}_step_metrics TEXT` (a JSON array) for three
reasons.

*No cross-step query need.* The analysis layer always fetches the whole trajectory by
sample and then processes it; it never does row-level aggregation like `SELECT * FROM
steps WHERE finish_reason='length'`. Every filter happens at sample granularity, and
normalising this data into a table would mean paying index/JOIN cost for queries that do
not exist. *Preserves the simplicity of one writer per model.* Switching to a long table
would require a second table, a second foreign key, and a second INSERT path; the writer
boundary would jump from "one-row upsert" to "multi-row transaction", which conflicts
with §6.5's "eliminate races via orchestration" principle. *JSON size is controllable.*
The step count per sample is bounded by `REACT_MAX_STEPS` (default 12); a single JSON is
typically < 1 KB; on v3 schema with `SAMPLING_N=3` and ~100 questions, the DB delta is
on the order of KB and WAL handles it easily.

The cost is that step-level aggregation that a long table could do has to be done by
reload + parse in Python. Since the analysis script is a one-shot tool
(`python -m forecast_eval.analysis`), this cost is acceptable.

### 6.5 One writer per model plus WAL

Concurrent writes to SQLite are a classic pitfall. The project's strategy.

*One async writer task per model DB.* Every worker's results are sent via
`asyncio.Queue` to the writer for that model. *`PRAGMA journal_mode=WAL` plus
`synchronous=NORMAL` plus `busy_timeout=5000`.* Ample throughput under
single-writer-multi-reader, with crash recovery still safe. *Batched commits.* Flush
every `DB_COMMIT_BATCH=10` entries or every 1 second.

The core idea is **eliminate races via orchestration, do not solve races with locks**.
Once we pin "one writer per DB", the concurrency problem degenerates into ordinary
single-threaded batch inserts. The alternative of lock-per-INSERT works but burns CPU
on contention with no amortisation; the orchestration approach moves the cost to enqueue
(cheap) instead of write (expensive).

### 6.6 The DB stores raw observations only

```text
DB:        raw observations only
├── correct (bool, NULL)
├── parse_ok (bool)
├── tool_calls_count
├── react_steps
├── tokens / latency
├── belief_final / belief_trace / belief_parse_ok  (v4, when BELIEF_PROTOCOL=true)
├── search_calls (with detector verdicts when leak filter is on)
└── error / created_at

analysis/: aggregations
├── pass@1 / pass_any@N / majority_vote
├── FSS / Cohen κ / Hamming / Fleiss κ
├── BI / NLL / MBS / ABI (probabilistic, when belief data exists)
├── parse_failure_rate / error_breakdown
└── per-correct cost / Pareto frontier / paired bootstrap
```

This is one of the project's most important architectural decisions and is the
operational embodiment of the paper's "metric-agnostic design" claim (paper §3.4).

### 6.7 v5 demotion of probabilistic metrics under K=5

At $K = 5$ parallel samples, the empirical probability $\hat{p} = n / K$ for each
(question, label) takes only six discrete values $\{0, 0.2, 0.4, 0.6, 0.8, 1.0\}$. This
pushes v4's Reliability Diagram, Murphy three-decomposition, and Platt-scaling LOO into
the "mathematically correct, statistically meaningless" position. v5 redirects the
analysis stack to the **discrete-native** metric family suited for $K=5$: BS / NLL /
MBS / BI / ABI are demoted to auxiliary columns with a `†` footnote and a $K$-disclaimer
in `per_model_summary.md`.

Concretely, `calibration.py` is deleted in v5, and its 5 artefacts
(`calibration_params.json`, `per_model_summary_calibrated.csv`, `reliability_data*.json`,
`brier_decomposition.csv`) are discontinued. The discrete family (FSS, Cohen $\kappa$,
Hamming, Fleiss $\kappa$, mean entropy, VCI, MVG) becomes the v5 main line, with
`entropy_accuracy_bins.csv` and `inter_trial_consistency.csv` as new v5 artefacts. If
$K$ is increased to $\ge 30$ in the future, calibration can be reintroduced in a new
change.

This decision is **not** a paper-level constraint; it is a v5 *engineering* choice
motivated by sample-size statistics. Reverting to $K = 30$ would re-enable the
probabilistic line; the demotion is by analyst convention rather than by hard-coded
gate.

---

## 7. Reproducibility and audit

### 7.1 The source database is checked into Git

`forecast_eval_set_example.db` goes straight into the repo. It is the evaluation's
gold-standard example dataset and must ship with the repo; anyone can `git clone` and
obtain the exact same questions. The filename (`SOURCE_DB`) and the internal question
table name (`SOURCE_TABLE`, default `forecast_eval_set_example`) are both exposed as
`.env` parameters; with a custom dataset, just change these two variables and the loader
splices `<SOURCE_TABLE>` into the SQL `FROM` clause at runtime. The table name is
whitelist-validated against `^[A-Za-z_][A-Za-z0-9_]*$` (config.py:L586) at the Settings
stage to foreclose SQL injection, since this is the only place we can be injected and
therefore the only place the validation needs to run.

The alternative of shipping the DB on a separate registry adds a network dependency to
reproduction; checking it into Git makes `git clone` the one and only setup step.

### 7.2 Six-part fingerprint pins down the inputs

Each run computes `source_db_hash` and writes it to `run_meta`; together with
`metadata_hash` and `prompt_templates_hash` and, when applicable,
`reflection_protocol_hash`, `belief_protocol_hash`, and `leak_detector_prompt_hash`,
this forms a multi-part fingerprint of "exactly which inputs this run is based on".
The three core hashes are computed at db.py:L385; the full set is assembled in
`evaluation.py`.

### 7.3 Three independent protocol fingerprints

`prompt_templates_hash`, `reflection_protocol_hash`, and `belief_protocol_hash` are
**three mutually independent** SHA-256 fingerprints kept side by side in `run_meta` and
the manifest. This independence is deliberate and load-bearing for ablation studies,
since the reflection-A/B paired-bootstrap analysis requires the exact-match-except-one-hash
invariant.

`prompt_templates_hash` reflects "how question content is rendered to the model": the
templates for the stem, options, instructions, question-type description, and so on.
Once a template changes, every question text changes, which makes this a coarse-grained
run-distinguishing key. Computed by `compute_prompt_templates_hash` (db.py:L397).
`reflection_protocol_hash` reflects "which meta-cognitive instruction was injected into
the model in the ReAct main loop", essentially a switch on a search-behaviour prior.
Its variation has only three axes: on/off, whether the text was modified, and version
number. `belief_protocol_hash` reflects "did the model emit a structured belief vector
before `\boxed{...}`", a switch on whether the probabilistic-family metrics are
populated.

The benefit of three separate hashes is that A/B comparisons across runs can require
"only `reflection_protocol_hash` differs, everything else equal", which is exactly what
an ablation study wants. The reflection-A/B pairing in
`analysis/behavior.py::find_paired_runs` *requires* this exact-match-except-one-hash
invariant, and unrelated runs are filtered out automatically.

The full text of each protocol coexists in `run_meta` (`reflection_protocol_text`,
`belief_protocol_text`), which enables post-hoc diffs without depending on the
`prompts.py` source code; for example, when releasing a report, the recipient receives
a redacted DB rather than the git repo.

The alternative of a single composite hash loses ablation discrimination: once any of
the three changes, everything looks "different", and we can no longer pair runs along
one axis. The alternative of a 6-way compound key including the detector hash and the
source DB hash was considered and rejected; the paper's design declares
$H_{\mathrm{aux}}$ outside the $\mathcal{R}$ tuple precisely because the detector is an
auxiliary engineering layer, and carrying it as a separate axis
(`leak_detector_prompt_hash`) preserves the "strict-equality except one axis" pairing
pattern at the ablation layer.

### 7.4 Config snapshot redaction

`run_meta.config_snapshot` stores the redacted `.env` as JSON, via `db.snapshot_settings`
(db.py:L429). Sensitive fields like `LLM_API_KEY` retain only the first 4 characters
plus length plus `sha256[:12]`; `TAVILY_API_KEY` is now `list[str]`, with each key
redacted independently and persisted as `[{prefix, sha256_12, length, provider}, ...]`.

The shape of the redaction balances two needs that would otherwise conflict. *Want to
know which parameters this run used?* Stored. *Want to know the plaintext of the key?*
Never stored. The point is that "auditable" and "non-leakable" coexist within the same
field. Pinned by `test_db.py` round-tripping through `snapshot_settings` and asserting
prefix length and digest length, never the raw value.

### 7.5 Progress logs and `messages_trace`

Every progress log line carries the question id, question_type, choice_type, model,
sample_idx, correctness, step count, tool-call count, and latency:

```text
12:03:44 | INFO | [run=20260424-120344-a7k3] [5/1610] q=69566c13 qt=binary_named ct=single model=openai/gpt-5 sample=2/5 correct=True steps=4 tool_calls=3 latency=8421ms
```

A single log line should fully describe what path a sample took through the system, so
that reading the log is reading the trace, with no DB join required.

The DB stores two large JSON blobs directly. `messages_trace` holds the complete ReAct
message sequence (LLM replies, `tool_call`, `tool_result`). `search_calls` holds each
`web_search` call's query, end_date, result count, and per-result published_date, plus,
when the leak filter is on, `n_results_raw`, `n_results_kept`, `detector_verdicts`,
`detector_latency_ms`, and `detector_error_kind`.

These are large (~80% of the DB), so the `WRITE_MESSAGES_TRACE=false` switch is
provided. But the default is on, because the value of debugging one failure far
outweighs the few extra MB of disk; without the trace, the leakage audit cannot be
redone post-hoc.

The detector audit fields go into `search_calls`, never into `messages_trace`, pinned by
`leak_filter.py:L25` and asserted by `test_leak_filter.py`. This is intentional: the
detector verdict is *audit metadata*, not LLM-visible content; if it were in
`messages_trace`, an unsuspecting downstream model could see it and bias its behaviour.

`loguru` writes to two channels: stderr for humans and a rotating file for machines,
with rotation 100 MB and retention 5. Humans and machines have different needs when
reading logs, so we serve them separately.

---

## 8. Error handling

### 8.1 Why some errors should not be retried

| Error                       | Retry?                       | Reason                                                                         |
| --------------------------- | ---------------------------- | ------------------------------------------------------------------------------ |
| Network / 5xx               | Yes, per backoff sequence    | Mostly transient                                                               |
| Rate limit                  | Yes, prefer Retry-After      | The provider has told you how long to wait                                     |
| Auth 401/403                | **Stop the entire run**      | The key is wrong; retrying is pointless and stopping early saves money         |
| Bad request                 | No                           | Things like `model_not_found` only run after a config change                   |
| Content policy              | No                           | The same prompt sent again returns the same result                             |
| Refusal / parse fail        | No                           | Not an error, but model behaviour                                              |
| Tavily itself               | Has its own retry sequence   | Once exhausted, return the error to the LLM                                    |
| Training-cutoff filter      | Not invoked                  | Write `skipped_training_cutoff` directly                                       |

### 8.2 Three independent backoff sequences

```bash
LLM_BACKOFF_NETWORK_S=2,5,15,30,60         # config.py:L236
LLM_BACKOFF_RATE_LIMIT_S=10,30,60,120,300   # config.py:L237
LLM_BACKOFF_SERVER_5XX_S=5,15,30,60,120     # config.py:L238
```

Different error types use different backoffs, since rate limit is much slower than
network: the former typically needs minute-level cooldowns while the latter usually
clears in a few seconds. The sequence length also determines the maximum retry count;
configuration is unified in `.env`.

The three sequences are tuned for OpenRouter's behaviour patterns: network errors clear
in seconds (e.g. transient TCP resets), rate limits clear in minutes (provider-side
cool-down), and 5xx surges clear in low minutes (provider-side recovery). Different
providers may need different sequences, which is why each is a separate `.env` knob
rather than a single `LLM_BACKOFF_S`.

### 8.3 Error classification codes are first-class report citizens

The `error` field is not "fill in a string when something errors" but a fixed finite
enum: `network`, `server_5xx`, `bad_request`, `content_policy`, `skipped_training_cutoff`
(see `errors.ErrorKind`).

`error_breakdown.csv` slices directly by this classification. The principle is **every
failure behaviour must be categorisable and aggregatable in the report**: an
`error="something went wrong"` is useless.

### 8.4 v5.1 classification expansion

Two common misclassifications surfaced during cross-provider evaluation, each driving a
specific v5.1 expansion.

*Aliyun content moderation (`data_inspection_failed`) was mis-bucketed as `bad_request`.*
v5.0's `_body_matches` only recognised English needles like `content_policy /
content_filter / safety`; the `code=data_inspection_failed` returned by DashScope
(`https://dashscope.aliyuncs.com`) fell through to the catch-all `bad_request`. v5.1
unified the needle list under `errors.CONTENT_POLICY_NEEDLES`, adding
`data_inspection_failed`, `inappropriate content`, and `sensitive`; on match, the error
is classified as `content_policy`, which preserves the "MUST NOT retry" semantics.

*Remote disconnect `RemoteProtocolError` was mis-bucketed as `unknown`.* v5.0's network
exception tuple only listed `ConnectError`, `ReadTimeout`, `ConnectTimeout`, and
`WriteTimeout`; `httpx.RemoteProtocolError` ("Server disconnected without sending a
response.") fell into `UNKNOWN`, and the entire sample failed without retry. v5.1
expanded the network exception family to align with httpx's existing `NetworkError`
subset by adding `RemoteProtocolError`, `WriteError`, and `PoolTimeout`, with parallel
expansion on the LLM side (`errors.classify`) and the Tavily side
(`search._single_request`).

The principle is that **a misclassified error is silently miscounted in the report**.
The v5.1 expansion was driven by cross-provider observations where these two patterns
surfaced once each per ~2K samples, which is small numerically but fatal to honest
reporting, because they would otherwise tip the `bad_request` vs `content_policy` ratio.

---

## 9. Configuration as contract

### 9.1 Almost every tunable lives in `.env`

The CLI exposes only three flags: `--question-type`, `--choice-type`, and
`--skip-analysis`; everything else goes through `.env`. Three reasons.

*Easy to re-run.* A single `.env` is enough to reproduce the entire configuration; CLI
flags scattered in shell history are easily lost. *CI- and scheduler-friendly.* Scripted
execution generally prefers managing a file rather than a command line. *Config / DB
self-consistency.* `config_snapshot` is written into `run_meta`, so reviewing a run
later tells you what its `.env` looked like at the time, after redaction.

### 9.2 OpenAI-compatible endpoint as integration surface

`LLM_BASE_URL` accepts any OpenAI-compatible endpoint: OpenRouter, Aliyun Bailian,
OpenAI, DeepSeek, SiliconFlow, and local vLLM all work. The integration surface is
deliberately small and standard. OpenAI's chat completion plus function calling protocol
has become the de facto standard, so this project does not build a provider-adaptation
layer but pushes adaptation responsibility to the endpoint. The paper's six-model
comparison spans across providers (DeepSeek, Z.ai, Alibaba, MiniMax, Moonshot,
ByteDance) precisely because of this neutrality.

### 9.3 Training-cutoff config is quality config

`MODEL_TRAINING_CUTOFFS=openai/gpt-5=2024-10-01,...` (config.py:L224) is not optional;
it is **part of evaluation fairness**. The docs explicitly recommend declaring an
explicit cutoff for every model under evaluation; an unspecified model is not filtered
(with a warning). The paper takes the most conservative interpretation: when a model
card discloses cutoff only at month-level granularity, adopt the *last day* of the
disclosed month as $\kappa_M$ (paper §4.1.2 footnote).

### 9.4 Startup validation as enforcement

`Settings._post_validate` runs at process start and aborts before any LLM/Tavily call
if any of the following fail.

* `:online` slug or `::` substring in `MODELS` or `LEAK_DETECTOR_MODEL` (config.py:L599);
* `MODEL_TRAINING_CUTOFFS` parse failures (config.py:L181);
* `REACT_MAX_SEARCH_CALLS` empty list (config.py:L580);
* `SOURCE_TABLE` failing the `^[A-Za-z_][A-Za-z0-9_]*$` whitelist (config.py:L586);
* `TAVILY_API_KEY` empty when `ENABLE_WEB_SEARCH=True`;
* `LEAK_DETECTOR_API_KEY` or `LEAK_DETECTOR_MODEL` empty when
  `ENABLE_SEARCH_LEAK_FILTER=True`;
* `COMPOSITE_WEIGHTS_QTYPE` or `COMPOSITE_WEIGHTS_CTYPE` containing unknown bucket names
  or all-zero weights (config.py:L515);
* `GRID_DEFAULT_R` or `GRID_DEFAULT_C` not in their respective lists.

The full set is enumerated in FRAME §7.1. The principle is **fail fast, before any
billable call**: an evaluation that would have been silently miscounted is much more
expensive than an evaluation that did not start. Pinned by `test_config.py`, where each
rule has at least one fixture exercising both the accept and reject path.

---

## 10. Testing as sentinel

### 10.1 Tests must not hit the network or burn the API

A complete run of the example dataset times the number of models times $N$ samples is
**tens to hundreds of dollars**. Tripping over a prompt, parser, or schema bug at that
scale just wastes the money. The core constraints of the test design:

* `tavily-python` must not actually send requests; `respx` mocks `httpx`.
* The OpenAI client must not actually send requests; fixture replacement.
* SQLite uses a temporary directory; `tmp_path` fixture.
* The dataset must be small yet "look real"; we use a few real questions from the source
  DB as fixtures.

### 10.2 Five CI red lines

```text
test_prompts / test_parser / test_training_cutoff /
test_llm_no_browsing / test_analysis
```

These five must always be green. They cover the parts of the project most likely to
silently break, and they are precisely the components that realise the framework
$\mathcal{R}$.

| Test                    | Invariant guarded                                              | Framework component       | If it breaks                                                                  |
| ----------------------- | -------------------------------------------------------------- | ------------------------- | ----------------------------------------------------------------------------- |
| `test_prompts`          | prompt template rendering correct for all three question_types | $R$                       | The `user_prompt` text drifts; `prompt_templates_hash` no longer pins inputs |
| `test_parser`           | letter parsing and strict-equality scoring                     | $\Psi$, $\phi$            | Items mark "wrong" while letters actually match, or vice versa                |
| `test_training_cutoff`  | training-cutoff filtering semantics and resume priority        | $\kappa_M$ admissibility  | Cutoff-skipped questions get billed, or completed rows get rebilled           |
| `test_llm_no_browsing`  | provider-native browsing is never silently turned on           | information barrier       | The whole evaluation contract becomes invalid                                |
| `test_analysis`         | report numbers reconcile with the raw DB                       | $\Gamma$                  | The CSV / MD numbers diverge from what the raw observations support          |

The principle is **pick the invariants whose breakage would be expensive, and use unit
tests as sentinels**. Each red line corresponds to one component of $\mathcal{R}$ that,
if broken, makes the entire run unit invalid. The "if it breaks" column is deliberately
blunt; this is the cost of skipping the test, not the cost of running it. The full
mapping (33 tests to 11 framework components) is in FRAME §15.1.

### 10.3 Dry-run smoke test

`test_smoke_dry_run.py` replaces OpenRouter and Tavily with `httpx` stubs and runs an
end-to-end pipeline of 3 questions × 1 model × 1 sample. It does not validate logic
details. It validates "is the pipe still flowing", checking that the schema, the wide
table, the `messages_trace` JSON, and the `search_calls` fields are all present. This expresses the e2e/unit test split:
unit tests validate local correctness, smoke tests validate that integration does not
blow up.

### 10.4 Tests as documentation

Every paper-level invariant has a test. When the prose in paper, FRAME, or DESIGN drifts
from the code, the test is the tie-breaker. The 33 test files in `tests/` (~13K LOC, all
offline) are the most authoritative form of "what this codebase actually does"; if you
cannot reconcile what you read here with what a test asserts, the test wins.

---

## 11. Evolution: the openspec change archive

The repo root contains `openspec/changes/`, where changes are recorded in spec form.
`bootstrap-forecast-eval` is the initial bootstrap record. The subsequent landmark
changes are `react-tavily-grid-search`, `harness-resilience-v1`,
`search-leak-filter-v1`, `add-exam-score-metric`, `composite-score-by-subtype`, and
`discrete-native-analysis-v5`. Each ships with a `proposal.md` (motivation), a
`design.md` (decision archive), a `specs/.../spec.md` (capability deltas), and a
`tasks.md` (implementation checklist).

Two principles drive this. **Write the spec before the code**, in order to avoid
discovering the design is wrong only after the code is merged. **Change archive and code
diff coexist**, so that when reviewing the architectural evolution later, you can see
*why* we changed it, not just *what* was changed.

### 11.1 Grid search via virtual slug

`react-tavily-grid-search` extends the $(Q \times M \times N)$ three-axis space to $(Q
\times M \times R \times C \times N)$, but **without** a schema upgrade and **without**
touching the runner core loop. The method: at the evaluation entrypoint, encode each
$(\text{real\_model}, R, C)$ triple as a **virtual model slug** `{real}::r{R}::c{C}`
(`db.compose_virtual_slug` at db.py:L477; reverse via `parse_virtual_slug` at
db.py:L500); the runner, DB, and analysis main pipeline treat it as an opaque string.
Existing artefacts naturally expand into multiple rows by virtual slug, while the new
module `forecast_eval/analysis/grid.py` decodes the triple, re-aggregates, and emits
paper long tables and figures. Full decision archive in
`openspec/changes/react-tavily-grid-search/design.md`; the 10 key decisions:

| ID  | Decision                                                                                     |
| --- | -------------------------------------------------------------------------------------------- |
| D1  | Pick option C (virtual slug + per-task settings view); reject A (single run, multi-(R, C) DB schema v5 rewrite) and B (one run_dir per cell, with `runs/` bloat and complex cross-run aggregation) |
| D2  | Virtual slug uses the `::r{R}::c{C}` suffix; `db.model_slug_safe` replaces `::` with `_` to land an fs-safe filename `openai__gpt-5__r5__c3.db`; the regex `^(?P<real>.+?)::r(?P<R>\d+)::c(?P<C>\d+)$` non-greedy captures real_model |
| D3  | `runner.Task` carries a cell-local `settings: Settings`; the dispatcher derives an immutable sub-view via `model_copy(update={...})`; `react.py` and `search.py` are byte-unchanged |
| D4  | Only raise when `REACT_MIN_SEARCH_CALLS > min(C_list)`; for a cell with `C < MIN`, silent clamp `effective_min = min(MIN, C)` and record it under `run_meta.config_snapshot.grid_origin` for audit |
| D5  | `run_meta.config_snapshot` writes **single-valued** R/C; add a `grid_origin = {real_model, R, C, effective_min_search_calls}` sub-key; manifest top-level adds a `grid` block (`r_list / c_list / default_r / default_c / real_models / n_cells`) so the analysis layer does not have to decode the triple per .db |
| D6  | `manifest.models` and `manifest.model_files` field semantics remain "list of virtual slugs"; the new `grid.real_models` is a deduped real-slug convenience field, so v4 analysis main path's contract of "read `manifest.models` as the db file list" is preserved |
| D7  | `analysis/__init__.py::run_analysis` main path is **zero-intrusive**; append a `grid.run_grid_analysis(...)` at the end wrapped in `try/except` (same best-effort pattern as reflection A/B); failures do not interrupt the existing pipeline |
| D8  | Grid CIs all go through `inference.paired_bootstrap` (5000 resamples, seed=42); BI-domain CIs are obtained via "BS-domain paired bootstrap + monotone transform $\mathrm{BI}=100(1-\sqrt{\mathrm{BS}})$"; **no** new statistical code introduced |
| D9  | Pareto frontier's cost dimension defaults to `mean_search_calls` (actual mean search count, more honest than the C ceiling), with `mean_latency_ms / C` fallback allowed; y-axis defaults to `bi_mean`, with `nll_mean` (minimisation direction) as an option |
| D10 | Fig 1 main figure pins $R = \texttt{GRID\_DEFAULT\_R}$ with one curve per real_model; other R values each get a same-format appendix figure to avoid main-figure unreadability after stacking $M \cdot |R|$ curves |

The three PRs Phase 0 / 1 / 2 ship sequentially; each phase passes `pytest -q` and
`openspec validate --strict`, and after deleting the phase's own code the system is
equivalent to the previous phase's completed state (the rollback strategy). Single-value
`.env` parses under the new code as a length-1 list, so a Cartesian product produces a
single virtual slug, with the only visible difference being the `__r{R}__c{C}` suffix on
the .db filename. For legacy v4 runs (manifest without a `grid` block), grid analysis
and the grid figure family early-exit altogether, with zero intrusion.

### 11.2 Phase-gated rollouts

Each openspec change ships as a sequence of phases (typically `Phase 0 schema → Phase 1
code → Phase 2 docs/CSV columns`). The discipline:

* **Each phase passes CI on its own.** No "one big PR with everything"; CI catches the
  schema/code mismatch on the wrong side of each gate.
* **Each phase is reversible.** Deleting the phase's own code returns the system to the
  previous phase, which is checked by re-running tests after a hypothetical revert.
* **The change archive accumulates.** `openspec/changes/archive/` is the project's own
  ledger of "why we got here"; it is more durable than the git log because each entry
  has a spec / design / tasks separation that survives squash-merge.

---

## 12. Knowing which knob you are turning

A repeating confusion in cross-team conversations: which knobs are "part of the
evaluation contract" and which are "engineering tuning that does not affect the
scientific claim". The distinction matters because changing a contract-knob invalidates
cross-run comparability, while changing an engineering-knob does not.

### 12.1 Contract knobs (changing these makes runs incomparable)

Every entry below corresponds to a fingerprint or a $\mathcal{R}$-tuple field; all are
written to `run_meta` and any change shows up as a hash mismatch on a paired comparison.

| Knob                              | Why it is a contract                                                  |
| --------------------------------- | --------------------------------------------------------------------- |
| `SOURCE_DB` + `SOURCE_TABLE`      | Defines $\mathcal{D}$; fingerprint via `source_db_hash`               |
| `MODEL_TRAINING_CUTOFFS`          | Defines $\kappa_M$; per-model admissibility                           |
| `TAVILY_END_DATE_OFFSET_DAYS`     | Defines $\delta$, hence $\chi_i$                                      |
| `REACT_MAX_STEPS`                 | Defines $T$                                                           |
| `REACT_MAX_SEARCH_CALLS`          | Defines $C$ (grid axis)                                               |
| `TAVILY_MAX_RESULTS`              | Defines $R_{\mathrm{tav}}$ (grid axis)                                |
| Prompt templates (8 keys)         | Defines $R$; fingerprint via `prompt_templates_hash`                  |
| Reflection protocol text + on/off | Defines part of $F_M$; fingerprint via `reflection_protocol_hash`     |
| Belief protocol text + on/off     | Defines part of $F_M$; fingerprint via `belief_protocol_hash`         |
| Detector prompt + version         | Defines $H_{\mathrm{aux}}$; fingerprint via `leak_detector_prompt_hash` |
| Composite weights                 | Defines $\Gamma$ (default subtype-weighted form); recorded in `run_meta` |
| `SAMPLING_N`                      | Defines $S$; pinned at table-creation time                            |

### 12.2 Engineering knobs (safe to change within one comparison)

Every entry below is purely about throughput, cost, or robustness; none affects
$\mathcal{R}$.

| Knob                                     | Why it is engineering                                                |
| ---------------------------------------- | -------------------------------------------------------------------- |
| `LLM_MAX_CONCURRENCY`                    | Throughput; numerically irrelevant to outcomes                       |
| `LLM_BACKOFF_*`                          | Resilience under provider-side noise; a long-enough sequence converges |
| `SEARCH_RETRY_MAX` / `_BACKOFF_S`        | Same as above for Tavily                                             |
| `LEAK_DETECTOR_RETRY_MAX` / `_BACKOFF_S` | Same as above for the detector                                       |
| `LEAK_DETECTOR_CONCURRENCY`              | Throughput on the detector stage                                     |
| `DB_COMMIT_BATCH`                        | Disk-write batching; numerically irrelevant                          |
| `WRITE_MESSAGES_TRACE`                   | Disk-size knob; affects post-hoc audit, not numerics                 |
| `LOG_LEVEL` / `LOG_DIR`                  | Logging verbosity / location                                         |
| `RUNS_ROOT`                              | Where artefacts land; does not change what they are                  |

The contract / engineering split makes reviewing a PR or a `.env` change mechanical:
any change in §12.1 is a *new evaluation*, any change in §12.2 is a *bug fix or tuning*.
The runtime fingerprint set (§7.2) makes this distinction observable on any pair of
runs.

---

## Closing principles

Condensing the full document's design philosophy into a single set of principles, for
review reference:

1. **Boundary at the data layer, not the prompt layer.** Sample admission ($\kappa_M$),
   temporal masking ($\delta$, $\chi_i$), and the detector ($H_{\mathrm{aux}}$) all live
   in evaluator-controlled code, never in the model's instructions.
2. **Honesty over prettiness.** The threat model declares what we cannot control in
   plain language; the leakage audit publishes its Wilson upper bound rather than just
   the point estimate.
3. **Skip is not fail.** Actively excluded samples are categorised independently and do
   not pollute the error rate.
4. **Raw over aggregated.** The DB stores observations only; statistics are deferred to
   `analysis/`. Metric definitions evolve faster than DB schemas.
5. **Strict by default, partial credit by design.** The headline metric is strict
   frozenset equality; the composite uses exam-style partial credit; FSS adds chance
   correction. All three coexist on the same raw samples.
6. **Reproducibility over convenience.** Source data goes into Git; each DB is
   self-contained; six-part hashes pin down the fingerprint.
7. **Observability over elegance.** Full `messages_trace` is on by default; the
   progress log is one line per sample; per-call audit fields persist detector verdicts.
8. **Categorise failures.** Errors use a finite enum; every kind has its own cell in
   the report.
9. **Config as contract.** `.env` alone decides everything; CLI flags are minimal;
   `config_snapshot` is redacted before persistence; the contract / engineering split
   (§12) tells you which `.env` change invalidates a comparison.
10. **Tests guard the expensive.** Five CI red lines map one-to-one to components of
    the run unit $\mathcal{R}$; dry-run smoke tests validate integration; expensive
    failures are shifted to local.
11. **The framework is the contract.** Every engineering decision is judged by whether
    it strengthens or weakens $\mathcal{R}$; convenience wins over the framework only
    in clearly bounded escape hatches (`LEAK_DETECTOR_FAIL_ACTION=keep`,
    `WRITE_MESSAGES_TRACE=false`, A/B switches), which themselves leave audit trails.
12. **Fail fast, before billable calls.** Startup validation rejects misconfigurations
    before any LLM/Tavily contact; an evaluation that did not start is cheaper than one
    silently miscounted.
13. **Eliminate races via orchestration, not locks.** One writer per DB, one queue per
    writer, batched commits.
14. **Phase-gated changes.** Every openspec change ships in reversible phases; the
    change archive is the project's ledger of "why we got here".
15. **Three independent fingerprints, not one.** Templates, reflection protocol, and
    belief protocol have separate hashes so ablation studies can pair runs along one
    axis.

---

## Appendix A. Rejected alternatives index

A consolidated list of every alternative considered and rejected in this codebase,
sorted by decision area. Each row references the section where the rationale appears in
detail.

| Decision area                | Rejected alternative                                       | Reason for rejection                                          | §       |
| ---------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------- | ------- |
| Tool-mediated boundary       | Expose `end_date` as an LLM tool argument                  | Trusts the model not to widen $\chi_i$; pin-test surface explodes | 2.2     |
| Tool-mediated boundary       | Rewrite query string to insert date filter                 | Brittle string-pass; providers can ignore inline date operators | 2.2     |
| $\delta$ default             | $\delta = 0$ (use resolution day)                          | Catches 30–50% of same-day news leakage on the example DB; too lax | 2.3     |
| $\delta$ default             | Per-question $\delta_i$                                    | Breaks "one $\delta$ defines one evaluation" contract           | 2.3     |
| Provider-native browsing     | Warn instead of refuse                                     | Warnings get filtered; refusals stop the run                   | 2.4     |
| Provider-native browsing     | Only enforce at startup                                    | Bypassable via `model_copy(update={...})`; per-call re-check needed | 2.4 |
| Cutoff filter                | Weighted exclusion (discount close-to-cutoff samples)      | Adds analytic complexity; binary in/out cleaner                 | 2.5     |
| Cutoff filter                | Skip dataset-wide if any model fails admissibility         | Discards 10–20% of corpus on heterogeneous panel                | 2.5     |
| Discrete answer space        | Open-ended NL outputs scored by LLM judge                  | Re-introduces contamination risk; non-deterministic scoring     | 3.1     |
| Discrete answer space        | Numerical probability outputs scored by Brier/NLL only     | At $K=5$ the empirical $\hat p$ is too discrete (see §6.7)     | 3.1     |
| Strict scoring               | Jaccard / F1 / partial credit at the strict level          | Weakens the headline number; soft companions added at composite layer instead | 3.3 |
| > 26-option labelling        | Skip > 26-option questions                                 | Loses dataset coverage; round-trip test mitigates              | 3.4     |
| Parse-failure handling       | Retry on parse failure                                     | Capability masking + cost; same answer expected on retry        | 3.5     |
| System message               | Split system / user                                        | Loses cross-provider consistency                                | 5.1     |
| Reflection                   | Hard-floor `REACT_MIN_SEARCH_CALLS` by default              | Conflates capability with enforcement                           | 5.3     |
| Harness resilience           | Single "harness rescue" switch                             | Loses ablation discrimination across the four orthogonal modes  | 5.4     |
| DB layout                    | One DB with `model` column                                 | Single-writer contention; one slow provider stalls all          | 6.1     |
| DB layout                    | Store paths instead of copies                              | Breaks self-contained property                                  | 6.2     |
| DB layout                    | Long table + composite key                                 | Loses simple resume queries; multi-row writes break orchestration | 6.3   |
| Step metrics                 | Long table for per-step rows                               | Pays index/JOIN cost for queries that do not exist              | 6.4     |
| Concurrency                  | Lock-per-INSERT                                            | Burns CPU on contention; orchestration cheaper                  | 6.5     |
| Reproducibility              | Ship DB on a separate registry                             | Adds network dependency to reproduction                          | 7.1     |
| Reproducibility              | Single composite hash                                      | Loses ablation discrimination on {template, reflection, belief} | 7.3     |
| Composite weights            | Equal weights across buckets                               | Erases discriminative buckets; doubles up near-random ones      | 4.4     |
| Composite weights            | Empirical-prevalence weights                               | Drowns out signal-bearing buckets                                | 4.4     |

---

## Appendix B. Reading roadmap

If you are new to the project, we suggest reading in this order. The order is
engineered: each step gives you the language for the next. Steps 1–4 build the
conceptual model; steps 5–8 ground it in code; steps 9–10 give you the change history
and the test contract.

1. `README.md`: figure out in 10 minutes what OracleProto is and how to run it.
2. This document (`DESIGN.md`): understand the motivation behind each trade-off.
3. `paper/main.tex` §§1–3: read the formal framework, and sit with the tuple
   $\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)$ until
   each symbol has a clear name in your head.
4. `FRAME.md`: the complete spec at field, interface, and pseudocode level. Use the
   §1.1 grand map as your symbol-to-code lookup table whenever DESIGN says "see
   file:line".
5. `forecast_eval/prompts.py` and `forecast_eval/parser.py`: the renderer $R$ and the
   parser $\Psi$; these two files are practically the heart of the project.
6. `forecast_eval/runner.py` and `forecast_eval/react.py`: orchestration and the ReAct
   loop. Pay special attention to react.py:L266, where the four-knob priority chain
   lives.
7. `forecast_eval/leak_filter.py` and `forecast_eval/search.py`: the temporal masking
   implementation and the Stage-2 detector. The prompt template at leak_filter.py:L55
   is what enforces the "no question fields" whitelist.
8. `forecast_eval/analysis/`: the metric layer. Start with `exam_score.py`, which is
   single-file, self-contained, and reads in five minutes; then `accuracy.py` for
   Tversky, FSS, and Cohen $\kappa$; then `composite.py` for subtype-weighted
   aggregation; then `consistency.py` for Fleiss $\kappa$, VCI, and MVG.
9. `tests/`: read tests to reverse-engineer the contracts. The five CI red lines
   (§10.2) are the highest-priority entry points.
10. `openspec/changes/archive/`: to find out *why* things became what they are today,
    come here. Each change has a `proposal.md` (motivation), a `design.md` (decision
    archive), a `specs/.../spec.md` (capability deltas), and a `tasks.md`
    (implementation checklist).

---

> **Summed up in one sentence:** OracleProto uses engineering discipline to safeguard
> scientific rigour. Every seemingly excessive constraint exists because the alternative
> is a number in the final report that does not actually mean anything. The information
> boundary is part of the data, not the prompt; the run unit $\mathcal{R}$ is the
> contract, not a configuration; the audit trail is the report's foundation, not its
> appendix.
