# OracleProto — Design Rationale

> This document explains *why* the codebase looks the way it does. For *how exactly* the
> Python modules implement each piece — field names, schemas, function signatures, error
> codes, line numbers — pair this with `FRAME.md`. The paper at `paper/main.tex` provides the
> formal framework; this document maps every formal object to a concrete engineering
> trade-off and records the constraints behind each "seemingly excessive" decision.
>
> The reading order is: **paper §§1–3 (formalism) → DESIGN (rationale) → FRAME (mechanics) →
> source code**. Skipping the rationale layer is the single fastest way to misread an
> engineering choice as either over-engineering or laziness — almost every "weird" decision
> in this codebase is the unique fixed point of two paper-level constraints meeting one
> empirical observation.

---

## 0. Foreword: the question OracleProto answers

> **If a model is never allowed to look at information published after an event has been
> resolved, how strong is its native forecasting capability across a leakage-controlled
> dataset?**

That is the single question OracleProto exists to answer. The paper's introduction
(`paper/main.tex` §1, §2.1, §2.3) articulates why the question is non-trivial: existing
evaluation practice sits on an unstable middle ground.

* **Prospective live evaluation** (ForecastBench, FutureX) admits only events whose answers
  do not yet exist when the forecast is submitted. This is the gold standard for
  contamination control, but the leaderboard is a one-way temporal stream — once a question
  resolves it is removed; the resulting evaluation is impermanent and not reusable.
* **Retrospective evaluation** (FutureX-Past, archived live questions) is auditable and
  comparable, but is highly prone to mistaking *factual recall* for *forecasting capability*.
  The dataset card of FutureX-Past itself warns that historical outcomes may already have
  entered newer models' training data, so the subset must not be used as an ordinary
  live-prediction benchmark.

The diagnostic literature surveyed in paper §2.3 has empirically shown that **simulated
ignorance** ("imagine you do not know the answer") and **true ignorance** ("you never knew")
are systematically different — reasoning-optimised models are particularly bad at the
simulation, and a 1–5% label-noise rate alone is enough to break proper scoring rules
(Paleka et al. 2025; Li et al. 2026). BLF (Murphy 2026) reaches the same conclusion from the
inference side: a single-inference defence does not generalise across runs; the discipline
must live one level deeper, *inside the dataset itself*.

OracleProto's response is to push the discipline into the dataset schema and the run unit
$\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)$ (paper §3.4).
Three almost "religious" hard constraints fall out immediately, and **every other decision
in this document is downstream of these three**:

1. **The information boundary must be enforced at the dataset/tool layer, not the prompt
   layer.** Sample admission ($\kappa_M \le \chi_i < \tau_i$ — paper Eq. 4) is an *upstream
   filter*, not instructions to the model. Tool-level temporal masking is *injected by the
   tool implementation*, not a parameter the model can fill in. The model can propose
   queries; it cannot alter the cutoff.
2. **Results must be byte-reproducible.** Same dataset + same configuration + same model →
   anyone can `git clone` and re-run to obtain comparable numbers. Source DB checked into
   git; `(source_db_hash, metadata_hash, prompt_templates_hash, reflection_protocol_hash,
   belief_protocol_hash, leak_detector_prompt_hash)` form the run's six-part fingerprint.
3. **Every leakage path that we *can* control is controlled; every path we *cannot* is
   declared.** The threat model (paper §3.5) is honest about what we cannot fix.

Once you internalise these three constraints, every "seemingly over-strict" choice in the
following sections looks natural. This document walks each constraint through the
engineering that realises it.

### How to read this document

Every major decision below follows the same five-line shape, even when not formatted as such:

1. **What** the decision is.
2. **Why** — which paper-level constraint or which empirical observation forces it.
3. **Rejected alternatives** — what we considered and why we did *not* pick it.
4. **Where** — code path that enforces it (file:line) and the test that pins it.
5. **What would break it** — a change in some other module that would silently invalidate
   this decision.

Sections that omit any of these lines do so because the corresponding slot is genuinely
empty (e.g. there is no rejected alternative worth recording).

---

## 1. The OracleProto framework: from formalism to code

The paper formalises a forecasting evaluation as a run unit
$\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)$ (paper §3.4,
Eq. 4-7). Every component maps to exactly one object in the codebase, and once those
objects are fixed, the entire pipeline — sample admission, input construction, tool
masking, output parsing, metric aggregation — has a single audit-replay path.

| Symbol             | Object                              | Implementation                                                                 | Pinning test                |
| ------------------ | ----------------------------------- | ------------------------------------------------------------------------------ | --------------------------- |
| $\mathcal{D}$      | Discrete forecasting dataset        | `SOURCE_DB` / `SOURCE_TABLE`; `loader.sync_questions`                          | `test_db.py`                |
| $M$                | Evaluated model                     | one entry of `MODELS`; one SQLite file per $M$ under `runs/{run_id}/db/`        | `test_runner_grid_model.py` |
| $\kappa_M$         | Knowledge cutoff                    | `MODEL_TRAINING_CUTOFFS[M]` (config.py:L224); `runner.build_task_plan`         | `test_training_cutoff.py`   |
| $\delta$           | Temporal masking offset             | `TAVILY_END_DATE_OFFSET_DAYS` (default `-1`, config.py:L273); `search.tavily_search` | `test_search.py`      |
| $T$                | Max ReAct steps                     | `REACT_MAX_STEPS` (default `12`, config.py:L279); `react.run_react` loop bound | `test_react.py`             |
| $C$                | Max search calls                    | `REACT_MAX_SEARCH_CALLS` (default `[8]`, config.py:L283); `react.py` budget gate | `test_react.py`           |
| $R$                | Input renderer                      | `prompts.render_user_prompt`                                                   | `test_prompts.py`           |
| $\Psi$             | Output parser & validity            | `parser.parse_answer` (parser.py:L40)                                          | `test_parser.py`            |
| $\phi$             | Answer normalization map            | letter encoding `A` / `A,B` (per question_type); `parser.parse_gt` (parser.py:L92) | `test_parser.py`         |
| $\Gamma$           | Aggregation rule                    | `analysis/*` (composite accuracy, FSS, κ, BI, …)                               | `test_analysis.py`          |
| $H_{\mathrm{aux}}$ | Auxiliary leakage detector          | `leak_filter.filter_search_result`; logged in `run_meta.config_snapshot` rather than $\mathcal{R}$ tuple | `test_leak_filter.py` |

The paper deliberately keeps $H_{\mathrm{aux}}$ outside the formal tuple (paper §3.4) and
binds it via SHA-256 fingerprint to run metadata, because the detector is a *replaceable
empirical engineering layer* that supports the boundary, not a primitive component of the
forecasting system itself. The codebase mirrors this distinction: `MODELS` /
`MODEL_TRAINING_CUTOFFS` / `REACT_MAX_*` enter `run_meta` directly, while
`LEAK_DETECTOR_*` enters via `run_meta.config_snapshot.detector_*` plus
`run_meta.leak_detector_prompt_hash`.

The information visible to model $M$ on question $q_i$ is (paper Eq. 8):

$$\mathcal{I}_{i,M}^{\mathrm{vis}} = \mathcal{K}^{M}_{\le\kappa_M} \cup \mathcal{T}_{\le\chi_i},$$

where $\mathcal{K}^{M}_{\le\kappa_M}$ is parametric knowledge before the model's training
cutoff and $\mathcal{T}_{\le\chi_i}$ is temporally masked external information. The
forecasting system $F_M$ produces

$$\widehat{Y}_{i,M} = F_M(q_i^{\mathrm{in}}; \mathcal{I}_{i,M}^{\mathrm{vis}}), \quad \widehat{Y}_{i,M} \subseteq \mathcal{A}_i.$$

Everything in this codebase is an enforcer of this equation: the LLM is asked to choose
from a finite candidate set $\mathcal{A}_i$ under bounded information, and every
engineering decision is judged by whether it strengthens or weakens that boundary.

### 1.1 The "framework as contract" mental model

There is one mental model that ties every section of this document together: **$\mathcal{R}$
is a contract, not configuration**. A run is "the same evaluation as another run" iff every
field of $\mathcal{R}$ matches and every fingerprint matches; otherwise it is — strictly —
a *different* evaluation, not a noisier estimate of the same one.

Concretely:

* Two runs with different $\delta$ are **not** comparable, because the admissibility frontier
  $\chi_i$ differs and the visible retrieval set $\mathcal{T}_{\le\chi_i}$ is different.
* Two runs with different $\kappa_M$ for the same model slug are **not** comparable, because
  the admissible question subset $\mathcal{D}^{\mathrm{pred}}_M = \{q_i : \kappa_M \le \chi_i
  < \tau_i\}$ differs.
* Two runs with different `prompt_templates_hash` are **not** comparable as evaluations of
  the *same* renderer $R$; they evaluate two different objects whose only shared label is
  the model under test.

Every fingerprint, audit field, and config snapshot in this codebase exists to make those
"different evaluations" observable, so reports cannot accidentally be averaged across
inequivalent contracts.

### 1.2 What the framework leaves underspecified — and why

Several engineering choices are *not* mandated by $\mathcal{R}$ and could legitimately be
different in an alternative implementation:

| Engineering choice               | Mandated by $\mathcal{R}$? | What's left to the implementor                                |
| -------------------------------- | -------------------------- | ------------------------------------------------------------- |
| Storage backend                  | No                         | We chose SQLite (one file per model); a row store like Postgres would work |
| Retrieval backend                | No                         | We chose Tavily; any time-filterable retrieval is permitted (paper §3.3) |
| Detector model                   | No                         | We default to `Qwen3.5-Flash` for the audit; any sufficiently strict model works |
| Concurrency model                | No                         | We chose `asyncio` + per-model writer task; threads / processes also work |
| Backoff sequences                | No                         | Three sequences in `LLM_BACKOFF_*`; values are tuned for OpenRouter, not formal |
| Logging stack                    | No                         | We chose `loguru`; any structured logger works                  |

Decisions in this column would change *which* implementation it is, but not *which
evaluation* it is. The fingerprint of $R$ ($\Psi$, $\phi$, $\Gamma$) covers the
evaluation; the fingerprint of the detector covers the auxiliary leakage barrier; everything
else is engineering. We document this distinction explicitly because it tells you exactly
which knobs are safe to swap when porting OracleProto to a different stack.

---

## 2. Information boundary: the project's first principle

Paper §3 organises the boundary along three controlled channels (paper §3.5: "Controlled
information channels") plus an uncontrollable residual:

1. **Parametric knowledge** ($\mathcal{K}^M_{\le \kappa_M}$): controlled via sample
   admission $\kappa_M \le \chi_i$ (paper Eq. 4).
2. **Tool-mediated knowledge**: controlled via $\chi_i = \tau_i + \delta$ injected at
   tool implementation (paper Eq. 7, §3.3.2).
3. **Retrieval-result content**: controlled via auxiliary detector $H_{\mathrm{aux}}$
   (paper §3.3.3 — the third channel).
4. **Provider-side residual**: provider-native browsing (`:online`, `plugins`) — declared
   uncontrollable for any provider that exposes it, and forbidden for any provider that
   doesn't.

The codebase organises §§2.1–2.6 below to mirror these four channels in order from
"strictly enforceable" to "honestly declared".

### 2.1 The model never sees $\chi_i$ — a tool-mediated channel invariant

The schema of `web_search` exposed to the LLM has only one parameter, `query` (tools.py:L7).
When Tavily is actually called, $\chi_i = \tau_i + \delta$ (with $\delta = $
`TAVILY_END_DATE_OFFSET_DAYS`, default $-1$ day, config.py:L273) is **hard-coded and
injected by the tool implementation** (`react._compute_end_date` at react.py:L182,
`search._build_request_payload` at search.py:L133–L162), derived from the current
question's `end_time`. The model can neither perceive nor bypass it.

Two design philosophies sit underneath:

* **Aligning capability boundaries with tool boundaries.** Capability ("knowing the world up
  to a particular day") is determined by system configuration, and should not be something
  prompt engineering or model behaviour can affect. By making the model unable to even see
  "which day I am cut off at", we prevent it from inferring or working around that boundary
  via prompt construction or parameter injection.
* **Single, controllable failure mode.** If we exposed `end_date` as an LLM tool argument,
  we'd have to assume the model could "forget to fill it in" or "deliberately fill in a
  future date". Holding the decision inside the tool implementation collapses the failure
  mode from "the model might make a mistake" to "our code might make a mistake" — the latter
  is testable, auditable, and unit-testable.

**Rejected alternative: expose `end_date` as a tool parameter.** This would let the model
"reason" about cutoffs, but at the cost of trusting the model never to widen them; pin
test would have to assert on every emitted tool call, increasing test surface from O(1) to
O(N·n).

**Rejected alternative: rewrite the query string to insert a date filter.** This pushes
boundary enforcement into a brittle string-manipulation pass that providers can ignore (e.g.
some search engines drop `before:2026-04-01` operators silently).

Pinned by `test_search.py` (the payload contract: the request body always contains
`end_date` derived from `q.end_time`, never from any LLM-supplied field) and `test_react.py`
(end-to-end injection: the schema the LLM sees has no date parameter at any step).

### 2.2 Default leans strict: $\delta = -1$ day

`TAVILY_END_DATE_OFFSET_DAYS=-1` is the project default. The reasoning: many questions
(sports events, central-bank decisions, Oscar nominations) get resolved on the same day, and
using the question's `end_time` as the search cutoff would likely surface news summaries
that already contain the answer. Pushing the search time forward by one day **trades a
little information granularity for strictness**.

Reports also default to comparison at $\delta=-1$ — this is itself a design constraint:
numbers under different offsets are not directly comparable, because $\chi_i$ defines a
different admissible information state for each value of $\delta$ (§1.1).

The paper's audit (paper §4.3.4 / Eq. 67) anchors on $\chi_i$ rather than $\tau_i$ when
classifying "leak / no leak" precisely because the audit definition must match the
operational cutoff actually enforced at the tool layer; any fact in $(\chi_i, \tau_i]$ is
therefore both system-filtered and audit-classified as a leak, eliminating the
otherwise-ambiguous border zone.

**Rejected alternative: $\delta = 0$.** Returns the question's resolution day as the
search cutoff. Empirically catches roughly 30–50% of same-day news leakage on the example
DB, depending on time zones and event types. Documented as too lax; the project switches to
$\delta = +1$ only as an ablation knob.

**Rejected alternative: per-question $\delta_i$.** A natural idea — give "end-of-day"
events a stricter offset than "monthly-resolution" events. Rejected for two reasons: (a)
introducing a per-question knob breaks the contract that one $\delta$ defines one
evaluation; (b) the cost of getting one event wrong is asymmetric (false-negative leak >
false-positive overstrict), and a single conservative default dominates a fragile
per-question heuristic.

### 2.3 Provider-native browsing is forcibly disabled

OpenRouter / OpenAI / Anthropic each have their own web tool or `:online` suffix. The
moment we go down that path, the time cutoff is completely out of control. The project
enforces this on three layers:

* **Startup layer.** `Settings._post_validate` (config.py:L599–L614) rejects any model slug
  containing `:online` or `::` and aborts before any LLM/Tavily call.
* **Per-call layer.** `llm.chat` (llm.py) only attaches our own `WEB_SEARCH_SCHEMA`; any
  `plugins` / `:online` / provider-native retrieval keyword in kwargs is intercepted by
  `_assert_no_browsing` (llm.py:L74). The detector path duplicates this assertion
  (`leak_filter._assert_detector_safe` at leak_filter.py:L139).
* **Test layer.** `test_llm_no_browsing.py` directly mocks the client and asserts the
  outbound payload contains none of those fields, on both the main-LLM and detector paths.

Design philosophy: **the "temptation" of external tools must be rejected at the earliest
possible stage.** If even one release "for convenience" turned this on once, the
comparability of the entire dataset would be ruined. The triple-layer enforcement —
startup, send-time, test — is intentional; any one of the three can prevent a regression,
but only the trio survives a refactor that bypasses one of them.

**Rejected alternative: only enforce at startup.** A test fixture or partial config drift
via `model_copy(update={...})` in dispatcher code could bypass startup-time validation;
re-checking at `llm.chat` send time defends against this exact failure mode.

**Rejected alternative: warn instead of refuse.** Warnings get filtered by log levels,
config templates, and CI noise. Refusals stop the run; they cannot be silently ignored.

### 2.4 Training-data contamination: filter, do not lie

The tool cutoff cannot constrain facts the model has already memorised in its parameters.
The project takes a very plain strategy (paper §3.2.2):

> Declare each model's **training cutoff date** $\kappa_M$; if a question's $\tau_i \le
> \kappa_M$ (equivalently $\chi_i < \kappa_M$ at $\delta=-1$ day-rounded timestamps), the
> question is simply skipped for that model.

Skipped samples still write a row into the DB with `error="skipped_training_cutoff"`
(runner.py:L132–L199; the row is built upstream by `_skipped_cutoff_row` at runner.py:L94):

* Reports can clearly show "how many questions were filtered out per model and how many
  remain comparable" (paper Table 2's "Excluded by Cutoff" column is built from exactly
  this signal).
* `resume` will not retry that row (distinguishing it from transient errors like `network`).
* It is **not** counted in `error rate by kind` — it is not a failure; it is *active data
  cleansing*.

Behind this is a design principle the project repeatedly invokes: **"filtered out" and
"failed" are two different semantics and must be separated at the data layer.** If we used
a boolean `skipped` field, future stratified reporting by cutoff would lose information.
The filter is applied during *task generation*, before the LLM is ever called
(runner.py:L181–L193), so cutoff exclusions consume **zero** API budget — a property
`test_training_cutoff.py` enforces by asserting that no `llm.chat` mock is hit on a
cutoff-skipped sample.

The paper takes the most conservative interpretation when a model card discloses cutoff
only at month-level granularity (paper §4.1.2 footnote): adopt the *last day* of the
disclosed month as $\kappa_M$. The codebase loads this date as a `datetime.date` in
`_parse_training_cutoffs` (config.py:L181–L202), so the comparison `q_end <= cutoff` is a
strict day-rounded equality match aligned with the paper convention.

**Rejected alternative: weighted exclusion (e.g. discount samples close to cutoff).** Adds
analytic complexity without removing the underlying contamination concern; either a
question is in the model's training horizon or it isn't, and the binary decision keeps the
admissibility set $\mathcal{D}^{\mathrm{pred}}_M$ a clean subset of $\mathcal{D}$.

**Rejected alternative: skip the question dataset-wide if any model fails admissibility.**
This would shrink $\mathcal{D}$ to $\mathcal{D}^{\mathrm{pred}}_{\bigcap_m M}$ — the
intersection over all models — and discard easily 10–20% of the corpus on a heterogeneous
model panel. The per-model admissibility set keeps each comparison fair without burning the
shared corpus.

### 2.5 Stage-2 LLM content audit (v5.2): semantic leakage Tavily's protocol layer cannot catch

§§2.1–2.4 are protocol-layer (schema / `end_date` injection / `:online` disabling / cutoff
skipping) defences. The class of leakage this layer cannot cover is **the body of a
Tavily-returned page describing events that happened after $\chi_i$**: Tavily's `end_date`
filter operates on a page's *crawl/index* time, not on the event time *described in the
page content*. A wiki / aggregator page / long article indexed before $\chi_i$ can
perfectly well reference future events in its body. The paper's empirical audit (paper
Table 5) places the residual leakage rate of the Tavily-only baseline in the 3–16% range —
high enough that single-digit accuracy gaps in paper §4.5's results table become
statistically meaningless without further filtering.

The `search-leak-filter-v1` solution adds an independent LLM audit layer (the "detector")
at the end of `tavily_search`, before the main LLM sees `tool_result`: each
`SearchResultItem` is sent to the detector individually
(`leak_filter.filter_search_result`), with verdict ∈ {keep, drop, failed:*}. Items with
verdict=drop are removed entirely — the main LLM never sees any field of a dropped item
(including title / url / content / raw_content); the detector verdict is surfaced ONLY
through `SearchResult.audit` (consumed by `react._record_search_call`), not through any
LLM-visible payload.

| Dimension          | Implementation                                                                                                          |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Cut point          | end of the 200 path in `forecast_eval/search.py:tavily_search`, before `return`                                          |
| Client             | `_detector_client: AsyncOpenAI` (leak_filter.py:L109), independent module-level singleton, **not shared** with the main LLM `_client` |
| Input fields       | whitelist: `title / url / published_date / content / raw_content / cutoff_date`; **MUST NOT contain any field of `Question`** (to prevent the detector from morphing into an "answer auditor") |
| Prompt strictness  | 6 principles pinned in `LEAK_DETECTOR_PROMPT_TEMPLATE` (leak_filter.py:L55–L92): cutoff_date placeholder, treat specific/scheduled/speculative future events equally, "ambiguous → drop", forbid parametric knowledge, strict JSON output, no awareness of the question |
| Parameters         | temperature `0.0`, max_tokens `512`, timeout `60s`, concurrency `5` (paper §4.3.2 / config.py:L345–L352)                 |
| Failure mode       | FAIL-RETRY → CLOSED: by default, K retries still failing → drop; AUTH errors are caught locally and immediately drop (no propagation, no aborting the whole run) |
| Observability      | `search_calls.detector_*` five fields + `run_meta.config_snapshot` detector three-key fingerprint                        |
| Master switch      | `ENABLE_SEARCH_LEAK_FILTER` (config.py:L337), default True; when off, byte-level rollback to v5.1                        |

Three design choices deserve particular emphasis:

**Why the detector does not see the question.** A detector that knows the question can morph
into an "answer auditor" (drop everything that argues against my answer) and produce
question-specific second-order leakage. The whitelist enforces the detector's role:
classify *temporal leakage of facts* — does this page mention an event after $\chi_i$? —
not *relevance to the answer*. This is paper §3.3.3's design rationale ("avoid framing as
an 'answer auditor'") encoded as an inviolable input contract. Pinned by
`test_leak_filter.py` asserting the detector's user message never contains question fields.

**Why fail-closed by default.** Detector hiccups (timeout, network) are uncorrelated with
item content; biasing the residual towards "drop on uncertainty" is the conservative
choice. The keep-on-failure mode (`LEAK_DETECTOR_FAIL_ACTION=keep`) exists only as an A/B
escape hatch for the (rare) case of comparing against the unfiltered baseline. AUTH
failures (401/403) skip retries and immediately drop, because retrying an auth failure is
both pointless and a billing footgun.

**Why an independent client singleton.** Reusing `forecast_eval/llm.py:_client` would couple
two error-budget pools (main-LLM retries inflate detector quota) and confuse log triage.
Two singletons, two backoff sequences, two log namespaces. The independent client also
allows the detector to use a *more advanced* model than the model under test — strictly
preferable, because a more capable detector is the cheapest place to spend extra capability
in the pipeline.

The detector's reference date is *the question's* $\chi_i = \tau_i + \delta$ (sharing the
same source as Tavily's `end_date`), independent of §2.4's `MODEL_TRAINING_CUTOFFS` (the
model training cutoff $\kappa_M$, indexed *per model*). Even after a question passes
through the admissibility filter and enters execution, the detector still audits the search
results for *that question* — the two do not substitute for each other.

Reference: the BLF paper (Murphy 2026, §B.1 Stage 2) used an LLM-based leak classifier on
top of Brave's date filter and reported the runtime filter catching 320/341 = 93.8% of
actual leakage, with residual leakage on the order of 1.5%. The paper's own audit
(paper Table 7) on $N=270$ items measures recall 98.7% (235 / (235 + 3)) and per-audit-item
residual rate 1.1% (3/270; Wilson 95% upper bound 3.2%), comparable to the lower end of
the Tavily-only baseline at two orders of magnitude lower marginal cost. Stage 2 is an
empirically validated "algorithmic-layer + semantic-layer double insurance" engineering
practice.

For the full spec see `openspec/changes/search-leak-filter-v1/specs/` (capabilities
`search-leak-filter` / `search-tool` / `information-barrier` / `results-persistence`).

### 2.6 What we can and cannot control

Paper §3.5 contains the threat model that this project takes as gospel — an honest
confession of the controllable / uncontrollable boundary:

| Leakage source                              | Controllable?                                              | Mitigation                                            | Audited?  |
| ------------------------------------------- | ---------------------------------------------------------- | ----------------------------------------------------- | --------- |
| Tavily returned content (date filter)       | ✅                                                         | $\chi_i$ injection (§2.1–2.2)                         | yes (§2.5) |
| Provider-native browsing                    | ✅                                                         | code + test ban (§2.3)                                | yes (test_llm_no_browsing) |
| Model parametric memory                     | ⚠️ Partial                                                 | $\kappa_M$ admissibility filter (§2.4)                | partial (model card disclosure) |
| Page bodies that mention post-$\chi_i$ events | ⚠️ Partial                                               | Stage-2 LLM detector (§2.5); audited residual ≈ 1.1%   | yes (§2.5 audit) |
| Time clues in the question text itself      | ❌                                                         | accepted as evaluation bias                            | no        |
| External knowledge backflow after training  | ❌                                                         | accepted as evaluation bias                            | no        |

Design philosophy: **acknowledging what we cannot control matters more than pretending we
can.** The uncontrollable parts are accepted as part of the evaluation bias and reported in
audit metadata; the controllable parts are locked down by code + tests + per-call audit
logs. Paper §3.5 makes the same admission verbatim — the codebase here is its operational
shadow.

The two ❌ rows above are the residual sources our claims do not cover. We do not pretend
otherwise, and any user who wishes to attack OracleProto's claims should attack one of
these two — not the four rows above them.

---

## 3. Dataset reconstruction: from resolved events to forecasting tasks

The paper's central conceptual move (paper §3.2) is to rewrite a resolved event $z_i$ as a
time-bounded prediction:

$$(x_i, \mathcal{A}_i, Y_i, \tau_i, \rho_i) \quad \Rightarrow \quad q_i^{\mathrm{in}} = (x_i, \mathcal{A}_i, \chi_i, \rho_i).$$

The ground truth $Y_i$ is retained only for scoring; the visible prompt is rendered
deterministically from the structured fields. This construction equips the dataset with
four properties that make the evaluation object *dataset-level* rather than *event-level*
(paper §3.2.3):

1. **Temporal reproducibility** — the same source row + same $\delta$ always yields the same
   $\chi_i$.
2. **Model-dependent admissibility** — admissible question sets vary by $\kappa_M$, but the
   underlying corpus is shared.
3. **Discrete scorability** — $\widehat{G}_{i,M}, G_i \subseteq \mathcal{L}_i$ are
   finite-cardinality sets, not free text.
4. **Audit-reproducibility across calendar years** — the same dataset can be replayed
   against models with later cutoffs, with the admissibility set automatically shifting.

### 3.1 Discrete answer space, not free generation

Each question must have a finite answer space $2 \le K_i < \infty$ and a verified answer
set $Y_i \subseteq \mathcal{A}_i$ (paper §3.2.2). This prevents open-ended generation from
bypassing the evaluation constraint. The three question types — `yes_no`, `binary_named`,
`multiple_choice` — all collapse to letter sets at the scoring layer, with single-answer
questions satisfying $|Y_i|=1$ and multi-answer questions satisfying $|Y_i| \ge 1$. The
structural constraint $\rho_i$ (single vs multi) is recorded in `choice_type` and consumed
by the parser to validate that the model's output cardinality is legal.

**Rejected alternative: open-ended natural-language outputs scored by an LLM judge.**
Imports a second LLM into the scoring path and re-introduces a contamination concern (the
judge may have its own training data overlap). Strict letter-set scoring keeps the entire
scoring path deterministic, audit-replayable, and judge-independent.

**Rejected alternative: numerical probability outputs scored by Brier / NLL only.** The
v4 belief protocol does collect probabilities, but as a *companion* to letter-set output,
not a substitute. Paper §4.4.2 documents the v5 demotion of probabilistic metrics under
the K=5 sampling regime (the empirical $\hat p$ takes only 6 discrete values, making
calibration parameters statistically meaningless).

### 3.2 Letter encoding is the canonical answer

The source DB `answer` field uniformly uses letters (`'A'` or `'A, B'`) rather than option
text:

* `yes_no`: `Yes=A, No=B`
* `binary_named`: the first entity = A, the second = B
* `multiple_choice`: A/B/C/... follow the order of the `options` JSON array

The model's output form varies by question_type (`Yes`, an entity name, a letter list are
all possible), but `parser.parse_answer` (parser.py:L40–L89) uniformly normalises to
`frozenset[str]`. This design completely **decouples "how the model says it" from "how the
system scores it"** — the same scoring code covers all three question types because they
all reduce to set equality on letter sets.

Pinned by `test_parser.py` (round-trip on representative inputs of each type, including
case variations and whitespace tolerance).

### 3.3 Strict frozenset equality is the scoring primitive

The entire scoring logic is one line at `parser.py:L102–L106`:

```python
predicted_letters == ground_truth_letters  # both are frozenset[str]
```

Missed selections, extra selections, ordering — all are scored as wrong by strict equality.
This is the project's most "fastidious" design, and it implements paper Eq. 14 verbatim.

#### Why not Jaccard / F1 / partial credit at the strict level?

* **Explanation cost is too high.** `pass@1=0.62` is far more intuitive than
  `mean Jaccard=0.74`; one number is enough for a paper.
* **Avoid half-credit masking the real problem.** If a model misses one or two selections
  every time, its average score can still be 70+, but fundamentally it has not really
  mastered that question. Strict matching scores this behaviour as 0, forcing the report to
  be honest.
* **Unifies the scoring interface across three question types.** All three reduce to
  frozenset equality at the scoring stage; write the code once, all three types work.

For multi-answer questions the project adds **two soft-penalty companions** alongside
strict equality (paper §4.2.4 / §4.2.7), without ever replacing it:

* **Exam-style partial credit** (paper Eq. 17, `analysis/exam_score.py:L62`): "any FP vetoes
  to 0; otherwise score $|TP|/|G|$". Equivalent to *Recall under a zero-FP gate*. Makes the
  headline composite-accuracy more nuanced for multi-answer buckets where strict equality
  has near-zero variance.
* **Format Skill Score (FSS)** (paper Eq. 22, `analysis/accuracy.py:L386`): chance-corrected
  Tversky similarity with $(\alpha, \beta) = (2.0, 0.5)$. Penalises false positives 4×
  more than false negatives — the prediction-task intuition that *claiming an event will
  happen* is more dangerous than *missing an event*. Single-select questions degenerate to
  strict 0/1; the asymmetry only matters in multi-answer buckets.

The two soft penalties coexist with strict equality precisely so that the choice of metric
becomes an analyst-side decision rather than a system-side bias. The audit trail
(`composite_meta.json`) records exactly which buckets each composite uses.

#### Why $(\alpha, \beta) = (2.0, 0.5)$ specifically

The Tversky penalty asymmetry encodes a *prediction-domain* prior: in forecasting, claiming
an event will happen carries more downstream risk (acted-upon false signal) than missing it
(opportunity cost). 4× FP vs FN penalty is one octave on the log scale — strong enough to
flip cross-model rankings on the multi-answer bucket (paper §4.5.3 documents Qwen
overtaking Kimi on FSS despite trailing on $\mathrm{pass@1}$, exactly because Qwen's
selection sets are more "restrained"), conservative enough to leave $\alpha = \beta = 1$
(Jaccard) recoverable as an ablation knob.

The values are configurable through the `tversky_score` and `tversky_baseline` `alpha` /
`beta` keyword arguments (accuracy.py:L289–L292, L320–L322), but the analysis pipeline
hard-codes the (2.0, 0.5) defaults; changing them requires a code edit, not a `.env`
change, because the *interpretation* of FSS depends on the asymmetry being kept fixed
across runs.

### 3.4 ASCII continuation: imperfect, but keeps the mapping stable

The example DB contains 4 `multiple_choice` questions with > 26 options, of which 3 have
ground-truth answers landing on ASCII continuation characters such as `[`, `\`, `]`, `^`,
`_`, `` ` ``, `a`, `b`, `c`, and so on. These characters are extremely unfriendly to LLMs
(backticks are eaten by markdown, lower/upper-case a/A are easily confused), but the
project still keeps them — in order to **preserve a one-to-one letter ↔ index mapping**.

The cost is mitigated by several defences:

1. `prompts.render_user_prompt` explicitly quotes or escapes labels when generating an
   `outcomes_block` for > 26 options.
2. `parser.parse_answer` (parser.py:L74–L87) iterates `tokens` of length 1 only and uses
   `letter_to_index` round-trip validation — pinned by `test_parser.py` round-trip cases on
   > 26 options.
3. Logs / reports record letters and labels in parallel for manual review.

If we later confirm LLM performance is meaningfully dragged down by the labelling scheme,
we'll migrate to a stable scheme like `AA/AB` or `A01/A02`. **This is a documented "debt",
not an ignored bug.**

**Rejected alternative: skip the > 26 questions.** Discards roughly 4 / 215 multi-choice
questions on the example DB. Acceptable but loses dataset-level coverage; preferred
mitigation is the round-trip test plus the option-stable encoding.

### 3.5 Parse failure ≠ error

When the LLM does not output `\boxed{...}`, or writes "I cannot predict the future" — **this
is not a system failure, it is part of the model's capability.** Concretely (paper §4.2.3):

* `parse_ok=0`, `correct=NULL`: parse failures / refusals are accumulated separately into
  "refusal rate" and surfaced in reports.
* No retry: when the model itself says it cannot answer, asking again yields the same answer
  — backoff retries only waste tokens.
* `error` field stays NULL: the `error rate by kind` report is not polluted by such "soft
  failures".

The same coupling rule is encoded as paper §4.2.4's four-state matrix; `exam_score`
(exam_score.py:L62–L91) implements the matrix in a 7-line decision tree. Pinned by
`test_exam_score.py` and `test_aggregation.py`.

Design philosophy: **every behaviour must have its own cell in the report.** "System error
rate" and "model refusal rate" are two different things and cannot be lumped under a single
total error rate.

**Rejected alternative: retry on parse failure.** Two reasons against: (a) capability
masking — a model that refuses to commit to an answer is a model that lacks forecasting
capability; retrying papers over the gap; (b) cost: retries on a 10K-item evaluation
multiply API spend without changing the population mean.

---

## 4. The hierarchical evaluation system

The paper's evaluation system (paper §3.4.6) is

$$\mathcal{E}_M = (\mathcal{E}^{\mathrm{valid}}_M, \mathcal{E}^{\mathrm{item}}_M, \mathcal{E}^{\mathrm{question}}_M, \mathcal{E}^{\mathrm{model}}_M).$$

Four levels, each with a distinct semantic, each computed from the same normalised discrete
answer space.

| Level                     | Object                              | What lives here                                    | Code                              |
| ------------------------- | ----------------------------------- | ------------------------------------------------- | --------------------------------- |
| **Validity**              | $v_{i,M} = \mathbb{1}[\Psi_i(o_{i,M}) \ne \bot]$ | parse_ok / parse_failure_rate                | `parser.parse_answer` / `analysis/aggregation.py` |
| **Item**                  | $r_{i,M} = \mathbb{1}[\widehat{G}_{i,M} = G_i]$ on a single trial | strict equality / exam-score | `parser.is_correct` / `analysis/exam_score.py`    |
| **Question**              | $\{\widehat{G}_{i,M}^{(s)}\}_{s=1}^{S}$ across $S$ trials | pass_any@N / pass_all@N / Fleiss κ / VCI / MV | `analysis/accuracy.py` / `analysis/consistency.py` |
| **Model**                 | $\Gamma(\{\mathcal{E}^{\mathrm{question}}_{i,M} \mid q_i \in \mathcal{D}^{\mathrm{pred}}_M\})$ | composite accuracy / FSS / BI / per-correct cost | `analysis/composite.py` / `analysis/__init__.py` |

Each level is captured by a separate column family in the analytics output, and the choice
of $\Gamma$ at the model level is the analyst's lever, not the system's. The same raw
observations support flat means, weighted composites, paired-bootstrap CIs, posterior
comparisons, etc. — because nothing is pre-aggregated in the DB.

### 4.1 Why no pre-aggregation in the DB

* **Metric definitions evolve.** Today `pass@3` is "1 of 3 counts as a pass"; tomorrow it
  might change to "at least 3 correct". If aggregation lands in the DB, every redefinition
  requires a backfill. Deferring all metrics to the analysis layer makes them re-computable
  at any time.
* **`analysis.py` is a pure function.** input = `runs/{run_id}/db/*.db`, output =
  `analysis/*.csv|md|json`. Can be re-run independently via `python -m forecast_eval.analysis`.
* **DB and paper/report are decoupled.** Raw records are an engineering artefact;
  statistics are a product / academic artefact — their cadences are completely different.

This is one of the project's most important architectural decisions and is the operational
embodiment of the paper's "metric-agnostic design" claim (paper §3.4.6, Eq. 13). Pinned by
`test_analysis.py` constructing a hand-crafted DB fixture and confirming `run_analysis`
neither writes back nor mutates.

### 4.2 Recalibrating the `pass@k` naming

In the wider community `pass@k` generally means "at least one correct in k". The project
historically used `pass@3 = sum(correct)≥3` (a threshold semantic), which caused ambiguity.
Now made explicit (paper §4.2.5 / Eq. 35–37):

* `pass_any@N` ≡ standard `pass@k`: at least one correct in N.
* `at_least_k_correct@N`: at least k correct in N (threshold analysis).
* `pass@1 avg`: average accuracy across N (stable capability).
* `majority vote correct`: whether the majority-vote frozenset across N is correct
  (self-consistency).

The four columns are emitted by `analysis/accuracy.py::Aggregate.as_ordered_dict`
(accuracy.py:L66–L100); their definitional identities ($\mathrm{pass\_all} \le
\mathrm{pass@1}_{\mathrm{avg}} \le \mathrm{pass\_any}$) hold by construction and are
asserted in `test_aggregation.py`.

Design philosophy: **a name must either be unambiguous or explicitly declare its
semantics.**

### 4.3 Question-level signals: stability is not the same as correctness

Paper §4.5.3 reports an instructive divergence among the six tested models that motivated
the entire question-level signal column family:

* DeepSeek and Kimi tie on $\mathrm{pass\_any}@N$ ($0.80$, best-of-3 hit upper bound), but
* Qwen leads on $\mathrm{pass\_all}@N$ ($0.39$) and Fleiss' κ ($0.45$ — consistently
  same-answer behaviour), while
* Doubao ranks 3rd on Fleiss ($0.42$) but last on $\mathrm{pass}@1$ — *consistently giving
  wrong answers*.

This is exactly the diagnostic the question-level signals are designed to expose: high
consistency does not imply correctness, and a high best-of-N ceiling can come from "three
different answers, one of which happens to hit" rather than from "consistently correct
each time". The project therefore reports both axes side-by-side rather than collapsing
them.

The Fleiss' κ implementation (consistency.py:L176–L420) follows paper §4.2.6 / Eq. 39–43
verbatim, including the per-stratum decomposition for single-answer questions (each $k_q$
stratum is its own κ, weighted by question count) and the per-label binary decomposition
for multi-answer questions. `test_consistency.py` pins both decompositions on hand-crafted
vote tables.

### 4.4 Composite accuracy: weighted by sub-question type

`per_model_summary.csv` reports a flat mixed mean for backwards compatibility. For headline
scoring the paper uses composite accuracy: per-bucket exam-score → subtype-weighted average
(paper Eq. 18 / §4.2.7). Two dimensions are computed independently (decoupled from each
other, since `multiple_choice` itself contains both single and multi — the two are not
orthogonal):

* `question_type` dimension (yes_no / binary_named / multiple_choice) →
  `per_model_composite_by_question_type.csv`;
* `choice_type` dimension (single / multi) → `per_model_composite_by_choice_type.csv`.

For each (model, dimension, metric):

$$\text{composite}_m = \frac{\sum_{b \in B_{\text{valid}}} w_{m,b} \cdot v_{m,b}}{\sum_{b \in B_{\text{valid}}} w_{m,b}}.$$

Missing buckets (slice unavailable or weight 0) are dropped, and the remaining weights
renormalised proportionally; they are *not* treated as 0. All None → composite = None. The
identical formula and renormalisation rule appear at `composite.py:L18–L29` (formula) and
`composite.py:L77–L127` (allowlist + per-metric override resolution).

#### "Harder questions discriminate better" rationale for default weights

| Dimension       | Bucket            | Default weight | Difficulty rationale                                        |
| --------------- | ----------------- | -------------- | ----------------------------------------------------------- |
| `question_type` | `yes_no`          | 0.15           | k=2, blind guess 50%, almost no inter-model discrimination  |
| `question_type` | `binary_named`    | 0.15           | k=2 as above, adds entity recognition                       |
| `question_type` | `multiple_choice` | 0.70           | k=2..N wide range, includes multi-select, highest discrimination |
| `choice_type`   | `single`          | 0.40           | overall easier (includes yes_no / binary_named)             |
| `choice_type`   | `multi`           | 0.60           | true multi-select, almost every model performs poorly, high discrimination |

Defaults at `config.py:L365–L373`. One sentence: **let buckets that discriminate model
capability contribute more.** To switch to "I care more about easy questions", just flip
the numbers via `COMPOSITE_WEIGHTS_QTYPE` / `COMPOSITE_WEIGHTS_CTYPE` in `.env`. This is an
opinionated default, not a "neutral" one — we believe the orientation above is more
reasonable for the vast majority of evaluation scenarios; users with a different view solve
it by overriding one line of `.env`.

**Rejected alternative: equal weights across buckets.** Sounds neutral but is not — the
empirical question-type prevalence on the paper's curated 80-question set is roughly
$\{\text{yes\_no}: 37/80, \text{binary}: 3/80, \text{mc}: 40/80\}$. Equal weights would
double-count yes_no relative to mc and erase the most discriminative bucket. The
"discrimination-aware" weights above are the principled choice.

**Rejected alternative: empirical-prevalence weights.** Equivalent to a flat unweighted
mean. Rejected for the same discrimination reason: when 50% of questions are
near-random-baseline, weighting the composite by prevalence drowns out the signal-bearing
buckets.

#### Why the chance baseline matters under the exam view

Under the exam view, the chance baseline on the multi-choice multi-answer bucket is
$T^{\text{chance}}_q = 2^{-(k_q - m_q + 1)}$ (paper Eq. 23), which lands in $[0.06, 0.25]$
for typical $(k_q, m_q)$. Compare with the strict-equality baseline $0.5^{k_q}$, which is
essentially zero for $k_q \ge 5$. The exam view places the multi-answer column on the same
order of magnitude as single-answer buckets in absolute terms, so the multi-answer signal —
which is the one that actually discriminates models — is no longer drowned out by its
near-zero strict-view variance. This is the core gain of the exam composite over a
strict-equality composite, documented in paper §4.2.7 as the explicit rationale for
choosing `exam_score_at_n_avg` (rather than strict $\mathrm{pass@1}$) as the *headline*
composite.

#### Configuration entrypoint (`Settings` / `.env`)

* `COMPOSITE_WEIGHTS_QTYPE` / `COMPOSITE_WEIGHTS_CTYPE` (config.py:L365, L372): global
  default weights, shared by all metrics that are not explicitly overridden.
* `COMPOSITE_WEIGHT_OVERRIDES_QTYPE` / `COMPOSITE_WEIGHT_OVERRIDES_CTYPE`: per-metric
  independent overrides. Form: `"fss=yes_no=0.05,multiple_choice=0.95"`, semicolon-separated
  for multiple metrics. Misspelled metric names raise from `compute_composite` during the
  analysis phase rather than "silently falling back to default" — this is intentional: when
  you misconfigure, we make sure you know.
* Startup-time validation (config.py:L515–L535): bucket name must ∈ the legal set, weight
  ≥ 0, at least one > 0.

#### Cost amortisation: the per-correct cost as a Pareto axis

The paper proposes (Eq. 25)
$$C^{\text{per-correct}}_m = \frac{C^{\text{total}}_m}{|\mathcal{D}^{\text{eval}}| \cdot n \cdot \text{Composite\,Accuracy}_m}.$$

Adopting OpenRouter's actual billing instead of a "published unit price × token usage"
calculation circumvents grey areas like "are reasoning tokens billed?", "how is the
prompt-cache discount accounted for?", "are tool calls billed?", "do prices differ across
provider routings?". The platform invoice is the single financial fact verifiable by third
parties.

Dividing by **difficulty-weighted notional correct count** (rather than raw correct count)
matters because it places "expensive but accurate" and "cheap but reckless" models on the
same scale of cost-effectiveness, avoiding the false low-cost illusion produced by "low
per-sample unit price but high error rate". Semantically, $C^{\text{per-correct}}$ is the
reciprocal of "how many difficulty-weighted correct predictions does one USD buy".

The paper's experimental table (Table 4) demonstrates the value of this axis: at composite
accuracy within 1.2pp of DeepSeek's lead ($0.6016$ vs $0.5896$), Qwen costs 1/8 the total
(\$0.45 vs \$3.60) — the joint (accuracy, cost-per-correct) Pareto frontier is the only
meaningful comparison surface, and ranking on accuracy alone or cost alone is misleading.
On this axis Qwen and DeepSeek jointly span the frontier; the other four models are
Pareto-dominated by at least one endpoint.

---

## 5. Reproducibility: every run is its own independent space-time

### 5.1 The source database is checked into Git

`forecast_eval_set_example.db` goes straight into the repo. It is the evaluation's
"gold-standard" example dataset and must ship with the repo; anyone can `git clone` and
obtain the exact same questions. The filename (`SOURCE_DB`) and the internal question table
name (`SOURCE_TABLE`, default `forecast_eval_set_example`) are both exposed as `.env`
parameters; with a custom dataset, just change these two variables and the loader splices
`<SOURCE_TABLE>` into the SQL `FROM` clause at runtime. The table name is whitelist-validated
(`^[A-Za-z_][A-Za-z0-9_]*$`, config.py:L586–L595) at the Settings stage to foreclose
SQL injection — the only place we *can* be injected and the only place the validation
runs.

Each run also computes `source_db_hash` and writes it to `run_meta`; together with
`metadata_hash` and `prompt_templates_hash` (and, when applicable,
`reflection_protocol_hash`, `belief_protocol_hash`, `leak_detector_prompt_hash`), this
forms a multi-part fingerprint of "exactly which inputs this run is based on" (db.py:L385–L399
for the three core hashes; full set assembled in `evaluation.py`).

**Rejected alternative: ship the DB on a separate registry.** Adds a network dependency to
reproduction; checking it into Git makes `git clone` the one and only setup step.

### 5.2 Each run gets its own directory

```text
runs/{run_id}/
  manifest.json          # run-level metadata
  db/{model_slug}.db     # one sqlite per model (one virtual slug per file under grid)
  analysis/              # post-hoc statistical artefacts
  logs/{run_id}.log
```

A few details of the design choice:

* **Directory per run, not single DB.** The early "single `results.db`" was replaced.
  Reason: with a single DB, the boundary between runs depended entirely on the `run_id`
  column, which made independent distribution hard and made it easy to mix data from other
  runs into analysis.
* **`run_id` defaults to `YYYYMMDD-HHMMSS-xxxx`.** `ls` naturally sorts by time, and since
  this is also the directory name you can tell "when it ran" at a glance.
* **`RUN_ID` empty → new run; same value → resume.** One variable handles both, no extra
  `--resume` CLI flag needed.

### 5.3 One SQLite per model

Why not "one big DB per run"? Three reasons, in order of importance:

1. **Independently distributable.** Hand `runs/{run_id}/db/openai__gpt-5.db` to someone else
   and they can replay just this one model, with no need to obtain the other models' results.
2. **Non-interfering write paths.** One async writer task per model, with
   single-writer-multi-reader WAL mode providing ample concurrency, and one model's stall
   cannot block another's.
3. **Easy schema-evolution isolation.** If some model needs to store a special field (e.g.
   reasoning trace), its schema can be extended independently without affecting others.

The cost is that the analysis layer must scan multiple files, but `analysis.py` already
encapsulates that.

**Rejected alternative: one DB with a `model` column.** Trips up all three properties
above; specifically, single-writer contention turns one slow provider into a global stall
because each writer holds the same DB-level lock under WAL.

### 5.4 Each DB self-contains `questions` + `prompt_templates` copies

Each model DB embeds copies of the source question set and prompt templates. This looks
redundant at first glance, but it serves "independent replay":

> Whoever receives `openai__gpt-5.db` does not need to track down
> `forecast_eval_set_example.db`, nor hunt for which metadata version was in use at the time —
> every input the evaluation needed is inside this single DB.

Consistency between copies is guaranteed by hash verification: three fields —
`run_meta.source_db_hash` / `metadata_hash` / `prompt_templates_hash` — pin down "the source
data at the time".

**Rejected alternative: store paths, not copies.** Breaks the self-contained property; once
the path moves, the DB becomes unreplayable.

### 5.5 The config is redacted before being written into the DB

`run_meta.config_snapshot` stores the redacted `.env` as JSON
(`db.snapshot_settings` at db.py:L429–L444). Sensitive fields like `LLM_API_KEY` only
retain the first 4 characters + length + `sha256[:12]`; `TAVILY_API_KEY` is now `list[str]`,
with each key redacted independently and persisted as
`[{prefix, sha256_12, length, provider}, ...]`.

Design philosophy:

* **Want to know which parameters (temperature, concurrency, retry sequence) this run
  used?** Stored.
* **Want to know the plaintext of the key that was used?** Never stored.
* "Auditable" and "non-leakable" coexist within the same field.

Pinned by `test_db.py` round-tripping through `snapshot_settings` and asserting prefix
length and digest length, never the raw value.

### 5.6 Three independent protocol fingerprints

`prompt_templates_hash`, `reflection_protocol_hash`, and `belief_protocol_hash` are three
**mutually independent** SHA-256 fingerprints kept side-by-side in `run_meta` and the
manifest. This independence is deliberate and load-bearing for ablation studies (paper
§4.6.2 documents the reflection-A/B paired-bootstrap analysis that requires this exact-match-except-one-hash invariant).

* `prompt_templates_hash` reflects "how question content is rendered to the model" — the
  *templates* for the stem, options, instructions, question-type description, and so on.
  Once a template changes, every question text changes — this is a coarse-grained
  run-distinguishing key. Computed by `compute_prompt_templates_hash` (db.py:L397).
* `reflection_protocol_hash` reflects "which meta-cognitive instruction was injected into
  the model in the ReAct main loop", essentially *a switch on a search-behaviour prior*.
  Its variation has only three axes: on/off, whether the text was modified, version number.
* `belief_protocol_hash` reflects "did the model emit a structured belief vector before
  `\boxed{...}`", a switch on whether the probabilistic-family metrics are populated.

The benefit of three separate hashes: when running A/B comparisons across runs, you can
choose "only `reflection_protocol_hash` differs, everything else equal" — exactly what an
ablation study wants. The reflection-A/B pairing in
`analysis/behavior.py::find_paired_runs` *requires* this exact-match-except-one-hash
invariant, and unrelated runs are filtered out automatically.

The full text of each protocol coexists in `run_meta` (`reflection_protocol_text`,
`belief_protocol_text`), to enable post-hoc diffs without depending on the `prompts.py`
source code (e.g. when releasing a report, the recipient receives a redacted DB rather than
the git repo).

**Rejected alternative: a single composite hash.** Loses ablation discrimination — once any
of the three changes, everything looks "different", and you can no longer pair runs along
one axis.

**Rejected alternative: 6-way compound key including detector hash + source DB hash.**
Considered and chosen-against; the paper's design declares $H_{\mathrm{aux}}$ outside the
$\mathcal{R}$ tuple precisely because the detector is an auxiliary engineering layer.
Carrying it as a separate axis (`leak_detector_prompt_hash`) preserves the same
"strict-equality except one axis" pairing pattern at the ablation layer.

---

## 6. Wide table + single writer + post-hoc analysis

### 6.1 Why a wide table

One row per question, with an `s{i}_*` group of columns per sample (since v3, 20 fields:
original 14 + 6 newly added observation columns; v4 adds 3 belief columns; v5 adds 1
final-answer-retry column). Compared to a "long table + (question_id, sample_idx) composite
primary key", the wide table's advantages:

* **Resume queries are naturally simple.** `SELECT question_id WHERE s{i}_created_at IS NOT
  NULL` simply scans one column, no group by needed.
* **Atomic single-row read.** The analysis script reads one row and has every sample; no
  join or aggregation needed.
* **Schema fixes N.** `SAMPLING_N` is pinned at table-creation time, so whenever the DB is
  reopened in the future, the structure matches what it was then.

The cost: `SAMPLING_N` must be determined before the run starts and cannot be expanded
mid-run; the schema also needs to "dynamically generate 20 × N columns". This cost **is
acceptable in evaluation scenarios** — `SAMPLING_N` is by nature part of the run config and
should not change mid-run.

**Rejected alternative: long table + composite key.** Wins on flexible $N$, loses on
trivial resume queries and on JOIN-free analysis. Net: long tables are better for
production telemetry; wide tables are better for evaluation artefacts.

### 6.2 Why `step_metrics` is a JSON column instead of a separate long table

ReAct's per-round step metrics are naturally 1-to-N (one sample → multiple steps), and at
first glance you'd want to factor them out into a long table. The project ultimately
compresses them into `s{i}_step_metrics TEXT` (a JSON array) for three reasons:

* **No cross-step query need.** The analysis layer always "fetches the whole trajectory by
  sample and then processes it"; it never does row-level aggregation like
  `SELECT * FROM steps WHERE finish_reason='length'` — every filter happens at sample
  granularity. Normalising this data into a table would mean paying index/JOIN cost for
  queries that don't exist.
* **Preserves the simplicity of one writer per model.** Switching to a long table would
  require a second table + a second foreign key + a second INSERT path, and the writer
  boundary would jump from "one-row upsert" to "multi-row transaction", conflicting with
  §6.4's "eliminate races via orchestration" principle.
* **JSON size is controllable.** The step count per sample is bounded by `REACT_MAX_STEPS`
  (default 12); a single JSON is typically < 1 KB; on v3 schema with SAMPLING_N=3 / ~100
  questions, the DB delta is on the order of KB — WAL handles it easily.

The cost: "step-level aggregation" that a long table could do has to be done by reload +
parse in Python here. Given that the analysis script is a one-shot script anyway
(`python -m forecast_eval.analysis`), this cost is acceptable.

### 6.3 Phase 2 calibration uses LOO instead of holdout (v4)

When the v4 belief protocol is enabled, both Platt scaling and temperature scaling in
`forecast_eval/analysis/calibration.py` (now removed in v5; see §6.6) used leave-one-out
cross-validation. Rationale:

* **N is small.** With 80 questions in the paper's main run and 50–150 per cell after
  stratification, holdout splits make calibration parameters high-variance — different
  splits can yield (a, b) differences of ±0.3, enough to render ECE comparisons
  meaningless. LOO uses every question and prevents that question from polluting its own
  calibration parameters.
* **Compute is cheap enough.** Platt with IRLS is ~10 Newton iterations per fit, each O(N)
  for the Hessian. LOO is N refits; a naive total cost of O(N²) ≈ 100k float ops, < 1s for
  319 questions.

### 6.4 One writer per model + WAL

Concurrent writes to SQLite are a classic pitfall. The project's strategy:

* **One async writer task per model DB.** Every worker's results are sent via
  `asyncio.Queue` to the writer for that model.
* **`PRAGMA journal_mode=WAL` + `synchronous=NORMAL` + `busy_timeout=5000`.** Ample
  throughput under single-writer-multi-reader, with crash recovery still safe.
* **Batched commits.** Flush every `DB_COMMIT_BATCH=10` entries or every 1 second.

The core idea behind this design: **eliminate races via orchestration, don't solve races
with locks.** Once we pin "one writer per DB", the concurrency problem degenerates into
ordinary single-threaded batch inserts.

**Rejected alternative: lock-per-INSERT.** Works but burns CPU on contention with no
amortisation; the orchestration approach moves the cost to enqueue (cheap) instead of
write (expensive).

### 6.5 The DB stores raw observations only; aggregation happens later in `analysis/`

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

This is one of the project's most important architectural decisions and is the operational
embodiment of the paper's "metric-agnostic design" claim (paper §3.4.6, Eq. 13).

### 6.6 v5 demotion of probabilistic metrics under K=5

v5 reorientation: at K=5 parallel samples, the empirical probability $\hat{p} = n/K$ for
each (question, label) takes only 6 discrete values $\{0, 0.2, 0.4, 0.6, 0.8, 1.0\}$. This
pushes v4's Reliability Diagram / Murphy three-decomposition / Platt scaling LOO into the
"mathematically correct, statistically meaningless" position. v5 redirects the analysis
stack to the **discrete-native** metric family suited for K=5; BS / NLL / MBS / BI / ABI
are demoted to auxiliary columns with a `†` footnote and a K-disclaimer in
`per_model_summary.md`.

Concretely:

* `calibration.py` is deleted in v5; its 5 artefacts (`calibration_params.json`,
  `per_model_summary_calibrated.csv`, `reliability_data*.json`,
  `brier_decomposition.csv`) are discontinued.
* The discrete family (FSS / Cohen κ / Hamming / Fleiss κ / mean entropy / VCI / MVG)
  becomes the v5 main line, with `entropy_accuracy_bins.csv` and
  `inter_trial_consistency.csv` as new v5 artefacts.
* If $K$ is increased to ≥30 in the future, calibration can be reintroduced in a new
  change.

This decision is *not* a paper-level constraint — paper §4.4.2 documents both the
discrete-native main line and the probabilistic auxiliary family — but a v5 *engineering*
choice motivated by sample-size statistics. Reverting to K=30 would re-enable the
probabilistic line; the demotion is by analyst convention, not by hard-coded gate.

---

## 7. ReAct + Tool Use: "unfolding" the model's reasoning

### 7.1 The whole prompt as a single user message

The template is one entire prompt block (agent_role + event + outcomes + format +
guidance); the project chooses to feed it in **as a single user message** rather than
splitting into system / user.

Reasoning:

* **Most faithfully reproduces the source metadata template.** The source data
  `dataset_metadata.features_json.prompt_reconstruction` is a single string; forcibly
  extracting a system part would lose the semantics of the original concatenation.
* **Cross-model consistency.** Different providers handle system messages differently
  (OpenAI hard-caches them, Anthropic uses an independent field). Going uniformly through
  a user message gives the most stable comparability — exactly the property paper §3.4
  demands of $R$ as a deterministic renderer.
* **Easy to hash and diff.** The entire prompt content is written directly into the
  `user_prompt` field, so any future template change is visible through a hash at a glance.

**Rejected alternative: split system / user.** Loses cross-provider consistency; the same
$R$ becomes effectively two different renderers depending on provider.

### 7.2 Hard ceilings on the ReAct loop

Each sample has two gates:

* `REACT_MAX_STEPS=12` (config.py:L279): the LLM may interact with the system at most 12
  rounds in total (enabling the reflection protocol or nudges adds 2-4 rounds beyond a
  single-step direct answer, so the default is slightly higher than the historical value).
* `REACT_MAX_SEARCH_CALLS=[8]` (config.py:L283; paper main-table runs use $C=4$ as the
  headline configuration; the rationale is $R_{\mathrm{tav}} \cdot C = 5 \cdot 4 = 20
  \approx$ two pages of Google search results): after $C$ cumulative `web_search` calls,
  the tool returns `search budget exceeded` directly to the LLM.

The design philosophy is to **define an upper bound on "the model's autonomous searching"
via a budget**:

* Without a cap, malicious / degenerate models could call indefinitely and burn through the
  API bill.
* Capping while returning an error rather than throwing lets the LLM still provide a
  "best-effort answer" based on existing information, and separates "out of budget" from
  "system crashed".

Exceeding step count without producing a boxed answer → `parse_ok=0`, treated the same as a
refusal (§3.5).

The codebase default $C=8$ is *deeper* than the paper main run $C=4$ — this is intentional.
The paper's main-table configuration is a deliberately-tight budget for discrimination
(paper §4.1.4, "two pages of Google"), while the example DB ships with a wider budget for
smoother behavioural analysis. To exactly reproduce the paper main run, override
`REACT_MAX_SEARCH_CALLS=4` in `.env` (FRAME §1.3 reconciliation table).

### 7.3 Reflection protocol: pulling the model off "one-shot direct answer"

We observed that some models give a final answer after only ~1.6 searches on average — this
"confident one-shot" behaviour drastically lowers `pass@1` on long-tail events. The project
responds with a three-part protocol family (paper §4.1.3 inference-protocol bullets):

* **Reflection protocol (`REACT_REFLECTION_PROTOCOL=true`, default on, config.py:L288).**
  Append a *Forecasting Protocol* to the end of each sample's user message: decompose the
  question → list ≥3 different retrieval angles → reflect after each search →
  cross-validate → check the opposite direction → state confidence.
* **Budget-awareness protocol (`REACT_BUDGET_AWARENESS_PROTOCOL=true`, default on,
  config.py:L313).** Front-load "total step count + total search count" in the prompt so
  the model can plan holistically and reserve the final step for emitting `\boxed{...}`.
* **Forced finalisation near the limit (`REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=true` with
  `LOOKAHEAD=2`, default on, config.py:L314–L315).** As the loop approaches its limit,
  user messages are actively injected: a soft reminder at the second-to-last step where
  tool calls are still permitted, and a hard cutover at the final step with an empty tool
  list, forcing content-only output of `\boxed{...}`.

These protocol additions are **not written into `dataset_metadata`**, so
`prompt_templates_hash` does not change; their existence is persisted through
`run_meta.config_snapshot` alongside each sample's `user_prompt` field, enabling
per-question diffing after the fact.

A complementary fallback — `REACT_MIN_SEARCH_CALLS` (soft minimum search count, default
`0`= off, config.py:L292) — exists for the rare case where prompt guidance alone cannot
pull a model off one-shot direct answers; when on, the system injects a user nudge asking
it to try a different angle and search again, with the per-sample nudge count capped by
`REACT_MAX_NUDGES`.

Design philosophy:

* **Prompt first, rules second.** The protocol family is "guidance"; the nudge is
  "restriction". First try better guidance to make the model walk a few more steps
  spontaneously, only impose a soft floor when the model still insists — to avoid mixing
  "the capability under evaluation" with "the system's enforcement".
* **Toggleable, comparable.** All switches have clear defaults, and turning them off
  degrades to the historical behaviour (the same code can run "protocol on vs off"
  controlled experiments).
* **Auditable.** The protocol text and nudges both appear in `messages_trace`; the on/off
  state is anchored by `config_snapshot`, so this is **not implicit behaviour**.

**Rejected alternative: hard-floor `REACT_MIN_SEARCH_CALLS` by default.** Conflates
capability (does the model search enough?) with enforcement (we made it search). Default
0 keeps the floor opt-in; the reflection protocol drives natural search depth.

### 7.4 v5.1 harness-resilience: the four-knob priority chain

For cross-model comparisons, **`parse_failure_rate` must reflect only the model's own
format failure, not upstream resource exhaustion in the harness.** In v5.0, after
`REACT_MAX_SEARCH_CALLS` was exhausted the `web_search` schema was still exposed to the
LLM; the model kept asking for the tool and hit the `REACT_MAX_STEPS` ceiling, with
`final_raw=""` becoming parse_ok=0 directly. That disguised "tool starvation" as "format
failure".

v5.1 added four orthogonal switches at `react.run_react` (react.py:L248–L340, with the
priority decision logic at L272–L334), each defending a different failure mode:

| # | Switch                                    | Default | What it does                                                                |
| - | ----------------------------------------- | ------- | --------------------------------------------------------------------------- |
| 1 | `REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT`     | True    | Last step → `tools=[]` + force-finalise text (graded by LOOKAHEAD)         |
| 2 | `REACT_BUDGET_EXCEEDED_DROP_TOOLS`        | True    | Once $\ge C$ searches done → subsequent rounds get `tools=[]` (in-loop)    |
| 3 | `REACT_FINAL_ANSWER_RETRY`                | False   | Loop ended cleanly but `final_raw=""` → one extra LLM call with `tools=[]` |
| 4 | `REACT_MIN_SEARCH_CALLS` / `MAX_NUDGES`   | 0 / 2   | Opt-in soft floor, nudge user message when below floor                     |

The four are evaluated each iteration in **strict priority order** (react.py:L266: "Priority
is (1) > (2) > (3) > (4)"):

1. **Last-step hard cutoff.** Detected via
   `(REACT_MAX_STEPS - step) <= REACT_FORCE_FINAL_ANSWER_LOOKAHEAD` with `remaining == 1`
   (react.py:L272–L296). Supersedes everything else; the model can ONLY emit content this
   turn.
2. **Penultimate soft warning.** Same `force_final_active` flag with `remaining ∈ [2,
   LOOKAHEAD]` (react.py:L297–L309). Tools still exposed; the warning text branches
   internally on whether the search budget is already spent.
3. **Budget-exhausted commit notice.** Fires once when
   `searches_done_now >= REACT_MAX_SEARCH_CALLS` AND
   `REACT_BUDGET_EXCEEDED_DROP_TOOLS=True` (react.py:L310–L319). After this fires, every
   subsequent round gets `tools=[]` regardless of which other branch fires.
4. **Continuation reminder.** Lowest priority: previous turn was content without `\boxed{...}`
   and nothing else needs to fire (react.py:L320–L333). Replaces the historical inline
   "Harness: step N complete" injection that could double-inject with later branches.

The new analysis column `final_answer_retry_rate` lands in `per_model_summary.csv`,
letting analysts see "how much the fallback caught" separately and decide, when necessary,
whether to deduct it from the `pass_at_1` denominator. Schema upgraded to v5: each sample
slot in `run_results` adds `s{i}_final_answer_retry_used INTEGER`; old v4 DBs auto-ALTER
ADD via `init_schema` (NULL-compatible).

The default for `REACT_FINAL_ANSWER_RETRY` is **False** — superseded by switch #1
(force-final-answer-near-limit, in-loop), and kept as an optional out-of-loop emergency
backstop. Enabling it costs one extra LLM step (`react_steps + 1`) but does NOT count
toward `nudges_used` (different semantics; nudges are about search depth, this is about
format compliance).

Design philosophy: **make every reason a sample fails to commit visible and separable.**
The four knobs are orthogonal because the failure modes they target are orthogonal; pinning
them to the priority chain prevents two switches from fighting each other.

**Rejected alternative: a single "harness rescue" switch.** Loses ablation discrimination.
Two of the four switches are pure rescue (1, 3); two are pure shaping (2, 4). Collapsing
them would erase the ability to A/B test which intervention actually moved
`parse_failure_rate`.

### 7.5 Graceful degradation for tool-call errors

Within the ReAct loop, several tool-related errors do not interrupt the whole sample:

| Situation                       | Handling                                                                        |
| ------------------------------- | ------------------------------------------------------------------------------- |
| Unknown tool name               | return `unknown tool` to the LLM and let it change tack                          |
| `arguments` JSON parse fails    | send the error back as tool_result; the LLM can retry                            |
| Search budget exhausted         | tool_result returns `search budget exceeded`                                     |
| Tavily itself errors            | go through `SEARCH_BACKOFF_S` retries; if still failing → stuff the error into tool_result |

Design philosophy: **let the LLM "see" its own failures from the system's perspective
rather than papering over for it.** The capability numbers this produces are closer to
reality — a model that cannot handle tool failures should naturally score lower.

### 7.6 "Forbidden words" for reasoning models

Some reasoning models (o1 / o3 / r1 / qwq …) directly return 400 on custom sampling
parameters like `temperature` / `top_p`. The project maintains the substring list
`LLM_REASONING_MODEL_PATTERNS=o1,o3,o4,r1,qwq` in `.env` (default at config.py:L242); for
matching models, those two parameters are **not passed** at call time.

This is a typical "shift maintenance cost forward" design: rather than identifying 400
inside retry / error handling, handle it at request construction time.

---

## 8. Error handling: slicing "failure" into 8 semantics

The error-handling table is `FRAME.md §9`, but the spirit can be condensed into a few
principles.

### 8.1 Not every error should be retried

| Error                       | Retry?               | Reason                                                                         |
| --------------------------- | -------------------- | ------------------------------------------------------------------------------ |
| Network/5xx                 | Yes (per backoff sequence) | Mostly transient                                                          |
| Rate limit                  | Yes (prefer Retry-After) | The provider has told you how long to wait                                  |
| Auth 401/403                | **Stop the entire run** | The key is wrong; retrying is pointless and stopping early saves money        |
| Bad request                 | No                   | Things like `model_not_found` only run after a config change                    |
| Content policy              | No                   | The same prompt sent again returns the same result                              |
| Refusal / parse fail        | No                   | Not an error — it is model behaviour                                            |
| Tavily itself               | Has its own retry sequence | Once exhausted, return the error to the LLM                                |
| Training-cutoff filter      | Not invoked          | Write `skipped_training_cutoff` directly                                        |

### 8.2 Three independent backoff sequences

```bash
LLM_BACKOFF_NETWORK_S=2,5,15,30,60       # config.py:L236
LLM_BACKOFF_RATE_LIMIT_S=10,30,60,120,300 # config.py:L237
LLM_BACKOFF_SERVER_5XX_S=5,15,30,60,120   # config.py:L238
```

Different error types use different backoffs — rate limit is much slower than network,
because the former typically needs minute-level cooldowns while the latter usually clears
in a few seconds. The sequence length also determines the "max retry count"; configuration
is unified in `.env`.

The three sequences are tuned for OpenRouter's behaviour patterns: network errors clear in
seconds (e.g. transient TCP resets), rate limits clear in minutes (provider-side
cool-down), and 5xx surges clear in low minutes (provider-side recovery). Different
providers may need different sequences — that's why each is a separate `.env` knob, not a
single `LLM_BACKOFF_S`.

### 8.3 Error classification codes are first-class citizens of the report

The `error` field is not "fill in a string when something errors" but a fixed finite enum:
`network` / `server_5xx` / `bad_request` / `content_policy` / `skipped_training_cutoff`
(`errors.ErrorKind`).

`error_breakdown.csv` slices directly by this classification. Design philosophy: **every
failure behaviour must be categorisable and aggregatable in the report** — an
`error="something went wrong"` is useless.

### 8.4 v5.1 harness-resilience: classification boundary expansion

Two common misclassifications encountered during cross-provider evaluation:

* **Aliyun content moderation (`data_inspection_failed`) mis-bucketed as `bad_request`.**
  v5.0's `_body_matches` only recognised English needles like `content_policy /
  content_filter / safety`; the `code=data_inspection_failed` returned by DashScope
  (`https://dashscope.aliyuncs.com`) fell through to the catch-all `bad_request`. v5.1
  unified the needle list under `errors.CONTENT_POLICY_NEEDLES`, adding
  `data_inspection_failed` / `inappropriate content` / `sensitive`; on match → classify
  as `content_policy`, preserving the `MUST NOT retry` semantics.
* **Remote disconnect `RemoteProtocolError` mis-bucketed as `unknown`.** v5.0's network
  exception tuple only listed `ConnectError` / `ReadTimeout` / `ConnectTimeout` /
  `WriteTimeout`; `httpx.RemoteProtocolError` ("Server disconnected without sending a
  response.") fell into `UNKNOWN`, and the entire sample failed without retry. v5.1
  expanded the network exception family to align with httpx's existing `NetworkError`
  subset: `+RemoteProtocolError / +WriteError / +PoolTimeout`, with parallel expansion on
  the LLM side (`errors.classify`) and the Tavily side (`search._single_request`).

Design philosophy: **a misclassified error is silently miscounted in the report.** The
v5.1 expansion was driven by paper §4.6 cross-provider observations where these two
patterns surfaced once each per ~2K samples — small numerically, but fatal to honest
reporting because they would otherwise tip the `bad_request` vs `content_policy` ratio.

---

## 9. Configuration as contract: `.env` is the single source of truth

### 9.1 Almost every tunable is in `.env`

The CLI exposes only three flags — `--question-type` / `--choice-type` / `--skip-analysis`;
everything else goes through `.env`.

Reasoning:

* **Easy to re-run.** A single `.env` is enough to reproduce the entire configuration; CLI
  flags scattered in shell history are easily lost.
* **CI/scheduler-friendly.** Scripted execution generally prefers managing a file rather
  than a command line.
* **Config ↔ DB self-consistency.** `config_snapshot` is written into `run_meta`, so
  reviewing a run later tells you "what its `.env` looked like at the time" (after
  redaction).

### 9.2 OpenAI-compatible endpoint: horizontal compatibility

`LLM_BASE_URL` accepts any OpenAI-compatible endpoint: OpenRouter, Aliyun Bailian, OpenAI,
DeepSeek, SiliconFlow, local vLLM all work.

Design philosophy: **the integration surface should be small and standard.** OpenAI's chat
completion + function calling protocol has become the de facto standard; this project does
not build a provider-adaptation layer, but pushes adaptation responsibility to the
endpoint. The paper's six-model comparison spans across providers (DeepSeek, Z.ai, Alibaba,
MiniMax, Moonshot, ByteDance) precisely because of this neutrality.

### 9.3 Training-cutoff config is quality config

`MODEL_TRAINING_CUTOFFS=openai/gpt-5=2024-10-01,...` (config.py:L224) is not optional — it
is **part of evaluation fairness**. The docs explicitly recommend declaring an explicit
cutoff for every model under evaluation; an unspecified model is not filtered (with a
warning). The paper takes the most conservative interpretation: when a model card discloses
cutoff only at month-level granularity, adopt the *last day* of the disclosed month as
$\kappa_M$ (paper §4.1.2 footnote).

### 9.4 Startup validation as an enforcement layer

`Settings._post_validate` runs at process start and aborts before any LLM/Tavily call if
any of the following fail:

* `:online` slug or `::` substring in `MODELS` or `LEAK_DETECTOR_MODEL` (config.py:L599–L614);
* `MODEL_TRAINING_CUTOFFS` parse failures (config.py:L181–L202);
* `REACT_MAX_SEARCH_CALLS` empty list (config.py:L580+);
* `SOURCE_TABLE` failing the `^[A-Za-z_][A-Za-z0-9_]*$` whitelist (config.py:L586–L595);
* `TAVILY_API_KEY` empty when `ENABLE_WEB_SEARCH=True`;
* `LEAK_DETECTOR_API_KEY` / `LEAK_DETECTOR_MODEL` empty when `ENABLE_SEARCH_LEAK_FILTER=True`;
* `COMPOSITE_WEIGHTS_QTYPE` / `COMPOSITE_WEIGHTS_CTYPE` containing unknown bucket names or
  all-zero weights (config.py:L515–L535);
* `GRID_DEFAULT_R` / `GRID_DEFAULT_C` not in their respective lists.

The full set is enumerated in FRAME §7.1. Design philosophy: **fail fast, before any
billable call.** An evaluation that would have been silently miscounted is much more
expensive than an evaluation that didn't start.

Pinned by `test_config.py` (each rule has at least one fixture that exercises both the
accept and reject path).

---

## 10. Testing: shifting expensive failures to cheap local runs

### 10.1 Tests must not hit the network or burn the API

A complete run of the example dataset × number of models × N samples is **tens to hundreds
of dollars**. Tripping over a prompt / parser / schema bug at that scale just wastes the
money.

Core constraints of the test design:

* `tavily-python` must not actually send requests → `respx` mocks httpx
* The OpenAI client must not actually send requests → fixture replacement
* SQLite uses a temporary directory → `tmp_path` fixture
* The dataset must be small yet "look real" → use a few real questions from the source DB
  as fixtures

### 10.2 Five CI red lines

```text
test_prompts / test_parser / test_training_cutoff /
test_llm_no_browsing / test_analysis
```

These five must always be green. They cover the parts of the project most likely to
"silently break", and they are precisely the components that realise the framework
$\mathcal{R}$:

| Test                    | Invariant guarded                                        | Framework component     | If it breaks                                                                  |
| ----------------------- | -------------------------------------------------------- | ----------------------- | ----------------------------------------------------------------------------- |
| `test_prompts`          | prompt template rendering correct for all three question_types | $R$               | The `user_prompt` text drifts; `prompt_templates_hash` no longer pins inputs |
| `test_parser`           | letter parsing and strict-equality scoring               | $\Psi$, $\phi$          | Items mark "wrong" while letters actually match; or vice versa                |
| `test_training_cutoff`  | training-cutoff filtering semantics and resume priority  | $\kappa_M$ admissibility | Cutoff-skipped questions get billed; or completed rows get rebilled           |
| `test_llm_no_browsing`  | provider-native browsing is never silently turned on     | information barrier     | The whole evaluation contract becomes invalid                                |
| `test_analysis`         | report numbers reconcile with the raw DB                 | $\Gamma$                | The CSV / MD numbers diverge from what the raw observations support          |

Design philosophy: **pick the invariants whose breakage would be expensive, and use unit
tests as sentinels.** Each red line corresponds to one component of $\mathcal{R}$ that, if
broken, makes the entire run unit invalid. The "if it breaks" column is deliberately blunt
— this is the cost of skipping the test, not the cost of running it.

The full mapping (33 tests → 11 framework components) is in FRAME §15.1.

### 10.3 dry-run smoke test

`test_smoke_dry_run.py` replaces OpenRouter + Tavily with httpx stubs and runs an
end-to-end pipeline of 3 questions × 1 model × 1 sample. It does not validate logic
details — it validates "is the pipe still flowing": schema, wide table, `messages_trace`
JSON, `search_calls` fields all present.

This expresses the e2e/unit test split: unit tests validate "local correctness", smoke
tests validate "integration doesn't blow up".

### 10.4 Tests as documentation

Every paper-level invariant has a test. When the prose in paper / FRAME / DESIGN drifts
from the code, the test is the tie-breaker. The 33 test files in `tests/` (~13K LOC, all
offline) are the most authoritative form of "what this codebase actually does"; if you
can't reconcile what you read here with what a test asserts, the test wins.

---

## 11. Observability: every sample is traceable

### 11.1 Progress log

```text
12:03:44 | INFO | [run=20260424-120344-a7k3] [5/1610] q=69566c13 qt=binary_named ct=single model=openai/gpt-5 sample=2/5 correct=True steps=4 tool_calls=3 latency=8421ms
```

Every log line carries: question id, question_type, choice_type, model, sample_idx,
correctness, step count, tool-call count, latency.

Design philosophy: **a single log line should fully describe what path a sample took
through the system.** Reading the log is reading the trace — no DB join required.

### 11.2 `messages_trace` and `search_calls`

The DB stores two JSON blobs directly:

* `messages_trace`: the complete ReAct message sequence (LLM replies, tool_call,
  tool_result).
* `search_calls`: each `web_search` call's query, end_date, result count, per-result
  published_date, plus (when the leak filter is on) `n_results_raw` / `n_results_kept` /
  `detector_verdicts` / `detector_latency_ms` / `detector_error_kind`.

These are large (~80% of the DB), so the `WRITE_MESSAGES_TRACE=false` switch is provided.
But the default is on — reasoning: **the value of debugging one failure far outweighs the
few extra MB of disk.** And without the trace, the leakage audit cannot be redone post-hoc.

The detector audit fields go into `search_calls`, never into `messages_trace` — pinned by
leak_filter.py:L25-L27 and asserted by `test_leak_filter.py`. This is intentional: the
detector verdict is *audit metadata*, not LLM-visible content; if it were in
`messages_trace` an unsuspecting downstream model could see it and bias its behaviour.

### 11.3 `loguru` + structured + dual output

stderr (for humans) + rotating file (for machines), two channels; rotation 100 MB /
retention 5. Design philosophy: **humans and machines have different needs when reading
logs — serve them separately.**

---

## 12. Evolution path: spec-first changes driven by openspec

The repo root contains `openspec/changes/`, where changes are recorded in spec form.
`bootstrap-forecast-eval` is the initial bootstrap record. Subsequent landmark changes —
`react-tavily-grid-search`, `harness-resilience-v1`, `search-leak-filter-v1`,
`add-exam-score-metric`, `composite-score-by-subtype`, `discrete-native-analysis-v5` —
each ship with a `proposal.md` (motivation), `design.md` (decision archive),
`specs/.../spec.md` (capability deltas), and a `tasks.md` (implementation checklist).

Design philosophy:

* **Write the spec before the code.** Avoid "discovering the design is wrong only after the
  code is merged".
* **Change archive and code diff coexist.** When reviewing the architectural evolution
  later, you can see "why we changed it", not just "what was changed".

### 12.1 Grid search via virtual slug (option C)

`react-tavily-grid-search` extends the *(Q × M × N)* three-axis space to *(Q × M × R × C ×
N)*, but **without** schema upgrade and **without** touching the runner core loop. The
method: at the evaluation entrypoint encode each `(real_model, R, C)` triple as a **virtual
model slug** `{real}::r{R}::c{C}` (`db.compose_virtual_slug` at db.py:L477; reverse via
`parse_virtual_slug` at db.py:L500); runner / DB / analysis main pipeline treats it as an
opaque string — existing artefacts naturally expand into multiple rows by virtual slug,
while the new module `forecast_eval/analysis/grid.py` decodes the triple, re-aggregates,
and emits paper long tables and figures. Full decision archive in
`openspec/changes/react-tavily-grid-search/design.md`; the 10 key decisions:

| ID  | Decision                                                                                     |
| --- | -------------------------------------------------------------------------------------------- |
| D1  | Pick option C (virtual slug + per-task settings view); reject A (single run, multi-(R, C) DB — schema v5 rewrite) and B (one run_dir per cell — `runs/` bloat + complex cross-run aggregation) |
| D2  | Virtual slug uses `::r{R}::c{C}` suffix; `db.model_slug_safe` replaces `::` with `_` to land an fs-safe filename `openai__gpt-5__r5__c3.db`; the regex `^(?P<real>.+?)::r(?P<R>\d+)::c(?P<C>\d+)$` non-greedy captures real_model |
| D3  | `runner.Task` carries a cell-local `settings: Settings`; the dispatcher derives an immutable sub-view via `model_copy(update={...})`; `react.py` / `search.py` are byte-unchanged |
| D4  | Only raise when `REACT_MIN_SEARCH_CALLS > min(C_list)`; for a cell with `C < MIN`, silent clamp `effective_min = min(MIN, C)` and record it under `run_meta.config_snapshot.grid_origin` for audit |
| D5  | `run_meta.config_snapshot` writes **single-valued** R/C; add a `grid_origin = {real_model, R, C, effective_min_search_calls}` sub-key; manifest top-level adds a `grid` block (`r_list / c_list / default_r / default_c / real_models / n_cells`) so the analysis layer doesn't have to decode the triple per .db |
| D6  | `manifest.models` / `manifest.model_files` field semantics remain "list of virtual slugs"; the new `grid.real_models` is a deduped real-slug convenience field — v4 analysis main path's contract of "read `manifest.models` as the db file list" is preserved |
| D7  | `analysis/__init__.py::run_analysis` main path is **zero-intrusive**; append a `grid.run_grid_analysis(...)` at the end wrapped in `try/except` (same best-effort pattern as reflection A/B), failures do not interrupt the existing pipeline |
| D8  | Grid CIs all go through `inference.paired_bootstrap` (5000 resamples, seed=42); BI-domain CIs are obtained via "BS-domain paired bootstrap + monotone transform $\mathrm{BI}=100(1-\sqrt{\mathrm{BS}})$" — **no** new statistical code introduced |
| D9  | Pareto frontier's cost dimension defaults to `mean_search_calls` (actual mean search count, more honest than the C ceiling), with `mean_latency_ms / C` fallback allowed; y-axis defaults to `bi_mean`, with `nll_mean` (minimisation direction) as an option |
| D10 | Fig 1 main figure pins `R = GRID_DEFAULT_R` with one curve per real_model; other R values each get a same-format appendix figure to avoid main-figure unreadability after stacking M·\|R\| curves |

The three PRs `Phase 0 / 1 / 2` ship sequentially — each phase passes `pytest -q` and
`openspec validate --strict`, and after deleting the phase's own code the system is
equivalent to the previous phase's completed state (Rollback Strategy). Single-value `.env`
parses under the new code as a length-1 list → Cartesian product produces a single virtual
slug, with the **only** visible difference being the `__r{R}__c{C}` suffix on the .db
filename; for legacy v4 runs (manifest without a `grid` block), grid analysis and the grid
figure family early-exit altogether — zero intrusion.

### 12.2 Why phase-gated rollouts

Each openspec change ships as a sequence of phases (typically `Phase 0 schema → Phase 1
code → Phase 2 docs/CSV columns`). The discipline is:

* **Each phase passes CI on its own.** No "one big PR with everything"; CI catches the
  schema/code mismatch on the wrong side of each gate.
* **Each phase is reversible.** Deleting the phase's own code returns the system to the
  previous phase. This is checked by re-running tests after a hypothetical revert.
* **The change archive accumulates.** `openspec/changes/archive/` is the project's own
  ledger of "why we got here"; it is more durable than git-log because each entry has a
  spec + design + tasks separation that survives squash-merge.

---

## 13. Engineering vs framework: knowing which knob you're turning

A repeating confusion in cross-team conversations: which knobs are "part of the
evaluation contract" and which are "engineering tuning that doesn't affect the scientific
claim". The distinction matters because changing a contract-knob invalidates cross-run
comparability, while changing an engineering-knob does not.

### 13.1 Contract knobs (changing these makes runs incomparable)

Every entry below corresponds to a fingerprint or a $\mathcal{R}$-tuple field; all are
written to `run_meta` and any change shows up as a hash mismatch on a paired comparison.

| Knob                              | Why it's a contract                                                  |
| --------------------------------- | -------------------------------------------------------------------- |
| `SOURCE_DB` + `SOURCE_TABLE`     | Defines $\mathcal{D}$; fingerprint via `source_db_hash`               |
| `MODEL_TRAINING_CUTOFFS`         | Defines $\kappa_M$; per-model admissibility                          |
| `TAVILY_END_DATE_OFFSET_DAYS`    | Defines $\delta$, hence $\chi_i$                                     |
| `REACT_MAX_STEPS`                | Defines $T$                                                          |
| `REACT_MAX_SEARCH_CALLS`         | Defines $C$ (grid axis)                                              |
| `TAVILY_MAX_RESULTS`             | Defines $R_{\mathrm{tav}}$ (grid axis)                               |
| Prompt templates (8 keys)        | Defines $R$; fingerprint via `prompt_templates_hash`                 |
| Reflection protocol text + on/off | Defines part of $F_M$; fingerprint via `reflection_protocol_hash`    |
| Belief protocol text + on/off    | Defines part of $F_M$; fingerprint via `belief_protocol_hash`        |
| Detector prompt + version        | Defines $H_{\mathrm{aux}}$; fingerprint via `leak_detector_prompt_hash` |
| Composite weights                | Defines $\Gamma$ (default subtype-weighted form); recorded in `run_meta` |
| `SAMPLING_N`                     | Defines $S$; pinned at table-creation time                           |

### 13.2 Engineering knobs (changing these is fine within one comparison)

Every entry below is purely about throughput, cost, or robustness; none affects $\mathcal{R}$.

| Knob                              | Why it's engineering                                                |
| --------------------------------- | ------------------------------------------------------------------- |
| `LLM_MAX_CONCURRENCY`             | Throughput; numerically irrelevant to outcomes                       |
| `LLM_BACKOFF_*`                   | Resilience under provider-side noise; a long-enough sequence converges |
| `SEARCH_RETRY_MAX` / `_BACKOFF_S` | Same as above for Tavily                                             |
| `LEAK_DETECTOR_RETRY_MAX` / `_BACKOFF_S` | Same as above for the detector                                |
| `LEAK_DETECTOR_CONCURRENCY`       | Throughput on the detector stage                                     |
| `DB_COMMIT_BATCH`                 | Disk-write batching; numerically irrelevant                          |
| `WRITE_MESSAGES_TRACE`            | Disk-size knob; affects post-hoc audit, not numerics                 |
| `LOG_LEVEL` / `LOG_DIR`           | Logging verbosity / location                                         |
| `RUNS_ROOT`                       | Where artefacts land; doesn't change what they are                   |

The contract / engineering split matters because reviewing a PR or a `.env` change becomes
mechanical: any change in §13.1 is a *new evaluation*, any change in §13.2 is a *bug fix or
tuning*. The runtime fingerprint set (§5.6) makes this distinction observable on any pair
of runs.

---

## 14. Rejected alternatives index

A consolidated list of every alternative considered and rejected in this codebase, sorted
by decision area. Each row references the §-section where the rationale is stated in
detail.

| Decision area                | Rejected alternative                                       | Reason for rejection                                          | §       |
| ---------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------- | ------- |
| Tool-mediated boundary       | Expose `end_date` as an LLM tool argument                  | Trusts the model not to widen $\chi_i$; pin-test surface explodes | 2.1     |
| Tool-mediated boundary       | Rewrite query string to insert date filter                  | Brittle string-pass; providers can ignore inline date operators | 2.1     |
| $\delta$ default             | $\delta = 0$ (use resolution day)                          | Catches 30–50% of same-day news leakage on example DB; too lax | 2.2     |
| $\delta$ default             | Per-question $\delta_i$                                    | Breaks "one $\delta$ defines one evaluation" contract           | 2.2     |
| Provider-native browsing     | Warn instead of refuse                                     | Warnings get filtered; refusals stop the run                   | 2.3     |
| Provider-native browsing     | Only enforce at startup                                    | Bypassable via `model_copy(update={...})`; per-call re-check needed | 2.3 |
| Cutoff filter                | Weighted exclusion (discount close-to-cutoff samples)      | Adds analytic complexity; binary in/out cleaner                 | 2.4     |
| Cutoff filter                | Skip dataset-wide if any model fails admissibility         | Discards 10–20% of corpus on heterogeneous panel               | 2.4     |
| Discrete answer space        | Open-ended NL outputs scored by LLM judge                   | Re-introduces contamination risk; non-deterministic scoring     | 3.1     |
| Discrete answer space        | Numerical probability outputs scored by Brier/NLL only      | At K=5 the empirical $\hat p$ is too discrete (paper §4.4.2)   | 3.1     |
| > 26 options labelling       | Skip > 26-option questions                                 | Loses dataset coverage; round-trip test mitigates              | 3.4     |
| Parse failure handling       | Retry on parse failure                                     | Capability masking + cost; same answer expected on retry        | 3.5     |
| DB layout                    | Long table + composite key                                 | Loses simple resume queries; multi-row writes break orchestration | 6.1   |
| DB layout                    | One DB with `model` column                                 | Single-writer contention; one slow provider stalls all          | 5.3     |
| DB layout                    | Store paths instead of copies                              | Breaks self-contained property                                  | 5.4     |
| Concurrency                  | Lock-per-INSERT                                            | Burns CPU on contention; orchestration cheaper                  | 6.4     |
| Composite weights            | Equal weights across buckets                               | Erases discriminative buckets; doubles up near-random ones      | 4.4     |
| Composite weights            | Empirical-prevalence weights                               | Drowns out signal-bearing buckets                                | 4.4     |
| Reflection                   | Hard-floor `REACT_MIN_SEARCH_CALLS` by default              | Conflates capability with enforcement                            | 7.3     |
| Harness resilience           | Single "harness rescue" switch                             | Loses ablation discrimination across the four orthogonal modes  | 7.4     |
| Reproducibility              | Single composite hash                                       | Loses ablation discrimination on {template, reflection, belief} | 5.6     |
| Reproducibility              | Ship DB on separate registry                                | Adds network dependency to reproduction                          | 5.1     |
| System message               | Split system / user                                         | Loses cross-provider consistency                                 | 7.1     |

---

## 15. Summary of design-consistency principles

Condensing the full document's design philosophy into a single set of principles, placed at
the end for review reference:

1. **Boundary at the data layer, not the prompt layer.** Sample admission ($\kappa_M$),
   temporal masking ($\delta$, $\chi_i$), and detector ($H_{\mathrm{aux}}$) all live in
   evaluator-controlled code, not in the model's instructions.
2. **Honesty > prettiness.** The threat model declares what we cannot control in plain
   language; the leakage audit publishes its Wilson upper bound rather than just the point
   estimate.
3. **Skip ≠ fail.** Actively excluded samples are categorised independently and do not
   pollute the error rate.
4. **Raw > aggregated.** The DB stores observations only; statistics are deferred to
   `analysis/`. Metric definitions evolve faster than DB schemas.
5. **Strict by default, partial credit by design.** The headline metric is strict frozenset
   equality; the composite uses exam-style partial credit; FSS adds chance correction. All
   three coexist on the same raw samples.
6. **Reproducibility > convenience.** Source data goes into Git; each DB is self-contained;
   six-part hashes pin down the fingerprint.
7. **Observability > elegance.** Full `messages_trace` is on by default; the progress log
   is one line per sample; per-call audit fields persist detector verdicts.
8. **Categorise failures.** Errors use a finite enum; every kind has its own cell in the
   report.
9. **Config as contract.** `.env` alone decides everything; CLI flags are minimal;
   `config_snapshot` is redacted before persistence; the contract / engineering split (§13)
   tells you which `.env` change invalidates a comparison.
10. **Tests guard the expensive.** Five CI red lines map one-to-one to components of the
    run unit $\mathcal{R}$; dry-run smoke tests validate integration; expensive failures
    are shifted to local.
11. **The framework is the contract.** Every engineering decision is judged by whether it
    strengthens or weakens $\mathcal{R}$; convenience wins over the framework only in
    clearly bounded escape hatches (`LEAK_DETECTOR_FAIL_ACTION=keep`,
    `WRITE_MESSAGES_TRACE=false`, A/B switches), and even those leave audit trails.
12. **Fail fast, before billable calls.** Startup validation rejects misconfigurations
    before any LLM/Tavily contact; an evaluation that didn't start is cheaper than one
    silently miscounted.
13. **Eliminate races via orchestration, don't solve them with locks.** One writer per DB,
    one queue per writer, batched commits.
14. **Phase-gated changes.** Every openspec change ships in reversible phases; the change
    archive is the project's ledger of "why we got here".
15. **Three independent fingerprints, not one.** Templates, reflection protocol, belief
    protocol have separate hashes so ablation studies can pair runs along one axis.

---

## 16. Reading roadmap

If you are new to the project, we suggest reading in this order:

1. `README.md` — figure out in 10 minutes what OracleProto is and how to run it.
2. This document (`DESIGN.md`) — understand the motivation behind each trade-off.
3. `paper/main.tex` §§1–3 — read the formal framework; sit with the tuple
   $\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)$ until each
   symbol has a clear name in your head.
4. `FRAME.md` — the complete spec at field, interface, and pseudocode level. Use the §1.1
   grand map as your symbol-to-code lookup table whenever DESIGN says "see file:line".
5. `forecast_eval/prompts.py` + `forecast_eval/parser.py` — the renderer $R$ and parser
   $\Psi$; these two files are practically the "heart" of the project.
6. `forecast_eval/runner.py` + `forecast_eval/react.py` — orchestration and the ReAct loop.
   Pay special attention to react.py:L248–L334 (the four-knob priority chain).
7. `forecast_eval/leak_filter.py` + `forecast_eval/search.py` — the temporal masking
   implementation and the Stage-2 detector. The prompt template at leak_filter.py:L55–L92
   is what enforces the "no question fields" whitelist.
8. `forecast_eval/analysis/` — the metric layer. Start with `exam_score.py` (single-file,
   self-contained, reads in five minutes), then `accuracy.py` (Tversky / FSS / Cohen κ),
   then `composite.py` (subtype-weighted aggregation), then `consistency.py` (Fleiss κ /
   VCI / MVG).
9. `tests/` — read tests to reverse-engineer the contracts. The five CI red lines (§10.2)
   are the highest-priority entry points.
10. `openspec/changes/archive/` — to find out *why* things became what they are today, come
    here. Each change has `proposal.md` (motivation) + `design.md` (decision archive) +
    `specs/.../spec.md` (capability deltas) + `tasks.md` (implementation checklist).

The order is engineered: each step gives you the language for the next. Steps 1–4 build
the conceptual model; steps 5–8 ground it in code; steps 9–10 give you the change history
and the test contract.

---

> **Summed up in one sentence:**
> OracleProto uses engineering discipline to safeguard scientific rigour — every seemingly
> excessive constraint exists because the alternative is a number in the final report that
> doesn't actually mean anything. The information boundary is part of the data, not part
> of the prompt; the run unit $\mathcal{R}$ is the contract, not a configuration; the audit
> trail is the report's foundation, not its appendix.
