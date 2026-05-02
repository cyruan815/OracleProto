<div align="center">

<h1>OracleProto</h1>

<em>A reproducible framework for benchmarking LLM native forecasting via knowledge cutoff and temporal masking</em>

[English](./README.md) | [中文文档](./README-ZH.md)

</div>

OracleProto reconstructs resolved events into time-bounded forecasting samples whose
evaluation lives at the dataset level, so a run is auditable, replayable, and comparable
across models and across calendar years.

> **In one sentence.** The codebase turns a forecasting evaluation into a single reproducible
> run unit $\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)$ that
> fixes every input from questions through aggregation rules, and emits scoring artefacts
> whose bytes match whenever the configuration matches.

This README is a ten-minute orientation. It explains what the project is, how to run it, and
where each artefact comes from. For the rationale behind every constraint, read `DESIGN.md`.
For the field-level specification mapping each symbol to a module, a database column, and a
pinning test, read `FRAME.md`.

---

## 1. The problem

Existing forecasting evaluations sit on unstable middle ground. Prospective live benchmarks
such as ForecastBench and FutureX are contamination-controlled by construction, yet they
evaporate the moment events resolve, so the leaderboard becomes a one-way temporal stream
rather than a reusable artefact. Retrospective benchmarks such as FutureX-Past or archived
live questions are reproducible, yet they readily mistake factual recall for forecasting
capability, since by evaluation time the answer already sits in the model's training corpus.

Prompt-time discipline of the form "imagine you do not know that the election has resolved"
cannot bridge this gap. Independent surveys empirically document a substantial systematic
gap between simulated ignorance and true ignorance, and show that a 1–5% label-noise rate
alone is enough to break proper scoring rules. The same conclusion arrives from the
inference side: a single-inference defence does not generalise across runs, so the
discipline must live one level deeper, inside the dataset itself.

OracleProto pushes that discipline into the dataset schema. A question is admitted for model
$M$ only when its prediction cutoff $\chi_i$ satisfies

$$\kappa_M \le \chi_i < \tau_i,$$

where $\kappa_M$ denotes the model's training cutoff and $\tau_i$ the event-resolution time.
The model's parametric knowledge is therefore no more recent than the permitted prediction
environment, while the resolution time has not yet arrived in the simulated information
state. Inadmissible questions are not counted as model errors. They are filtered out at
`runner.build_task_plan` (runner.py:L132), audited separately, and pinned by
`tests/test_training_cutoff.py`.

---

## 2. The framework

OracleProto rests on two artefacts: a single run unit $\mathcal{R}$ that names every input to
the evaluation, and a four-channel information boundary that controls every path by which the
model could learn the answer.

### 2.1 The run unit $\mathcal{R}$

Every invocation of `evaluation.py` materialises a single run unit
$\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)$. Each symbol
resolves to one configuration knob, one code path, and one pinning test.

| Symbol             | Object                       | Config / code path                                                                                          | Pin test                          |
| ------------------ | ---------------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------- |
| $\mathcal{D}$      | Discrete forecasting dataset | `SOURCE_DB` / `SOURCE_TABLE` (config.py:L391/L395); `loader.sync_questions` (loader.py:L77)                  | `tests/test_db.py`                |
| $M$                | Evaluated model              | one entry of `MODELS` (config.py:L223); one SQLite per model under `runs/{run_id}/db/`                       | `tests/test_runner_grid_model.py` |
| $\kappa_M$         | Knowledge cutoff             | `MODEL_TRAINING_CUTOFFS[M]` (config.py:L224); admissibility filter at `runner.build_task_plan` (runner.py:L132) | `tests/test_training_cutoff.py`   |
| $\delta$           | Temporal masking offset      | `TAVILY_END_DATE_OFFSET_DAYS` default `-1` (config.py:L273); injected at the tool layer in `react.py`        | `tests/test_search.py`, `tests/test_react.py` |
| $T$                | Max ReAct steps              | `REACT_MAX_STEPS` default `12` (config.py:L279); outer loop `react.run_react` (react.py:L162)                | `tests/test_react.py`             |
| $C$                | Max search calls             | `REACT_MAX_SEARCH_CALLS` default `[8]` (config.py:L283); budget gate (react.py:L276–L279)                    | `tests/test_react.py`             |
| $R$                | Input renderer               | `forecast_eval/prompts.py::render_user_prompt`                                                              | `tests/test_prompts.py`           |
| $\Psi$             | Output parser and validity   | `forecast_eval/parser.py::parse_answer` (parser.py:L40)                                                     | `tests/test_parser.py`            |
| $\phi$             | Answer normalization map     | letter encoding `A` or `A,B` etc. defined per `question_type`; `parser.parse_gt` (parser.py:L92)             | `tests/test_parser.py`            |
| $\Gamma$           | Aggregation rule             | `forecast_eval/analysis/*` (composite accuracy, FSS, κ, BI, …)                                              | `tests/test_analysis.py`          |
| $H_{\mathrm{aux}}$ | Auxiliary leakage detector   | `leak_filter.filter_search_result`; logged in `run_meta.config_snapshot` rather than inside the $\mathcal{R}$ tuple itself | `tests/test_leak_filter.py` |

The auxiliary detector $H_{\mathrm{aux}}$ lies outside the formal tuple by design. It is a
replaceable empirical engineering layer that supports the boundary, not a primitive
component of the forecasting system. Its prompt SHA-256 is stored in
`run_meta.config_snapshot.leak_detector_prompt_hash`, so the leakage barrier is itself
byte-reproducible (`leak_filter.py:L55–L104`).

The information visible to model $M$ on question $q_i$ is

$$\mathcal{I}_{i,M}^{\mathrm{vis}} = \mathcal{K}^{M}_{\le\kappa_M} \cup \mathcal{T}_{\le\chi_i},$$

where $\mathcal{K}^{M}_{\le \kappa_M}$ is the parametric knowledge available before the
model's cutoff and $\mathcal{T}_{\le\chi_i}$ is the temporally masked external information.
The forecasting system $F_M$ produces $\widehat{Y}_{i,M} = F_M(q_i^{\mathrm{in}}; \mathcal{I}_{i,M}^{\mathrm{vis}})$
with $\widehat{Y}_{i,M}\subseteq\mathcal{A}_i$. The time-masked discrete forecasting loop is
implemented in `react.run_react` (react.py:L162).

For the field-level grand map mapping every symbol to a module, a DB column, and a test, see
`FRAME.md` §1.1.

### 2.2 The four-channel information boundary

Residual leakage is decomposed into three controlled channels and a fourth provider-side
residual. The three controlled channels are parametric, tool-mediated, and
retrieval-content; the fourth lies outside the evaluator's control. Each channel maps to a
mechanical defence with declared coverage.

| Channel                                       | Defence layer                                                | Where (code)                                                                                                                                | Default                                  | Residual leakage rate                                                                                       |
| --------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **L0 manual curation**                        | Upstream dataset construction                                | `forecast_eval_set_example.db` curation                                                                                                     | always                                   | 0% (manual annotation floor)                                                                                |
| **L1 parametric (admissibility filter)**      | $\kappa_M \le \chi_i$ check at task generation               | `runner.build_task_plan` (runner.py:L132–L199)                                                                                              | `MODEL_TRAINING_CUTOFFS` per model       | filters parametric-memory leakage upstream                                                                  |
| **L2 tool-mediated (Tavily)**                 | `end_date = \chi_i` injected at the tool layer               | `react._compute_end_date` (react.py:L39); `search.tavily_search`                                                                            | $\delta=-1$ day                          | non-trivial on its own; Tavily index/crawl-date metadata is noisy                                            |
| **L3 retrieval-content (Stage-2 detector)**   | Independent LLM auditor on each Tavily item                  | `leak_filter.filter_search_result` (leak_filter.py:L348)                                                                                    | `claude-sonnet-4.6`                      | drives the per-audit-item residual to the low-single-digit-percent range against L2 alone                   |
| **L4 provider-side residual (declared)**      | Provider-native browsing forbidden                           | `Settings._post_validate` (config.py:L602–L606, L747–L751); `llm._assert_no_browsing` (llm.py:L74–L98); `leak_filter._assert_detector_safe` (leak_filter.py:L139) | always                                   | declared as evaluation bias rather than pretended-away                                                      |

The triple-layer enforcement of L4 first rejects `:online` slugs and the `::` reserved
delimiter at startup, then re-asserts on the wire inside `llm.chat`, and finally duplicates
the same checks in the detector client. Both `tests/test_llm_no_browsing.py` and
`tests/test_config.py` pin the contract. Structurally, L4 is the one defence that must pass
before any billable LLM call leaves the process.

For the threat model and the broader "what we can and cannot control" decomposition, see
`DESIGN.md` §2. For the eight hard constraints derived from $\mathcal{R}$, see `FRAME.md`
§1.2.

---

## 3. Quickstart

### 3.1 Create the conda environment

```bash
conda env create -f environment.yml
conda activate forecast
```

### 3.2 Configure `.env`

```bash
cp .env.example .env
# Edit .env and fill in:
#   LLM_API_KEY (and LLM_BASE_URL: any OpenAI-compatible endpoint such as
#                OpenRouter, Aliyun Bailian, OpenAI, DeepSeek, SiliconFlow,
#                or local vLLM)
#   TAVILY_API_KEY (single value or CSV multi-key for higher quota)
#   LEAK_DETECTOR_API_KEY (Stage-2 auditor; can reuse LLM_API_KEY by leaving
#                          LEAK_DETECTOR_BASE_URL empty; see §8.4)
#   MODELS, MODEL_TRAINING_CUTOFFS: list every model under evaluation and its κ_M
```

Declaring $\kappa_M$ for every model is mandatory for a fair run, since the framework's
admissibility filter is what separates "the model failed to forecast" from "the model already
knew the answer". Models without a declared cutoff are not filtered, a warning is emitted,
and their numbers are not directly comparable to the rest. Cutoffs may be written at month
granularity; the recommended convention is to use the **last day of the disclosed month** as
$\kappa_M$, which is the conservative choice (admits fewer questions, never falsely admits a
question whose answer the model could have memorised).

`Settings._post_validate` (config.py:L598) runs before any LLM call leaves the process. It
fails fast on empty `LLM_API_KEY`, empty `MODELS`, `:online` suffixes, `::` in slugs,
`MIN_SEARCH > min(C)`, disabled `ENABLE_SEARCH_LEAK_FILTER` paired with a present
`LEAK_DETECTOR_API_KEY`, and `GRID_DEFAULT_R/C` outside the configured cells, so a
misconfigured `.env` cannot waste budget.

### 3.3 Run tests (no API calls required)

```bash
pytest tests/ -q
```

The CI baseline is `test_prompts / test_parser / test_training_cutoff /
test_llm_no_browsing / test_analysis`, and these five must stay green. They guard the
renderer $R$, the parser $\Psi$, the admissibility filter $\kappa_M$, the
provider-native-browsing ban from §2.2 L4, and the aggregation rule $\Gamma$, respectively.
The full suite spans 33 test files and roughly 13k lines, covering the v3/v4 DB migrations,
the leak filter, exam-score and composite weights, the grid dispatcher, the react budget
chain, and behavioural diagnostics.

### 3.4 Run an evaluation

```bash
# Smoke: cheapest model, single sample, yes_no only
MODELS=openai/gpt-4o-mini SAMPLING_N=1 \
    python evaluation.py --question-type yes_no

# Full eval with all models, all samples
python evaluation.py

# Filter combinations (AND across flags, OR within each flag)
python evaluation.py --question-type multiple_choice --choice-type multi

# Skip the post-run analysis pass; raw DBs still land in db/
python evaluation.py --skip-analysis
```

Every invocation creates a fresh folder under `RUNS_ROOT` (default `./runs`), named after
the auto-generated `run_id` of the form `YYYYMMDD-HHMMSS-{4-char hex}`. Resuming with the
same `run_id` continues into the existing folder; see §8.1.

---

## 4. Tight-budget configuration

The codebase defaults trade a wider search budget for smoother behavioural analysis. For
discrimination-focused runs, the framework supports a "tight" preset at
$R_{\mathrm{tav}}\cdot C = 5\cdot 4 = 20$, equivalent to "two pages of Google search
results". The tight configuration is reproduced by the following `.env` overrides:

```ini
SOURCE_DB=./forecast_eval_set_example.db
SOURCE_TABLE=forecast_eval_set_example
SAMPLING_N=3                                # codebase default 5
REACT_MAX_STEPS=12                          # matches default
REACT_MAX_SEARCH_CALLS=4                    # codebase default 8
TAVILY_MAX_RESULTS=5                        # matches default
TAVILY_END_DATE_OFFSET_DAYS=-1              # one-day temporal masking buffer
REACT_REFLECTION_PROTOCOL=true
REACT_BUDGET_AWARENESS_PROTOCOL=true
REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=true
REACT_FORCE_FINAL_ANSWER_LOOKAHEAD=2
REACT_BUDGET_EXCEEDED_DROP_TOOLS=true
REACT_FINAL_ANSWER_RETRY=false              # v5.1 backstop, off by default; see §8.3
ENABLE_SEARCH_LEAK_FILTER=true              # Stage-2 detector on
BELIEF_PROTOCOL=false                       # strict-letter mode (no companion belief)
```

Each model's $\kappa_M$ must then be declared via `MODEL_TRAINING_CUTOFFS`.

The rationale for which knobs are contract knobs that change cross-run comparability and
which knobs are pure engineering knobs lives in `DESIGN.md` §12.

---

## 5. Bring your own dataset

The repository ships with `forecast_eval_set_example.db` so that a fresh clone is enough to
reproduce a non-trivial run. The bundled DB contains 80 curated questions: 37 yes/no, 3
binary-named, and 40 multiple-choice, of which 8 are multi-answer, with event-resolution
dates spanning 2026-03-12 to 2026-04-14.

To plug in a different corpus, point `SOURCE_DB` and `SOURCE_TABLE` at any SQLite file or
table that follows the same seven-column schema
`id / choice_type / question_type / event / options / answer / end_time` (see `FRAME.md`
§2.1), with a `dataset_metadata` row carrying the eight prompt template keys (see `FRAME.md`
§2.3):

```bash
SOURCE_DB=./my_questions.db
SOURCE_TABLE=my_questions
```

`SOURCE_TABLE` is whitelist-validated against `^[A-Za-z_][A-Za-z0-9_]*$` at startup
(config.py:L586–L595), so a typo fails fast instead of leaking into the SQL layer.

Once the schema is satisfied, the evaluation is dataset-agnostic. Domain-specific corpora
such as medical, scientific, or engineering forecasting can be dropped in directly, and the
same run unit $\mathcal{R}$ guarantees the same audit and replay properties. $\mathcal{D}$
is a replaceable input component of $\mathcal{R}$, so cross-domain extension is a natural
unfolding of the framework rather than an internal defect of it.

---

## 6. Outputs

The output directory is the run unit's persisted form. Anyone receiving
`runs/{run_id}/db/{model_slug}.db` can replay the model's evaluation without any other
artefact.

### 6.1 Directory layout

```text
runs/
  {run_id}/
    manifest.json           # run-level metadata: run_id, schema_version, analysis_schema,
                            #   sampling_n, models, filters, source/metadata/templates hashes,
                            #   reflection_protocol_hash, belief_protocol_hash, started_at,
                            #   finished_at; plus a `grid` block when multi-(R, C) is enabled
    db/
      {model_slug}.db       # one SQLite per model; self-contains questions + prompt_templates
                            #   + run_meta + run_results (see §6.2). Independently distributable.
    analysis/               # generated by forecast_eval.analysis after the run finishes
      per_model_summary.csv         # main scoring table: composite accuracy + v5 discrete
                                    #   family (FSS / Cohen κ / Hamming / Fleiss κ / mean
                                    #   entropy / VCI / MVG) + v4 probabilistic companion
                                    #   (BI / BI_dec / NLL / MBS / ABI_crowd / ABI_uniform /
                                    #    fallback_share)
      per_model_summary.md          # markdown table with v5 main columns; probabilistic
                                    #   columns flagged with `†` and a K disclaimer
      per_model_by_question_type.csv # sliced by yes_no / binary_named / multiple_choice
      per_model_by_choice_type.csv   # sliced by single / multi
      per_model_composite_by_question_type.csv  # subtype-weighted composite; defaults
                                                #   0.15 / 0.15 / 0.70 (see §7.1)
      per_model_composite_by_choice_type.csv    # subtype-weighted composite; defaults
                                                #   0.40 / 0.60 (see §7.1)
      composite_meta.json             # composite-score audit trail: per (model, metric)
                                      #   buckets_used / weights_used_normalized / value /
                                      #   bucket_values
      per_model_by_difficulty.csv     # γ-tertile slice (low / mid / high)
      error_breakdown.csv             # by error kind: network / server_5xx / bad_request /
                                      #   content_policy / skipped_training_cutoff / <ok>
      finish_reason_breakdown.csv     # by ChatCompletion finish_reason
      overall.json                    # full structured aggregate, with `probabilistic`
                                      #   sub-object and `analysis_schema` mirrored from
                                      #   manifest
      # ---- v5 K-trial consistency ----
      inter_trial_consistency.csv     # per-model Fleiss κ / mean entropy / VCI / MVG
      entropy_accuracy_bins.csv       # per-model × tertile (Acc / MV Acc / Fleiss κ)
      pairwise_bootstrap.csv          # multi-metric paired bootstrap: FSS / Acc / MV_Acc /
                                      #   Fleiss κ / EBI × pairs × ΔMean / 95% CI / p / Cohen's d
      # ---- v4 probabilistic (companion, K=5 disclaimer) ----
      shrinkage_alpha_curve.csv       # per-(model, ctype) LOO α scan
      paired_delta_bi.csv             # BS-paired ΔBS + Holm-adjusted p + posterior
      pairwise_significance.csv       # α=0.05 flag (raw + Holm)
      posterior_pairwise.csv          # P(BI_A > BI_B)
      paired_delta_bi_by_difficulty.csv
      # ---- Phase 3 behavioural diagnostics (require BELIEF_PROTOCOL=true) ----
      belief_evolution.csv            # per-(model, q, k): volatility, inter-trial variance,
                                      #   convergence_step, evidence_efficiency,
                                      #   counterevidence_engaged
      reflection_ab.csv               # paired A/B (when sibling runs share every hash except
                                      #   the reflection-protocol hash)
      tool_usage_pdp.csv              # per-(model, feature, value) PDP for Pr(correct|x) and
                                      #   E[NLL|x]
      confidence_calibration.csv      # subjective confidence vs hit rate
      numeric_confidence_calibration.csv  # max_p binning vs hit rate
      # ---- grid search (only when manifest.grid is present) ----
      grid_summary.csv                # per (real_model, R, C) main table:
                                      #   acc/BI/NLL + 95% CI + cost columns
      grid_marginal_C.csv             # fixed R = grid.default_r, varying C
      grid_marginal_R.csv             # fixed C = grid.default_c, varying R
      grid_pareto.csv                 # `dominated_by` empty for Pareto-frontier cells, else
                                      #   the lex-smallest dominator slug
      grid_winrate.csv                # pairwise (R, C)-cell wins + significant-cell tally
      figs/                           # only after `python scripts/plot_analysis.py`;
                                      #   matplotlib not in core deps, on-demand
    logs/
      {run_id}.log
```

Model-slug filesystem safety maps `/` to `__`, and any character outside `[A-Za-z0-9._-]` to
`_`. So `openai/gpt-4o-mini` becomes `openai__gpt-4o-mini.db`. Grid virtual slugs add
`__r{R}__c{C}` suffixes; see §8.2.

### 6.2 Per-model database schema

Each model DB holds three tables.

* **`questions`** and **`prompt_templates`** are copies of the source data, so every DB is
  independently replayable without the original `SOURCE_DB`.
* **`run_meta`** holds a single row containing `run_id, model, sampling_n, config_snapshot
  (redacted), filters_snapshot, source/metadata/templates hashes, training_cutoff,
  reflection_protocol_text/hash, belief_protocol_text/hash, started_at, finished_at`. The
  two protocol fingerprints are independent of `prompt_templates_hash` and of each other.
  `DESIGN.md` §7.3 explains why three independent fingerprints, namely template,
  reflection, and belief, enable three-axis ablation A/B pairing without collisions.
* **`run_results`** is a wide table with one row per question. For each $i$ in
  `0..SAMPLING_N-1` there is an `s{i}_*` group of 20 columns at the v3 base, plus 3 belief
  columns added at v4 and 1 retry column added at v5.1. The full set is `final_answer_letters
  / final_answer_raw / correct / parse_ok / tool_calls_count / react_steps / prompt_tokens /
  completion_tokens / reasoning_tokens / latency_ms / messages_trace / search_calls / error /
  created_at` for the v2 base, `finish_reason / nudges_used / step_metrics / response_id /
  system_fingerprint / service_tier` for v3 observability, `belief_final / belief_trace /
  belief_parse_ok` for v4 belief, and `final_answer_retry_used` for v5.1; see §8.3. Old DBs
  auto-migrate via `ALTER TABLE ADD COLUMN` on first re-open. Setting
  `Settings.BELIEF_PROTOCOL=false` keeps the v4 belief columns NULL and leaves all v3
  accuracy metrics byte-identical to pre-v4 runs.

The DB stores raw observations only. No aggregates are pre-computed; pass@1, pass_any@N,
majority vote, FSS, BI and the rest all come from the `analysis/` pass, which runs
automatically at the end of `evaluation.py` and can also be invoked standalone:

```bash
python -m forecast_eval.analysis runs/{run_id}
```

This separation between raw observations and aggregated metrics is one of the project's most
load-bearing architectural decisions. `DESIGN.md` §4.1 lays out the rationale: metric
definitions evolve faster than DB schemas, so deferring all aggregation to the analysis
layer means a metric redefinition never requires a DB backfill. The contract is pinned by
`tests/test_analysis.py`, which runs the entire analysis on a hand-crafted DB fixture
without re-touching it.

---

## 7. Scoring

### 7.1 Composite accuracy with exam-style partial credit

`per_model_summary.csv` reports a flat mixed mean (`pass_at_1_avg`) for backwards
compatibility. For the headline scoring recommended for cross-model comparison,
`per_model_composite_*.csv` performs a weighted composition by sub-question type along two
dimensions:

* `per_model_composite_by_question_type.csv` buckets by `yes_no` / `binary_named` /
  `multiple_choice`;
* `per_model_composite_by_choice_type.csv` buckets by `single` / `multi`.

Per-bucket scoring uses exam-style partial credit, implemented at
`forecast_eval/analysis/exam_score.py:L62`:

$$\text{exam-score}(\hat{S}, G) = \begin{cases} |\hat{S} \cap G| / |G|, & \hat{S} \setminus G = \varnothing,\\ 0, & \hat{S} \setminus G \ne \varnothing.\end{cases}$$

Intuitively, any false positive vetoes the score to zero; otherwise the score is the
proportion correctly recovered, $|TP|/|G|$. This is recall under a zero-FP hard gate.
Single-answer questions where $m_q = 1$ degenerate to the strict-equality $\{0, 1\}$ case,
and multi-answer questions retain the asymmetry "rather miss than wrongly select". The
composite formula (implemented at `analysis/composite.py`) is

$$\text{composite}_m = \frac{\sum_{b \in B_{\text{valid}}(m)} w_{m,b}\cdot v_{m,b}}{\sum_{b \in B_{\text{valid}}(m)} w_{m,b}}.$$

$B_{\text{valid}}$ is the set of buckets where the measurement is non-None and the weight is
positive. Missing buckets are dropped and the remaining weights renormalised; they are not
treated as zero. This contract is pinned by `tests/test_composite_score.py` and
`tests/test_exam_score.py`.

Default weights follow the *harder questions discriminate better* principle, defined at
config.py:L365–L374:

| Dimension       | Bucket            | Default weight | Difficulty rationale                                       |
| --------------- | ----------------- | -------------- | ---------------------------------------------------------- |
| `question_type` | `yes_no`          | 0.15           | k=2, blind guess 50%, low cross-model discrimination       |
| `question_type` | `binary_named`    | 0.15           | k=2, adds entity recognition but still binary              |
| `question_type` | `multiple_choice` | 0.70           | k=2..N wide range, includes multi-select, highest signal   |
| `choice_type`   | `single`          | 0.40           | overall easier; includes yes_no and binary_named           |
| `choice_type`   | `multi`           | 0.60           | true multi-select; near-zero strict baseline → high signal |

Override these via `COMPOSITE_WEIGHTS_QTYPE` and `COMPOSITE_WEIGHTS_CTYPE` in `.env`. For
per-metric overrides use `COMPOSITE_WEIGHT_OVERRIDES_QTYPE` and `..._CTYPE` (see
`.env.example` comments). When any metric in a model row hits an override, its
`weights_kind` column is flagged `overridden`. `composite_meta.json` records `buckets_used`,
`weights_used_normalized`, and `bucket_values` for each (model, metric), giving a
one-to-one reproducible audit trail.

The exam-vs-strict difference matters only on the multi-choice multi-answer bucket. The
three single-answer buckets satisfy
$\text{exam}_{\text{avg}}^{(b)} \equiv \text{pass@1}_{\text{avg}}^{(b)}$, so the composite
formula's value depends on the exam-vs-strict choice only through the multi-multi bucket,
which carries the largest discrimination signal under the tight preset in §4.

The per-correct cost is

$$C^{\text{per-correct}}_m = \frac{C^{\text{total}}_m}{|\mathcal{D}^{\text{eval}}|\cdot n \cdot \text{Composite\,Accuracy}_m},$$

i.e. the platform's actual invoice divided by the difficulty-weighted notional
correct-sample count. This places expensive-but-accurate and cheap-but-reckless models on the
same cost-effectiveness scale, avoiding the false-low-cost illusion of a low per-sample unit
price paired with a high error rate.

### 7.2 Hierarchical scoring suite

Scoring is organised as a hierarchical decomposition validity → item → question → model.
The headline composite accuracy from §7.1 is one column of `per_model_summary.csv`; the
companion suite covers stability, consistency, and chance-corrected skill.

| Metric                                                 | What it measures                                                                                                  | Code                                                      |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| $\text{pass@1}_{\text{avg}}$                           | Single-trial strict-equality hit rate                                                                             | `analysis/accuracy.py`                                    |
| $\text{pass}^{\text{any}}@n$                           | Best-of-$n$ hit upper bound                                                                                       | `analysis/accuracy.py`                                    |
| $\text{pass}^{\text{all}}@n$                           | All-of-$n$ stability lower bound                                                                                  | `analysis/accuracy.py`                                    |
| Cohen's $\kappa$                                       | Chance-corrected strict accuracy against a question-type-conditional baseline                                     | `analysis/accuracy.py::cohen_kappa`                       |
| Fleiss' $\kappa$                                       | Inter-trial agreement across $K^{\mathrm{eff}}_q$ samples                                                          | `analysis/consistency.py`                                 |
| Tversky $T$                                            | Set similarity with FP penalty $\alpha$ and FN penalty $\beta$                                                    | `analysis/accuracy.py::tversky_score` (accuracy.py:L286)  |
| FSS                                                    | Tversky-based, chance-corrected skill score; default $(\alpha, \beta) = (2.0, 0.5)$ penalises FP four times more than FN | `analysis/accuracy.py::fss` (accuracy.py:L386)            |
| MV-Acc / MVG / VCI / Hamming / mean-entropy            | Discrete-native consistency family                                                                                | `analysis/consistency.py`                                 |

FSS is designed to surface a class of cross-model reorderings that strict
$\text{pass@1}_{\text{avg}}$ misses. Two models can tie on strict accuracy yet differ
substantially in how restrained their multi-answer selection sets are; under FSS at
$(\alpha, \beta) = (2.0, 0.5)$ the more FP-conservative model is correctly preferred. This
is the empirical justification for the asymmetric Tversky weights: encoding "rather miss
than wrongly select" directly into the score function flips real cross-model rankings that
a symmetric Jaccard would not have caught.

### 7.3 On-demand plots and FSS sensitivity

`matplotlib` is not in `environment.yml`, since the analysis path stays dependency-light. To
render the analytics CSV/JSON into PNGs:

```bash
pip install matplotlib
python scripts/plot_analysis.py runs/{run_id}
```

This populates `runs/{run_id}/analysis/figs/`, which is gitignored, with:

* v5 main figures: FSS bar with CI, ΔFSS forest, per-model entropy-Acc grid (3 buckets ×
  3 metrics: Acc / MV Acc / Fleiss κ);
* Companion figures: BI bar with CI (BLF anchor), ΔBI forest, difficulty-grid heatmap,
  per-question belief trajectories on 5 sample questions, tool-usage PDP per feature.

v5 removed the reliability-diagram and Murphy-three-decomposition figures, since at $K=5$
they are statistically meaningless given only six unique probability levels per label.

Each plot is best-effort: when the corresponding CSV or JSON is missing, the plot is
silently skipped instead of failing the pipeline.

`per_model_summary.csv` reports a single canonical FSS at $(\alpha, \beta) = (2, 0.5)$.
Reviewers asking "why not Jaccard $(1, 1)$ or strict $(3, 0.5)$?" run the sensitivity sweep
on demand:

```bash
python scripts/fss_sensitivity.py runs/{run_id}                      # 4-tier sweep
python scripts/fss_sensitivity.py runs/{run_id} --alpha 1 --beta 1   # single point
```

| (α, β)    | Semantics                                                  |
| --------- | ---------------------------------------------------------- |
| (1, 1)    | Jaccard, symmetric: FP and FN equally penalised            |
| (1, 0.5)  | Mild asymmetry: multi-selection error 2× missed            |
| (2, 0.5)  | v5 default: multi-selection error 4× missed                |
| (3, 0.5)  | Strict: multi-selection error 6× missed                    |

The script is not invoked by `run_analysis`. The sensitivity CSV carries a provenance
comment on top so a reviewer reading the bare file will not mistake it for the main metric,
with the contract pinned by `tests/test_fss_sensitivity.py`.

---

## 8. Operational features

### 8.1 Resume semantics

Each `(question_id, sample_idx)` slot is judged independently:

* `s{i}_created_at IS NOT NULL` and `s{i}_error IS NULL` means finished and not retried.
* `s{i}_error = 'skipped_training_cutoff'` was actively excluded by the
  $\kappa_M \le \chi_i$ check, and is not retried, since it was never a model failure.
* Any other `s{i}_error` value such as `network`, `server_5xx`, `bad_request`, or
  `content_policy` is retried on the next run, which reuses the existing DB. Error
  classification lives in `forecast_eval/errors.py:classify` (errors.py:L86); the bucket
  list is `network / rate_limit / server_5xx / bad_request / content_policy`, plus the
  synthetic `skipped_training_cutoff`.

Set `RUN_ID=<existing-run-id>` in `.env`, or as a CLI env var, to resume into the same
folder. Leaving it blank mints a fresh `YYYYMMDD-HHMMSS-xxxx` id.
`tests/test_runner_resume.py` pins the behaviour: completed rows are never re-emitted,
`skipped_training_cutoff` rows are never re-run, and every other error class is retried
under the original retry policy.

### 8.2 Grid search via virtual model slug

`TAVILY_MAX_RESULTS` (which is $R_{\mathrm{tav}}$) and `REACT_MAX_SEARCH_CALLS` (which is
$C$) accept comma-separated lists of positive integers. Setting both to multi-value lists
produces $\lvert\text{MODELS}\rvert \cdot \lvert R\rvert \cdot \lvert C\rvert$ independent
virtual model slugs of the form `{real_model}::r{R}::c{C}` (`db.compose_virtual_slug` and
`db.parse_virtual_slug`, db.py:L477/L500). Each cell lives in its own DB file at
`runs/<id>/db/<real>__r{R}__c{C}.db` and re-uses every existing analysis stage. An extra
grid pass writes 5 `grid_*.csv` long tables plus a per-cell figure family under
`analysis/figs/`.

The runner, the DB schema, and the analysis main pipeline are byte-unchanged.
`forecast_eval/analysis/grid.py` decodes the triple from the slug, re-aggregates, and emits
the long-form grid tables. See `DESIGN.md` §11.1 for the design archive of decisions D1–D10,
together with `tests/test_grid_dispatcher.py` and `tests/test_grid_analysis.py` for the
contract pinning.

```bash
MODELS=openai/gpt-5,anthropic/claude-sonnet-4.5
TAVILY_MAX_RESULTS=5,10
REACT_MAX_SEARCH_CALLS=1,3,5,8
GRID_DEFAULT_R=5    # main figure anchor; must be in TAVILY_MAX_RESULTS
GRID_DEFAULT_C=5    # symmetric, must be in REACT_MAX_SEARCH_CALLS

python evaluation.py
python scripts/plot_analysis.py runs/<run_id>
```

A single-value `.env` such as `TAVILY_MAX_RESULTS=5` is parsed as a length-1 list, so
existing setups stay byte-equivalent except for the new `__r{R}__c{C}` suffix on DB
filenames. Legacy v4 runs without a `manifest.grid` block exit the grid path early.
`MODELS` entries cannot contain `::` (config.py:L610–L614), so virtual-slug round-tripping
never collides with a real model name.

### 8.3 Harness resilience switches (v5.1)

Two opt-in resilience levers, with `REACT_FINAL_ANSWER_RETRY` defaulting OFF and
`REACT_BUDGET_EXCEEDED_DROP_TOOLS` defaulting ON; see
`openspec/changes/harness-resilience-v1/`.

* **`REACT_FINAL_ANSWER_RETRY`** defaults to `false` (config.py:L301). When the ReAct loop
  exits cleanly with empty `final_raw`, meaning the model spent all steps on `tool_calls`
  and never produced content, the harness makes one extra `llm_chat` call with `tools=[]`
  and a fixed "commit your `\boxed{...}` answer" user nudge. This switch is superseded by
  the in-loop force-final-answer-near-limit chain below and is kept as an optional
  out-of-loop emergency backstop. When enabled, the retry counts as one step in
  `react_steps` and `step_metrics` but not in `nudges_used`. The per-sample column
  `final_answer_retry_used` (0/1) records the outcome and rolls up to
  `final_answer_retry_rate` in `per_model_summary.csv`. The motivation is that cross-model
  comparisons require `parse_failure_rate` to reflect only the model's own format failure,
  not upstream tool-budget exhaustion bookkept by the harness.
* **`REACT_BUDGET_EXCEEDED_DROP_TOOLS`** defaults to `true` (config.py:L302). Once
  cumulative `web_search` calls reach `REACT_MAX_SEARCH_CALLS`, every subsequent LLM call
  drops the tool schema by setting `tools=[]`. The model can no longer request more
  searches; it must finalise its answer or the bail-out retry above mops up.

The in-loop priority chain introduced by `force-final-answer-near-limit-v1`
(config.py:L313–L315) is what really drives termination at the budget edge, and it is the
reason the post-hoc `REACT_FINAL_ANSWER_RETRY` is now off by default. At the top of every
iteration of `react.run_react` (react.py:L162) the harness picks at most one of four
injections, in this priority order (react.py:L266 priority comment, L272–L334 logic):

1. The **last-step hard cutoff** fires when `REACT_MAX_STEPS - step == 1` and
   `REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=true`. It injects force-finalise text and
   `tools=[]`, so the model can ONLY emit content.
2. The **penultimate soft warning** fires when
   `remaining ∈ [2, REACT_FORCE_FINAL_ANSWER_LOOKAHEAD]`. It injects reminder text, with
   tools still allowed unless the search budget is already gone.
3. The **budget-exhausted commit notice** fires when cumulative searches
   `>= REACT_MAX_SEARCH_CALLS` and `REACT_BUDGET_EXCEEDED_DROP_TOOLS=true`, only once per
   run. It tells the model that the search tool is now gone and asks it to finalise.
4. The **continuation reminder** fires when the previous turn was content without
   `\boxed{...}` and nothing else needs to fire. It says "your last reply had no
   `\boxed{...}`, here is the live status".

Defaults are `REACT_BUDGET_AWARENESS_PROTOCOL=true`,
`REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=true`, and `REACT_FORCE_FINAL_ANSWER_LOOKAHEAD=2`
(config.py:L313–L315), pinned by the priority-chain section of `tests/test_react.py`. To
run the v5.0 baseline with no in-loop intervention, flip all three to `false` and the
harness reverts to the legacy "single-shot per turn until budget" behaviour.

Error classification in `forecast_eval/errors.py` was widened in v5.1: HTTP 400 bodies
containing any of `data_inspection_failed`, `inappropriate content`, or `sensitive`, in
addition to the legacy `content_policy` / `content_filter` / `safety` /
`content_policy_violation` needles (see `errors.CONTENT_POLICY_NEEDLES` at
errors.py:L39–L48), classify as `content_policy` rather than `bad_request`. The
transient-network family at errors.py:L97–L111 now also covers `httpx.RemoteProtocolError`,
`WriteError`, `WriteTimeout`, and `PoolTimeout`. Both the LLM client and the Tavily search
client retry these instead of treating them as fatal.

### 8.4 Search leak filter (v5.2)

Tavily filters by crawl or index date, not by content time. A page indexed before $\chi_i$
may still describe events that happened after it, such as wiki updates, aggregator pages,
or "looking ahead" sections. To plug that hole the framework adds a Stage-2 LLM-based
semantic audit. Every Tavily result is sent through an independent `detector` LLM that
returns `keep` or `drop` per item. Items the detector flags `drop` are removed before the
main LLM sees the search payload. Input fields are whitelisted to title, URL,
published_date, content, raw_content, and cutoff_date; the question text, options, and
ground truth are deliberately withheld so the detector is a leakage classifier rather than
an answer auditor.

Defaults (see `.env.example` for the full annotated block):

* `ENABLE_SEARCH_LEAK_FILTER=true` (config.py:L337) is required to enable the filter. Pair
  it with `LEAK_DETECTOR_API_KEY` and `LEAK_DETECTOR_MODEL`. It mutually requires
  `ENABLE_WEB_SEARCH=true`, otherwise the detector path is dead code and startup fails fast
  at config.py:L752–L757.
* `LEAK_DETECTOR_BASE_URL` is optional; an empty value falls back to `LLM_BASE_URL`. The
  detector client is independent of the main LLM client even when the endpoints coincide
  (`leak_filter.get_detector_client`, leak_filter.py:L112), so it has separate quota,
  timeout, and backoff bookkeeping.
* `LEAK_DETECTOR_FAIL_ACTION=drop` (config.py:L351) is the fail-closed default. Detector
  errors over HTTP, on timeout, or on invalid JSON, after `LEAK_DETECTOR_RETRY_MAX` retries
  with `LEAK_DETECTOR_BACKOFF_S`, drop the item. Set this to `keep` only as an A/B escape
  hatch when comparing against the unfiltered baseline.
* `LEAK_DETECTOR_RETRY_MAX` and `LEAK_DETECTOR_BACKOFF_S` are independent of the main LLM's
  retry settings, so detector hiccups never push back on the main LLM's quota window.

Audit fields are persisted per `web_search` call inside the `run_results.search_calls` JSON
entry:

```text
{ "query": ..., "end_date": ..., "n_results": <kept>,
  "published_dates": [<raw-order, length == n_results_raw>],
  "n_results_raw": <int>, "n_results_kept": <int>,
  "detector_verdicts": ["keep", "drop", "failed:network", ...],
  "detector_latency_ms": <int>, "detector_error_kind": str | null }
```

`run_meta.config_snapshot` additionally records the detector fingerprint triplet
`leak_detector_enabled`, `leak_detector_model`, and `leak_detector_prompt_hash`, which is a
sha256 of the prompt template at `leak_filter.py:L55–L92` truncated to the first 16 hex.
The leakage barrier is therefore byte-reproducible. It is pinned by
`tests/test_leak_filter.py` and the on-the-wire smoke `scripts/smoke_leak_filter.py` and
`scripts/verify_leak_filter_e2e.py`.

Disable path: set `ENABLE_SEARCH_LEAK_FILTER=false` and the detector layer is bypassed
entirely; behaviour is byte-identical to v5.1. All upstream barriers remain unaffected:
the web_search schema, the `end_date` injection, the Tavily `end_date` filter, the
`MODEL_TRAINING_CUTOFFS` admissibility check, and the `:online` ban.

---

## 9. Documentation

### 9.1 Layered documents

The repository's documentation is layered. Each layer answers a different question; pick
the layer that matches your question and skip the others.

| You want to know…                                                                            | Read…                                |
| -------------------------------------------------------------------------------------------- | ------------------------------------ |
| What this project is and how to run it                                                       | this README                          |
| Why every constraint exists, and what was rejected                                           | `DESIGN.md`                          |
| Field-level and interface-level specification mapping each symbol to a module, a DB column, and a test | `FRAME.md`                  |
| The exact rationale for each schema-change proposal                                          | `openspec/changes/<change-id>/`      |
| The exact rationale for each archived schema-change proposal                                 | `openspec/changes/archive/`          |
| Tight-budget configuration recipe                                                            | this README §4                       |
| Contract knobs vs engineering knobs (which `.env` changes invalidate cross-run comparability)| `DESIGN.md` §12                      |
| Three independent fingerprints and the manifest layout                                       | `FRAME.md` §6.3; `evaluation.py::_compute_*_protocol` |

The three document layers form a bidirectional contract: DESIGN's rationale → FRAME's
specification → code's implementation, each with its own pinning test. Each layer can be
read in isolation, but a contradiction between any pair indicates a bug. The test suite
exists to catch such contradictions early.

### 9.2 Reading order for newcomers

If you are new to the project, we suggest reading in this order:

1. **`README.md`** (this file) for what OracleProto is and how to run it.
2. **`DESIGN.md`** for the rationale: why every constraint exists, the threat model, the
   trade-offs between strict matching and partial credit, and why the DB stores raw
   observations only. §1 (framework and code map) gives the fastest entry; §12 sorts
   contract knobs from engineering knobs.
3. **`FRAME.md`** for the technical specification at field, interface, and pseudocode
   level. §1.1 (the grand map) is the cross-reference scaffold; §2–6 walk top-down from
   data to pipeline.
4. **`forecast_eval/prompts.py` and `forecast_eval/parser.py`** for the renderer $R$ and
   the parser $\Psi$, which are the heart of the project's information boundary.
5. **`forecast_eval/runner.py` and `forecast_eval/react.py`** for orchestration. The
   admissibility filter sits at runner.py:L132 and the ReAct loop at react.py:L162, with
   the priority chain at react.py:L266.
6. **`tests/`** to reverse-engineer the contracts. The 33 test files cover the v3/v4/v5
   schemas, the leak filter, exam-score, the grid dispatcher, and behavioural diagnostics.
7. **`openspec/changes/archive/`** to find out why things became what they are today.

---

## 10. Version history

| Version | Headline change                                                                                    | Default behaviour                                                  |
| ------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| v3      | Wide-table schema, per-sample observability columns                                                | strict-letter scoring, no belief, no detector                      |
| v4      | Belief protocol via companion JSON block; probabilistic suite (BI / NLL / MBS / ABI)                | `BELIEF_PROTOCOL=false`; v3 byte-equivalent until enabled          |
| v5      | Discrete-native pivot: FSS / Cohen κ / Fleiss κ / Hamming as primary; dropped reliability and Murphy figures at $K=5$ | exam-style and composite weights are the headline   |
| v5.1    | Harness resilience: in-loop force-final, budget-drop tools, retry backstop; widened error needles  | force-final on, retry-backstop off by default                      |
| v5.2    | Stage-2 LLM detector for retrieval-content leakage                                                 | `ENABLE_SEARCH_LEAK_FILTER=true` (default-strict, fail-closed)     |

Migrations are forward-only: every old DB auto-migrates via `ALTER TABLE ADD COLUMN` on
first re-open. Pinned by `tests/test_db_v4_migration.py` and
`tests/test_db_v5_migration.py`.

---

> **One sentence.** OracleProto turns LLM forecasting evaluation from a one-off live
> competition into a dataset-level, auditable, reusable, and trainable capability, by making
> the information boundary part of the data rather than part of the prompt.
