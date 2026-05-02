# OracleProto

**A reproducible framework for benchmarking LLM native forecasting via knowledge cutoff and temporal masking.**

This repository is the reference implementation of the paper *OracleProto: A Reproducible
Framework for Benchmarking LLM Native Forecasting via Knowledge Cutoff and Temporal Masking*
(Ma, Ruan, Huang, Yang & Zhou; BUPT — `paper/main.tex`). The framework reconstructs **resolved
events** into **time-bounded forecasting samples** so that the evaluation object lives at the
dataset level — auditable, replayable, and comparable across models and across calendar years.

> **Summed up in one sentence.** This codebase turns a forecasting evaluation into a single,
> reproducible run unit
> $\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)$ (paper §3.5,
> "Run definition"), fixes every input — questions, knowledge cutoffs, temporal masking, ReAct
> budgets, prompt rendering, output parsing, label normalization, and aggregation rules — and
> emits scoring artefacts that are bit-identical from byte-identical configurations.

This README is the **10-minute orientation**: what the project is, how to run it, where each
artefact comes from. For *why* each constraint exists, read `DESIGN.md`; for the field-level
specification (every symbol → module → DB column → test) read `FRAME.md`; for the formal
framework and the FutureX-Past evaluation read `paper/main.tex`.

---

## 1. The problem this project solves

Existing forecasting evaluations sit on an unstable middle ground (paper §1, §2.3):

* **Prospective live benchmarks** (ForecastBench, FutureX) are contamination-controlled by
  construction but evaporate the moment the event resolves; the leaderboard is a one-way
  temporal stream rather than a reusable artefact.
* **Retrospective benchmarks** (FutureX-Past, archived live questions) are reproducible but
  highly prone to mistaking *factual recall* for *forecasting capability*: by the time the
  paper is written, the answer is sitting in the model's training corpus.

Prompt-time discipline ("imagine you do not know that the election has resolved") cannot
bridge this gap. The diagnostic literature surveyed in paper §2.3 (Paleka et al. 2025; Li et
al. 2026) has empirically shown a substantial systematic gap between **simulated ignorance**
and **true ignorance**, and that a 1–5% label-noise rate alone is enough to break proper
scoring rules. BLF (Murphy 2026) reaches the same conclusion from the inference side: a
single-inference defence does not generalise across runs; the discipline must live one level
deeper, *inside the dataset itself*.

OracleProto's response is to push the discipline **one level deeper, into the dataset
schema**. For a given (model, question) pair, the question is admitted only if its prediction
cutoff $\chi_i$ satisfies (paper §3.1, Eq. 4 `eq:pred-set`)

$$\kappa_M \le \chi_i < \tau_i,$$

where $\kappa_M$ is the model's training cutoff and $\tau_i$ is the event-resolution time —
the model's parametric knowledge is not more recent than the permitted prediction
environment, and the resolution time has not yet arrived in the simulated information state.
Inadmissible questions are *not counted as model errors*; they are filtered out and audited
separately (`runner.build_task_plan`, runner.py:L132–L199; pinned by
`tests/test_training_cutoff.py`).

## 2. Three contributions

Following paper §1, this repository delivers:

1. **A formal dataset-level framework for LLM forecasting evaluation.** The evaluation object
   is no longer a one-shot live result, but a dataset-level task that is definable, auditable,
   and reproducible — anyone can re-run the same
   $\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)$ and obtain
   comparable numbers (paper §3, materialised across `forecast_eval/*`).
2. **The OracleProto unified evaluation protocol.** Knowledge cutoffs (sample admission),
   tool-level temporal masking, content-level leakage detection (a Stage-2 LLM auditor),
   discrete answer normalization, and hierarchical scoring (validity → item → question →
   model) are wired into one pipeline (paper §3.3, §3.4, §3.5; `forecast_eval/runner.py`,
   `react.py`, `leak_filter.py`, `parser.py`, `analysis/`).
3. **A systematic evaluation benchmark and a trainable forecasting harness.** The example
   dataset bundled with the repo (`forecast_eval_set_example.db`, 319 questions), plus the
   FutureX-Past instantiation reported in the paper (curated 80-question subset, paper
   §4.1.1), together provide a leakage-controlled forecasting evaluation set. Outputs
   (per-sample raw records, per-model SQLite databases, hierarchical analytics) are
   immediately reusable as signals for SFT, RL, and forecasting-agent training (paper §1, §5).

## 3. The run unit $\mathcal{R}$

Every invocation of `evaluation.py` materialises a single run unit
$\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)$ (paper §3.5,
"Run definition"; the run unit is the 34th equation of the paper, immediately followed by the
stochastic-trial definition $\widehat{Y}_{i,M}^{(s)}$). Each symbol resolves to **one**
configuration knob, **one** code path, and **one** test that pins the contract:

| Symbol             | Object                       | Config / code path                                                     | Pin test                       |
| ------------------ | ---------------------------- | --------------------------------------------------------------------- | ------------------------------ |
| $\mathcal{D}$      | Discrete forecasting dataset | `SOURCE_DB` / `SOURCE_TABLE` (config.py:L391/L395); `loader.sync_questions` (loader.py:L77) | `tests/test_db.py`             |
| $M$                | Evaluated model              | one entry of `MODELS` (config.py:L223); one SQLite per model under `runs/{run_id}/db/` | `tests/test_runner_grid_model.py` |
| $\kappa_M$         | Knowledge cutoff             | `MODEL_TRAINING_CUTOFFS[M]` (config.py:L224); admissibility filter at `runner.build_task_plan` (runner.py:L132) | `tests/test_training_cutoff.py` |
| $\delta$           | Temporal masking offset      | `TAVILY_END_DATE_OFFSET_DAYS` default `-1` (config.py:L273); injected at the tool layer in `react.py` | `tests/test_search.py`, `tests/test_react.py` |
| $T$                | Max ReAct steps              | `REACT_MAX_STEPS` default `12` (config.py:L279); outer loop `react.run_react` (react.py:L248) | `tests/test_react.py`          |
| $C$                | Max search calls             | `REACT_MAX_SEARCH_CALLS` default `[8]` (config.py:L283); budget gate (react.py:L276–L279) | `tests/test_react.py`          |
| $R$                | Input renderer               | `forecast_eval/prompts.py::render_user_prompt`                         | `tests/test_prompts.py`        |
| $\Psi$             | Output parser & validity     | `forecast_eval/parser.py::parse_answer` (parser.py:L40)                 | `tests/test_parser.py`         |
| $\phi$             | Answer normalization map     | letter encoding (`A`, `A,B` …) defined per `question_type`; `parser.parse_gt` (parser.py:L92) | `tests/test_parser.py`         |
| $\Gamma$           | Aggregation rule             | `forecast_eval/analysis/*` (composite accuracy, FSS, κ, BI, …)          | `tests/test_analysis.py`       |
| $H_{\mathrm{aux}}$ | Auxiliary leakage detector   | `leak_filter.filter_search_result`; logged in `run_meta.config_snapshot` rather than in the $\mathcal{R}$ tuple (paper §3.5 "Run definition") | `tests/test_leak_filter.py` |

The auxiliary detector $H_{\mathrm{aux}}$ is deliberately kept **outside** the formal tuple
in the paper (§3.5) and bound via SHA-256 fingerprint to run metadata, because the detector
is a *replaceable empirical engineering layer* that supports the boundary, not a primitive
component of the forecasting system itself. Its prompt SHA-256 is stored in
`run_meta.config_snapshot.leak_detector_prompt_hash`, so the leakage barrier is itself
byte-reproducible (`leak_filter.py:L55`–L104).

The information visible to model $M$ on question $q_i$ is, by paper §3.3 Eq. 16
(`eq:visible-info`),

$$\mathcal{I}_{i,M}^{\mathrm{vis}} = \mathcal{K}^{M}_{\le\kappa_M} \cup \mathcal{T}_{\le\chi_i},$$

with $\mathcal{K}^{M}_{\le\kappa_M}$ the parametric knowledge before the model's cutoff and
$\mathcal{T}_{\le\chi_i}$ the temporally masked external information. The forecasting system
$F_M$ produces $\widehat{Y}_{i,M} = F_M(q_i^{\mathrm{in}}; \mathcal{I}_{i,M}^{\mathrm{vis}})$
with $\widehat{Y}_{i,M}\subseteq\mathcal{A}_i$ — see paper Algorithm 1 for the time-masked
discrete forecasting loop, mirrored almost line-for-line in `react.run_react` (react.py:L248).

For the field-level grand map (every symbol → module → DB column → test), see `FRAME.md` §1.1.

## 4. The four-channel information boundary

Paper §3.5 decomposes the residual leakage surface into **three controlled channels** —
parametric, tool-mediated, and retrieval-content — plus a **fourth provider-side residual**
that is not under the evaluator's control. The implementation maps each channel to a
mechanical defence with declared coverage, summarised below; the empirical residual rates
come from paper §4.1.5 Table 3 (270-item manual audit, paper §4.3.4).

| Channel (paper §3.5)        | Defence layer                                       | Where (code)                                              | Default            | Residual leakage rate (paper §4.1.5) |
| --------------------------- | --------------------------------------------------- | --------------------------------------------------------- | ------------------ | ------------------------------------ |
| **L0 manual curation**      | Upstream dataset construction                       | `forecast_eval_set_example.db` / FutureX-Past curation     | always             | 0% (manual annotation floor)          |
| **L1 parametric (admissibility filter)** | $\kappa_M \le \chi_i$ check at task generation | `runner.build_task_plan` (runner.py:L132–L199)            | `MODEL_TRAINING_CUTOFFS` declared per model | filters parametric-memory leakage upstream |
| **L2 tool-mediated (Tavily)** | `end_date = \chi_i` injected at the tool layer    | `react._compute_end_date` (react.py:L182); `search.tavily_search` | $\delta=-1$ day  | 3%–16% on its own (Tavily metadata noise) |
| **L3 retrieval-content (Stage-2 detector)** | Independent LLM auditor on each Tavily item | `leak_filter.filter_search_result` (leak_filter.py:L348) | `claude-sonnet-4.6` | **1.1%** per-audit-item; **1.3%** leak-conditional (Wilson 95% UB **3.2%**) |
| **L4 provider-side residual (declared)**  | Provider-native browsing **forbidden**           | `Settings._post_validate` (config.py:L602–L606, L747–L751); `llm._assert_no_browsing` (llm.py:L74–L98); `leak_filter._assert_detector_safe` (leak_filter.py:L139) | always             | declared as evaluation bias rather than pretended-away |

The triple-layer enforcement of L4 — (a) startup validation rejects `:online` slugs and the
`::` reserved delimiter, (b) `llm.chat` re-asserts on the wire, (c) the detector client
duplicates the same checks — is pinned by `tests/test_llm_no_browsing.py` and
`tests/test_config.py`. It is structurally the one defence that must pass *before* any
billable LLM call leaves the process.

The 270-item audit (paper §4.3.4, Table 8) was sampled across 3 models × 30 questions × 3
trials. The detector recall is 98.7% (235 TP / 238 real leaks), specificity 96.9% (31 TN /
32 real non-leaks), per-audit-item residual rate 3/270 ≈ 1.1%, with the Wilson 95% upper
bound landing at ≈ 3.2% — an order of magnitude below the Tavily-only baseline and
approaching the manual-annotation floor at two orders of magnitude lower marginal cost.

For the threat model and the "what we can/cannot control" decomposition, see DESIGN.md §2;
for the eight hard constraints derived from $\mathcal{R}$, see FRAME.md §1.2.

---

## 5. Quickstart

### 5.1 Create the conda environment

```bash
conda env create -f environment.yml
conda activate forecast
```

### 5.2 Configure `.env`

```bash
cp .env.example .env
# Edit .env and fill in:
#   LLM_API_KEY (and LLM_BASE_URL: any OpenAI-compatible endpoint — OpenRouter / Aliyun
#                Bailian / OpenAI / DeepSeek / SiliconFlow / local vLLM)
#   TAVILY_API_KEY (single value or CSV multi-key for higher quota)
#   LEAK_DETECTOR_API_KEY (Stage-2 auditor; can reuse LLM_API_KEY by leaving
#                          LEAK_DETECTOR_BASE_URL empty — see §16)
#   MODELS, MODEL_TRAINING_CUTOFFS — list every model under evaluation and its κ_M
```

Declaring $\kappa_M$ for **every** model is mandatory for a fair run: the framework's
admissibility filter is what separates "the model failed to forecast" from "the model already
knew the answer" (paper §3.1, Eq. 4). Models without a declared cutoff are not filtered (a
warning is emitted) and their numbers are not directly comparable to the rest. Cutoffs may be
written at month granularity; FRAME.md §1.3 records the convention "use the last day of the
disclosed month as $\kappa_M$" for safety.

`Settings._post_validate` (config.py:L597) runs **before** any LLM call leaves the process —
empty `LLM_API_KEY`, empty `MODELS`, `:online` suffixes, `::` in slugs, `MIN_SEARCH > min(C)`,
disabled `ENABLE_SEARCH_LEAK_FILTER` paired with a present `LEAK_DETECTOR_API_KEY`,
`GRID_DEFAULT_R/C` outside the configured cells, etc. — all fail-fast so a misconfigured
`.env` cannot waste budget.

### 5.3 Run tests (no API calls required)

```bash
pytest tests/ -q
```

The CI baseline is `test_prompts / test_parser / test_training_cutoff / test_llm_no_browsing /
test_analysis` — those five must stay green. They guard, respectively, the renderer $R$, the
parser $\Psi$, the admissibility filter $\kappa_M$, the provider-native-browsing ban (§4 L4),
and the aggregation rule $\Gamma$. The full suite (33 test files, ~14k lines) covers the
v3/v4 DB migrations, leak filter, exam-score/composite weights, grid dispatcher, react budget
chain, and behavioural diagnostics.

### 5.4 Run an evaluation

```bash
# Smoke: cheapest model, single sample, yes_no only
MODELS=openai/gpt-4o-mini SAMPLING_N=1 \
    python evaluation.py --question-type yes_no

# Full eval with all models, all samples
python evaluation.py

# Filter combinations (AND across flags, OR within each flag)
python evaluation.py --question-type multiple_choice --choice-type multi

# Skip the post-run analysis pass (raw DBs still land in db/)
python evaluation.py --skip-analysis
```

Every invocation creates a fresh folder under `RUNS_ROOT` (default `./runs`), named after the
auto-generated `run_id` `YYYYMMDD-HHMMSS-{4-char hex}`. Resuming with the same `run_id`
continues into the existing folder (§14).

---

## 6. Reproducing the paper's main run

The example DB and the codebase ship with a *deeper search-budget configuration* than the
paper's main run. This is intentional — paper main is tight for discrimination
($R_{\mathrm{tav}}\cdot C = 5\cdot 4 = 20$, "two pages of Google search results"), while
codebase defaults trade a wider budget for smoother behavioural analysis. The paper's main
run (paper §4.1.3 inference protocol + §4.1.4 search-tool configuration) is reproduced by
these `.env` overrides:

```ini
SOURCE_DB=./forecast_eval_set_example.db
SOURCE_TABLE=forecast_eval_set_example
SAMPLING_N=3                                # paper n=3; codebase default 5
REACT_MAX_STEPS=12                          # paper T=12 (matches default)
REACT_MAX_SEARCH_CALLS=4                    # paper C=4; codebase default 8
TAVILY_MAX_RESULTS=5                        # paper R_tav=5 (matches default)
TAVILY_END_DATE_OFFSET_DAYS=-1              # paper δ=1 day, sign per FRAME §1.3
REACT_REFLECTION_PROTOCOL=true              # paper main run on
REACT_BUDGET_AWARENESS_PROTOCOL=true
REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=true
REACT_FORCE_FINAL_ANSWER_LOOKAHEAD=2
REACT_BUDGET_EXCEEDED_DROP_TOOLS=true
REACT_FINAL_ANSWER_RETRY=false              # v5.1 backstop, off by default (§15)
ENABLE_SEARCH_LEAK_FILTER=true              # Stage-2 detector on
BELIEF_PROTOCOL=false                       # paper strict-letter mode
```

Then declare each model's $\kappa_M$ via `MODEL_TRAINING_CUTOFFS` per paper Table 2 (the six
published cutoffs cover $\kappa_M$ ∈ {2025-09-29, 2026-02-11, 2026-02-25, 2026-02-12,
2026-01-27, 2026-03-10}), corresponding to **DeepSeek-V3.2-Exp / GLM 5 / Qwen3.5-Flash /
MiniMax M2.5 / Kimi K2.5 / Doubao Seed 2.0 Lite**.

The paper-vs-default knob diff is reproduced and audited in `FRAME.md` §1.3; the rationale
(why these are the right knobs to vary as a contract, and which knobs are pure engineering
that do not change comparability) is in `DESIGN.md` §13.

---

## 7. Bring your own dataset

The repository ships with `forecast_eval_set_example.db` so that a `git clone` is enough to
reproduce a non-trivial run. The example DB has **319 questions** spanning 2026-01-15 to
2026-04-14; the FutureX-Past instantiation reported in the paper is a curated **80-question
subset** of the same source format, partitioned across yes/no (37) / binary-named (3) /
multiple-choice (40, of which 8 are multi-answer) — paper §4.1.1, Table 1. To plug in a
different corpus,
point `SOURCE_DB` / `SOURCE_TABLE` at any SQLite file/table that follows the same 7-column
schema (`id / choice_type / question_type / event / options / answer / end_time` — see
FRAME.md §2.1) plus a `dataset_metadata` row carrying the eight prompt template keys
(FRAME.md §2.3):

```bash
SOURCE_DB=./my_questions.db
SOURCE_TABLE=my_questions
```

`SOURCE_TABLE` is whitelist-validated against `^[A-Za-z_][A-Za-z0-9_]*$` at startup
(config.py:L586–L595), so a typo fails fast instead of leaking into the SQL layer.

Once the schema is satisfied the paper's evaluation is **dataset-agnostic** — drop in
domain-specific corpora (medical, scientific, engineering forecasting) and the same run unit
$\mathcal{R}$ guarantees the same audit / replay properties (paper §6 Limitations:
$\mathcal{D}$ is a replaceable input component of $\mathcal{R}$, so cross-domain extension is
"a natural unfolding of the framework rather than an internal defect of it").

## 8. Output layout

The output directory **is** the run unit's persisted form. Anyone receiving
`runs/{run_id}/db/{model_slug}.db` can replay the model's evaluation without any other
artefact.

```text
runs/
  {run_id}/
    manifest.json           # run-level metadata: run_id, schema_version, analysis_schema,
                            #   sampling_n, models, filters, source/metadata/templates hashes,
                            #   reflection_protocol_hash, belief_protocol_hash, started_at /
                            #   finished_at — plus a `grid` block when multi-(R, C) is enabled
    db/
      {model_slug}.db       # one SQLite per model; self-contains questions + prompt_templates
                            #   + run_meta + run_results (see §9). Independently distributable.
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
      per_model_composite_by_question_type.csv  # subtype-weighted composite (default 0.15 /
                                                #   0.15 / 0.70 — see §10)
      per_model_composite_by_choice_type.csv    # subtype-weighted composite (default 0.40 /
                                                #   0.60)
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
      figs/                           # only after `python scripts/plot_analysis.py`
                                      #   (matplotlib not in core deps; on-demand)
    logs/
      {run_id}.log
```

Model-slug filesystem safety: `/` → `__`, anything outside `[A-Za-z0-9._-]` → `_`. So
`openai/gpt-4o-mini` becomes `openai__gpt-4o-mini.db`. Grid virtual slugs add `__r{R}__c{C}`
suffixes (§13).

---

## 9. Database schema (per-model, self-contained)

Each model DB holds:

* **`questions`** / **`prompt_templates`** — copies of the source data, so every DB is
  independently replayable without the original `SOURCE_DB`.
* **`run_meta`** — single row: `run_id, model, sampling_n, config_snapshot (redacted),
  filters_snapshot, source/metadata/templates hashes, training_cutoff,
  reflection_protocol_text/hash, belief_protocol_text/hash, started_at, finished_at`.
  The two protocol fingerprints are independent of `prompt_templates_hash` and of each other
  — see DESIGN.md §5 for why three independent fingerprints (template / reflection / belief)
  enable three-axis ablation A/B pairing without collisions.
* **`run_results`** — wide table, **one row per question**. For each $i$ in
  `0..SAMPLING_N-1` a `s{i}_*` group of columns (v3 = 20 columns; v4 adds 3 belief columns;
  v5.1 adds 1 retry column):
  `final_answer_letters / final_answer_raw / correct / parse_ok / tool_calls_count /
  react_steps / prompt_tokens / completion_tokens / reasoning_tokens / latency_ms /
  messages_trace / search_calls / error / created_at` (v2 base) +
  `finish_reason / nudges_used / step_metrics / response_id / system_fingerprint /
  service_tier` (v3 observability) + `belief_final / belief_trace / belief_parse_ok` (v4
  belief) + `final_answer_retry_used` (v5.1, see §15). Old DBs are auto-migrated via
  `ALTER TABLE ADD COLUMN` on first re-open; `Settings.BELIEF_PROTOCOL=false` keeps the v4
  belief columns NULL and leaves all v3 accuracy metrics byte-identical to pre-v4 runs.

**The DB stores raw observations only.** No aggregates are pre-computed — pass@1, pass_any@N,
majority vote, FSS, BI, etc. all come from the `analysis/` pass, which runs automatically at
the end of `evaluation.py` and can also be invoked standalone:

```bash
python -m forecast_eval.analysis runs/{run_id}
```

This separation (raw vs. aggregated) is one of the project's most load-bearing architectural
decisions — DESIGN.md §6 catalogues it as Principle 5 ("Metric definitions evolve faster than
DB schemas; deferring all aggregation to the analysis layer means a metric redefinition never
requires a DB backfill"). Pinned by `tests/test_analysis.py`, which runs the entire analysis
on a hand-crafted DB fixture without re-touching it.

---

## 10. Composite accuracy and the subtype weighting

`per_model_summary.csv` reports a flat mixed mean (`pass_at_1_avg`) for backwards
compatibility. For the headline scoring used in the paper (and recommended for cross-model
comparison), `per_model_composite_*.csv` performs a **weighted composition by sub-question
type** along two dimensions:

* `per_model_composite_by_question_type.csv` — buckets = `yes_no` / `binary_named` /
  `multiple_choice`;
* `per_model_composite_by_choice_type.csv` — buckets = `single` / `multi`.

Per-bucket scoring uses **exam-style partial credit** (paper §4.2.2, Eq. 37
`eq:exam-score`), implemented at `forecast_eval/analysis/exam_score.py:L62`:

$$\text{exam-score}(\hat{S}, G) = \begin{cases} |\hat{S} \cap G| / |G|, & \hat{S} \setminus G = \varnothing,\\ 0, & \hat{S} \setminus G \ne \varnothing.\end{cases}$$

Intuitively, **"any false positive vetoes the score to 0; otherwise score by the proportion
correctly recovered, $|TP|/|G|$"** — i.e., Recall under a zero-FP hard gate. Single-answer
questions ($m_q = 1$) degenerate to the strict-equality $\{0, 1\}$ case (paper §4.2.4
Eq. 40 `eq:strict-equiv`); multi-answer questions retain the asymmetry "rather miss than
wrongly select". The composite formula (paper §4.2.1, Eq. 35 `eq:composite`,
`analysis/composite.py`) is

$$\text{composite}_m = \frac{\sum_{b \in B_{\text{valid}}(m)} w_{m,b}\cdot v_{m,b}}{\sum_{b \in B_{\text{valid}}(m)} w_{m,b}}.$$

$B_{\text{valid}}$ is the set of buckets where the measurement is non-None **and** the
weight is > 0 (paper Eq. 36 `eq:bvalid`); missing buckets are dropped and the remaining
weights renormalised — they are **not** treated as 0. This contract is pinned by
`tests/test_composite_score.py` and `tests/test_exam_score.py`.

**Default weights** follow the *"harder questions discriminate better"* principle
(`config.py:L365–L374`):

| Dimension       | Bucket            | Default weight | Difficulty rationale                                        |
| --------------- | ----------------- | -------------- | ----------------------------------------------------------- |
| `question_type` | `yes_no`          | 0.15           | k=2, blind guess 50%, low cross-model discrimination        |
| `question_type` | `binary_named`    | 0.15           | k=2, adds entity recognition but still binary               |
| `question_type` | `multiple_choice` | 0.70           | k=2..N wide range, includes multi-select, highest signal    |
| `choice_type`   | `single`          | 0.40           | overall easier (includes yes_no / binary_named)             |
| `choice_type`   | `multi`           | 0.60           | true multi-select; near-zero strict baseline → high signal  |

Override these via `COMPOSITE_WEIGHTS_QTYPE` / `COMPOSITE_WEIGHTS_CTYPE` in `.env`; for
per-metric overrides use `COMPOSITE_WEIGHT_OVERRIDES_QTYPE` / `..._CTYPE` (see `.env.example`
comments). When any metric in a (model) row hits an override, its `weights_kind` column is
flagged `overridden`. `composite_meta.json` records buckets_used / weights_used_normalized /
bucket_values for each (model, metric) — a one-to-one reproducible audit trail.

The exam-vs-strict difference matters only on the multi-choice multi-answer bucket (paper
§4.2.4 Eq. 40): the three single-answer buckets satisfy
$\text{exam}_{\text{avg}}^{(b)} \equiv \text{pass@1}_{\text{avg}}^{(b)}$, so the composite
formula's value depends on the exam-vs-strict choice **only** through the multi-multi bucket
— which carries the largest discrimination signal in the paper's main run.

Per-correct cost (paper §4.2.8, Eq. 57 `eq:per-correct`):

$$C^{\text{per-correct}}_m = \frac{C^{\text{total}}_m}{|\mathcal{D}^{\text{eval}}|\cdot n \cdot \text{Composite\,Accuracy}_m},$$

i.e. the platform's actual invoice divided by the *difficulty-weighted notional correct-sample
count*; this places "expensive but accurate" and "cheap but reckless" models on the same
cost-effectiveness scale, avoiding the false-low-cost illusion of "low per-sample unit price
but high error rate".

## 11. v5 hierarchical scoring suite

Scoring follows paper §3.4 as a **hierarchical decomposition** validity → item → question →
model, with the metric definitions developed in §4.2. The headline composite accuracy (§10)
is one column in `per_model_summary.csv`; the companion suite covers stability, consistency,
and chance-corrected skill:

| Metric (paper §, Eq.)               | What it measures                                                       | Code                                                      |
| ----------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------- |
| $\text{pass@1}_{\text{avg}}$ (§4.2.5, Eq. 42) | Single-trial strict-equality hit rate                       | `analysis/accuracy.py`                                    |
| $\text{pass}^{\text{any}}@n$ (§4.2.5, Eq. 44) | Best-of-$n$ hit upper bound                                  | `analysis/accuracy.py`                                    |
| $\text{pass}^{\text{all}}@n$ (§4.2.5, Eq. 45) | All-of-$n$ stability lower bound                             | `analysis/accuracy.py`                                    |
| Cohen's $\kappa$ (§4.2.6, Eq. 46)   | Chance-corrected strict accuracy vs question-type-conditional baseline | `analysis/accuracy.py::cohen_kappa`                       |
| Fleiss' $\kappa$ (§4.2.6, Eq. 49)   | Inter-trial agreement across $K^{\mathrm{eff}}_q$ samples              | `analysis/consistency.py`                                 |
| Tversky $T$ (§4.2.7, Eq. 51)        | Set similarity with FP penalty $\alpha$, FN penalty $\beta$            | `analysis/accuracy.py::tversky_score` (accuracy.py:L286)  |
| FSS (§4.2.7, Eq. 56)                | Tversky-based, chance-corrected skill score; default $(\alpha, \beta) = (2.0, 0.5)$ → FP 4× FN | `analysis/accuracy.py::fss` (accuracy.py:L386) |
| MV-Acc / MVG / VCI / Hamming / mean-entropy (§4.2.10) | Discrete-native consistency family            | `analysis/consistency.py`                                 |

The **FSS reordering** is paper §4.3.3's headline finding: under strict
$\text{pass@1}_{\text{avg}}$ Kimi K2.5 ranks above Qwen3.5-Flash, but under FSS at
$(\alpha, \beta) = (2.0, 0.5)$ Qwen overtakes Kimi because Qwen's selection sets in the
multi-choice multi-answer bucket are more restrained (FP-conservative). This is the single
empirical justification for the asymmetric Tversky weights — encoding "rather miss than
wrongly select" directly into the score function flips a real cross-model ranking, which a
symmetric Jaccard would not have caught.

## 12. On-demand plots and FSS sensitivity

`matplotlib` is **not** in `environment.yml` because the analysis path stays
dependency-light. To render the analytics CSV/JSON into PNGs:

```bash
pip install matplotlib
python scripts/plot_analysis.py runs/{run_id}
```

This populates `runs/{run_id}/analysis/figs/` (gitignored) with:

* **v5 main figures** (paper §C): FSS bar with CI, ΔFSS forest, per-model entropy-Acc grid
  (3 buckets × 3 metrics: Acc / MV Acc / Fleiss κ);
* **Companion / appendix figures**: BI bar with CI (BLF anchor), ΔBI forest, difficulty-grid
  heatmap, per-question belief trajectories (5 sample questions), tool-usage PDP per feature.
  v5 removed reliability-diagram and Murphy-three-decomposition figures because at $K=5$ they
  are statistically meaningless (only 6 unique probability levels per label).

Each plot is best-effort: when the corresponding CSV/JSON is missing, the plot is silently
skipped instead of failing the pipeline.

`per_model_summary.csv` reports a single canonical FSS at $(\alpha, \beta) = (2, 0.5)$.
Reviewers asking "why not Jaccard $(1, 1)$ or strict $(3, 0.5)$?" run the sensitivity sweep
on demand:

```bash
python scripts/fss_sensitivity.py runs/{run_id}              # 4-tier sweep
python scripts/fss_sensitivity.py runs/{run_id} --alpha 1 --beta 1   # single point
```

| (α, β)    | Semantics                                        |
| --------- | ------------------------------------------------ |
| (1, 1)    | Jaccard / symmetric — FP and FN equally penalised |
| (1, 0.5)  | Mild asymmetry — multi-selection error 2× missed |
| (2, 0.5)  | **v5 default** — multi-selection error 4× missed |
| (3, 0.5)  | Strict — multi-selection error 6× missed         |

The script is **not** invoked by `run_analysis`; the sensitivity CSV carries a provenance
comment on top so a reviewer reading the bare file won't mistake it for the main metric
(pinned by `tests/test_fss_sensitivity.py`).

---

## 13. Grid search via virtual model slug

`TAVILY_MAX_RESULTS` ($R_{\mathrm{tav}}$) and `REACT_MAX_SEARCH_CALLS` ($C$) accept
comma-separated lists of positive integers. Setting both to multi-value lists produces
$\lvert\text{MODELS}\rvert \cdot \lvert R\rvert \cdot \lvert C\rvert$ independent **virtual
model slugs** of the form `{real_model}::r{R}::c{C}` (`db.compose_virtual_slug` /
`parse_virtual_slug`, db.py:L477/L500). Each cell lives in its own DB file
(`runs/<id>/db/<real>__r{R}__c{C}.db`) and re-uses every existing analysis stage; an extra
grid pass writes 5 `grid_*.csv` long tables plus a paper figure family under
`analysis/figs/`.

The trick: the runner / DB schema / analysis main pipeline are **byte-unchanged** —
`forecast_eval/analysis/grid.py` decodes the triple from the slug, re-aggregates, and emits
paper long tables. See `DESIGN.md` §10.1 for the design archive (Decisions D1–D10) and
`tests/test_grid_dispatcher.py` / `test_grid_analysis.py` for the contract pinning.

```bash
MODELS=openai/gpt-5,anthropic/claude-sonnet-4.5
TAVILY_MAX_RESULTS=5,10
REACT_MAX_SEARCH_CALLS=1,3,5,8
GRID_DEFAULT_R=5    # main figure anchor; must be in TAVILY_MAX_RESULTS
GRID_DEFAULT_C=5    # symmetric, in REACT_MAX_SEARCH_CALLS

python evaluation.py
python scripts/plot_analysis.py runs/<run_id>
```

A single-value `.env` (e.g. `TAVILY_MAX_RESULTS=5`) is parsed as a length-1 list, so existing
setups stay byte-equivalent except for the new `__r{R}__c{C}` suffix on DB filenames; legacy
v4 runs without a `manifest.grid` block exit the grid path early. `MODELS` entries cannot
contain `::` (config.py:L610–L614) so virtual-slug round-tripping never collides with a real
model name.

---

## 14. Resume semantics

Each `(question_id, sample_idx)` slot is judged independently:

* `s{i}_created_at IS NOT NULL` and `s{i}_error IS NULL` → finished, not retried.
* `s{i}_error = 'skipped_training_cutoff'` → actively excluded by $\kappa_M \le \chi_i$
  check; not retried (it was never a model failure).
* Any other `s{i}_error` value (`network`, `server_5xx`, `bad_request`, `content_policy`, …)
  → next run reuses the DB and retries that slot. Error classification lives in
  `forecast_eval/errors.py:classify` (errors.py:L86); the bucket list (paper §4.1.6) is
  `network / rate_limit / server_5xx / bad_request / content_policy` plus the synthetic
  `skipped_training_cutoff`.

Set `RUN_ID=<existing-run-id>` in `.env` (or CLI env) to resume into the same folder; leaving
it blank mints a fresh `YYYYMMDD-HHMMSS-xxxx` id. `tests/test_runner_resume.py` pins the
behaviour: "completed" rows are never re-emitted, "skipped_training_cutoff" rows are never
re-run, every other error class is retried under the original retry policy.

## 15. Harness resilience switches (v5.1)

Two opt-in resilience levers (default OFF for `REACT_FINAL_ANSWER_RETRY`, default ON for
`REACT_BUDGET_EXCEEDED_DROP_TOOLS`); see
`openspec/changes/harness-resilience-v1/`:

* **`REACT_FINAL_ANSWER_RETRY`** — default **`false`** (config.py:L301). When the ReAct loop
  exits cleanly with empty `final_raw` (model spent all steps on tool_calls and never
  produced content), make one extra `llm_chat` call with `tools=[]` and a fixed "commit your
  `\boxed{...}` answer" user nudge. **Superseded by the in-loop force-final-answer-near-limit
  switch chain below; kept as an optional out-of-loop emergency backstop.** When enabled, the
  retry counts as one step in `react_steps` / `step_metrics` but NOT in `nudges_used`. The
  per-sample column `final_answer_retry_used` (0/1) records the outcome and rolls up to
  `final_answer_retry_rate` in `per_model_summary.csv`. The motivation: cross-model
  comparisons require `parse_failure_rate` to reflect only the model's own format failure,
  not upstream tool-budget exhaustion bookkept by the harness.
* **`REACT_BUDGET_EXCEEDED_DROP_TOOLS`** — default **`true`** (config.py:L302). Once
  cumulative `web_search` calls reach `REACT_MAX_SEARCH_CALLS`, every subsequent LLM call
  drops the tool schema (`tools=[]`). The model can no longer request more searches; it must
  finalise its answer or the bail-out retry above mops up.

The **in-loop priority chain** introduced by `force-final-answer-near-limit-v1` (config.py:L313–L315)
is what really drives termination at the budget edge, and is the reason the post-hoc
`REACT_FINAL_ANSWER_RETRY` is now off by default. At the top of every iteration of
`react.run_react` (react.py:L248) the harness picks **at most one** of four injections, in
this priority order (react.py:L266 priority comment, L272–L334 logic):

1. **Last-step hard cutoff** (`REACT_MAX_STEPS - step == 1` and
   `REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=true`): inject force-finalise text + `tools=[]`,
   the model can ONLY emit content.
2. **Penultimate soft warning** (`remaining ∈ [2, REACT_FORCE_FINAL_ANSWER_LOOKAHEAD]`):
   reminder text, tools still allowed unless the search budget is already gone.
3. **Budget-exhausted commit notice** (cumulative searches `>= REACT_MAX_SEARCH_CALLS` and
   `REACT_BUDGET_EXCEEDED_DROP_TOOLS=true`, fired ONCE per run): tells the model the search
   tool is now gone, please finalise.
4. **Continuation reminder** (previous turn was content without `\boxed{...}` and nothing
   else needs to fire): "your last reply had no `\boxed{...}`, here is the live status".

Defaults: `REACT_BUDGET_AWARENESS_PROTOCOL=true`, `REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=true`,
`REACT_FORCE_FINAL_ANSWER_LOOKAHEAD=2` (config.py:L313–L315). Pinned by the
``tests/test_react.py`` priority-chain section. To run the v5.0 baseline (no in-loop
intervention) flip all three to `false` and the harness reverts to the legacy
"single-shot per turn until budget" behaviour.

Error classification (`forecast_eval/errors.py`) was widened in v5.1: HTTP 400 bodies
containing any of `data_inspection_failed`, `inappropriate content`, or `sensitive` (in
addition to the legacy `content_policy` / `content_filter` / `safety` /
`content_policy_violation` needles, see `errors.CONTENT_POLICY_NEEDLES` at errors.py:L39–L48)
classify as `content_policy`, not `bad_request`. The transient-network family
(errors.py:L97–L111) now also covers `httpx.RemoteProtocolError`, `WriteError`,
`WriteTimeout`, `PoolTimeout` — both the LLM client and the Tavily search client retry these
instead of treating them as fatal.

## 16. Search leak filter (v5.2)

Tavily filters by *crawl/index* date, not by content time. A page indexed before $\chi_i$
may still describe events that happened after it (wiki updates, aggregator pages, "looking
ahead" sections). To plug that hole the framework adds the Stage-2 LLM-based audit
described in paper §3.5 ("controlled information channels") and §4.1.5 ("Semantic Layer"):
every Tavily result is sent through an independent
`detector` LLM (input fields whitelisted to title / URL / published_date / content /
raw_content / cutoff_date — the question text, options, and ground truth are deliberately
withheld so the detector is a leakage classifier, not an answer auditor) that returns
`keep` / `drop` per item. Items the detector flags `drop` are removed before the main LLM
sees the search payload.

Defaults (see `.env.example` for the full annotated block):

* `ENABLE_SEARCH_LEAK_FILTER=true` (config.py:L337) — required to enable the filter; pair
  with `LEAK_DETECTOR_API_KEY` + `LEAK_DETECTOR_MODEL`. Mutually requires
  `ENABLE_WEB_SEARCH=true` (otherwise the detector path is dead code; startup fails fast at
  config.py:L752–L757).
* `LEAK_DETECTOR_BASE_URL` — optional; empty falls back to `LLM_BASE_URL`. The detector
  client is independent of the main LLM client even when the endpoints coincide
  (`leak_filter.get_detector_client`, leak_filter.py:L112) — separate quota / timeout /
  backoff bookkeeping.
* `LEAK_DETECTOR_FAIL_ACTION=drop` (config.py:L351) — fail-closed by default (paper §3.5
  recommendation). Detector errors (HTTP / timeout / invalid-JSON, after
  `LEAK_DETECTOR_RETRY_MAX` retries with `LEAK_DETECTOR_BACKOFF_S`) drop the item. Set to
  `keep` only as an A/B escape hatch when comparing against the unfiltered baseline.
* `LEAK_DETECTOR_RETRY_MAX` / `LEAK_DETECTOR_BACKOFF_S` — independent from the main LLM's
  retry settings, so detector hiccups never push back on the main LLM's quota window.

Audit fields persisted per `web_search` call (`run_results.search_calls` JSON entry):

```text
{ "query": ..., "end_date": ..., "n_results": <kept>,
  "published_dates": [<raw-order, length == n_results_raw>],
  "n_results_raw": <int>, "n_results_kept": <int>,
  "detector_verdicts": ["keep","drop","failed:network", ...],
  "detector_latency_ms": <int>, "detector_error_kind": str | null }
```

`run_meta.config_snapshot` additionally records the detector fingerprint triplet
`leak_detector_enabled` / `leak_detector_model` / `leak_detector_prompt_hash` (sha256 of the
prompt template at `leak_filter.py:L55–L92`, first 16 hex), so the leakage barrier itself is
byte-reproducible. Pinned by `tests/test_leak_filter.py` and the on-the-wire smoke
`scripts/smoke_leak_filter.py` / `scripts/verify_leak_filter_e2e.py`.

Disable path: set `ENABLE_SEARCH_LEAK_FILTER=false` and the detector layer is bypassed
entirely; behaviour is byte-identical to v5.1. The four upstream barriers (web_search schema
/ `end_date` injection / Tavily `end_date` filter / `MODEL_TRAINING_CUTOFFS` / `:online`
ban) remain unaffected.

The paper's $N=270$ audit (paper §4.3.4, Table 8) measured this filter at recall **98.7%**
and per-audit-item residual rate **1.1%** (Wilson 95% upper bound **3.2%**) — comparable to
the lower end of the Tavily-only baseline (3%–16%) and approaching the manual-curation floor
at two orders of magnitude lower marginal cost.

---

## 17. Where to find what (cross-document matrix)

This repository's documentation is layered. Each layer answers a different question; pick
the layer that matches your question and skip the others.

| You want to know…                               | Read…                                   |
| ----------------------------------------------- | --------------------------------------- |
| What this project is, how to run it             | This README                             |
| The paper's formal framework + experimental results | `paper/main.tex` (1115 lines)        |
| *Why* every constraint exists, what was rejected | `DESIGN.md` (1695 lines, 17 sections, 27 rejected-alternative entries) |
| Field-level / interface-level specification (every symbol → module → DB column → test) | `FRAME.md` (2168 lines) |
| The exact rationale for each schema-change proposal | `openspec/changes/<change-id>/`       |
| The exact rationale for each archived schema-change proposal | `openspec/changes/archive/`   |
| Reproducing the paper's main run                | This README §6 + `FRAME.md` §1.3        |
| Contract knobs vs engineering knobs (which `.env` changes invalidate cross-run comparability) | `DESIGN.md` §13 |
| Three independent fingerprints + manifest layout | `FRAME.md` §5; `evaluation.py:_compute_*_protocol` |
| The paper-vs-default knob diff                  | `FRAME.md` §1.3 (this README §6 quotes the override block) |

The four document layers form a **bidirectional contract**: the paper's symbol →
DESIGN's rationale → FRAME's spec → code's implementation, each with its own pinning test.
Each layer can be read in isolation but a contradiction between any pair indicates a bug —
the test suite exists to catch such contradictions early.

## 18. Reading roadmap

If you are new to the project we suggest reading in this order:

1. **`README.md` (this file)** — figure out in 10 minutes what OracleProto is and how to run it.
2. **`paper/main.tex` §§1–3** — the formal framework, the FutureX-Past instantiation, and
   the leakage audit numbers. The §3.3 visible-info, §3.4 evaluation-system, and §3.5
   run-unit + controlled-channels definitions are load-bearing for everything else.
3. **`DESIGN.md`** — the rationale: *why* every constraint exists, the threat model, the
   trade-offs between strict matching and partial credit, why the DB stores raw observations
   only, etc. §0 (foreword) and §1 (framework ↔ code map) give the fastest entry; §13 sorts
   contract knobs from engineering knobs.
4. **`FRAME.md`** — the technical specification at field, interface, and pseudocode level.
   §1.1 (grand map) is the cross-reference scaffold; §2–6 walk top-down from data to
   pipeline.
5. **`forecast_eval/prompts.py` + `forecast_eval/parser.py`** — the renderer $R$ and the
   parser $\Psi$; the heart of the project's information boundary.
6. **`forecast_eval/runner.py` + `forecast_eval/react.py`** — orchestration (admissibility
   filter at runner.py:L132) and the ReAct loop (react.py:L248 main loop, L266 priority
   chain).
7. **`tests/`** — read tests to reverse-engineer the contracts; the 33 test files cover the
   v3/v4/v5 schemas, leak filter, exam-score, grid dispatcher, and behavioural diagnostics.
8. **`paper/main.tex` §4–6** — the experimental setup (§4.1), the metric definitions (§4.2),
   the six-model results (§4.3), and the leakage audit (§4.3.4).
9. **`openspec/changes/archive/`** — to find out *why* things became what they are today.

## 19. Version history (high-level)

| Version | Headline change                                           | Default behaviour                              |
| ------- | --------------------------------------------------------- | ---------------------------------------------- |
| v3      | Wide-table schema, per-sample observability columns       | strict-letter scoring, no belief, no detector  |
| v4      | Belief protocol (companion JSON block), probabilistic suite (BI / NLL / MBS / ABI) | `BELIEF_PROTOCOL=false`; v3 byte-equivalent until enabled |
| v5      | Discrete-native pivot: FSS / Cohen κ / Fleiss κ / Hamming as primary; dropped reliability/Murphy at $K=5$ | exam-style + composite weights are the headline |
| v5.1    | Harness resilience: in-loop force-final + budget-drop tools + retry backstop; widened error needles | force-final on, retry-backstop **off** by default |
| v5.2    | Stage-2 LLM detector for retrieval-content leakage        | `ENABLE_SEARCH_LEAK_FILTER=true` (default-strict, fail-closed) |

Migrations are forward-only: every old DB auto-migrates via `ALTER TABLE ADD COLUMN` on
first re-open. Pinned by `tests/test_db_v4_migration.py` and `tests/test_db_v5_migration.py`.

## 20. Citation

If you use OracleProto in your research, please cite:

```bibtex
@article{ma2026oracleproto,
  title  = {OracleProto: A Reproducible Framework for Benchmarking LLM
            Native Forecasting via Knowledge Cutoff and Temporal Masking},
  author = {Ma, Yiding and Ruan, Chengyun and Huang, Kaibo and
            Yang, Zhongliang and Zhou, Linna},
  year   = {2026},
  note   = {Beijing University of Posts and Telecommunications}
}
```

---

> **One sentence.** OracleProto turns LLM forecasting evaluation from a one-off live
> competition into a dataset-level, auditable, reusable, and trainable capability — by making
> the information boundary part of the data, not part of the prompt.
