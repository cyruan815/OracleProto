# OracleProto

**A reproducible framework for benchmarking LLM native forecasting via knowledge cutoff and temporal masking.**

This repository is the reference implementation of the paper *OracleProto: A Reproducible
Framework for Benchmarking LLM Native Forecasting via Knowledge Cutoff and Temporal Masking*
(Ma, Ruan, Huang, Yang & Zhou; BUPT). The framework reconstructs **resolved events** into
**time-bounded forecasting samples** so that the evaluation object lives at the dataset level —
auditable, replayable, and comparable across models and across calendar years.

> **Summed up in one sentence.** This codebase turns a forecasting evaluation into a single,
> reproducible run unit
> $\mathcal{R}=(\mathcal{D},M,\kappa_M,\delta,T,C,R,\Psi,\phi,\Gamma)$, fixes every input
> (questions, knowledge cutoffs, temporal masking, ReAct budgets, prompt rendering, output
> parsing, label normalization, and aggregation rules), and emits scoring artefacts that are
> bit-identical to byte-identical configurations.

---

## 1. The problem this project solves

Existing forecasting evaluations sit on an unstable middle ground:

* **Prospective live benchmarks** (ForecastBench, FutureX) are contamination-controlled but
  evaporate the moment the event resolves; the leaderboard is a one-way temporal stream rather
  than a reusable artefact.
* **Retrospective benchmarks** (FutureX-Past, archived live questions) are reproducible but are
  highly prone to mistaking *factual recall* for *forecasting capability*: by the time the
  paper is written, the answer is sitting in the model's training corpus.

Prompt-time discipline ("imagine you do not know that the election has resolved") cannot bridge
this gap — the diagnostic literature (Paleka et al., Li et al.) has empirically shown a
substantial systematic gap between **simulated ignorance** and **true ignorance**, and that a
1–5% label-noise rate alone is enough to break proper scoring rules.

OracleProto's response is to push the discipline **one level deeper, into the dataset itself**.
For a given (model, question) pair, the question is admitted only if its prediction cutoff
$\chi_i$ satisfies $\kappa_M \le \chi_i < \tau_i$ — i.e. the model's parametric knowledge is
not more recent than the permitted prediction environment, and the resolution time has not yet
arrived in the simulated information state. Inadmissible questions are *not counted as model
errors*; they are filtered out and audited separately.

## 2. Three contributions

Following the paper's structure, this repository delivers:

1. **A formal dataset-level framework for LLM forecasting evaluation.** The evaluation object
   is no longer a one-shot live result, but a dataset-level task that is definable, auditable,
   and reproducible — anyone can re-run the same `(D, M, κ_M, δ, T, C, R, Ψ, φ, Γ)` and obtain
   comparable numbers.
2. **The OracleProto unified evaluation protocol.** Knowledge cutoffs (sample admission),
   tool-level temporal masking, content-level leakage detection (a Stage-2 LLM auditor),
   discrete answer normalization, and hierarchical scoring (validity → item → question → model)
   are wired into one pipeline.
3. **A systematic evaluation benchmark and a trainable forecasting harness.** The example
   dataset bundled with the repo, plus the FutureX-Past instantiation reported in the paper,
   provide a leakage-controlled forecasting evaluation set. Outputs (per-sample raw records,
   per-model SQLite databases, hierarchical analytics) are immediately reusable as signals for
   SFT, RL, and forecasting-agent training.

## 3. The run unit

Every invocation of `evaluation.py` materialises a single run unit
$\mathcal{R}=(\mathcal{D},M,\kappa_M,\delta,T,C,R,\Psi,\phi,\Gamma)$:

| Symbol         | Object                       | Where it lives in the codebase                                               |
| -------------- | ---------------------------- | --------------------------------------------------------------------------- |
| $\mathcal{D}$  | Discrete forecasting dataset | `SOURCE_DB` / `SOURCE_TABLE` (default `forecast_eval_set_example.db`)        |
| $M$            | Evaluated model              | one entry of `MODELS`; one SQLite file per $M$                               |
| $\kappa_M$     | Knowledge cutoff             | `MODEL_TRAINING_CUTOFFS[M]` (sample admission)                                |
| $\delta$       | Temporal masking offset      | `TAVILY_END_DATE_OFFSET_DAYS` (default `-1`); injected at the tool layer     |
| $T$            | Max ReAct steps              | `REACT_MAX_STEPS`                                                            |
| $C$            | Max search calls             | `REACT_MAX_SEARCH_CALLS`                                                     |
| $R$            | Input renderer               | `forecast_eval/prompts.py::render_user_prompt`                               |
| $\Psi$         | Output parser / validity     | `forecast_eval/parser.py::parse_answer`                                      |
| $\phi$         | Answer normalization map     | letter encoding (`A`, `A,B` …) defined per question_type                      |
| $\Gamma$       | Aggregation rule             | `forecast_eval/analysis/*` (composite accuracy, FSS, κ, etc.)                |

The auxiliary leakage detector $H_{\mathrm{aux}}$ (Stage 2 of the three-layer barrier) is logged
as run-configuration metadata alongside $\mathcal{R}$, with its prompt SHA-256 stored in
`run_meta.config_snapshot`, so the leakage barrier is itself byte-reproducible.

## 4. The three-layer leakage barrier

Tool-level date filtering alone leaves ~3–16% of real leaks reaching the main LLM (cached
pages, aggregator pages, "looking ahead" sections still passing Tavily's `published_date`
filter). The paper's audit on $N=270$ items measured the residual rate at $\mathrm{FN}/N
\approx 1.1\%$ once the Stage-2 auditor is wired in, with the Wilson 95% upper bound landing at
$\approx 3.2\%$ — an order of magnitude below the Tavily-only baseline.

| Layer                      | Where                                       | Default          | Target leakage source                                          |
| -------------------------- | ------------------------------------------- | ---------------- | -------------------------------------------------------------- |
| **L0 manual curation**     | upstream dataset construction               | always           | event description leaking the answer in the question text       |
| **L1 admissibility filter** | $\kappa_M \le \chi_i$ check at task generation | `MODEL_TRAINING_CUTOFFS` | parametric-memory leakage (the model "remembers" the answer)    |
| **L2 algorithmic (Tavily)** | `end_date = \tau_i + \delta` injected at tool layer | $\delta=-1$ day  | retrieval results indexed after $\chi_i$                        |
| **L3 semantic (detector)** | independent LLM auditor on each Tavily item | `claude-sonnet-4.6` | retrieved page bodies that mention events after $\chi_i$        |

Together with the **provider-native browsing ban** (no `:online`, no `plugins`, no
provider-specific web tool — `forecast_eval/llm.py` enforces this) the four channels close the
controllable leakage surface. The remaining residual (question-text temporal cues, post-training
external knowledge backflow) is acknowledged as evaluation bias rather than pretended away.

---

## 5. Bring your own dataset

The repository ships with `forecast_eval_set_example.db` so that a `git clone` is enough to
reproduce a non-trivial run. The example DB has 319 questions; the FutureX-Past instantiation
reported in the paper is a curated 80-question subset of the same source format. To plug in a
different corpus, point `SOURCE_DB` / `SOURCE_TABLE` at any SQLite file/table that follows the
same 7-column schema (`id / choice_type / question_type / event / options / answer / end_time`,
see `FRAME.md` §2.1) plus a `dataset_metadata` row carrying the prompt templates:

```bash
SOURCE_DB=./my_questions.db
SOURCE_TABLE=my_questions
```

`SOURCE_TABLE` is whitelist-validated against `[A-Za-z_][A-Za-z0-9_]*` at startup, so a typo
fails fast instead of leaking into the SQL layer.

## 6. Quickstart

### 6.1 Create the conda environment

```bash
conda env create -f environment.yml
conda activate forecast
```

### 6.2 Configure `.env`

```bash
cp .env.example .env
# Edit .env and fill in:
#   LLM_API_KEY (and LLM_BASE_URL: any OpenAI-compatible endpoint — OpenRouter / Aliyun
#                Bailian / OpenAI / DeepSeek / SiliconFlow / local vLLM)
#   TAVILY_API_KEY (single value or CSV multi-key for higher quota)
#   LEAK_DETECTOR_API_KEY (Stage-2 auditor; can reuse LLM_API_KEY by leaving
#                          LEAK_DETECTOR_BASE_URL empty)
#   MODELS, MODEL_TRAINING_CUTOFFS — list every model under evaluation and its κ_M
```

Declaring $\kappa_M$ for **every** model is mandatory for a fair run: the framework's
admissibility filter is what separates "the model failed to forecast" from "the model already
knew the answer". Models without a declared cutoff are not filtered (a warning is emitted) and
their numbers are not directly comparable to the rest.

### 6.3 Run tests (no API calls required)

```bash
pytest tests/ -q
```

The CI baseline is `test_prompts / test_parser / test_training_cutoff / test_llm_no_browsing /
test_analysis` — those five must stay green. They guard, respectively, the renderer $R$, the
parser $\Psi$, the admissibility filter, the provider-native-browsing ban, and the aggregation
rule $\Gamma$.

### 6.4 Run an evaluation

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
continues into the existing folder.

---

## 7. Output layout

The output directory **is** the run unit's persisted form. Anyone receiving
`runs/{run_id}/db/{model_slug}.db` can replay the model's evaluation without any other artefact.

```text
runs/
  {run_id}/
    manifest.json           # run-level metadata: run_id, schema_version, analysis_schema,
                            #   sampling_n, models, filters, source/metadata/templates hashes,
                            #   reflection_protocol_hash, belief_protocol_hash, started_at /
                            #   finished_at — plus a `grid` block when multi-(R, C) is enabled
    db/
      {model_slug}.db       # one SQLite per model; self-contains questions + prompt_templates
                            #   + run_meta + run_results (see §8). Independently distributable.
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
                                                #   0.15 / 0.70 — see §9)
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
`openai/gpt-4o-mini` becomes `openai__gpt-4o-mini.db`.

---

## 8. Database schema (per-model, self-contained)

Each model DB holds:

* **`questions`** / **`prompt_templates`** — copies of the source data, so every DB is
  independently replayable without the original `SOURCE_DB`.
* **`run_meta`** — single row: `run_id, model, sampling_n, config_snapshot (redacted),
  filters_snapshot, source/metadata/templates hashes, training_cutoff,
  reflection_protocol_text/hash, belief_protocol_text/hash, started_at, finished_at`.
  The two protocol fingerprints are independent of `prompt_templates_hash` and of each other
  (see DESIGN.md §5 for why three independent fingerprints enable three-axis ablation).
* **`run_results`** — wide table, **one row per question**. For each $i$ in
  `0..SAMPLING_N-1` a `s{i}_*` group of columns (v3 = 20 columns; v4 adds 3 belief columns):
  `final_answer_letters / final_answer_raw / correct / parse_ok / tool_calls_count /
  react_steps / prompt_tokens / completion_tokens / reasoning_tokens / latency_ms /
  messages_trace / search_calls / error / created_at` (v2 base) +
  `finish_reason / nudges_used / step_metrics / response_id / system_fingerprint /
  service_tier` (v3 observability) + `belief_final / belief_trace / belief_parse_ok` (v4
  belief). Old DBs are auto-migrated via `ALTER TABLE ADD COLUMN` on first re-open;
  `Settings.BELIEF_PROTOCOL=false` keeps the new columns NULL and leaves all v3 accuracy
  metrics byte-identical to pre-v4 runs.

**The DB stores raw observations only.** No aggregates are pre-computed — pass@1, pass_any@N,
majority vote, FSS, BI, etc. all come from the `analysis/` pass, which runs automatically at the
end of `evaluation.py` and can also be invoked standalone:

```bash
python -m forecast_eval.analysis runs/{run_id}
```

This separation (raw vs. aggregated) is one of the project's most load-bearing architectural
decisions. Metric definitions evolve faster than DB schemas; deferring all aggregation to the
analysis layer means a metric redefinition never requires a DB backfill.

---

## 9. Composite accuracy and the subtype weighting

`per_model_summary.csv` reports a flat mixed mean (`pass_at_1_avg`) for backwards compatibility.
For the headline scoring used in the paper (and recommended for cross-model comparison),
`per_model_composite_*.csv` performs a **weighted composition by sub-question type** along two
dimensions:

* `per_model_composite_by_question_type.csv` — buckets = `yes_no` / `binary_named` /
  `multiple_choice`;
* `per_model_composite_by_choice_type.csv` — buckets = `single` / `multi`.

Per-bucket scoring uses **exam-style partial credit** (paper §3.2): any false positive vetoes
the score to 0; otherwise score $\lvert\hat S \cap G\rvert / \lvert G\rvert$. The composite
formula (paper Eq. 18) is

$$\text{composite}_m = \frac{\sum_{b \in B_{\text{valid}}} w_{m,b}\cdot v_{m,b}}{\sum_{b \in B_{\text{valid}}} w_{m,b}}.$$

$B_{\text{valid}}$ is the set of buckets where the measurement is non-None **and** the weight is
> 0; missing buckets are dropped and the remaining weights renormalised (they are not treated
as 0).

**Default weights** follow the *"harder questions discriminate better"* principle:

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
flagged `overridden`. `composite_meta.json` records buckets_used /
weights_used_normalized / bucket_values for each (model, metric) — a one-to-one
reproducible audit trail.

---

## 10. On-demand plots and FSS sensitivity (Phase 3)

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
  v5 removed reliability-diagram and Murphy-three-decomposition figures because at K=5 they are
  statistically meaningless (only 6 unique probability levels per label).

Each plot is best-effort: when the corresponding CSV/JSON is missing, the plot is silently
skipped instead of failing the pipeline.

`per_model_summary.csv` reports a single canonical FSS at $(α, β) = (2, 0.5)$. Reviewers asking
"why not Jaccard $(1, 1)$ or strict $(3, 0.5)$?" run the sensitivity sweep on demand:

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
comment on top so a reviewer reading the bare file won't mistake it for the main metric.

---

## 11. Grid search via virtual model slug

`TAVILY_MAX_RESULTS` (R) and `REACT_MAX_SEARCH_CALLS` (C) accept comma-separated lists of
positive integers. Setting both to multi-value lists produces $\lvert\text{MODELS}\rvert \cdot
\lvert R\rvert \cdot \lvert C\rvert$ independent **virtual model slugs** of the form
`{real_model}::r{R}::c{C}`. Each cell lives in its own DB file
(`runs/<id>/db/<real>__r{R}__c{C}.db`) and re-uses every existing analysis stage; an extra
grid pass writes 5 `grid_*.csv` long tables plus a paper figure family under `analysis/figs/`.

The trick: the runner / DB schema / analysis main pipeline are **byte-unchanged** —
`forecast_eval/analysis/grid.py` decodes the triple from the slug, re-aggregates, and emits
paper long tables. See `DESIGN.md` §10.1 for the design archive (Decisions D1–D10).

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
setups stay byte-equivalent except for the new `__r{R}__c{C}` suffix on DB filenames; legacy v4
runs without a `manifest.grid` block exit the grid path early.

---

## 12. Resume semantics

Each `(question_id, sample_idx)` slot is judged independently:

* `s{i}_created_at IS NOT NULL` and `s{i}_error IS NULL` → finished, not retried.
* `s{i}_error = 'skipped_training_cutoff'` → actively excluded by $\kappa_M \le \chi_i$ check;
  not retried (it was never a model failure).
* Any other `s{i}_error` value (`network`, `server_5xx`, `bad_request`, `content_policy`, …) →
  next run reuses the DB and retries that slot.

Set `RUN_ID=<existing-run-id>` in `.env` (or CLI env) to resume into the same folder; leaving it
blank mints a fresh `YYYYMMDD-HHMMSS-xxxx` id.

## 13. Harness resilience switches (v5.1)

Two opt-out switches default ON; toggle to `false` in `.env` only for A/B controls (see
`openspec/changes/harness-resilience-v1/`):

* **`REACT_FINAL_ANSWER_RETRY`** — when the ReAct loop exits cleanly with empty `final_raw`
  (model spent all steps on tool_calls and never produced content), make one extra `llm_chat`
  call with `tools=[]` and a fixed "commit your `\boxed{...}` answer" user nudge. The retry
  counts as one step in `react_steps` / `step_metrics` but NOT in `nudges_used`. The new
  per-sample column `final_answer_retry_used` (0/1) records the outcome and rolls up to
  `final_answer_retry_rate` in `per_model_summary.csv`. The motivation: cross-model
  comparisons require `parse_failure_rate` to reflect only the model's own format failure, not
  upstream tool-budget exhaustion bookkept by the harness.
* **`REACT_BUDGET_EXCEEDED_DROP_TOOLS`** — once cumulative `web_search` calls reach
  `REACT_MAX_SEARCH_CALLS`, every subsequent LLM call drops the tool schema (`tools=[]`). The
  model can no longer request more searches; it must finalise its answer or the bail-out retry
  above mops up.

Error classification (`forecast_eval/errors.py`) was widened in v5.1:
HTTP 400 bodies containing any of `data_inspection_failed`, `inappropriate content`, or
`sensitive` (in addition to the legacy `content_policy` / `content_filter` / `safety` /
`content_policy_violation` needles, see `errors.CONTENT_POLICY_NEEDLES`) classify as
`content_policy`, not `bad_request`. The transient-network family now also covers
`httpx.RemoteProtocolError`, `WriteError`, `WriteTimeout`, `PoolTimeout` — both the LLM client
and the Tavily search client retry these instead of treating them as fatal.

## 14. Search leak filter (v5.2)

Tavily filters by *crawl/index* date, not by content time. A page indexed before $\chi_i$ may
still describe events that happened after it (wiki updates, aggregator pages, "looking ahead"
sections). To plug that hole the framework adds a Stage-2 LLM-based audit: every Tavily result
is sent through an independent `detector` LLM (input fields whitelisted to title / URL /
published_date / content / raw_content / cutoff_date — the question text, options, and ground
truth are deliberately withheld so the detector is a leakage classifier, not an answer
auditor) that returns `keep` / `drop` per item. Items the detector flags `drop` are removed
before the main LLM sees the search payload.

Defaults (see `.env.example` for the full annotated block):

* `ENABLE_SEARCH_LEAK_FILTER=true` — required to enable the filter; pair with
  `LEAK_DETECTOR_API_KEY` + `LEAK_DETECTOR_MODEL`. Mutually requires `ENABLE_WEB_SEARCH=true`;
  otherwise startup fails.
* `LEAK_DETECTOR_BASE_URL` — optional; empty falls back to `LLM_BASE_URL`. The detector client
  is independent of the main LLM client even when the endpoints coincide (separate quota /
  timeout / backoff bookkeeping).
* `LEAK_DETECTOR_FAIL_ACTION=drop` — fail-closed by default. Detector errors (HTTP / timeout /
  invalid-JSON, after `LEAK_DETECTOR_RETRY_MAX` retries with `LEAK_DETECTOR_BACKOFF_S`) drop
  the item. Set to `keep` only as an A/B escape hatch when comparing against the unfiltered
  baseline.
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
prompt template, first 16 hex), so the leakage barrier itself is byte-reproducible.

Disable path: set `ENABLE_SEARCH_LEAK_FILTER=false` and the detector layer is bypassed
entirely; behaviour is byte-identical to v5.1. The four upstream barriers (web_search schema /
`end_date` injection / Tavily `end_date` filter / `MODEL_TRAINING_CUTOFFS` / `:online` ban)
remain unaffected.

The paper's `N=270` audit measured this filter at recall 98.7% and per-audit-item residual rate
1.1% (Wilson 95% upper bound 3.2%) — comparable to the lower end of the Tavily-only baseline
and approaching the manual-curation floor at two orders of magnitude lower marginal cost.

---

## 15. Reading roadmap

If you are new to the project we suggest reading in this order:

1. **`README.md` (this file)** — figure out in 10 minutes what OracleProto is and how to run it.
2. **`DESIGN.md`** — the rationale: *why* every constraint exists, the threat model, the
   trade-offs between strict matching and partial credit, why the DB stores raw observations
   only, etc.
3. **`FRAME.md`** — the technical specification at field, interface, and pseudocode level. The
   bridge between the paper's framework $\mathcal{R}$ and the Python implementation.
4. **`forecast_eval/prompts.py` + `forecast_eval/parser.py`** — the renderer $R$ and the parser
   $\Psi$; the heart of the project's information boundary.
5. **`forecast_eval/runner.py` + `forecast_eval/react.py`** — orchestration and the ReAct loop.
6. **`tests/`** — read tests to reverse-engineer the contracts.
7. **`paper/main.tex`** — the formal framework, the FutureX-Past instantiation, and the leakage
   audit numbers.
8. **`openspec/changes/archive/`** — to find out *why* things became what they are today.

## 16. Citation

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

> **One sentence.** OracleProto turns LLM forecasting evaluation from a one-off live competition
> into a dataset-level, auditable, reusable, and trainable capability — by making the
> information boundary part of the data, not part of the prompt.
