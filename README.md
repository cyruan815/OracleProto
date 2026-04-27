# Forecast Evaluation

LLM forecast evaluation harness against 319 time-bounded prediction questions
from `forecast_eval_set_example.db` (the bundled example dataset). The core
guarantee: the LLM's only information channel is a `web_search` tool whose
`end_date` is injected by the tool layer from each question's `end_time`, so
the model cannot see information published after the event resolution date.

See `FRAME.md` for the full technical framework and
`openspec/changes/bootstrap-forecast-eval/` for the spec-driven change record.

## Bring your own dataset

The shipped example dataset is `forecast_eval_set_example.db` and the question
table inside it is also called `forecast_eval_set_example`. Both names are
configurable — point at any other SQLite file / table that follows the same
7-column schema (`id / choice_type / question_type / event / options / answer
/ end_time`, see `FRAME.md` §2.1) plus a `dataset_metadata` row carrying the
prompt templates:

```bash
SOURCE_DB=./my_questions.db
SOURCE_TABLE=my_questions
```

`SOURCE_TABLE` only accepts a bare SQLite identifier
(`[A-Za-z_][A-Za-z0-9_]*`) and is validated at startup, so a typo there fails
fast instead of leaking into the SQL layer. The defaults
`./forecast_eval_set_example.db` + `forecast_eval_set_example` are used in
every example below; substitute your own as needed.

## Quickstart

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate forecast
```

### 2. Configure `.env`

```bash
cp .env.example .env
# Edit .env and fill LLM_API_KEY + TAVILY_API_KEY.
# LLM_BASE_URL accepts any OpenAI-compatible endpoint (OpenRouter / 阿里百炼 /
# OpenAI / DeepSeek / SiliconFlow / local vLLM — see .env.example comments).
# Also adjust MODELS and MODEL_TRAINING_CUTOFFS for the models you want to
# compare; every model you evaluate should have a cutoff declared so that
# training-data leakage is filtered consistently.
```

### 3. Run tests (no API calls required)

```bash
pytest tests/ -q
```

The CI baseline is `test_prompts / test_parser / test_training_cutoff /
test_llm_no_browsing / test_analysis` — those five must stay green.

### 4. Run an evaluation

```bash
# Smoke: cheapest model, single sample, yes_no only (93 questions)
MODELS=openai/gpt-4o-mini SAMPLING_N=1 \
    python evaluation.py --question-type yes_no

# Full eval with all models, all samples
python evaluation.py

# Filter combinations (AND across flags, OR within each flag)
python evaluation.py --question-type multiple_choice --choice-type multi

# Skip the post-run analysis pass (raw DBs still land in db/)
python evaluation.py --skip-analysis
```

Each invocation of `evaluation.py` creates a fresh folder under `RUNS_ROOT`
(default `./runs`), named after the `run_id`. Resuming with the same `run_id`
continues into that same folder.

## Output layout

```
runs/
  {run_id}/
    manifest.json           # run-level metadata: run_id, schema_version,
                            #   analysis_schema, sampling_n, models, filters,
                            #   source/metadata/templates hashes,
                            #   reflection_protocol_hash, belief_protocol_hash,
                            #   started_at / finished_at
    db/
      {model_slug}.db       # one SQLite per model (see schema below)
    analysis/               # generated after the run finishes
      per_model_summary.csv         # v3 accuracy + v5 discrete-native
                                    #   (FSS / Cohen κ / Hamming / Fleiss κ /
                                    #    mean entropy / VCI / MVG) +
                                    #   v4 companion probabilistic
                                    #   (bi / bi_dec / nll / mbs /
                                    #    abi_crowd / abi_uniform /
                                    #    fallback_share)
      per_model_summary.md          # markdown table with v5 main columns;
                                    #   probabilistic columns flagged with
                                    #   `†` and a K=5-resolution disclaimer
      per_model_by_question_type.csv
      per_model_by_choice_type.csv
      per_model_by_difficulty.csv   # γ-tertile slice (low/mid/high)
      error_breakdown.csv           # byte-regression-tested vs v3
      finish_reason_breakdown.csv   # byte-regression-tested vs v3
      overall.json                  # full structured aggregate, with
                                    #   `probabilistic` sub-object and
                                    #   `analysis_schema` mirrored from
                                    #   manifest
      # ---- v5 K-trial consistency ----
      inter_trial_consistency.csv   # per-model Fleiss κ / mean entropy /
                                    #   VCI / MVG
      entropy_accuracy_bins.csv     # per-model × tertile (Acc / MV Acc /
                                    #   Fleiss κ); per-model bucket
                                    #   boundaries differ by design
      pairwise_bootstrap.csv        # multi-metric paired bootstrap:
                                    #   FSS / Acc / MV_Acc / Fleiss κ / EBI ×
                                    #   model pairs × ΔMean / 95% CI /
                                    #   p-value / Cohen's d / sig flag
      # ---- v4 probabilistic (companion) ----
      shrinkage_alpha_curve.csv     # per-(model, ctype) LOO α scan
      paired_delta_bi.csv           # BS-paired ΔBS + Holm-adj p + posterior
      pairwise_significance.csv     # α=0.05 flag (raw + Holm)
      posterior_pairwise.csv        # P(BI_A > BI_B)
      paired_delta_bi_by_difficulty.csv  # paired bootstrap per tertile
      # ---- Phase 3 behavior ----
      belief_evolution.csv          # per-(model, q, k): volatility,
                                    #   inter-trial variance,
                                    #   convergence_step, evidence_efficiency,
                                    #   counterevidence_engaged
      reflection_ab.csv             # paired A/B (when sibling runs share
                                    #   every hash except reflection)
      tool_usage_pdp.csv            # per-(model, feature, value) PDP for
                                    #   Pr(correct|x) and E[NLL|x]
      confidence_calibration.csv    # subjective confidence vs hit rate
      numeric_confidence_calibration.csv  # max_p binning vs hit rate
      # ---- grid search (only when manifest.grid is present) ----
      grid_summary.csv              # per (real_model, R, C) main table:
                                    #   acc/BI/NLL + 95% CI + cost columns
      grid_marginal_C.csv           # fixed R = grid.default_r, varying C
      grid_marginal_R.csv           # fixed C = grid.default_c, varying R
      grid_pareto.csv               # every cell with `dominated_by` =
                                    #   "" on the Pareto frontier else the
                                    #   lex-smallest dominator slug
      grid_winrate.csv              # pairwise (R, C)-cell win counts +
                                    #   significant-cell tally
      figs/                         # only after `python scripts/plot_analysis.py`
                                    #   (matplotlib not in core deps)
                                    # grid family (multi-cell runs only):
                                    #   grid_pareto_C.png         (Fig 1 main, fix R=default_r)
                                    #   grid_pareto_C_R{R}.png    (per-R appendix)
                                    #   grid_heatmap_RC_<rm>.png  (Fig 2 per real_model)
                                    #   grid_curve_C.png          (Fig 3, BI/NLL/Acc vs C)
                                    #   grid_curve_R.png          (Fig 3, BI/NLL/Acc vs R)
                                    #   grid_winrate_matrix.png   (Fig 4 winrate)
    logs/
      {run_id}.log
```

Model slug safety: `/` → `__`, any character outside `[A-Za-z0-9._-]` → `_`.
So `openai/gpt-4o-mini` becomes `openai__gpt-4o-mini.db`.

### DB schema (per-model, self-contained)

Each model DB holds:

* `questions` / `prompt_templates` — copies of the source data (so every DB is
  independently replayable).
* `run_meta` — single row: `run_id, model, sampling_n, config/filters
  snapshot, source/metadata/templates hashes, training_cutoff, started_at,
  finished_at, reflection_protocol_text/hash, belief_protocol_text/hash`.
  The two protocol fingerprints are independent of `prompt_templates_hash`
  and of each other — see DESIGN.md §5 for why.
* **`run_results` wide table** — one row per question:
  - `question_id` (PK), `user_prompt` (rendered once per question)
  - for each `i` in `0..SAMPLING_N-1`, a `s{i}_*` group of columns
    (v3 = 20 columns; v4 adds 3 belief columns):
    `final_answer_letters / final_answer_raw / correct / parse_ok /
    tool_calls_count / react_steps / prompt_tokens / completion_tokens /
    reasoning_tokens / latency_ms / messages_trace / search_calls / error /
    created_at` (v2 base) +
    `finish_reason / nudges_used / step_metrics / response_id /
    system_fingerprint / service_tier` (v3 observability) +
    `belief_final / belief_trace / belief_parse_ok` (v4 belief).
  - Old DBs are auto-migrated via `ALTER TABLE ADD COLUMN` on first re-open;
    `Settings.BELIEF_PROTOCOL=false` keeps the new belief columns NULL and
    leaves all v3 accuracy metrics byte-identical to pre-v4 runs.

**The DB stores raw observations only.** No aggregates are pre-computed —
all metrics (pass@1, pass_any@N, majority vote, etc.) come from the
`analysis/` pass, which runs automatically at the end of `evaluation.py` and
can also be invoked standalone:

```bash
python -m forecast_eval.analysis runs/{run_id}
```

### On-demand plots (Phase 3)

`matplotlib` is **not** in `environment.yml` because the analysis path is
expected to stay dependency-light. To turn the structured outputs above into
PNGs, install matplotlib locally and run the script that ships under
`scripts/`:

```bash
pip install matplotlib
python scripts/plot_analysis.py runs/{run_id}
```

This populates `runs/{run_id}/analysis/figs/` (already in `.gitignore`) with
**v5 main figures**: FSS bar with CI, ΔFSS forest, per-model entropy-Acc
grid (3 buckets × 3 metrics: Acc / MV Acc / Fleiss κ); plus **appendix /
companion figures**: BI bar with CI (BLF anchor), ΔBI forest, difficulty
grid heatmap, per-question belief trajectories (5 sample questions), and
tool-usage PDP per feature. v5 removed the reliability-diagram and Murphy
three-decomposition figures (Decision 2: K=5 makes them statistically
meaningless). Each plot is best-effort: when the corresponding CSV/JSON
is missing, the plot is silently skipped instead of failing the pipeline.

### On-demand FSS sensitivity (v5)

`per_model_summary.csv` reports a single canonical FSS at the published
$(α, β) = (2, 0.5)$. Reviewers asking "why not Jaccard $(1, 1)$ or strict
$(3, 0.5)$?" run the sensitivity sweep on demand:

```bash
python scripts/fss_sensitivity.py runs/{run_id}              # 4-tier sweep
python scripts/fss_sensitivity.py runs/{run_id} --alpha 1 --beta 1   # single point
```

This writes `runs/{run_id}/analysis/fss_sensitivity.csv` with one row per
(model, $(α, β)$). The default sweep covers four tiers:

| (α, β)    | Semantics                                        |
| --------- | ------------------------------------------------ |
| (1, 1)    | Jaccard / symmetric — FP and FN equally penalised |
| (1, 0.5)  | Mild asymmetry — multi-selection error 2× missed |
| (2, 0.5)  | **v5 default** — multi-selection error 4× missed |
| (3, 0.5)  | Strict — multi-selection error 6× missed         |

The script is **not** invoked by `run_analysis`; the sensitivity CSV
carries a provenance comment on top so a reviewer reading the bare file
won't mistake it for the main metric.

## Grid search quickstart

`TAVILY_MAX_RESULTS` (R) and `REACT_MAX_SEARCH_CALLS` (C) accept a comma-
separated list of positive integers. Setting both to multi-value lists
produces `|MODELS| · |R| · |C|` independent **virtual model slugs** of
the form `{real_model}::r{R}::c{C}`. Each cell lives in its own DB file
(`runs/<id>/db/<real>__r{R}__c{C}.db`) and re-uses every existing
analysis stage; an extra grid pass writes 5 `grid_*.csv` long-tables
plus a paper figure family under `analysis/figs/`.

Example `.env` snippet:

```
MODELS=openai/gpt-5,anthropic/claude-sonnet-4.5
TAVILY_MAX_RESULTS=5,10
REACT_MAX_SEARCH_CALLS=1,3,5,8
GRID_DEFAULT_R=5    # main figure anchor; must be in TAVILY_MAX_RESULTS
GRID_DEFAULT_C=5    # symmetric, in REACT_MAX_SEARCH_CALLS
```

Then run a single command:

```bash
python evaluation.py
python scripts/plot_analysis.py runs/<run_id>
```

Single-value `.env` (the legacy default, e.g. `TAVILY_MAX_RESULTS=5`) is
parsed as a length-1 list, so existing setups stay byte-equivalent
except for the new `__r{R}__c{C}` suffix on DB filenames. See
`DESIGN.md` "grid search via virtual slug (C 方案)" for why we encode
the grid in slug strings rather than introducing a new schema axis.

## Resume semantics

- `s{i}_created_at IS NOT NULL` and `s{i}_error IS NULL` → finished, not retried.
- `s{i}_error = 'skipped_training_cutoff'` → actively filtered by
  `MODEL_TRAINING_CUTOFFS`, not retried.
- Any other `s{i}_error` value (`network`, `server_5xx`, `bad_request`,
  `content_policy`, …) → next run reuses the DB and retries that
  `(question_id, sample_idx)` cell.

## Harness resilience switches (v5.1)

Two opt-out switches default ON; toggle to `false` in `.env` only for A/B
controls (`openspec/changes/harness-resilience-v1/`):

- `REACT_FINAL_ANSWER_RETRY` — when the ReAct loop exits cleanly with an
  empty `final_raw` (model spent all steps on tool_calls and never produced
  content), make one extra `llm_chat` call with `tools=[]` and a fixed
  "commit your `\boxed{...}` answer" user nudge. The retry counts as one
  step in `react_steps` / `step_metrics` but NOT in `nudges_used`. The new
  per-sample column `final_answer_retry_used` (0/1) records the outcome and
  rolls up to `final_answer_retry_rate` in `per_model_summary.csv`.
- `REACT_BUDGET_EXCEEDED_DROP_TOOLS` — once cumulative `web_search` calls
  reach `REACT_MAX_SEARCH_CALLS`, every subsequent LLM call drops the tool
  schema (`tools=[]`). The model can no longer request more searches; it
  must finalise its answer or the bail-out retry above mops up.

Error classification (`forecast_eval/errors.py`) was also widened: HTTP 400
bodies containing any of `data_inspection_failed`, `inappropriate content`,
`敏感`, `违规`, `不当内容`, `审核未通过` (in addition to the legacy
`content_policy` / `content_filter` / `safety` / `content_policy_violation`
needles, see `errors.CONTENT_POLICY_NEEDLES`) classify as `content_policy`,
not `bad_request`. The transient-network family now also covers
`httpx.RemoteProtocolError`, `WriteError`, `WriteTimeout`, `PoolTimeout` —
both the LLM client and the Tavily search client retry these instead of
treating them as fatal.

Set `RUN_ID=<existing-run-id>` in `.env` (or CLI env) to resume into the same
folder; leaving it blank mints a fresh `YYYYMMDD-HHMMSS-xxxx` id.

## Search leak filter (v5.2)

Tavily filters by *crawl/index* date, not by content time. A page indexed
before `q.end_time` may still describe events that happened after it (wiki
updates, aggregator pages, "looking ahead" sections). To plug that hole we
add a Stage-2 LLM-based audit: every Tavily result is sent through an
independent `detector` LLM that returns `keep` / `drop` per item. Items the
detector flags `drop` are removed before the main LLM sees the search payload.

Defaults (see `.env.example` for the full annotated block):

- `ENABLE_SEARCH_LEAK_FILTER=true` — required to enable the filter; pair with
  `LEAK_DETECTOR_API_KEY` + `LEAK_DETECTOR_MODEL`. Mutually requires
  `ENABLE_WEB_SEARCH=true`; otherwise startup fails.
- `LEAK_DETECTOR_BASE_URL` — optional; empty falls back to `LLM_BASE_URL`.
  The detector client is independent of the main LLM client even when the
  endpoints coincide (separate quota / timeout / backoff bookkeeping).
- `LEAK_DETECTOR_FAIL_ACTION=drop` — fail-closed by default: detector errors
  exhaust retries → item is dropped. Set to `keep` only as an A/B escape
  hatch when you need to compare against the unfiltered baseline.
- `LEAK_DETECTOR_RETRY_MAX` / `LEAK_DETECTOR_BACKOFF_S` — independent from
  the main LLM's retry settings, so detector hiccups never push back on the
  main LLM's quota window.

Audit fields persisted per `web_search` call (`run_results.search_calls`
JSON entry):

```
{ "query": ..., "end_date": ..., "n_results": <kept>,
  "published_dates": [<raw-order, length == n_results_raw>],
  "n_results_raw": <int>, "n_results_kept": <int>,
  "detector_verdicts": ["keep","drop","failed:network", ...],
  "detector_latency_ms": <int>, "detector_error_kind": str | null }
```

`run_meta.config_snapshot` additionally records the detector fingerprint
triplet `leak_detector_enabled` / `leak_detector_model` /
`leak_detector_prompt_hash` (sha256 of the prompt template, first 16 hex).

Disable path: set `ENABLE_SEARCH_LEAK_FILTER=false` and the detector layer is
bypassed entirely; behaviour is byte-identical to v5.1. The five existing
information-barrier layers (web_search schema / `end_date` injection / Tavily
`end_date` filter / `MODEL_TRAINING_CUTOFFS` / `:online` ban) remain
unaffected.

## Historical smoke baseline

Prior to the per-run directory refactor, early smoke runs wrote to a single
`results.db`. That baseline has been archived and the raw DB removed; the
first real-API runs against the new layout will become the new reference.
