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
      per_model_summary.csv         # accuracy + v4 probabilistic columns
                                    #   (bi / bi_dec / nll / mbs /
                                    #    abi_crowd / abi_uniform /
                                    #    fallback_share)
      per_model_summary.md          # markdown table; Phase 2 adds
                                    #   BI_cal / NLL_cal / ECE_uncal /
                                    #   ECE_cal columns + `cal*` overfit
                                    #   marker when LOO calibration looks
                                    #   over-confident
      per_model_summary_calibrated.csv  # Phase 2: BI/NLL/ECE/ABI uncal+cal
      per_model_by_question_type.csv
      per_model_by_choice_type.csv
      per_model_by_difficulty.csv   # Phase 2: γ-tertile slice (low/mid/high)
      error_breakdown.csv           # byte-regression-tested vs v3
      finish_reason_breakdown.csv   # byte-regression-tested vs v3
      overall.json                  # full structured aggregate, with
                                    #   `probabilistic` sub-object and
                                    #   `analysis_schema` mirrored from
                                    #   manifest
      # ---- Phase 2 calibration ----
      calibration_params.json       # per-(model, cell) Platt / temperature
      reliability_data.json
      reliability_data_calibrated.json
      brier_decomposition.csv       # Murphy rel/res/unc, uncal + cal
      # ---- Phase 2 aggregation ----
      shrinkage_alpha_curve.csv     # per-(model, ctype) LOO α scan
      # ---- Phase 2 inference ----
      paired_delta_bi.csv           # pairwise ΔBS + Holm-adj p + posterior
      pairwise_significance.csv     # α=0.05 flag (raw + Holm)
      posterior_pairwise.csv        # P(BI_A > BI_B)
      paired_delta_bi_by_difficulty.csv  # paired bootstrap per tertile
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

## Resume semantics

- `s{i}_created_at IS NOT NULL` and `s{i}_error IS NULL` → finished, not retried.
- `s{i}_error = 'skipped_training_cutoff'` → actively filtered by
  `MODEL_TRAINING_CUTOFFS`, not retried.
- Any other `s{i}_error` value (`network`, `server_5xx`, `bad_request`,
  `content_policy`, …) → next run reuses the DB and retries that
  `(question_id, sample_idx)` cell.

Set `RUN_ID=<existing-run-id>` in `.env` (or CLI env) to resume into the same
folder; leaving it blank mints a fresh `YYYYMMDD-HHMMSS-xxxx` id.

## Historical smoke baseline

Prior to the per-run directory refactor, early smoke runs wrote to a single
`results.db`. That baseline has been archived and the raw DB removed; the
first real-API runs against the new layout will become the new reference.
