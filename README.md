# Forecast Evaluation

LLM forecast evaluation harness against 322 time-bounded prediction questions
from `forecast_eval_set.db`. The core guarantee: the LLM's only information
channel is a `web_search` tool whose `end_date` is injected by the tool layer
from each question's `end_time`, so the model cannot see information published
after the event resolution date.

See `FRAME.md` for the full technical framework and
`openspec/changes/bootstrap-forecast-eval/` for the spec-driven change record.

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
test_llm_no_browsing` — those four must stay green.

### 4. Run an evaluation

```bash
# Smoke: cheapest model, single sample, yes_no only (93 questions)
MODELS=openai/gpt-4o-mini SAMPLING_N=1 \
    python evaluation.py --question-type yes_no

# Full eval with all models, all samples
python evaluation.py

# Filter combinations (AND across flags, OR within each flag)
python evaluation.py --question-type multiple_choice --choice-type multi
```

The evaluation process writes every completed sample to `results.db`; rerunning
with the same `RUN_ID` resumes from the last successful row. Progress logs go to
stderr and `logs/{run_id}.log`.

## Smoke test baseline

First real-API smoke on 2026-04-24 against `qwen3.6-plus-2026-04-02` (via
dashscope OpenAI-compatible endpoint) + Tavily search, `SAMPLING_N=1`,
`REACT_MAX_STEPS=10`, `REACT_MAX_SEARCH_CALLS=8`.

| run_id | filter | cutoff | eligible | pass@1 | wall | avg latency | avg steps | avg tool calls | tokens (p/c/r) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `20260424-104322-35e4` | `--question-type binary_named` | 2026-04-02 | 1 | 1/1 (100%) | 68s | 67.5s | 2.0 | 3.0 | 5.5k / 3.4k / 2.9k |
| `20260424-104547-252c` | `--question-type yes_no` | 2026-04-11 | 3 | 2/3 (66.7%) | 6m33s | 333.4s | 5.7 | 5.7 | 161k / 52.5k / 50.8k |

Notes:
- `skipped_training_cutoff` rows: 10 + 90; all cutoff rows persist without
  calling the LLM, as expected.
- `runs.finished_at` populated; `pass@1` computed as `correct / eligible`.
- `messages_trace` is valid JSON (6 / 12–13 turns per sample); every
  `search_calls` entry carries `end_date` injected from `q.end_time`.
- 0 row-level errors across 4 live samples.
