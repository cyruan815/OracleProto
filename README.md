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

_To be filled after first real-API smoke run (see task 11.3)._
