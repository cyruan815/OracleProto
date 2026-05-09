<div align="center">

<img src="static/images/OracleProto_Logo_Horizontal.png" alt="OracleProto Logo" width="100%">

<em>A reproducible framework for benchmarking LLM native forecasting via knowledge cutoff and temporal masking</em>

</div>

$$\Large \text{Forecasting} = \text{Gathering} \times \text{Synthesis} \times \text{Judgment} \times \text{Decision}$$

<div align="center">

Traditional benchmarks ask: “Can you recall the answer?”<br>
OracleProto asks: “Can you predict the future?”

<b>May every forecast be reproducible, may AI truly become decision support</b><br>
In service of every person’s judgments and choices for a good life

</div>

---

## Overview

- **Background & Challenges:** Evaluating LLM forecasting faces a dilemma: live benchmarks **expire easily**, and retrospective benchmarks suffer from **data leakage**. Prompting cannot establish a genuine **knowledge boundary**.
  
- **Architecture & Methods:** The OracleProto framework combines model knowledge cutoffs and temporal masking to rigorously reconstruct historical events into **reproducible, time-bounded forecasting samples**.
  
- **Experimental Results:** Tests on six contemporary LLMs show that OracleProto effectively distinguishes models' forecasting quality, stability, and cost efficiency. It reduces the leakage rate to 1%, providing a controlled signal source for **model comparison, supervised fine-tuning, and reinforcement learning**.

<div align="center">

<img src="static/images/Framework.png" alt="Framework of OracleProto" width="100%">

Framework of OracleProto

</div>

---

## 1. Code map

```
forecast_eval/                       # core package
├─ runner.py                         # build_task_plan + scheduler
├─ react.py                          # ReAct loop + Tavily end_date injection
├─ leak_filter.py                    # retrieval-content auditor
├─ llm.py                            # OpenAI-compatible client; enforces no provider-side browsing
├─ search.py                         # Tavily wrapper
├─ analysis/                         # scoring and diagnostics: accuracy, FSS, BI, composite, behavior
├─ prompts.py / parser.py            # input renderer R / output parser Ψ
├─ types.py / errors.py / config.py  # data models / typed exceptions / Settings
├─ db.py / loader.py                 # SQLite schema migrations / dataset sync
└─ tavily_keys.py / tools.py         # API-key rotation / tool schemas
evaluation.py                        # entrypoint
scripts/                             # offline tooling
tests/                               # tests
runs/, logs/                         # run artefacts
forecast_eval_set_example.db         # bundled example dataset
```

---

## 2. Quickstart

### 2.1 Environment

Use `uv` :

```bash
uv sync
source .venv/bin/activate
```

or use `Conda` :

```bash
conda env create -f environment.yml
conda activate oracleproto
```

### 2.2 Configure `.env`

```bash
cp .env.example .env
```

Fill `LLM_API_KEY`, `LLM_BASE_URL`, `MODELS`, `MODEL_TRAINING_CUTOFFS`,
`TAVILY_API_KEY`, `LEAK_DETECTOR_API_KEY`, `LEAK_DETECTOR_BASE_URL`,
`LEAK_DETECTOR_MODEL`. The inline notes in [`.env.example`](./.env.example)
cover the rest.

### 2.3 Tests

```bash
pytest tests/ -q
```

### 2.4 Run

```bash
python evaluation.py
```

Each invocation creates `runs/{run_id}/` with `run_id` of the form
`YYYYMMDD-HHMMSS-{4-char hex}`. Set `RUN_ID=<existing-id>` in `.env` to resume that
run in place; completed or ineligible questions are skipped, and transient errors
retry under the original backoff policy.

---

## 3. Bring your own dataset

The bundled `forecast_eval_set_example.db` carries 80 curated questions across
three question types, with dates spanning 2026-03-12 to 2026-04-14. To plug in
another corpus, swap `SOURCE_DB` and `SOURCE_TABLE` in `.env`.

---

## 4. Outputs

```
runs/{run_id}/
├─ manifest.json          # run-level metadata and hash chain
├─ db/{model_slug}.db     # one SQLite per model, independently replayable
├─ analysis/              # CSV/JSON regenerated from the raw DB
└─ logs/{run_id}.log
```

The DB stores raw observations only. Every aggregate ($`\text{pass@1}`$, FSS, BI,
composite, …) is recomputed by `forecast_eval/analysis/`, which runs at the end of
`evaluation.py` and can also be invoked standalone:

```bash
python -m forecast_eval.analysis runs/{run_id}
```