<div align="center">

<img src="static/images/OracleProto_Logo_Horizontal.png" alt="OracleProto Logo" width="100%">

<em>A reproducible framework for benchmarking LLM native forecasting via knowledge cutoff and temporal masking</em>

[![GitHub Stars](https://img.shields.io/github/stars/MaYiding/OracleProto?style=flat-square)](https://github.com/MaYiding/OracleProto/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/MaYiding/OracleProto?style=flat-square)](https://github.com/MaYiding/OracleProto/network)
[![GitHub Issues](https://img.shields.io/github/issues/MaYiding/OracleProto?style=flat-square)](https://github.com/MaYiding/OracleProto/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/MaYiding/OracleProto?style=flat-square)](https://github.com/MaYiding/OracleProto/pulls)

![GitHub License](https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge)
![Python Version](https://img.shields.io/badge/Python-3.12-brightgreen?style=for-the-badge)

[English](./README.md) | [中文文档](./README-ZH.md) | [Hugging Face](https://huggingface.co/datasets/MaYiding/OracleProto) | Paper

View Paper: [OracleProto: A Reproducible Framework for Benchmarking LLM Native Forecasting via Knowledge Cutoff and Temporal Masking](/static/paper/OracleProto.pdf)

Visit Our Leaderboards: [oracleproto.pages.dev](https://oracleproto.pages.dev)

</div>

---

## Overview

As Large Language Models (LLMs) evolve toward real-world decision-support systems, evaluating their "native forecasting capability" faces a fundamental tension: prospective live benchmarks offer a gold standard for contamination control but expire immediately upon event resolution, while reproducible retrospective benchmarks are highly prone to mistaking pre-training memorization for genuine forecasting. To address this challenge, we propose OracleProto, a reproducible evaluation framework for LLM native forecasting capability. By jointly enforcing model knowledge cutoff alignment, tool-level temporal masking, content-level leakage detection, and standardized hierarchical scoring, the framework reconstructs resolved events into forecasting samples with strict temporal boundaries. Evaluations across six mainstream LLMs demonstrate that OracleProto effectively distinguishes forecasting quality, stability, and cost efficiency under controlled information boundaries, while reducing the residual leakage rate to the 1% level. Ultimately, OracleProto transforms one-off forecasting evaluation into an auditable, reusable dataset-level capability that provides a controlled signal source for subsequent supervised fine-tuning (SFT) and reinforcement learning (RL).

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

---