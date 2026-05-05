<div align="center">

<img src="static/images/OracleProto_Logo_Horizontal.png" alt="OracleProto Logo" width="100%">

<em>A reproducible framework for benchmarking LLM native forecasting via knowledge cutoff and temporal masking</em>

[![GitHub Stars](https://img.shields.io/github/stars/MaYiding/OracleProto?style=flat-square)](https://github.com/MaYiding/OracleProto/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/MaYiding/OracleProto?style=flat-square)](https://github.com/MaYiding/OracleProto/network)
[![GitHub Issues](https://img.shields.io/github/issues/MaYiding/OracleProto?style=flat-square)](https://github.com/MaYiding/OracleProto/issues)

![GitHub License](https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge)
![Python Version](https://img.shields.io/badge/Python-3.12-brightgreen?style=for-the-badge)

[English](./README.md) | [中文文档](./README-ZH.md) | [Hugging Face](https://huggingface.co/datasets/MaYiding/OracleProto)

View Our Paper: [OracleProto: A Reproducible Framework for Benchmarking LLM Native Forecasting via Knowledge Cutoff and Temporal Masking](/static/paper/OracleProto.pdf)

Visit Our Leaderboards: [oracleproto.pages.dev](https://oracleproto.pages.dev)

</div>

---

## Overview

- **Background & Challenges**: As LLMs evolve into real-world decision-support systems, evaluating their forecasting capability faces a dilemma: live benchmarks are pristine but expire once events resolve, while reproducible retrospective benchmarks struggle to differentiate genuine forecasting from facts models memorized during pre-training, as simply prompting them to "pretend not to know" cannot replace a genuine knowledge boundary.

- **Architecture & Methods**: We propose OracleProto, a reproducible framework that rigorously reconstructs resolved events into time-bounded forecasting samples by combining model-cutoff-aligned sample admission, tool-level temporal masking, content-level leakage detection, discrete answer normalization, and hierarchical scoring.

- **Experimental Results**: Instantiated on a FutureX-Past-derived dataset with six contemporary LLMs, experiments show that OracleProto distinguishes forecasting quality, sampling stability, and cost efficiency under controlled information boundaries. It successfully reduces residual leakage to the 1% level (an order of magnitude below tool-only temporal filtering), turning LLM forecasting from a one-off evaluation into an auditable, reusable dataset-level capability, and providing a unified interface for fair cross-model comparison and a controlled signal source for downstream SFT and RL.

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

## 5. Contact

For questions about code usage, dataset construction, or reproducing results, please reach out to the developers directly:
- **Yiding Ma**: [yidingma@bupt.edu.cn](mailto:yidingma@bupt.edu.cn)
- **Chengyun Ruan**: [ruanchengyun815@bupt.edu.cn](mailto:ruanchengyun815@bupt.edu.cn)

For joint research, dataset and benchmark co-development, or paper collaboration, please contact the principal investigators:
- **Kaibo Huang** (corresponding author): [huangkaibo@bupt.edu.cn](mailto:huangkaibo@bupt.edu.cn)
- **Zhongliang Yang** (corresponding author): [yangzl@bupt.edu.cn](mailto:yangzl@bupt.edu.cn)

---

## 6. Paper

View Our Paper: [OracleProto: A Reproducible Framework for Benchmarking LLM Native Forecasting via Knowledge Cutoff and Temporal Masking](/static/paper/OracleProto.pdf)

---

## 7. Citation

If you use this project in your research, please cite our paper:

```
@article{OracleProto,
  title={OracleProto: A Reproducible Framework for Benchmarking LLM Native Forecasting via Knowledge Cutoff and Temporal Masking},
  author={Yiding Ma, Chengyun Ruan, Kaibo Huang, Zhongliang Yang, Linna Zhou},
  year={2026}
}
```

---