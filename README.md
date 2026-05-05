<div align="center">

<img src="static/images/OracleProto_Logo_Horizontal.png" alt="OracleProto Logo" width="100%">

<em>A reproducible framework for benchmarking LLM native forecasting via knowledge cutoff and temporal masking</em>

[English](./README.md) | [中文文档](./README-ZH.md) | [Hugging Face](https://huggingface.co/datasets/MaYiding/OracleProto)

Visit Our Leaderboards: [oracleproto.pages.dev](https://oracleproto.pages.dev)

</div>

---

## Overview

Forecasting benchmarks live with a structural tension: forward-looking ones expire the moment their events resolve, while retrospective ones risk testing recall, because prompt-level instructions cannot push a model back across a knowledge boundary it has already crossed. OracleProto closes the gap at the dataset layer rather than the prompt layer, rewriting each resolved event into a forecasting sample bounded by the model's own knowledge cutoff and packaging it as a self-contained run that stays byte-comparable across models, years, and teams. The dataset itself becomes the unit of evaluation, the training signal, and the audit trail.

---

## 1. Code map

```
forecast_eval/                       # core package
├─ runner.py                         # build_task_plan + scheduler (L1: κ_M ≤ χ_i admissibility)
├─ react.py                          # ReAct loop + Tavily end_date injection (L2: temporal masking)
├─ leak_filter.py                    # retrieval-content auditor (L3)
├─ llm.py                            # OpenAI-compatible client; enforces no provider-side browsing (L4)
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

L1–L4 mark the four channels through which residual leakage is controlled: parametric
memory, tool-mediated retrieval, retrieval-content audit, and the
provider-native-browsing ban. `tests/` pins each contract; touching any of the four
files above without rerunning the matching pin test breaks reproducibility.

---

## 2. Quickstart

### 2.1 Environment

```bash
conda env create -f environment.yml
conda activate oracleproto
```

Python 3.12. Core dependencies: `openai`, `tavily-python`, `pydantic>=2.6`, `loguru`,
`httpx`, `tenacity`, `pytest`. `matplotlib` stays out of `environment.yml` and is
installed on demand for plotting.

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
yes/no, binary-named, and single-/multi-answer multiple-choice, with resolution
dates spanning 2026-03-12 to 2026-04-14. To plug in another corpus, point
`SOURCE_DB` and `SOURCE_TABLE` at a SQLite database that follows the seven-column
schema in [`FRAME.md`](./FRAME.md) §2.1, plus a `dataset_metadata` row carrying the
eight prompt template keys (§2.3). $`\mathcal{D}`$ is a replaceable input to
$`\mathcal{R}`$, so the rest of the framework runs unchanged.

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

The DB schema, the CSV catalogue under `analysis/`, and the model-slug filename
mapping are specified in [`FRAME.md`](./FRAME.md) §6 and §9.

---

## 5. Documentation map

| Where to look                                                | Read                                |
| ------------------------------------------------------------ | ----------------------------------- |
| Why each constraint exists; threat model; contract knobs     | [`DESIGN.md`](./DESIGN.md)          |
| Field-level spec: symbol → module → DB column → pin test     | [`FRAME.md`](./FRAME.md)            |
| Every option's default and validation rule                   | [`.env.example`](./.env.example)    |
