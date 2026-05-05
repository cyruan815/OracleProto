<div align="center">

<img src="static/images/OracleProto_Logo_Horizontal.png" alt="OracleProto Logo" width="100%">

<em>A reproducible framework for benchmarking LLM native forecasting via knowledge cutoff and temporal masking</em>

[English](./README.md) | [中文文档](./README-ZH.md) | [Hugging Face](https://huggingface.co/datasets/MaYiding/OracleProto)

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
evaluation.py                        # entrypoint: one R per (model × question)
scripts/                             # offline tooling: dataset build, sensitivity sweeps, plots
tests/                               # pytest pin tests — assertion-as-contract
runs/, logs/                         # run artefacts (gitignored)
forecast_eval_set_example.db         # bundled 80-question dataset (intentionally not ignored)
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

Fill `LLM_API_KEY` (with `LLM_BASE_URL` for any OpenAI-compatible endpoint),
`TAVILY_API_KEY`, `LEAK_DETECTOR_API_KEY`, `MODELS`, and `MODEL_TRAINING_CUTOFFS`.
Declare $`\kappa_M`$ for every evaluated model. When a model card discloses the
cutoff only at month granularity, take the **last day of the disclosed month** as
the conservative choice: this convention never admits a question whose answer the
model could already have memorised. `Settings._post_validate` (`config.py`) fails
fast on missing keys, `:online` slugs, and other configuration errors before any
LLM call is issued. The annotated [`.env.example`](./.env.example) is the single
source of truth for every option.

### 2.3 Tests

```bash
pytest tests/ -q
```

`tests/` is the contract layer. The CI baseline (`test_prompts`, `test_parser`,
`test_training_cutoff`, `test_llm_no_browsing`, `test_analysis`) pins the renderer,
the parser, the L1 admissibility filter, the L4 browsing ban, and the aggregator.

### 2.4 Run

```bash
# smoke: cheapest model, single sample, yes_no only
MODELS=openai/gpt-4o-mini SAMPLING_N=1 python evaluation.py --question-type yes_no

# full sweep across MODELS × SAMPLING_N
python evaluation.py

# filter combinations: AND across flags, OR within each flag
python evaluation.py --question-type multiple_choice --choice-type multi
```

Each invocation creates `runs/{run_id}/` with `run_id` of the form
`YYYYMMDD-HHMMSS-{4-char hex}`. Set `RUN_ID=<existing-id>` in `.env` to resume that
run in place; finished slots are skipped, `skipped_training_cutoff` rows are never
retried, and transient errors retry under the original backoff policy.

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
