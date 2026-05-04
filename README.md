<div align="center">

<h1>OracleProto</h1>

<em>A reproducible framework for benchmarking LLM native forecasting via knowledge cutoff and temporal masking</em>

[English](./README.md) | [中文文档](./README-ZH.md) | [Hugging Face](https://huggingface.co/datasets/MaYiding/OracleProto)

</div>

OracleProto reconstructs resolved events into time-bounded forecasting samples. Every
invocation of `evaluation.py` materialises one run unit

$`\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)`$,

and admits question $`q_i`$ to model $`M`$ only when $`\kappa_M \le \chi_i < \tau_i`$,
where $`\kappa_M`$ is the model training cutoff and $`\tau_i`$ the event resolution
time. This README covers the code layout and how to run an evaluation. The rationale
for every constraint lives in [`DESIGN.md`](./DESIGN.md); the field-level mapping
symbol → module → DB column → pin test in [`FRAME.md`](./FRAME.md).

---

## Overview

**Challenge.** Forecasting evaluation for LLMs sits between two unstable extremes. Prospective live benchmarks stay contamination-free but lose validity once events resolve, so each leaderboard is a one-way temporal stream. Retrospective benchmarks are easy to replay, yet a resolved question whose outcome already lives in the model's parameters tests recall rather than forecasting. Prompt-level "act as if you were on date X" instructions cannot close the gap between simulated ignorance and the true knowledge boundary the model never crossed.

**Method.** OracleProto encodes the boundary into the dataset rather than the prompt. Within $`\mathcal{R}`$, four channels are gated independently before any output is scored:
- **L1, parametric memory.** Admit $`q_i`$ to $`M`$ only when $`\kappa_M \le \chi_i < \tau_i`$; questions whose answer the model could have memorised are filtered before scheduling rather than recorded as a forecasting failure.
- **L2, tool-mediated retrieval.** Every search call carries the tool-side cutoff $`\chi_i = \tau_i - \delta`$, fixed by the evaluator and never by the model.
- **L3, retrieval-content audit.** An auxiliary LLM detector reads each snippet against $`\chi_i`$ and drops anything that leaks the post-cutoff outcome; its prompt SHA-256 is part of the run hash.
- **L4, provider-side browsing ban.** A route allow-list and a request-time check reject any model or endpoint that performs unbounded native browsing.

Predictions are constrained to a finite answer space, normalised by $`\phi`$, and scored at four levels (parseability, item, question, model), so one run admits cross-model comparison without per-model rescaling.

**What is built.** A resolved event from FutureX-Past becomes a replayable forecasting sample for any model with $`\kappa_M \le \chi_i`$. Each invocation of `evaluation.py` produces one self-contained `runs/{run_id}/` directory: a `manifest.json` carrying the full configuration hash chain, one SQLite per model that any third party can re-score offline, and a CSV catalogue regenerated from the raw observations. The bundled `forecast_eval_set_example.db` ships 80 curated questions with resolution dates spanning 2026-03-12 to 2026-04-14, ready to drop into any OpenAI-compatible endpoint. The same dataset stays byte-comparable across the model panel, across calendar years, and across teams; the per-step retrieval trace and the boxed final answer can be reused as SFT and outcome-based RL signal without altering the formal contract.

**Vision.** Forecasting is the native capability LLMs need on the path from text generation to real-world decision support: finance, policy, public safety, scientific research. Making the dataset itself the central object of evaluation, rather than a single live snapshot or a particular agent stack, turns one-shot scoring into a cumulative data asset that can be audited, reused, extended across model knowledge cutoffs, and recycled as training signal. The same artefact then serves at once as evaluation set, training corpus, and audit trail for the forecasting capability behind real decisions.

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
memory, tool-mediated retrieval, retrieval-content semantics, and the
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
$`\kappa_M`$ is mandatory for every evaluated model; the conservative convention is
the **last day of the disclosed month**, which never admits a question whose answer
the model could already have memorised. `Settings._post_validate` (`config.py`)
fails fast on missing keys, `:online` slugs, and other configuration errors before
any LLM call leaves the process. The annotated [`.env.example`](./.env.example) is
the single source of truth for every option.

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
`YYYYMMDD-HHMMSS-{4-char hex}`. Set `RUN_ID=<existing-id>` in `.env` to resume into
the same folder; finished slots are skipped, `skipped_training_cutoff` is never
retried, and transient errors retry under the original policy.

---

## 3. Bring your own dataset

The bundled `forecast_eval_set_example.db` carries 80 curated questions across
yes/no, binary-named, and single-/multi-answer multiple-choice, with resolution
dates spanning 2026-03-12 to 2026-04-14. To plug in another corpus, point
`SOURCE_DB` and `SOURCE_TABLE` at a SQLite that follows the seven-column schema in
[`FRAME.md`](./FRAME.md) §2.1, with a `dataset_metadata` row carrying the eight
prompt template keys (§2.3). $`\mathcal{D}`$ is a replaceable input of
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
