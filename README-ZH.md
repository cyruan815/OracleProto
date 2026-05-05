<div align="center">

<img src="static/images/OracleProto_Logo_Horizontal.png" alt="OracleProto Logo" width="100%">

<em>通过知识截止与时间掩码，对 LLM 原生预测能力进行基准评测的可复现框架</em>

[English](./README.md) | [中文文档](./README-ZH.md) | [Hugging Face](https://huggingface.co/datasets/MaYiding/OracleProto)

</div>

---

## 概述

预测基准始终面临一道结构性张力：前瞻式基准一旦事件解算便会失效，回顾式基准则容易把事实回忆当成预测能力来度量——prompt 层面的指令无法把模型推回它已经越过的知识边界。OracleProto 在数据集层面而非 prompt 层面合上这道缺口：每一道已解算事件都被改写为受模型自身知识截止约束的预测样本，封装为自包含的运行单元，在模型、年份与团队之间保持字节级可比。数据集由此成为评测的单元、训练信号与审计轨迹。

---

## 1. 代码地图

```
forecast_eval/                       # 核心 Python 包
├─ runner.py                         # build_task_plan + 调度（L1：κ_M ≤ χ_i 可允许性过滤）
├─ react.py                          # ReAct 循环 + Tavily end_date 注入（L2：时间掩码）
├─ leak_filter.py                    # 检索内容审计（L3）
├─ llm.py                            # OpenAI 兼容客户端；强制禁止供应商原生浏览（L4）
├─ search.py                         # Tavily 包装
├─ analysis/                         # 评分与诊断：accuracy、FSS、BI、composite、behavior
├─ prompts.py / parser.py            # 输入渲染器 R / 输出解析器 Ψ
├─ types.py / errors.py / config.py  # 数据模型 / 类型化异常 / Settings
├─ db.py / loader.py                 # SQLite schema 迁移 / 数据集同步
└─ tavily_keys.py / tools.py         # API key 轮转 / 工具 schema
evaluation.py                        # 入口：为每个 (model × question) 编排一个 R
scripts/                             # 离线工具：数据集构建、灵敏度扫描、绘图
tests/                               # pytest pin test —— 断言即契约
runs/, logs/                         # 运行产物（已 gitignore）
forecast_eval_set_example.db         # 随仓 80 题样例数据集（故意不忽略）
```

L1–L4 标注残余泄漏被控制的四条通道：参数化记忆、工具中介检索、检索内容审计、
供应商原生浏览禁令。`tests/` 固定每条契约；改动以上任一文件而不重跑对应
pin test 即会破坏可复现性。

---

## 2. 快速开始

### 2.1 环境

```bash
conda env create -f environment.yml
conda activate oracleproto
```

Python 3.12。核心依赖：`openai`、`tavily-python`、`pydantic>=2.6`、`loguru`、
`httpx`、`tenacity`、`pytest`。`matplotlib` 不在 `environment.yml` 中，绘图时
按需安装。

### 2.2 配置 `.env`

```bash
cp .env.example .env
```

填入 `LLM_API_KEY`（配合 `LLM_BASE_URL` 使用任何 OpenAI 兼容端点）、
`TAVILY_API_KEY`、`LEAK_DETECTOR_API_KEY`、`MODELS`、`MODEL_TRAINING_CUTOFFS`。
$`\kappa_M`$ 对每个待评测模型必填；当 model card 仅以月级粒度披露 cutoff
时，保守约定为**所披露月份的最后一日**，这样可以确保不让答案已可能被模型
记忆的问题进入评测样本。`Settings._post_validate`（`config.py`）在任何
LLM 调用发出之前就会对缺失 key、`:online` slug 与其他配置错误快速失败。
带注释的 [`.env.example`](./.env.example) 是每个选项的唯一权威。

### 2.3 测试

```bash
pytest tests/ -q
```

`tests/` 即契约层。CI 基线（`test_prompts`、`test_parser`、`test_training_cutoff`、
`test_llm_no_browsing`、`test_analysis`）固定渲染器、解析器、L1 可允许性过滤、
L4 浏览禁令与聚合器。

### 2.4 运行

```bash
# 冒烟：最便宜模型、单样本、仅 yes_no
MODELS=openai/gpt-4o-mini SAMPLING_N=1 python evaluation.py --question-type yes_no

# 全量扫描 MODELS × SAMPLING_N
python evaluation.py

# 过滤组合：标志间为 AND，标志内为 OR
python evaluation.py --question-type multiple_choice --choice-type multi
```

每次调用创建 `runs/{run_id}/`，`run_id` 形如 `YYYYMMDD-HHMMSS-{4-char hex}`。
在 `.env` 中设置 `RUN_ID=<existing-id>` 即可在同一目录中续跑该运行；已完成
槽位被跳过，`skipped_training_cutoff` 行永不重试，瞬时错误按原退避策略重试。

---

## 3. 接入自有数据集

仓库随附 `forecast_eval_set_example.db`，包含 80 道人工策展的问题，覆盖 yes/no、
binary-named、单/多答案多项选择，事件解算日期跨越 2026-03-12 至 2026-04-14。
若要接入其他语料，把 `SOURCE_DB` 与 `SOURCE_TABLE` 指向一个遵循
[`FRAME-ZH.md`](./FRAME-ZH.md) §2.1 七列 schema 的 SQLite 数据库，并提供一行
`dataset_metadata`，其中包含八条 prompt 模板键（§2.3）。$`\mathcal{D}`$ 是
$`\mathcal{R}`$ 的可替换输入，框架其余部分无需改动即可继续运行。

---

## 4. 输出

```
runs/{run_id}/
├─ manifest.json          # 运行级元数据与哈希链
├─ db/{model_slug}.db     # 每模型一份 SQLite，可独立重放
├─ analysis/              # 由原始 DB 重算的 CSV/JSON
└─ logs/{run_id}.log
```

DB 仅存原始观测。每一项聚合（$`\text{pass@1}`$、FSS、BI、composite 等）由
`forecast_eval/analysis/` 重算，该步骤在 `evaluation.py` 末尾自动运行，亦可
独立调用：

```bash
python -m forecast_eval.analysis runs/{run_id}
```

DB schema、`analysis/` 下的 CSV 目录、模型 slug 文件名映射，见
[`FRAME-ZH.md`](./FRAME-ZH.md) §6 与 §9。

---

## 5. 文档导航

| 想知道                                                      | 读                                  |
| ----------------------------------------------------------- | ----------------------------------- |
| 每道约束为何存在；威胁模型；契约旋钮                          | [`DESIGN-ZH.md`](./DESIGN-ZH.md)          |
| 字段级规范：符号 → 模块 → DB 列 → pin test                  | [`FRAME-ZH.md`](./FRAME-ZH.md)            |
| 每个选项的默认值与校验规则                                  | [`.env.example`](./.env.example)    |
