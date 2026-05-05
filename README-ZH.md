<div align="center">

<img src="static/images/OracleProto_Logo_Horizontal.png" alt="OracleProto Logo" width="100%">

<em>通过知识截止与时间掩码，对 LLM 原生预测能力进行基准评测的可复现框架</em>

[English](./README.md) | [中文文档](./README-ZH.md) | [Hugging Face](https://huggingface.co/datasets/MaYiding/OracleProto)

访问我们的排行榜：[oracleproto.pages.dev](https://oracleproto.pages.dev)

</div>

---

## 概述

随着大语言模型（LLM）向现实世界的决策支持系统演进，评估其“原生预测能力”面临着一个根本性矛盾：实时基准能完美避免数据污染但事件结束后即刻失效，而回顾性基准虽可复现，却极易将预训练记忆误判为真正的预测。为了解决这一挑战，我们提出了 OracleProto，一个针对 LLM 原生预测能力的可复现评估框架。该框架通过联合执行模型知识截止期对齐、工具级时间掩蔽、内容级泄露检测，以及标准化的层级评分机制，将已完结事件完美重构为具有严格时间边界的预测样本。基于六款主流大模型的评估表明，OracleProto 能够在受控信息边界下精准区分模型的预测质量、稳定性与成本效率，并将残余泄露率降至 1% 的数量级。最终，OracleProto 成功将一次性的预测评估转化为可审计、可复用且支持后续监督微调（SFT）与强化学习（RL）的数据集级能力。

<div align="center">

<img src="static/images/Framework.png" alt="Framework of OracleProto" width="100%">

OracleProto 框架图

</div>

---

## 1. 代码地图

```
forecast_eval/                       # 核心代码
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
evaluation.py                        # 入口
scripts/                             # 离线工具
tests/                               # 测试
runs/, logs/                         # 运行产物
forecast_eval_set_example.db         # 样例数据集
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

填入 `LLM_API_KEY`、`LLM_BASE_URL`、`MODELS`、`MODEL_TRAINING_CUTOFFS`、`TAVILY_API_KEY`、`LEAK_DETECTOR_API_KEY`、`LEAK_DETECTOR_BASE_URL`、`LEAK_DETECTOR_MODEL`。其他解释说明见 [`.env.example`](./.env.example) 中的注释。

### 2.3 测试

```bash
pytest tests/ -q
```

### 2.4 运行

```bash
python evaluation.py
```

每次调用创建 `runs/{run_id}/`，`run_id` 形如 `YYYYMMDD-HHMMSS-{4-char hex}`。
在 `.env` 中设置 `RUN_ID=<existing-id>` 即可在同一目录中续跑该运行；已完成的题目或不符合条件的题目将被跳过，瞬时错误按原退避策略重试。

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
