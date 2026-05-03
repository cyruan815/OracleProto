<div align="center">

<h1>OracleProto</h1>

<em>通过知识截止与时间掩码,对 LLM 原生预测能力进行基准评测的可复现框架</em>

[English](./README.md) | [中文文档](./README-ZH.md)

</div>

OracleProto 将已解算的事件重构为带时间边界的预测样本。`evaluation.py` 的每一次
调用物化为一个运行单元

$`\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)`$,

并仅当 $`\kappa_M \le \chi_i < \tau_i`$ 时将问题 $`q_i`$ 准入给模型 $`M`$,其中
$`\kappa_M`$ 是模型训练截止,$`\tau_i`$ 是事件解算时间。本 README 只讲代码结构
与如何运行一次评测。每道约束的论证见 [`DISIGN-ZH.md`](./DISIGN-ZH.md);每个符号 →
模块 → DB 列 → pin test 的字段级映射见 [`FRAME-ZH.md`](./FRAME-ZH.md)。

---

## 1. 代码地图

```
forecast_eval/                       # 核心 Python 包
├─ runner.py                         # build_task_plan + 调度 (L1: κ_M ≤ χ_i 可允许性过滤)
├─ react.py                          # ReAct 循环 + Tavily end_date 注入 (L2: 时间掩码)
├─ leak_filter.py                    # 检索内容审计 (L3)
├─ llm.py                            # OpenAI 兼容客户端;强制禁止供应商原生浏览 (L4)
├─ search.py                         # Tavily 包装
├─ analysis/                         # 评分与诊断:accuracy、FSS、BI、composite、behavior
├─ prompts.py / parser.py            # 输入渲染器 R / 输出解析器 Ψ
├─ types.py / errors.py / config.py  # 数据模型 / 类型化异常 / Settings
├─ db.py / loader.py                 # SQLite schema 迁移 / 数据集同步
└─ tavily_keys.py / tools.py         # API key 轮转 / 工具 schema
evaluation.py                        # 入口:为每个 (model × question) 编排一个 R
scripts/                             # 离线工具:数据集构建、灵敏度扫描、绘图
tests/                               # pytest pin test —— 断言即契约
runs/, logs/                         # 运行产物(已 gitignore)
forecast_eval_set_example.db         # 随仓 80 题样例数据集(故意不忽略)
```

L1–L4 标注残余泄漏被控制的四条通道:参数化记忆、工具中介检索、检索内容语义、
供应商原生浏览禁令。`tests/` 固定每条契约;改动以上任一文件而不重跑对应
pin test 即破坏可复现性。

---

## 2. 快速开始

### 2.1 环境

```bash
conda env create -f environment.yml
conda activate forecast
```

Python 3.12。核心依赖:`openai`、`tavily-python`、`pydantic>=2.6`、`loguru`、
`httpx`、`tenacity`、`pytest`。`matplotlib` 不在 `environment.yml` 中,绘图时
按需安装。

### 2.2 配置 `.env`

```bash
cp .env.example .env
```

填入 `LLM_API_KEY`(配合 `LLM_BASE_URL` 使用任何 OpenAI 兼容端点)、
`TAVILY_API_KEY`、`LEAK_DETECTOR_API_KEY`、`MODELS`、`MODEL_TRAINING_CUTOFFS`。
$`\kappa_M`$ 对每个待评测模型必填;保守约定为**所披露月份的最后一日**,这样
绝不会准入答案可能已被模型记忆的问题。`Settings._post_validate`(`config.py`)
在任何 LLM 调用离开进程之前对缺失 key、`:online` slug 与其他配置错误快速失败。
带注释的 [`.env.example`](./.env.example) 是每个选项的唯一权威。

### 2.3 测试

```bash
pytest tests/ -q
```

`tests/` 即契约层。CI 基线(`test_prompts`、`test_parser`、`test_training_cutoff`、
`test_llm_no_browsing`、`test_analysis`)固定渲染器、解析器、L1 可允许性过滤、
L4 浏览禁令与聚合器。

### 2.4 运行

```bash
# 冒烟:最便宜模型、单样本、仅 yes_no
MODELS=openai/gpt-4o-mini SAMPLING_N=1 python evaluation.py --question-type yes_no

# 全量扫描 MODELS × SAMPLING_N
python evaluation.py

# 过滤组合:标志间为 AND,标志内为 OR
python evaluation.py --question-type multiple_choice --choice-type multi
```

每次调用创建 `runs/{run_id}/`,`run_id` 形如 `YYYYMMDD-HHMMSS-{4-char hex}`。
在 `.env` 中设置 `RUN_ID=<existing-id>` 续接到同一文件夹;已完成槽位被跳过,
`skipped_training_cutoff` 永不重试,瞬时错误按原策略重试。

---

## 3. 接入自有数据集

仓库随附 `forecast_eval_set_example.db`,包含 80 道人工策展的问题,覆盖 yes/no、
binary-named、单/多答案多项选择,事件解算日期跨越 2026-03-12 至 2026-04-14。
接入其他语料,把 `SOURCE_DB` 与 `SOURCE_TABLE` 指向遵循 [`FRAME-ZH.md`](./FRAME-ZH.md)
§2.1 七列 schema 的 SQLite,并附带一行携带八条 prompt 模板键(§2.3)的
`dataset_metadata`。$`\mathcal{D}`$ 是 $`\mathcal{R}`$ 的可替换输入,框架其余
部分原样运行。

---

## 4. 输出

```
runs/{run_id}/
├─ manifest.json          # 运行级元数据与哈希链
├─ db/{model_slug}.db     # 每模型一份 SQLite,可独立重放
├─ analysis/              # 由原始 DB 重算的 CSV/JSON
└─ logs/{run_id}.log
```

DB 仅存原始观测。每一项聚合($`\text{pass@1}`$、FSS、BI、composite 等)由
`forecast_eval/analysis/` 重算,该步骤在 `evaluation.py` 末尾自动运行,亦可
独立调用:

```bash
python -m forecast_eval.analysis runs/{run_id}
```

DB schema、`analysis/` 下的 CSV 目录、模型 slug 文件名映射,见
[`FRAME-ZH.md`](./FRAME-ZH.md) §6 与 §9。

---

## 5. 文档导航

| 想知道                                                      | 读                                  |
| ----------------------------------------------------------- | ----------------------------------- |
| 每道约束为何存在;威胁模型;契约旋钮                          | [`DISIGN-ZH.md`](./DESIGN-ZH.md)          |
| 字段级规范:符号 → 模块 → DB 列 → pin test                  | [`FRAME-ZH.md`](./FRAME-ZH.md)            |
| 每个选项的默认值与校验规则                                  | [`.env.example`](./.env.example)    |
