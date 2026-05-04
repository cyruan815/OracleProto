<div align="center">

<h1>OracleProto</h1>

<em>通过知识截止与时间掩码,对 LLM 原生预测能力进行基准评测的可复现框架</em>

[English](./README.md) | [中文文档](./README-ZH.md) | [Hugging Face](https://huggingface.co/datasets/MaYiding/OracleProto)

</div>

OracleProto 将已解算的事件重构为带时间边界的预测样本。`evaluation.py` 的每一次
调用物化为一个运行单元

$`\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)`$,

并仅当 $`\kappa_M \le \chi_i < \tau_i`$ 时将问题 $`q_i`$ 准入给模型 $`M`$,其中
$`\kappa_M`$ 是模型训练截止,$`\tau_i`$ 是事件解算时间。本 README 只讲代码结构
与如何运行一次评测。每道约束的论证见 [`DISIGN-ZH.md`](./DISIGN-ZH.md);每个符号 →
模块 → DB 列 → pin test 的字段级映射见 [`FRAME-ZH.md`](./FRAME-ZH.md)。

---

## 概述

**挑战。** LLM 预测能力评测处在两个不稳定的极端之间。前瞻式实时基准能挡住污染,但事件一旦解算便失效,排行榜成为单向时间流;回顾式基准便于重放,但答案已经在模型参数里的"已解算问题",测的是事实回忆而非预测。Prompt 层面"假装你在 X 日"这类指令也弥合不了"模拟无知"(simulated ignorance)与模型从未越过的真实知识边界之间的差距。

**方法。** OracleProto 把信息边界编码进数据集,而不是 prompt。在运行单元 $`\mathcal{R}`$ 之内,四条通道在任何输出被评分之前各自被独立守住:
- **L1,参数化记忆。** 仅当 $`\kappa_M \le \chi_i < \tau_i`$ 时把 $`q_i`$ 准入给 $`M`$;模型可能已记住答案的题目在调度前就被排除,而不是被记为预测失败。
- **L2,工具中介检索。** 每一次 search 调用都携带工具侧截止 $`\chi_i = \tau_i - \delta`$,由评测端固定,模型不可改写。
- **L3,检索内容审计。** 辅助 LLM detector 在 $`\chi_i`$ 之下逐条阅读检索结果,丢弃泄露截止后结果的片段;其 prompt 的 SHA-256 进入运行哈希。
- **L4,禁止供应商原生浏览。** 路由白名单与请求时校验拒绝任何会绕过边界进行原生浏览的模型或 endpoint。

预测被约束到有限答案空间,经 $`\phi`$ 归一化,并在 parseability、item、question、model 四个层次上评分,一次运行就允许跨模型直接对比,不需要逐模型再做缩放。

**做成了什么。** FutureX-Past 中一道已解算的事件,只要某个模型满足 $`\kappa_M \le \chi_i`$,就能作为可重放的预测样本对它评测。`evaluation.py` 的每一次调用产出一个自包含的 `runs/{run_id}/` 目录:`manifest.json` 携带完整配置哈希链,每个模型一份可被任意第三方离线重算的 SQLite,以及由原始观测重新生成的 CSV 目录。仓库内随附的 `forecast_eval_set_example.db` 包含 80 道人工策展问题,事件解算日期跨越 2026-03-12 至 2026-04-14,可直接对接任意 OpenAI 兼容端点。同一份数据集在模型面板之间、日历年份之间、不同团队之间保持字节级可比;同一份逐步检索轨迹与 boxed 终态答案,无需改动形式契约就能转作 SFT 与 outcome-based RL 的训练信号。

**愿景。** LLM 从文本生成走向金融、政策、公共安全、科研等真实决策支持场景的过程中,预测是必须具备的原生能力。把数据集本身作为预测能力评测的中心对象,而不是某一次实时快照或某一种 agent 栈,把一次性打榜转化为可审计、可复用、可跨模型截止延展、可回收为训练信号的累积数据资产。同一份产物因此一身三用:既是评测集、又是训练语料、又是真实决策背后预测能力的审计轨迹。

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
conda activate oracleproto
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
