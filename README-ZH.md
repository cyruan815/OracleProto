<div align="center">

<h1>OracleProto</h1>

<em>通过知识截止与时间掩码，对 LLM 原生预测能力进行基准评测的可复现框架</em>

[English](./README.md) | [中文文档](./README-ZH.md)

</div>

OracleProto 将已解算的事件重构为带时间边界的预测样本，并将评测置于数据集层面，
从而使一次运行在跨模型、跨日历年度上可审计、可重放、可比较。

> **一句话总结。** 本代码库将一次预测评测转化为单一的可复现运行单元
> $`\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)`$，
> 固定从问题到聚合规则的每一项输入，并输出在配置匹配时字节也匹配的评分制品。

本 README 是一份十分钟的入门指南。它解释本项目是什么、如何运行、以及每一份制品来自何处。
每一道约束背后的论证，见 `DESIGN.md`。将每个符号映射到一个模块、一个数据库列与一个固定
测试的字段级规范，见 `FRAME.md`。

---

## 1. 问题

现有的预测评测位于不稳定的中间地带。前瞻型实时基准如 ForecastBench 与 FutureX 在构造
上即受到污染控制，但事件一旦解算它们便会蒸发，因此排行榜成为单向的时间流而不是可复用的
制品。回溯型基准如 FutureX-Past 或归档的实时问题虽然可复现，却很容易将事实回忆误判为
预测能力，因为评测之时答案早已嵌入模型的训练语料。

「想象你不知道选举已经解算」之类的提示词层面纪律，无法弥合这一鸿沟。独立调研以经验数据
证实了模拟无知与真实无知之间存在显著的系统性偏差，并表明仅 1–5% 的标签噪声率便足以
击穿严格评分规则。从推理侧得到同样的结论：单次推理的防御无法在多次运行间泛化，因此
纪律必须下沉一层、嵌入数据集本身。

OracleProto 将这一纪律推入数据集模式。问题 $`q_i`$ 仅当其预测截止 $`\chi_i`$ 满足

$$\kappa_M \le \chi_i < \tau_i,$$

方可对模型 $`M`$ 准入，其中 $`\kappa_M`$ 表示模型的训练截止，$`\tau_i`$ 表示事件解算时间。
模型的参数化知识因此不会比所允许的预测环境更新，而解算时间在模拟的信息状态中尚未到达。
不可纳入的问题不计入模型错误。它们在 `runner.build_task_plan`（runner.py:L132）处被
过滤掉、单独审计，并由 `tests/test_training_cutoff.py` 固定。

---

## 2. 框架

OracleProto 依赖两件制品：单一的运行单元 $`\mathcal{R}`$ 命名评测的每一项输入，以及四
通道信息边界控制模型可能习得答案的每一条路径。

### 2.1 运行单元 $`\mathcal{R}`$

`evaluation.py` 的每一次调用物化为单一的运行单元
$`\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)`$。
每个符号解析为一个配置旋钮、一条代码路径、一个固定测试。

| 符号               | 对象                         | 配置 / 代码路径                                                                                                              | 固定测试                          |
| ------------------ | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| $`\mathcal{D}`$      | 离散预测数据集               | `SOURCE_DB` / `SOURCE_TABLE`(config.py:L391/L395);`loader.sync_questions`(loader.py:L77)                                    | `tests/test_db.py`                |
| $`M`$                | 待评测模型                   | `MODELS`（config.py:L223）的一条记录；每个模型在 `runs/{run_id}/db/` 下对应一份 SQLite                                         | `tests/test_runner_grid_model.py` |
| $`\kappa_M`$         | 知识截止                     | `MODEL_TRAINING_CUTOFFS[M]`（config.py:L224）；可纳入性过滤位于 `runner.build_task_plan`(runner.py:L132)                       | `tests/test_training_cutoff.py`   |
| $`\delta`$           | 时间掩码偏移                 | `TAVILY_END_DATE_OFFSET_DAYS` 默认 `-1`（config.py:L273）；在 `react.py` 的工具层注入                                          | `tests/test_search.py`、`tests/test_react.py` |
| $`T`$                | ReAct 最大步数               | `REACT_MAX_STEPS` 默认 `12`（config.py:L279）；外循环 `react.run_react`(react.py:L162)                                         | `tests/test_react.py`             |
| $`C`$                | 最大搜索调用数               | `REACT_MAX_SEARCH_CALLS` 默认 `[8]`（config.py:L283）；预算闸门（react.py:L276–L279）                                            | `tests/test_react.py`             |
| $`R`$                | 输入渲染器                   | `forecast_eval/prompts.py::render_user_prompt`                                                                              | `tests/test_prompts.py`           |
| $`\Psi`$             | 输出解析器与有效性           | `forecast_eval/parser.py::parse_answer`(parser.py:L40)                                                                      | `tests/test_parser.py`            |
| $`\phi`$             | 答案规范化映射               | 按 `question_type` 定义的字母编码 `A` 或 `A,B` 等；`parser.parse_gt`(parser.py:L92)                                          | `tests/test_parser.py`            |
| $`\Gamma`$           | 聚合规则                     | `forecast_eval/analysis/*`(综合准确率、FSS、κ、BI 等)                                                                       | `tests/test_analysis.py`          |
| $`H_{\mathrm{aux}}`$ | 辅助泄漏探测器               | `leak_filter.filter_search_result`；记录在 `run_meta.config_snapshot` 中，而非 $`\mathcal{R}`$ 元组内部                          | `tests/test_leak_filter.py`       |

辅助探测器 $`H_{\mathrm{aux}}`$ 按设计置于形式元组之外。它是支撑边界的可替换经验工程层，
不是预测系统的原语组件。其 prompt 的 SHA-256 存储于
`run_meta.config_snapshot.leak_detector_prompt_hash`，因此泄漏屏障本身亦可字节复现
（`leak_filter.py:L55–L104`）。

模型 $`M`$ 在问题 $`q_i`$ 上可见的信息为

$$\mathcal{I}_{i,M}^{\mathrm{vis}} = \mathcal{K}^{M}_{\le\kappa_M} \cup \mathcal{T}_{\le\chi_i},$$

其中 $`\mathcal{K}^{M}_{\le\kappa_M}`$ 是模型截止之前可用的参数化知识，
$`\mathcal{T}_{\le\chi_i}`$ 是经过时间掩码的外部信息。预测系统 $`F_M`$ 产出
$`\widehat{Y}_{i,M} = F_M(q_i^{\mathrm{in}}; \mathcal{I}_{i,M}^{\mathrm{vis}})`$,
且 $`\widehat{Y}_{i,M}\subseteq\mathcal{A}_i`$。时间掩码下的离散预测循环实现于
`react.run_react`（react.py:L162）。

将每个符号映射到模块、DB 列与测试的字段级总图，见 `FRAME.md` §1.1。

### 2.2 四通道信息边界

残余泄漏分解为三条受控通道与第四条供应商侧残余。三条受控通道分别为参数化、工具中介、
检索内容；第四条位于评测者控制之外。每条通道对应一道带有声明覆盖范围的机械化防御。

| 通道                                          | 防御层                                                       | 位置（代码）                                                                                                                                  | 默认                                  | 残余泄漏率                                                                       |
| --------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- | -------------------------------------------------------------------------------- |
| **L0 人工策展**                               | 上游数据集构造                                               | `forecast_eval_set_example.db` 策展                                                                                                          | 始终                                  | 0%(人工标注下限)                                                                |
| **L1 参数化（可纳入性过滤）**                   | 在任务生成处校验 $`\kappa_M \le \chi_i`$                       | `runner.build_task_plan`(runner.py:L132–L199)                                                                                                | 按模型设置 `MODEL_TRAINING_CUTOFFS`   | 在上游过滤参数化记忆泄漏                                                         |
| **L2 工具中介（Tavily）**                       | 在工具层注入 `end_date = \chi_i`                             | `react._compute_end_date`(react.py:L39);`search.tavily_search`                                                                              | $`\delta=-1`$ 天                        | 单独使用时残余非平凡；Tavily 的索引/抓取日期元数据带噪                            |
| **L3 检索内容（二阶段探测器）**                 | 对每条 Tavily 结果项进行独立 LLM 审计                        | `leak_filter.filter_search_result`(leak_filter.py:L348)                                                                                      | `claude-sonnet-4.6`                   | 相对于仅有 L2 的情形，将每条审计项的残余压到个位数百分点低段                      |
| **L4 供应商侧残余（声明性）**                   | 禁止供应商原生浏览                                           | `Settings._post_validate`(config.py:L602–L606、L747–L751);`llm._assert_no_browsing`(llm.py:L74–L98);`leak_filter._assert_detector_safe`(leak_filter.py:L139) | 始终                                  | 声明为评测偏差，而非佯装消除                                                      |

L4 的三层强制首先在启动时拒绝 `:online` 后缀的 slug 与 `::` 保留分隔符，再在
`llm.chat` 的在线调用中再断言一次，最后在探测器客户端重复同样的校验。
`tests/test_llm_no_browsing.py` 与 `tests/test_config.py` 共同固定该契约。结构上，
L4 是任何会产生计费的 LLM 调用离开进程之前必须通过的那一道防御。

威胁模型以及更宽泛的「我们能控与不能控」分解，见 `DESIGN.md` §2。从 $`\mathcal{R}`$
推导的八条硬性约束，见 `FRAME.md` §1.2。

---

## 3. 快速开始

### 3.1 创建 conda 环境

```bash
conda env create -f environment.yml
conda activate forecast
```

### 3.2 配置 `.env`

```bash
cp .env.example .env
# 编辑 .env 并填入：
#   LLM_API_KEY(以及 LLM_BASE_URL：任何 OpenAI 兼容端点，如
#                OpenRouter、阿里云百炼、OpenAI、DeepSeek、SiliconFlow,
#                或本地 vLLM)
#   TAVILY_API_KEY(单值或 CSV 多键以提高额度)
#   LEAK_DETECTOR_API_KEY(二阶段审计器；可通过留空 LEAK_DETECTOR_BASE_URL
#                          复用 LLM_API_KEY；见 §8.4)
#   MODELS、MODEL_TRAINING_CUTOFFS：列出每个待评测模型及其 κ_M
```

为每个模型声明 $`\kappa_M`$ 是公平运行的强制要求，因为框架的可纳入性过滤正是用以区分
「模型预测失败」与「模型已经知道答案」。未声明截止的模型不会被过滤，会发出警告，且其
数字与其余模型不直接可比。截止可写到月份粒度；推荐约定为以**所披露月份的最后一日**作为
$`\kappa_M`$，这是保守选择（准入更少问题，绝不会错误地准入答案可能被模型记忆的问题）。

`Settings._post_validate`（config.py:L598）在任何 LLM 调用离开进程之前运行。它对空
`LLM_API_KEY`、空 `MODELS`、`:online` 后缀、slug 中的 `::`、`MIN_SEARCH > min(C)`、
`ENABLE_SEARCH_LEAK_FILTER` 关闭却存在 `LEAK_DETECTOR_API_KEY`、以及 `GRID_DEFAULT_R/C`
不在所配置网格内等情形快速失败，确保配置错误的 `.env` 不会浪费预算。

### 3.3 运行测试（无需 API 调用）

```bash
pytest tests/ -q
```

CI 基线为 `test_prompts / test_parser / test_training_cutoff /
test_llm_no_browsing / test_analysis`，这五项必须保持绿色。它们分别守护渲染器 $`R`$、
解析器 $`\Psi`$、可纳入性过滤 $`\kappa_M`$、§2.2 L4 中的供应商原生浏览禁令、以及聚合
规则 $`\Gamma`$。完整套件覆盖 33 个测试文件、约 13k 行，囊括 v3/v4 DB 迁移、泄漏过滤、
考试式评分与综合权重、网格调度器、ReAct 预算链、以及行为诊断。

### 3.4 运行一次评测

```bash
# 冒烟：最便宜模型、单样本、仅 yes_no
MODELS=openai/gpt-4o-mini SAMPLING_N=1 \
    python evaluation.py --question-type yes_no

# 全量评测：所有模型、所有样本
python evaluation.py

# 过滤组合（标志间为 AND，标志内为 OR）
python evaluation.py --question-type multiple_choice --choice-type multi

# 跳过运行后的分析步骤；原始 DB 仍落到 db/
python evaluation.py --skip-analysis
```

每次调用在 `RUNS_ROOT`（默认 `./runs`）下创建一个新文件夹，以自动生成的形如
`YYYYMMDD-HHMMSS-{4-char hex}` 的 `run_id` 命名。以同一 `run_id` 续接将复用现有
文件夹；详见 §8.1。

---

## 4. 紧预算配置

代码库默认以更宽的搜索预算换取更平滑的行为分析。对于聚焦区分度的运行，框架支持
$`R_{\mathrm{tav}}\cdot C = 5\cdot 4 = 20`$ 的「紧」预设，相当于「两页谷歌搜索结果」。
该紧配置可通过以下 `.env` 覆盖复现：

```ini
SOURCE_DB=./forecast_eval_set_example.db
SOURCE_TABLE=forecast_eval_set_example
SAMPLING_N=3                                # 代码库默认 5
REACT_MAX_STEPS=12                          # 与默认一致
REACT_MAX_SEARCH_CALLS=4                    # 代码库默认 8
TAVILY_MAX_RESULTS=5                        # 与默认一致
TAVILY_END_DATE_OFFSET_DAYS=-1              # 一日的时间掩码缓冲
REACT_REFLECTION_PROTOCOL=true
REACT_BUDGET_AWARENESS_PROTOCOL=true
REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=true
REACT_FORCE_FINAL_ANSWER_LOOKAHEAD=2
REACT_BUDGET_EXCEEDED_DROP_TOOLS=true
REACT_FINAL_ANSWER_RETRY=false              # v5.1 兜底，默认关闭；见 §8.3
ENABLE_SEARCH_LEAK_FILTER=true              # 二阶段探测器开启
BELIEF_PROTOCOL=false                       # 严格字母模式（无伴随信念）
```

每个模型的 $`\kappa_M`$ 仍须通过 `MODEL_TRAINING_CUTOFFS` 声明。

哪些旋钮属于会改变跨运行可比性的契约旋钮、哪些旋钮属于纯工程旋钮，其论证见
`DESIGN.md` §12。

---

## 5. 接入自有数据集

仓库随附 `forecast_eval_set_example.db`，使新克隆即可复现一次非平凡的运行。随附 DB
包含 80 道精心策展的问题：37 道是非题、3 道二元具名题、40 道多项选择题（其中 8 道为
多答案），事件解算日期跨越 2026-03-12 至 2026-04-14。

接入其他语料时，将 `SOURCE_DB` 与 `SOURCE_TABLE` 指向遵循同一七列模式
`id / choice_type / question_type / event / options / answer / end_time`（见
`FRAME.md` §2.1）的任何 SQLite 文件或表，并附带一行携带八条 prompt 模板键的
`dataset_metadata`(见 `FRAME.md` §2.3):

```bash
SOURCE_DB=./my_questions.db
SOURCE_TABLE=my_questions
```

`SOURCE_TABLE` 在启动时（config.py:L586–L595）按白名单 `^[A-Za-z_][A-Za-z0-9_]*$`
校验，因此手误会快速失败，而不会泄漏到 SQL 层。

模式满足之后，评测对数据集是无关的。医学、科学、工程预测等领域语料可直接接入，同一
运行单元 $`\mathcal{R}`$ 保证相同的审计与重放性质。$`\mathcal{D}`$ 是 $`\mathcal{R}`$ 的
可替换输入组件，因此跨域扩展是框架的自然展开，而非其内部缺陷。

---

## 6. 输出

输出目录就是运行单元的持久化形式。任何拿到 `runs/{run_id}/db/{model_slug}.db` 的人
都可在不依赖任何其他制品的情况下重放该模型的评测。

### 6.1 目录布局

```text
runs/
  {run_id}/
    manifest.json           # 运行级元数据：run_id、schema_version、analysis_schema、
                            #   sampling_n、models、filters、source/metadata/templates 哈希、
                            #   reflection_protocol_hash、belief_protocol_hash、started_at、
                            #   finished_at；启用多 (R, C) 时另带 `grid` 块
    db/
      {model_slug}.db       # 每个模型一份 SQLite；自含 questions + prompt_templates
                            #   + run_meta + run_results（见 §6.2）。可独立分发。
    analysis/               # 由 forecast_eval.analysis 在运行结束后生成
      per_model_summary.csv         # 主评分表：综合准确率 + v5 离散家族
                                    #   (FSS / Cohen κ / Hamming / Fleiss κ /
                                    #   平均熵 / VCI / MVG)+ v4 概率伴随
                                    #   (BI / BI_dec / NLL / MBS / ABI_crowd /
                                    #    ABI_uniform / fallback_share)
      per_model_summary.md          # 携带 v5 主列的 markdown 表；概率列以 `†` 标记
                                    #   并附 K 免责声明
      per_model_by_question_type.csv # 按 yes_no / binary_named / multiple_choice 切片
      per_model_by_choice_type.csv   # 按 single / multi 切片
      per_model_composite_by_question_type.csv  # 按子类型加权的综合分；默认
                                                #   0.15 / 0.15 / 0.70(见 §7.1)
      per_model_composite_by_choice_type.csv    # 按子类型加权的综合分；默认
                                                #   0.40 / 0.60(见 §7.1)
      composite_meta.json             # 综合分审计轨迹：每 (model, metric) 的
                                      #   buckets_used / weights_used_normalized / value /
                                      #   bucket_values
      per_model_by_difficulty.csv     # γ-三分位切片（low / mid / high）
      error_breakdown.csv             # 按错误种类分解：network / server_5xx / bad_request /
                                      #   content_policy / skipped_training_cutoff / <ok>
      finish_reason_breakdown.csv     # 按 ChatCompletion finish_reason 分解
      overall.json                    # 完整结构化聚合，含 `probabilistic` 子对象，
                                      #   并镜像 manifest 中的 `analysis_schema`
      # ---- v5 K 次试验一致性 ----
      inter_trial_consistency.csv     # 每模型 Fleiss κ / 平均熵 / VCI / MVG
      entropy_accuracy_bins.csv       # 每模型 × 三分位（Acc / MV Acc / Fleiss κ）
      pairwise_bootstrap.csv          # 多指标配对 bootstrap:FSS / Acc / MV_Acc /
                                      #   Fleiss κ / EBI × 配对 × ΔMean / 95% CI / p / Cohen's d
      # ---- v4 概率（伴随，K=5 免责声明） ----
      shrinkage_alpha_curve.csv       # 每 (model, ctype) 的 LOO α 扫描
      paired_delta_bi.csv             # BS 配对 ΔBS + Holm 调整 p + 后验
      pairwise_significance.csv       # α=0.05 标记（原始 + Holm）
      posterior_pairwise.csv          # P(BI_A > BI_B)
      paired_delta_bi_by_difficulty.csv
      # ---- 阶段 3 行为诊断（需要 BELIEF_PROTOCOL=true） ----
      belief_evolution.csv            # 每 (model, q, k):volatility、试验间方差、
                                      #   convergence_step、evidence_efficiency、
                                      #   counterevidence_engaged
      reflection_ab.csv               # 配对 A/B(当兄弟运行除反思协议哈希外的每个哈希
                                      #   均一致时)
      tool_usage_pdp.csv              # 每 (model, feature, value) 对 Pr(correct|x) 与
                                      #   E[NLL|x] 的 PDP
      confidence_calibration.csv      # 主观置信度对比命中率
      numeric_confidence_calibration.csv  # max_p 分箱对比命中率
      # ---- 网格搜索（仅当 manifest.grid 存在） ----
      grid_summary.csv                # 每 (real_model, R, C) 主表：
                                      #   acc/BI/NLL + 95% CI + 成本列
      grid_marginal_C.csv             # 固定 R = grid.default_r，扫描 C
      grid_marginal_R.csv             # 固定 C = grid.default_c，扫描 R
      grid_pareto.csv                 # Pareto 前沿单元的 `dominated_by` 为空，
                                      #   否则为字典序最小的支配者 slug
      grid_winrate.csv                # 配对 (R, C) 单元胜次 + 显著单元计数
      figs/                           # 仅在执行 `python scripts/plot_analysis.py` 后生成；
                                      #   matplotlib 不在核心依赖中，按需安装
    logs/
      {run_id}.log
```

模型 slug 的文件系统安全映射将 `/` 映为 `__`,`[A-Za-z0-9._-]` 之外的任何字符映为
`_`。因此 `openai/gpt-4o-mini` 变为 `openai__gpt-4o-mini.db`。网格虚拟 slug 追加
`__r{R}__c{C}` 后缀；详见 §8.2。

### 6.2 每模型数据库模式

每个模型 DB 持有三张表。

* **`questions`** 与 **`prompt_templates`** 是源数据的副本，因此每个 DB 在没有原始
  `SOURCE_DB` 的情况下也可独立重放。
* **`run_meta`** 持有单行，包含 `run_id, model, sampling_n, config_snapshot
  (已脱敏), filters_snapshot, source/metadata/templates 哈希， training_cutoff,
  reflection_protocol_text/hash, belief_protocol_text/hash, started_at, finished_at`。
  两个协议指纹与 `prompt_templates_hash` 彼此独立。`DESIGN.md` §7.3 解释为何模板、
  反思、信念三个独立指纹能够无碰撞地启用三轴消融 A/B 配对。
* **`run_results`** 是宽表，每问题一行。对每个 $`i \in 0..\text{SAMPLING\_N}-1`$,
  存在一组 `s{i}_*` 列：v3 基线 20 列，v4 新增 3 列信念列，v5.1 新增 1 列重试列。
  完整集合在 v2 基线上为 `final_answer_letters / final_answer_raw / correct /
  parse_ok / tool_calls_count / react_steps / prompt_tokens / completion_tokens /
  reasoning_tokens / latency_ms / messages_trace / search_calls / error /
  created_at`,v3 可观测性新增 `finish_reason / nudges_used / step_metrics /
  response_id / system_fingerprint / service_tier`,v4 信念新增 `belief_final /
  belief_trace / belief_parse_ok`,v5.1 新增 `final_answer_retry_used`；详见 §8.3。
  旧 DB 在首次重新打开时通过 `ALTER TABLE ADD COLUMN` 自动迁移。设置
  `Settings.BELIEF_PROTOCOL=false` 使 v4 信念列保持 NULL，并使所有 v3 准确率指标
  与 v4 之前的运行字节一致。

DB 仅存储原始观测，不预先计算任何聚合；pass@1、pass_any@N、多数投票、FSS、BI 及其余
指标全部来自 `analysis/` 步骤，该步骤在 `evaluation.py` 末尾自动运行，亦可独立调用：

```bash
python -m forecast_eval.analysis runs/{run_id}
```

原始观测与聚合指标的此种分离，是本项目最承重的架构决策之一。`DESIGN.md` §4.1 阐述了
其论证：指标定义比 DB 模式演化更快，因此将所有聚合推迟到分析层意味着指标重定义永远
不会要求 DB 回填。该契约由 `tests/test_analysis.py` 固定，它在一份手工构造的 DB 夹具上
运行整套分析而不再触碰它。

---

## 7. 评分

### 7.1 带考试式部分得分的综合准确率

`per_model_summary.csv` 报告一项扁平的混合均值（`pass_at_1_avg`）以保持向后兼容。
对于跨模型比较所推荐的头条评分，`per_model_composite_*.csv` 沿两个维度按子题型作
加权合成：

* `per_model_composite_by_question_type.csv` 按 `yes_no` / `binary_named` /
  `multiple_choice` 分桶；
* `per_model_composite_by_choice_type.csv` 按 `single` / `multi` 分桶。

每桶评分使用考试式部分得分，实现于 `forecast_eval/analysis/exam_score.py:L62`:

$$\text{exam-score}(\hat{S}, G) = \begin{cases} |\hat{S} \cap G| / |G|, & \hat{S} \setminus G = \varnothing,\\\\ 0, & \hat{S} \setminus G \ne \varnothing.\end{cases}$$

直观上，任何假阳性即否决得分至零；否则得分为正确恢复的比例 $`|TP|/|G|`$。这是在零 FP
硬性闸门下的召回率。$`m_q = 1`$ 的单答案问题退化为 $`\{0, 1\}`$ 的严格相等情形，多答案
问题保留「宁愿漏选也不错选」的非对称。综合公式（实现于 `analysis/composite.py`）为

$$\text{composite}_m = \frac{\sum_{b \in B_{\text{valid}}(m)} w_{m,b}\cdot v_{m,b}}{\sum_{b \in B_{\text{valid}}(m)} w_{m,b}}.$$

$`B_{\text{valid}}`$ 是测量非 None 且权重为正的桶集合。缺失桶被丢弃并对剩余权重重新
归一化；它们不被视作零。该契约由 `tests/test_composite_score.py` 与
`tests/test_exam_score.py` 固定。

默认权重遵循 *题越难区分度越好* 原则，定义于 config.py:L365–L374:

| 维度            | 桶                | 默认权重 | 难度论证                                                   |
| --------------- | ----------------- | -------- | ---------------------------------------------------------- |
| `question_type` | `yes_no`          | 0.15     | k=2，盲猜 50%，跨模型区分度低                               |
| `question_type` | `binary_named`    | 0.15     | k=2，加入实体识别但仍为二元                                 |
| `question_type` | `multiple_choice` | 0.70     | k=2..N 区间宽，包含多选，信号最强                           |
| `choice_type`   | `single`          | 0.40     | 整体更易；包含 yes_no 与 binary_named                       |
| `choice_type`   | `multi`           | 0.60     | 真实多选；严格基线接近零 → 信号高                           |

可在 `.env` 中通过 `COMPOSITE_WEIGHTS_QTYPE` 与 `COMPOSITE_WEIGHTS_CTYPE` 覆盖。
按指标覆盖请用 `COMPOSITE_WEIGHT_OVERRIDES_QTYPE` 与 `..._CTYPE`（见 `.env.example`
注释）。当某模型行的某指标命中覆盖，其 `weights_kind` 列标记为 `overridden`。
`composite_meta.json` 记录每 (model, metric) 的 `buckets_used`、
`weights_used_normalized` 与 `bucket_values`，提供一比一可复现的审计轨迹。

考试式与严格相等的差异仅在多选多答案桶上重要。三个单答案桶满足
$`\text{exam}_{\text{avg}}^{(b)} \equiv \text{pass@1}_{\text{avg}}^{(b)}`$，因此综合
公式的取值仅通过多选多答案桶感知考试式与严格相等的选择，而该桶在 §4 紧预设下承担
最大区分信号。

每正确答案的成本为

$$C^{\text{per-correct}}_m = \frac{C^{\text{total}}_m}{|\mathcal{D}^{\text{eval}}|\cdot n \cdot \text{Composite\,Accuracy}_m},$$

即平台真实账单除以按难度加权的名义正确样本数。这将「贵但准」与「便宜但乱」的模型
置于同一性价比尺度，避免低单样本单价配高错误率制造的虚低成本错觉。

### 7.2 分层评分套件

评分组织为「有效性 → 项 → 题 → 模型」的分层分解。§7.1 的头条综合准确率是
`per_model_summary.csv` 的一列；伴随套件覆盖稳定性、一致性与随机校正后的技能。

| 指标                                                   | 测量内容                                                                                                        | 代码                                                      |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| $`\text{pass@1}_{\text{avg}}`$                           | 单试验严格相等命中率                                                                                            | `analysis/accuracy.py`                                    |
| $`\text{pass}^{\text{any}}@n`$                           | best-of-$`n`$ 命中上界                                                                                            | `analysis/accuracy.py`                                    |
| $`\text{pass}^{\text{all}}@n`$                           | all-of-$`n`$ 稳定性下界                                                                                           | `analysis/accuracy.py`                                    |
| Cohen's $`\kappa`$                                       | 相对题型条件随机基线的随机校正严格准确率                                                                        | `analysis/accuracy.py::cohen_kappa`                       |
| Fleiss' $`\kappa`$                                       | 跨 $`K^{\mathrm{eff}}_q`$ 个样本的试验间一致性                                                                     | `analysis/consistency.py`                                 |
| Tversky $`T`$                                            | 带 FP 惩罚 $`\alpha`$ 与 FN 惩罚 $`\beta`$ 的集合相似度                                                             | `analysis/accuracy.py::tversky_score`(accuracy.py:L286)  |
| FSS                                                    | 基于 Tversky 的随机校正技能分；默认 $`(\alpha, \beta) = (2.0, 0.5)`$ 对 FP 的惩罚是 FN 的 4 倍                     | `analysis/accuracy.py::fss`(accuracy.py:L386)            |
| MV-Acc / MVG / VCI / Hamming / 平均熵                  | 离散原生一致性家族                                                                                              | `analysis/consistency.py`                                 |

FSS 旨在显化严格 $`\text{pass@1}_{\text{avg}}`$ 漏掉的一类跨模型重排序。两个模型可能
在严格准确率上打平，却在多答案选择集的克制程度上有显著差异；在
$`(\alpha, \beta) = (2.0, 0.5)`$ 下的 FSS 会正确偏向更 FP 保守的模型。这是非对称 Tversky
权重的经验论证：将「宁愿漏选也不错选」直接编码进评分函数，会翻转对称 Jaccard 抓不到的
真实跨模型排序。

### 7.3 按需作图与 FSS 灵敏度

`matplotlib` 不在 `environment.yml` 中，因为分析路径保持依赖轻量。要将分析的
CSV/JSON 渲染为 PNG:

```bash
pip install matplotlib
python scripts/plot_analysis.py runs/{run_id}
```

这会填充 `runs/{run_id}/analysis/figs/`（已被 gitignore），包含：

* v5 主图：带 CI 的 FSS 柱状图、ΔFSS 森林图、每模型熵-Acc 网格（3 桶 × 3 指标：
  Acc / MV Acc / Fleiss κ）;
* 伴随图：带 CI 的 BI 柱状图（BLF 锚点）、ΔBI 森林图、按难度网格的热图、5 道样本
  问题的每问题信念轨迹、每特征的工具使用 PDP。

v5 移除了可靠性图与 Murphy 三分解图，因为在 $`K=5`$ 时每标签仅有六个唯一概率级，统计上
无意义。

每张图为尽力而为：当对应 CSV 或 JSON 缺失时，该图被静默跳过而不让流水线失败。

`per_model_summary.csv` 仅报告单一规范化的 FSS，取 $`(\alpha, \beta) = (2, 0.5)`$。
审稿人若问「为何不取 Jaccard $`(1, 1)`$ 或严格 $`(3, 0.5)`$?」可按需运行灵敏度扫描：

```bash
python scripts/fss_sensitivity.py runs/{run_id}                      # 4 档扫描
python scripts/fss_sensitivity.py runs/{run_id} --alpha 1 --beta 1   # 单点
```

| (α, β)    | 语义                                                       |
| --------- | ---------------------------------------------------------- |
| (1, 1)    | Jaccard，对称：FP 与 FN 等罚                                |
| (1, 0.5)  | 轻度非对称：多选错误为漏选错误的 2 倍                       |
| (2, 0.5)  | v5 默认：多选错误为漏选错误的 4 倍                          |
| (3, 0.5)  | 严格：多选错误为漏选错误的 6 倍                             |

该脚本不被 `run_analysis` 调用。灵敏度 CSV 顶部携带溯源注释，使审稿人单独阅读裸文件
时不会误以为是主指标，契约由 `tests/test_fss_sensitivity.py` 固定。

---

## 8. 运维特性

### 8.1 续接语义

每个 `(question_id, sample_idx)` 槽位独立判定：

* `s{i}_created_at IS NOT NULL` 且 `s{i}_error IS NULL` 表示已完成且未重试。
* `s{i}_error = 'skipped_training_cutoff'` 表示主动被 $`\kappa_M \le \chi_i`$ 校验
  排除，不会重试，因为它从来不是模型失败。
* 其余 `s{i}_error` 取值，如 `network`、`server_5xx`、`bad_request` 或
  `content_policy`，将在下一次运行中重试，复用现有 DB。错误分类位于
  `forecast_eval/errors.py:classify`（errors.py:L86）；桶列表为
  `network / rate_limit / server_5xx / bad_request / content_policy`，外加合成的
  `skipped_training_cutoff`。

在 `.env` 或 CLI 环境变量中设置 `RUN_ID=<existing-run-id>` 以续接到同一文件夹。
留空将铸造新的 `YYYYMMDD-HHMMSS-xxxx` id。`tests/test_runner_resume.py` 固定该行为：
已完成行从不重新下发，`skipped_training_cutoff` 行从不重新运行，其余每个错误类按
原重试策略重试。

### 8.2 通过虚拟模型 slug 的网格搜索

`TAVILY_MAX_RESULTS`（即 $`R_{\mathrm{tav}}`$）与 `REACT_MAX_SEARCH_CALLS`(即 $`C`$)
接受逗号分隔的正整数列表。两者均设为多值列表会产生
$`\lvert\text{MODELS}\rvert \cdot \lvert R\rvert \cdot \lvert C\rvert`$ 个独立的虚拟
模型 slug，形如 `{real_model}::r{R}::c{C}`（`db.compose_virtual_slug` 与
`db.parse_virtual_slug`,db.py:L477/L500）。每个单元在自己的 DB 文件
`runs/<id>/db/<real>__r{R}__c{C}.db` 中存活，并复用每个现有分析阶段。一个额外的
网格步骤会写出 5 张 `grid_*.csv` 长表以及 `analysis/figs/` 下的每单元图族。

runner、DB 模式与分析主流水线字节不变。`forecast_eval/analysis/grid.py` 从 slug
解码三元组，重新聚合，并输出长格式网格表。设计档案 D1–D10 见 `DESIGN.md` §11.1,
连同 `tests/test_grid_dispatcher.py` 与 `tests/test_grid_analysis.py` 共同固定契约。

```bash
MODELS=openai/gpt-5,anthropic/claude-sonnet-4.5
TAVILY_MAX_RESULTS=5,10
REACT_MAX_SEARCH_CALLS=1,3,5,8
GRID_DEFAULT_R=5    # 主图锚点；必须出现在 TAVILY_MAX_RESULTS 中
GRID_DEFAULT_C=5    # 对称要求，必须出现在 REACT_MAX_SEARCH_CALLS 中

python evaluation.py
python scripts/plot_analysis.py runs/<run_id>
```

形如 `TAVILY_MAX_RESULTS=5` 的单值 `.env` 被解析为长度为 1 的列表，因此既有设置除
DB 文件名新增 `__r{R}__c{C}` 后缀外字节等价。无 `manifest.grid` 块的旧 v4 运行会
提早退出网格路径。`MODELS` 条目不可包含 `::`（config.py:L610–L614），因此虚拟 slug
的往返绝不会与真实模型名碰撞。

### 8.3 框架韧性开关（v5.1）

两道可选的韧性杠杆，`REACT_FINAL_ANSWER_RETRY` 默认关闭、
`REACT_BUDGET_EXCEEDED_DROP_TOOLS` 默认开启；详见
`openspec/changes/harness-resilience-v1/`。

* **`REACT_FINAL_ANSWER_RETRY`** 默认 `false`（config.py:L301）。当 ReAct 循环以空
  `final_raw` 干净退出，意味着模型把所有步数都花在 `tool_calls` 而从未产出内容时，
  框架会以 `tools=[]` 与一段固定的「请提交你的 `\boxed{...}` 答案」用户提示再发一次
  `llm_chat`。本开关被下文的循环内强制收尾链取代，作为可选的循环外应急兜底保留。
  启用时该重试在 `react_steps` 与 `step_metrics` 中计为一步，但不计入 `nudges_used`。
  每样本列 `final_answer_retry_used`（0/1）记录结果，并汇总为
  `per_model_summary.csv` 中的 `final_answer_retry_rate`。其动机是：跨模型比较要求
  `parse_failure_rate` 仅反映模型自身的格式失败，而非框架记账的上游工具预算耗尽。
* **`REACT_BUDGET_EXCEEDED_DROP_TOOLS`** 默认 `true`（config.py:L302）。一旦累计
  `web_search` 调用达到 `REACT_MAX_SEARCH_CALLS`，后续每次 LLM 调用会通过 `tools=[]`
  丢掉工具模式。模型不能再请求更多搜索；它必须收尾，否则由上述兜底重试收拾残局。

由 `force-final-answer-near-limit-v1`（config.py:L313–L315）引入的循环内优先级链
才是真正在预算边缘驱动收尾的机制，也正因如此事后的 `REACT_FINAL_ANSWER_RETRY` 现已
默认关闭。在 `react.run_react`（react.py:L162）每次迭代起点，框架按以下优先级
（react.py:L266 的优先级注释、L272–L334 的逻辑）至多挑选一种注入：

1. **末步硬切断**在 `REACT_MAX_STEPS - step == 1` 且
   `REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=true` 时触发。它注入强制收尾文本与
   `tools=[]`，模型只能产出内容。
2. **倒数次步软警告**在 `remaining ∈ [2, REACT_FORCE_FINAL_ANSWER_LOOKAHEAD]`
   时触发。它注入提醒文本，工具仍允许，除非搜索预算已用完。
3. **预算耗尽提交通告**在累计搜索 `>= REACT_MAX_SEARCH_CALLS` 且
   `REACT_BUDGET_EXCEEDED_DROP_TOOLS=true` 时触发，每次运行只触发一次。它告知模型
   搜索工具现已不可用，并请它收尾。
4. **续作提醒**在上一轮是不带 `\boxed{...}` 的内容且其他都不需要触发时触发。它告诉
   模型「上次回复无 `\boxed{...}`，这是当前实时状态」。

默认值为 `REACT_BUDGET_AWARENESS_PROTOCOL=true`、
`REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=true`、`REACT_FORCE_FINAL_ANSWER_LOOKAHEAD=2`
（config.py:L313–L315），由 `tests/test_react.py` 的优先级链段固定。要运行没有循环
内干预的 v5.0 基线，将这三项都翻转为 `false`，框架便回到「每轮单发直至预算耗尽」
的旧行为。

`forecast_eval/errors.py` 中的错误分类在 v5.1 被加宽：HTTP 400 响应体含有
`data_inspection_failed`、`inappropriate content` 或 `sensitive` 之一时（除了
`errors.CONTENT_POLICY_NEEDLES` 在 errors.py:L39–L48 中已有的 `content_policy` /
`content_filter` / `safety` / `content_policy_violation`），归类为 `content_policy`
而非 `bad_request`。errors.py:L97–L111 的瞬时网络家族现在还覆盖
`httpx.RemoteProtocolError`、`WriteError`、`WriteTimeout`、`PoolTimeout`。LLM 客户端
与 Tavily 搜索客户端均会重试这些错误，而非视作致命。

### 8.4 搜索泄漏过滤（v5.2）

Tavily 按抓取或索引日期过滤，不按内容时间过滤。在 $`\chi_i`$ 之前被索引的页面仍可能
描述其后发生的事件，例如维基更新、聚合页或「展望」段落。为堵上这一漏洞，框架增加了
一道二阶段、基于 LLM 的语义审计。每条 Tavily 结果被送入独立的 `detector` LLM，按条
返回 `keep` 或 `drop`。被探测器标记为 `drop` 的条目会在主 LLM 看到搜索载荷之前被
剔除。输入字段被白名单限定在标题、URL、published_date、content、raw_content、
cutoff_date；问题文本、选项与真值被有意隐瞒，使探测器是一个泄漏分类器而非答案审计器。

默认值（完整带注释段落见 `.env.example`）:

* `ENABLE_SEARCH_LEAK_FILTER=true`（config.py:L337）启用过滤所必需。需配合
  `LEAK_DETECTOR_API_KEY` 与 `LEAK_DETECTOR_MODEL`。它与 `ENABLE_WEB_SEARCH=true`
  互为前提，否则探测器路径为死代码，启动会在 config.py:L752–L757 快速失败。
* `LEAK_DETECTOR_BASE_URL` 可选；空值回退至 `LLM_BASE_URL`。即使端点重合，探测器
  客户端仍独立于主 LLM 客户端（`leak_filter.get_detector_client`,
  leak_filter.py:L112），因此具有独立的额度、超时与退避记账。
* `LEAK_DETECTOR_FAIL_ACTION=drop`（config.py:L351）是失败即丢弃的默认。HTTP 错误、
  超时或无效 JSON 在经 `LEAK_DETECTOR_RETRY_MAX` 次重试与 `LEAK_DETECTOR_BACKOFF_S`
  退避后，将丢弃该条目。仅当与未过滤基线作 A/B 对照时，将其置为 `keep`。
* `LEAK_DETECTOR_RETRY_MAX` 与 `LEAK_DETECTOR_BACKOFF_S` 独立于主 LLM 的重试设置，
  因此探测器抖动绝不会回压到主 LLM 的额度窗口。

每次 `web_search` 调用的审计字段持久化于 `run_results.search_calls` 的 JSON 条目内：

```text
{ "query": ..., "end_date": ..., "n_results": <kept>,
  "published_dates": [<原始顺序，长度 == n_results_raw>],
  "n_results_raw": <int>, "n_results_kept": <int>,
  "detector_verdicts": ["keep", "drop", "failed:network", ...],
  "detector_latency_ms": <int>, "detector_error_kind": str | null }
```

`run_meta.config_snapshot` 还记录探测器指纹三元组 `leak_detector_enabled`、
`leak_detector_model`、`leak_detector_prompt_hash`，后者是 `leak_filter.py:L55–L92`
处 prompt 模板 sha256 的前 16 位十六进制截断。泄漏屏障由此可字节复现。该机制由
`tests/test_leak_filter.py` 与在线烟雾 `scripts/smoke_leak_filter.py`、
`scripts/verify_leak_filter_e2e.py` 共同固定。

关闭路径：设置 `ENABLE_SEARCH_LEAK_FILTER=false` 即可完全旁路探测器层；行为与 v5.1
字节一致。所有上游屏障保持不变：web_search 模式、`end_date` 注入、Tavily `end_date`
过滤、`MODEL_TRAINING_CUTOFFS` 可纳入性校验与 `:online` 禁令。

---

## 9. 文档

### 9.1 分层文档

仓库的文档采取分层组织。每一层回答不同的问题；按问题选择对应层，跳过其他层。

| 你想知道…                                                                                       | 阅读…                                |
| ----------------------------------------------------------------------------------------------- | ------------------------------------ |
| 本项目是什么、如何运行                                                                          | 本 README                            |
| 每道约束为何存在，哪些方案被否决                                                                 | `DESIGN.md`                          |
| 将每个符号映射到一个模块、一个 DB 列、一个测试的字段级与接口级规范                              | `FRAME.md`                           |
| 每份模式变更提案的精确论证                                                                      | `openspec/changes/<change-id>/`      |
| 每份归档模式变更提案的精确论证                                                                  | `openspec/changes/archive/`          |
| 紧预算配置配方                                                                                  | 本 README §4                         |
| 契约旋钮 vs 工程旋钮（哪些 `.env` 改动会让跨运行可比性失效）                                      | `DESIGN.md` §12                      |
| 三个独立指纹与 manifest 布局                                                                    | `FRAME.md` §6.3;`evaluation.py::_compute_*_protocol` |

三层文档构成双向契约：DESIGN 的论证 → FRAME 的规范 → 代码的实现，每一层都有自己的
固定测试。每一层均可独立阅读，但任何两层之间的矛盾都意味着 bug。测试套件存在的意义
即是早期捕获此类矛盾。

### 9.2 新人阅读顺序

如果你刚接触本项目，建议按以下顺序阅读：

1. **`README.md`**（本文件），了解 OracleProto 是什么、如何运行。
2. **`DESIGN.md`**，了解论证：每道约束为何存在、威胁模型、严格匹配与部分得分之间
   的权衡、以及为何 DB 仅存原始观测。§1（框架与代码地图）是最快入口；§12 将契约
   旋钮与工程旋钮分离开。
3. **`FRAME.md`**，了解字段级、接口级与伪代码级的技术规范。§1.1（总图）是交叉引用
   的脚手架；§2–6 自顶向下从数据走到流水线。
4. **`forecast_eval/prompts.py` 与 `forecast_eval/parser.py`**，了解渲染器 $`R`$ 与
   解析器 $`\Psi`$，它们是项目信息边界的核心。
5. **`forecast_eval/runner.py` 与 `forecast_eval/react.py`**，了解编排。可纳入性
   过滤位于 runner.py:L132,ReAct 循环位于 react.py:L162，优先级链位于
   react.py:L266。
6. **`tests/`**，反向逆推契约。33 个测试文件覆盖 v3/v4/v5 模式、泄漏过滤、考试式
   评分、网格调度器与行为诊断。
7. **`openspec/changes/archive/`**，了解今日的设计为何成为今日的样子。

---

> **一句话总结。** OracleProto 通过将信息边界作为数据的一部分（而非 prompt 的一部
> 分），把 LLM 预测评测从一次性的实时竞赛变为数据集层面、可审计、可复用、可训练的
> 能力。
