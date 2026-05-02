# OracleProto 技术框架

> 本文档是 OracleProto 参考实现的工程规约。它将运行单元
> $`\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)`$
> 以及辅助探测器 $`H_{\mathrm{aux}}`$ 的每一个符号，逐一映射到模块、
> 函数、环境变量、SQLite 列以及固定该不变量的单元测试。请配合
> `DESIGN.md` 阅读以了解每项权衡背后的设计依据；面向新读者的两页入门
> 指引位于 `README.md`。

## 如何阅读本文档

本文档是一份自顶向下的参考资料。第 1–4 节确立"度量的对象是什么"，
第 5–8 节描述流水线如何产出度量值，第 9–12 节涵盖将这些度量值转化为
可上报数字的分析机制与运维机制。每一节都可以独立检索，但靠后的章节
会假定读者已熟悉 §1 的符号表。

文中始终使用两种引用形式：

* `module.py:Lnnn` 指向当前主干上的行号。当给出行号区间时，引用的符号
  或契约即跨越这些行。
* `test_<name>.py` 指向 `tests/` 下的文件。仓库共附带 33 个测试文件，
  约包含 560 个独立用例，全部离线运行。

正文中的记号 $`X \to Y`$ 表示"X 解析为 / 产生 Y"；$`X = Y`$ 保留其
数学含义。

---

## 1. 运行单元

本代码库是 **OracleProto** 的参考实现，OracleProto 是一套通过知识截止
与时间掩码对 LLM *原生预测能力* 进行基准评测的可复现框架。

实现承担两项职责：其一是物化运行单元 $`\mathcal{R}`$，使同一份配置在
中间产物上字节等价、仅在最终答案文本上保留随机性差异；其二是通过
SHA-256 指纹将辅助泄漏探测器 $`H_{\mathrm{aux}}`$ 与运行元数据绑定，
从而使泄漏屏障本身可字节复现。后续每一节都回答同一个问题：实现是
如何兑现 $`\mathcal{R}`$ 的某个组成的，又由哪一个测试固定该契约？

### 1.1 符号到实现的映射

运行单元 $`\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R,
\Psi, \phi, \Gamma)`$，连同作为运行元数据记录的辅助探测器
$`H_{\mathrm{aux}}`$，按下表映射到代码库。每个符号对应一个配置旋钮、
一条代码路径、一列数据库列（如适用），以及一个固定该契约的测试。

| 符号               | 对象                              | 环境变量 / 配置键                                | 代码路径                                                                       | 数据库列 / 产物                                  | 固定测试                          |
| ------------------ | --------------------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------- | --------------------------------- |
| $`\mathcal{D}`$      | 离散预测数据集                    | `SOURCE_DB`、`SOURCE_TABLE`                     | `loader.sync_questions`（loader.py:L77）                                        | `questions` 表；`manifest.hashes.source_db`        | `test_db.py`、`test_evaluation.py` |
| $`M`$                | 被评测模型（slug）                | `MODELS`（CSV）                                 | `runner._resolve_settings`（runner.py:L160）；按模型调用 `llm.chat`             | `run_meta.model`；按模型一份 DB 文件               | `test_runner_grid_model.py`        |
| $`\kappa_M`$         | 各模型的知识截止                  | `MODEL_TRAINING_CUTOFFS=<slug>=YYYY-MM-DD,...`  | `runner.build_task_plan` 可纳入性过滤（runner.py:L132–L199）                   | `run_meta.training_cutoff`；`s{i}_error="skipped_training_cutoff"` | `test_training_cutoff.py`         |
| $`\delta`$           | 时间掩码偏移（天）                | `TAVILY_END_DATE_OFFSET_DAYS`（默认 `-1`）      | `react._compute_end_date`（react.py:L182）；`search.tavily_search`               | `s{i}_search_calls[*].end_date`                    | `test_search.py`、`test_react.py`  |
| $`T`$                | ReAct 步数上限                    | `REACT_MAX_STEPS`（默认 `12`）                  | `react.run_react` 外层循环（react.py:L248）                                     | `s{i}_react_steps`；`s{i}_step_metrics`            | `test_react.py`                    |
| $`C`$                | 搜索调用上限（网格轴）            | `REACT_MAX_SEARCH_CALLS`（CSV；默认 `[8]`）     | 预算门控 `react.py:L276–L279`；工具调用校验 L429–L503                          | 虚拟 slug `::c{C}`；`s{i}_tool_calls_count`         | `test_react.py`、`test_grid_slug.py` |
| $`R`$                | 输入渲染器                        | `dataset_metadata.features_json.prompt_reconstruction` | `prompts.render_user_prompt`（prompts.py:L447）                          | `s{i}_user_prompt`；`manifest.hashes.prompt_templates` | `test_prompts.py`                |
| $`\Psi`$             | 输出解析器与有效性                | （无环境变量旋钮）                              | `parser.parse_answer`（parser.py:L40）                                          | `s{i}_final_answer_letters`、`s{i}_parse_ok`        | `test_parser.py`                   |
| $`\phi`$             | 答案归一化映射                    | （字母编码规则，见 §4.8）                       | `parser.parse_gt`、`parser.is_correct`（parser.py:L102）                        | `s{i}_correct`                                     | `test_parser.py`                   |
| $`\Gamma`$           | 聚合规则                          | `COMPOSITE_WEIGHTS_*`、`SAMPLING_N` 等          | `forecast_eval/analysis/*`（由 `evaluation.py` 自动调用）                       | `runs/{run_id}/analysis/` 下的 CSV / MD / JSON      | `test_analysis.py`                 |
| $`H_{\mathrm{aux}}`$ | 泄漏探测器（Stage 2）             | `ENABLE_SEARCH_LEAK_FILTER`、`LEAK_DETECTOR_*`  | `leak_filter.filter_search_result`（leak_filter.py:L348）                        | `s{i}_search_calls[*].audit.detector_*`             | `test_leak_filter.py`              |
| $`\hat{p}_{q,j}`$    | 信念向量（v4 配套）               | `BELIEF_PROTOCOL`（默认 `False`）               | `parser.parse_belief`（parser.py:L117）；`react.run_react` 收尾阶段              | `s{i}_belief_final`、`s{i}_belief_trace`、`s{i}_belief_parse_ok` | `test_parser_belief.py`、`test_react_reflection.py` |

辅助轴 $`R_{\mathrm{tav}}`$ 表示 Tavily 单次调用返回的结果数，与渲染器
符号 $`R`$ 不同。它对应 `TAVILY_MAX_RESULTS`，取 CSV 值，默认 `[5]`。
它与 $`C`$ 一起张成网格，编码进虚拟 slug `{real_model}::r{R}::c{C}`
（§10）。

### 1.2 不变量

下述八条是框架级不变量，必须由实现保持。每一条都由代码强制并由至少
一个测试固定，因此任何一条测试失败都会使运行单元失效。

1. **LLM 永远看不到 $`\chi_i`$。** 暴露给 LLM 的 `web_search` 工具
   schema 仅声明一个 `query` 参数（tools.py:L7–L24）；$`\chi_i = \tau_i +
   \delta`$ 由工具实现层硬编码（react.py:L182、search.py:L133）。
   由 `test_search.py` 固定载荷契约，由 `test_react.py` 固定端到端注入。
2. **禁止厂商内置的浏览。** 以 `:online` 结尾的 slug 在启动时被拒绝
   （config.py:L599–L614）并在传输层再次断言（llm.py:L74–L98），
   `extra_body.plugins` 与除已声明 `web_search` 之外的任何工具 schema
   亦被拒绝。由 `test_llm_no_browsing.py` 与 `test_config.py` 固定。
3. **样本准入先于任何 LLM 调用。** 检查 $`\kappa_M \le \chi_i`$ 在任务
   计划生成阶段进行；不满足准入条件的样本直接写入
   `error="skipped_training_cutoff"` 行，不消耗任何 LLM 或 Tavily
   预算（runner.py:L132–L199）。由 `test_training_cutoff.py` 固定。
4. **以严格 frozenset 相等性评分答案。** `parser.is_correct(pred, gt)`
   就是一行 `pred == gt`（parser.py:L102–L106），三种题型都归约到这一
   等式判断。由 `test_parser.py` 固定。
5. **数据库只存原始观测。** 不存任何聚合值或派生指标；§9 所有指标都由
   `forecast_eval.analysis` 在事后读宽表计算。由 `test_analysis.py`
   固定，该测试在不再触碰 DB 固定件的前提下对其执行分析。
6. **Stage-2 探测器 $`H_{\mathrm{aux}}`$ 拥有封闭的输入白名单。** 仅
   `title`、`url`、`published_date`、`content`、`raw_content` 与
   `cutoff_date` 进入探测器 prompt（leak_filter.py:L212–L227）；问题
   文本、选项与正确答案永远不会被传入。由 `test_leak_filter.py` 固定。
7. **三个独立指纹而非单一指纹。** `prompt_templates_hash`、
   `reflection_protocol_hash` 与 `belief_protocol_hash` 并排存放于
   `run_meta` 与 manifest 顶层（db.py:L143–L150、evaluation.py:L171–L178），
   因此在 {模板, 反思, 信念} 任一轴向上的消融实验互不冲撞。
8. **综合准确率（Composite Accuracy）是头条指标。** 默认子类型权重
   为 `yes_no=0.15`、`binary_named=0.15`、`multiple_choice=0.70`
   （config.py:L365）；按指标的覆写值会针对已知指标名称白名单做校验，
   错拼立刻失败（config.py:L515–L535、composite.py:L77–L127）。
   由 `test_composite_score.py` 固定。

---

## 2. 数据集 $`\mathcal{D}`$

### 2.1 源数据库

示例数据集以 `forecast_eval_set_example.db` 形式发布，主表为
`forecast_eval_set_example`。两个名称都可通过 `.env` 中的 `SOURCE_DB`
与 `SOURCE_TABLE` 配置。自定义数据集必须保留以下七列模式与
`dataset_metadata` 结构。`SOURCE_TABLE` 仅接受合法的 SQLite 标识符
（匹配 `^[A-Za-z_][A-Za-z0-9_]*$`），并在启动时校验
（config.py:L586–L595），因为它会被原样拼入查询，否则会构成 SQL 注入
向量。

主表包含 $`N`$ 行、七列：

| 字段            | 类型    | 描述                                                                                                                  |
| --------------- | ------- | --------------------------------------------------------------------------------------------------------------------- |
| `id`            | TEXT PK | 唯一问题 ID，来源于 HuggingFace。                                                                                       |
| `choice_type`   | TEXT    | `single` 或 `multi`，由答案字母数量计算得出。                                                                            |
| `question_type` | TEXT    | `yes_no`、`binary_named` 或 `multiple_choice`；选择 prompt 模板族。                                                       |
| `event`         | TEXT    | 事件描述 $`x_i`$，不携带选项、角色设定或格式要求。                                                                          |
| `options`       | TEXT    | 选项集 $`\mathcal{A}_i`$，以 JSON 数组表示。`yes_no` 为 `["Yes","No"]`；`binary_named` 为两个实体名；`multiple_choice` 为带标签的选项列表。 |
| `answer`        | TEXT    | $`Y_i`$ 以字母编码：单选为 `"A"`，多选为 `"A, B"`（逗号 + 空格）。字母→索引的规则见 §4.8。                                  |
| `end_time`      | TEXT    | 解算时间 $`\tau_i`$（Asia/Shanghai），格式为 `YYYY-MM-DD`。                                                                  |

索引：`idx_<table>_choice_type`、`idx_<table>_question_type`、
`idx_<table>_end_time`。

辅助表 `dataset_metadata` 仅含一行，其 `features_json` 字段记录所有
prompt 模板、列描述与转换日志。渲染器 $`R`$ 在运行时从该表读取模板；
模板有意不在源代码中硬编码。`prompt_templates_hash` 指纹仅覆盖 §2.3
列出的八个模板键，而反思、预算感知、信念这些协议补充则作为运行时
槽位存在，不进入该指纹（§4.7）。

### 2.2 示例数据集

`forecast_eval_set_example.db` 包含 80 道题，跨度从 2026-03-12 到
2026-04-14：

| question_type / choice_type | single | multi | total |
| --------------------------- | -----: | ----: | ----: |
| `yes_no`                    |     37 |     0 |    37 |
| `binary_named`              |      3 |     0 |     3 |
| `multiple_choice`           |     32 |     8 |    40 |
| **total**                   |   **72** |  **8** |    **80** |

本示例中 `multiple_choice` 的选项数从 3 到 14 不等，但解析器支持 §4.8
描述的完整 ASCII 续接编码方案，因此自定义数据集即使包含至多 35 个
选项也无需改动代码即可有效。

只要满足七列契约与 `dataset_metadata` 结构，框架本身与具体数据集无关。

### 2.3 `prompt_reconstruction` 契约

渲染器 $`R`$ 严格要求以下八个键，缺失任一会在加载时抛错
（loader.py:L13–L22）：

```text
agent_role
guidance
prompt_template
outcomes_block_rule
yes_no_output_format
binary_named_output_format
multiple_choice_single_output_format
multiple_choice_multi_output_format
```

`db.compute_prompt_templates_hash(templates)`（db.py:L397–L399）仅在
这些键上计算
$`\text{sha256}(\text{canonical\_kv\_string}(\text{templates}))`$。
本次运行是否启用反思、信念或预算感知，对该指纹是不可见的；这些文本
分别独立哈希到 `reflection_protocol_hash` 与 `belief_protocol_hash`
（§6.3）。

### 2.4 示例

`yes_no`：

```yaml
event:    "2026 a dream year for trump?"
options:  ["Yes","No"]
answer:   "B"            # B = No
end_time: "2026-01-31"
```

`binary_named`：

```yaml
event:    "Golden Knights vs. Kings"
options:  ["Golden Knights","Kings"]
answer:   "A"            # A = Golden Knights
end_time: "2026-01-15"
```

`multiple_choice`（单选）：

```yaml
event:    "Bank of Brazil decision in January?"
options:  ["No change in the Selic rate ...", "the Bank of Brazil raise ...", "the Bank of Brazil lower ..."]
answer:   "A"
end_time: "2026-01-27"
```

`multiple_choice`（多选）：

```yaml
event:    "Oscars 2026: Achievement in Casting Nominations"
options:  [<12 nominee list entries>]
answer:   "A, B, D, E"
end_time: "2026-01-22"
```

按惯例，`event` 字段不携带选项或格式要求；这些在调用时由渲染器 $`R`$
拼入（§4.7）。

---

## 3. 端到端流水线

流水线包含七个阶段，从 `.env` 一路走到分析写出器。读者若想了解某项
保证在流水线的何处被强制执行，应查阅 §4 的信息边界阶段以及 §5 的
循环内控制流。

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   OracleProto 评测流水线                                │
└────────────────────────────────────────────────────────────────────────┘

[.env]  →  [python evaluation.py [--question-type ...] [--choice-type ...]]
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │ 1. 加载 Settings       │
                          │    并初始化 run_id     │
                          └────────────────────────┘
                                      │
                                      ▼
                          ┌──────────────────────────────────┐
                          │ 2. 同步源数据                    │
                          │    forecast_eval_set_example.db  │
                          │      → questions 表              │
                          │      → prompt_templates 表       │
                          │    计算四个哈希                  │
                          └──────────────────────────────────┘
                                      │
                                      ▼
                          ┌────────────────────────────────────┐
                          │ 3. 续跑检测                        │
                          │    各模型执行                      │
                          │    load_completed_samples：跳过    │
                          │    已处于不可重试终止态的行        │
                          └────────────────────────────────────┘
                                      │
                                      ▼
                 ┌────────────────────────────────────────┐
                 │ 4. 任务计划（D × M × N）               │
                 │    应用 κ_M 可纳入性过滤               │
                 │    写入 skipped_training_cutoff 行     │
                 │    asyncio.Semaphore 用于 LLM/Search/  │
                 │    探测器三条通道                      │
                 └────────────────────────────────────────┘
                                      │
                      ┌───────────────┼───────────────┐
                      ▼               ▼               ▼
                ┌──────────┐    ┌──────────┐    ┌──────────┐
                │ Worker 1 │    │ Worker 2 │ …  │ Worker N │
                └────┬─────┘    └────┬─────┘    └────┬─────┘
                     │               │               │
                     └───────────────┼───────────────┘
                                     ▼
                      ┌────────────────────────────┐
                      │ ReAct 循环 F_M（每样本，   │
                      │ 详见 §5）：                │
                      │   render(q) → user_prompt  │
                      │   while step < T:          │
                      │     步前注入               │
                      │     llm.chat(messages,     │
                      │               tools=...)   │
                      │     for each tool_call:    │
                      │       u_t = tavily.search( │
                      │              query, χ_i)   │
                      │       ũ_t = AuxLeakFilter( │
                      │              u_t, χ_i)     │
                      │   v5.1 末次答案重试        │
                      │   parser.parse_answer      │
                      └──────────┬─────────────────┘
                                 │
                                 ▼
                      ┌────────────────────────┐
                      │ 5. 评分                │
                      │    在解析后的字母集合  │
                      │    上应用 Ψ ∘ φ        │
                      └──────────┬─────────────┘
                                 │
                                 ▼
                      ┌────────────────────────┐
                      │ 6. 入队 → 写出器       │
                      │    每模型一个          │
                      │    AsyncWriter；WAL +  │
                      │    批量 UPSERT         │
                      └──────────┬─────────────┘
                                 │
                                 ▼
                      ┌────────────────────────┐
                      │ 7. analysis.run        │
                      │    聚合 Γ：            │
                      │    Composite Accuracy、│
                      │    FSS、Cohen κ、      │
                      │    Fleiss κ、pass@k、  │
                      │    BI、NLL …           │
                      │    → CSV / MD / JSON   │
                      └────────────────────────┘
```

七个阶段的叙述形式：

1. **加载与校验。** `Settings()` 通过 pydantic-settings 解析 `.env`，
   并执行 §7.2 列出的全部检查。`run_id` 被生成（或在用户提供时复用），
   并创建运行目录 `RUNS_ROOT/{run_id}/`，下设 `db/`、`analysis/` 与
   `logs/` 子目录。
2. **同步源数据。** `loader.sync_questions` 与
   `loader.sync_prompt_templates` 将源数据库中的表复制进每个模型的
   运行 DB，随后计算并存储四个可复现性哈希（`source_db_hash`、
   `metadata_hash`、`prompt_templates_hash`，以及在条件满足时的
   `reflection_protocol_hash` 与 `belief_protocol_hash`）。
3. **续跑。** 对每个模型，`db.load_completed_samples` 扫描已有的
   `run_results` 行，输出 `(question_id, sample_idx)` 的集合：要么
   正常填充完毕，要么已被标记为 `skipped_training_cutoff`。
4. **任务计划。** `runner.build_task_plan` 生成
   $`\mathcal{D} \times M \times \{0, \dots, n-1\}`$ 的笛卡尔积，扣除
   续跑集合，再施加 $`\kappa_M`$ 可纳入性过滤（§4.2）。不被纳入的样本
   直接写为 `skipped_training_cutoff`，绝不接触 LLM 或 Tavily。
5. **Worker 扇出。** 三个独立的 `asyncio.Semaphore` 对象限制 LLM、
   Tavily 与探测器三条通道的并发。每个 Worker 通过 ReAct 循环 $`F_M`$
   驱动一条样本（§5），该循环交替进行 LLM 轮次与 Tavily 调用，后者
   本身又会被 $`H_{\mathrm{aux}}`$ 审计。
6. **评分与持久化。** 每个完成的 `SampleResult` 被入队到该模型的
   `AsyncWriter`（§6.5），后者批量进行 UPSERT，每攒满
   `DB_COMMIT_BATCH` 条或每秒一次 flush。
7. **分析。** 除非传入了 `--skip-analysis`，否则
   `forecast_eval.analysis.run_analysis(run_dir)` 会遍历每个模型 DB，
   运行 §9 的指标栈，并把 §9.12 的产物写入 `analysis/`。

### 3.1 并发模型

三个信号量分别对外部受限资源进行划分：`LLM_MAX_CONCURRENCY`、
`SEARCH_MAX_CONCURRENCY` 与 `LEAK_DETECTOR_CONCURRENCY`，默认均为 5。
之所以使用三个独立信号量而非共享单一预算，是因为三个后端各自独立
地限速，且探测器的 QPS 预算只是主 LLM 预算的一部分。

写出器侧采用每模型一个单线程异步写出器：`runner.run`
（runner.py:L362）为每个模型打开一个 `db.AsyncWriter`，并把该模型的
结果路由到它。因此每个单模型 DB 都是一个写出者多个读取者的场景，这
在 SQLite WAL 模式下是安全的。

### 3.2 续跑语义

续跑按样本槽独立判定。写出器执行的查询是：

```sql
SELECT question_id FROM run_results
 WHERE s{i}_created_at IS NOT NULL
   AND (s{i}_error IS NULL OR s{i}_error = 'skipped_training_cutoff');
```

因此凡是 `created_at` 已设、且 `error` 要么为 NULL 要么为预设排除值
的槽位，都被视为已完成并从任务队列中移除。下游使用的状态分类如下：

| `error` 取值                        | 含义                                | 下次续跑是否重试？                                  |
| ----------------------------------- | ----------------------------------- | --------------------------------------------------- |
| `NULL`                              | 正常完成                            | 否                                                  |
| `'skipped_training_cutoff'`         | 由 §4.2 排除                         | 否                                                  |
| `'network'`、`'server_5xx'`         | 退避后仍然失败                      | 是                                                  |
| `'rate_limit'`                      | 速率限制，退避用尽                  | 是                                                  |
| `'bad_request'`                     | `model_not_found` 等                | 是（在配置修复后）                                  |
| `'content_policy'`                  | 厂商拒绝                            | 可选；默认重试一次并覆写                            |

使用相同 `run_id` 重新运行将续跑进既有的
`runs/{run_id}/db/<slug>.db`；不同的 `run_id` 会产出全新的
`runs/{new_run_id}/`。覆写原语为 `INSERT ... ON
CONFLICT(question_id) DO UPDATE SET s{i}_* = excluded.s{i}_*`，
`user_prompt` 通过 `COALESCE` 保留首个样本的取值。由
`test_runner_resume.py` 固定。

---

## 4. 信息边界

框架围绕三条受控信息通道与一面有据可查的残余面来组织泄漏控制。代码库
将每条通道实现于流水线的特定层；本节按样本所遭遇的顺序逐一展开。

### 4.1 通道与残余

| 通道                | 对象                                  | 所在层                      | 防御                                                                            |
| ------------------- | ------------------------------------- | --------------------------- | ------------------------------------------------------------------------------- |
| 1. 参数化            | 训练前已存在的知识                    | 样本准入                    | $`\kappa_M`$ 可纳入性过滤（§4.2）                                                  |
| 2. 工具中介          | Tavily 请求载荷                       | 工具层                      | 在 `end_date` 上注入 $`\delta`$ 偏移（§4.3）                                       |
| 3. 检索内容          | Tavily 响应正文                       | LLM 之前的 Stage-2 审计     | 独立探测器 $`H_{\mathrm{aux}}`$，配合白名单与失败即丢弃（§4.4）                    |
| 4. 厂商侧            | 内置浏览或增强                        | 传输层断言                  | `:online` 禁令、`plugins` 禁令、单工具白名单（§4.5）                             |
| 残余 A               | 问题文本中的时间线索                  | 无                          | 数据本身固有；作为评测偏差接受（§4.6）                                          |
| 残余 B               | 训练后的知识回流                      | 无                          | 作为评测偏差接受（§4.6）                                                        |

### 4.2 通道 1：参数化知识

若一道题的解算时间早于模型的训练截止，则其答案很可能已经处于训练
语料中；模型是在"记起"答案而非预测它。这类样本无法反映原生预测能力，
应当从模型的可评测子集 $`\mathcal{D}^{\mathrm{pred}}_M`$ 中移除。

每个模型的 $`\kappa_M`$ 在 `.env` 中通过 `MODEL_TRAINING_CUTOFFS` 声明，
该项是 `<slug>=YYYY-MM-DD` 对的 CSV，由 `config._parse_cutoffs`
（config.py:L479）解析。任务计划阶段，`runner.build_task_plan` 对每对
`(question, model)` 执行过滤：

```python
cutoff = MODEL_TRAINING_CUTOFFS.get(real_model)   # None 表示未声明，不做过滤
if cutoff is not None and q.end_time <= cutoff:
    # 在 δ = -1 day 下与 χ_i < κ_M 等价的天粒度判断：
    # 为每个 sample_idx 写入一行，error="skipped_training_cutoff"
    enqueue_skipped_cutoff_rows(q, model)
```

被过滤的 `(question, model, sample_idx)` 行依然落在 `run_results`
中，`error="skipped_training_cutoff"`、`parse_ok=0`、`correct=NULL`，
所有数值字段为零。这样"每个模型过滤掉了多少题"可直接从 DB 审计，
并喂给报表中按模型的排除计数列。续跑永远不会重试这些行。

`test_training_cutoff.py` 分三部分固定该契约：对于
`q.end_time <= cutoff` 的题目，所有样本槽都写入
`skipped_training_cutoff`；未声明截止日的模型不被过滤；续跑优先于
截止过滤，已完成的行不会被排除行替换。

### 4.3 通道 2：工具中介知识

暴露给 LLM 的 `web_search` schema 仅声明一个 `query` 参数
（tools.py:L7–L24）：

```python
{
  "type": "function",
  "function": {
    "name": "web_search",
    "description": "Search the web for information relevant to the question.",
    "parameters": {
      "type": "object",
      "properties": {"query": {"type": "string", "description": "Search query"}},
      "required": ["query"],
      "additionalProperties": false
    }
  }
}
```

实际调用 Tavily 时，截止 $`\chi_i = \tau_i + \delta`$ 由工具实现硬编码
（react.py:L182、search.py:L133–L162）：

```python
end_date = (date.fromisoformat(q.end_time)
            + timedelta(days=settings.TAVILY_END_DATE_OFFSET_DAYS)).isoformat()
result = await search.tavily_search(query=args["query"], end_date=end_date, settings=cfg)
```

默认 $`\delta = -1`$ 天。在 `end_time` 为 `YYYY-MM-DD` 粒度下，该偏移
排除当日信息：

```text
question.end_time (τ_i) = 2026-01-18
→ Tavily end_date (χ_i) = 2026-01-17
```

$`\delta = 0`$（宽松）与 $`\delta \in \{-2, -3\}`$（更保守）也都是合法
配置，但报表默认使用 $`\delta = -1`$ 之下的对比，因此不同偏移下得到的
数值不可直接比较。

`test_search.py` 固定 LLM 可见的 `web_search` schema 不包含
`end_date`，且 `tavily_search` 在被调用时会注入正确的 `end_date`；
`test_react.py` 端到端固定循环内的连接关系。

### 4.4 通道 3：检索内容审计

工具级过滤约束的是请求侧，但返回的摘要、缓存页面与聚合摘要仍可能
携带 $`\chi_i`$ 之后的内容。`leak_filter.py` 中的 Stage-2 探测器在每条
Tavily 结果进入主 LLM 上下文之前对其进行审计。

探测器运行在一个**独立客户端**之上：`_detector_client`
（leak_filter.py:L108–L133）是与主 `llm._client` 不同的另一个
`AsyncOpenAI` 实例，通过 `LEAK_DETECTOR_*` 环境变量配置。如果
`LEAK_DETECTOR_BASE_URL` 为空，则回退到 `LLM_BASE_URL`。切入点是
`search.tavily_search` 中 HTTP-200 路径的末尾、`return` 之前；裁决
随后由 `leak_filter.filter_search_result`（leak_filter.py:L348）应用，
后者遍历结果列表并按裁决丢弃条目。

探测器 prompt 拥有由六个字段构成的**封闭输入白名单**，这是契约的
承重之处（leak_filter.py:L212–L227）：

```text
title           — result.title
url             — result.url
published_date  — result.published_date 或 "(unknown)"
content         — result.content
raw_content     — result.raw_content 或 "(empty)"
cutoff_date     — 调用方传入的 χ_i
```

问题文本、选项与正确答案绝不会被传入。把探测器框定为"答案审计器"会
制造二阶泄漏，因为探测器可能据此合理化"伪造证据与已知答案一致"而
放行。

输出 schema 是严格的 JSON，包含两个字段：

```json
{"verdict": "keep" | "drop", "reason": "<sentence>"}
```

`drop` 裁决会在主 LLM 看到任何东西之前删除整条结果（包含 title、URL、
content、raw_content）；审计字段则保留以供事后审查。

探测器**默认失败即丢弃**。重试序列为
`max_attempts = LEAK_DETECTOR_RETRY_MAX + 1`，默认三次重试，退避序列
`[2, 5, 15]` 秒。401 或 403 的 AUTH 错误在本地捕获、转换为
`failed:auth` 且永不向上传播；该条目立即被丢弃
（leak_filter.py:L281–L288）。其他可重试错误耗尽序列后，裁决变为
`failed:<kind>`，并由 `LEAK_DETECTOR_FAIL_ACTION` 接管：默认 `drop`
丢弃条目，`keep` 则放行。该默认值将残余偏向"不确定就丢弃"，因为
探测器抖动与条目内容不相关。

每次调用都会向 `s{i}_search_calls[*].audit` 追加一个审计字典
（leak_filter.py:L380–L387）：

| 字段                   | 类型         | 含义                                                  |
| ---------------------- | ------------ | ----------------------------------------------------- |
| `n_results_raw`        | int          | 过滤前的条目数                                        |
| `n_results_kept`       | int          | 过滤后的条目数                                        |
| `published_dates_raw`  | list[str]    | 所有条目的原始发布日期（审计不变量）                  |
| `detector_verdicts`    | list[str]    | 逐条裁决；取值：`keep` / `drop` / `failed:*`          |
| `detector_latency_ms`  | int          | 探测器墙钟延迟                                        |
| `detector_error_kind`  | str \| null  | 该批次中占主导的失败类型                              |

`run_meta.config_snapshot` 内有三个键描述探测器运行情况，外加一个
顶层槽位：

```text
leak_detector_enabled         — bool
leak_detector_model           — str
leak_detector_prompt_hash     — sha256[:16]，对应 LEAK_DETECTOR_PROMPT_TEMPLATE
leak_detector_prompt_version  — 人类可读标签，默认 "v1"
```

当 `ENABLE_SEARCH_LEAK_FILTER=False` 时，探测器路径在字节级别回滚，
行为与未启用探测器的 v5.1 完全一致。

`test_leak_filter.py`（550 LOC）固定五项契约：探测器输入白名单、
重试耗尽时失败即丢弃、AUTH 立即失败即丢弃且不传播、`search_calls`
上每个审计字段的存在、以及禁用路径与 v5.1 字节等价。

### 4.5 通道 4：厂商侧残余

模型服务可能附带一个绕开 Tavily 层的内置浏览工具。传输路径上设有两道
防御。

第一道是 **slug 禁令**。以 `:online` 结尾的 slug（OpenRouter 的
"在线增强"变体命名）在启动时被 `Settings` 校验拒绝
（config.py:L599–L614），并由 `llm._assert_no_browsing`
（llm.py:L74–L98）在传输层再次断言。由 `test_llm_no_browsing.py`
固定。

第二道是 **`plugins` 字段禁令**。`extra_body.plugins` 在传输层被拒
（llm.py:L97），且 `tools=[...]` 中只允许 `[WEB_SEARCH_SCHEMA]` 一个
工具。

如果某厂商强制附加无法关闭的浏览能力，应在 README 与报告中将其标记为
"不适合严格评测"，因为框架无法防御 API 不暴露的能力。

### 4.6 威胁模型与残余面

框架识别的六个泄漏来源中，前面四条可通过上述通道控制；剩下的两条是
数据本身固有的，以及训练后的知识回流，作为评测偏差接受：

| 泄漏来源                                     | 是否可控？    | 缓解                                                                              |
| -------------------------------------------- | ------------- | --------------------------------------------------------------------------------- |
| 工具搜索内容（Tavily 返回文本）               | 是            | 在工具层注入 $`\delta`$ 偏移（§4.3）                                                |
| 厂商内置浏览或 web 工具                       | 是            | `:online` 禁令、`plugins` 禁令、单工具白名单（§4.5）                              |
| 提及 $`\chi_i`$ 后事件的页面正文                | 部分           | Stage-2 探测器，配合白名单输入与默认失败即丢弃（§4.4）                            |
| 模型参数化记忆                                | 部分           | $`\kappa_M`$ 可纳入性过滤排除 $`\chi_i < \kappa_M`$ 的样本（§4.2）                     |
| 问题文本中的时间线索                          | 否             | 数据固有；作为评测偏差接受                                                        |
| 训练后的外部知识回流                          | 否             | 作为评测偏差接受                                                                  |

立场是明确的：这是一个可审计、可复现、可比较的框架，并非"所有泄漏
都被堵死"的证明。

### 4.7 渲染器 $`R`$

源数据库只存储原始素材（`event`、`options`、`question_type`、
`end_time`）。系统派生样本时，
`prompts.render_user_prompt`（prompts.py:L447）从
`dataset_metadata.features_json.prompt_reconstruction` 读取模板，按
`question_type` 装配 user 消息：

```text
{agent_role} The event to be predicted: "{event} (resolved around {end_time} (GMT+8)).{outcomes_block}"

IMPORTANT: Your final answer MUST end with this exact format:
{output_format}
{guidance}
```

各槽位的渲染规则：

| 槽位              | 规则                                                                                                                                                |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `agent_role`      | 常量 `"You are an agent that can predict future events."`，原样插入。                                                                                  |
| `event`           | `<SOURCE_TABLE>.event` 的原始文本。                                                                                                                  |
| `end_time`        | `<SOURCE_TABLE>.end_time` 的原始文本，格式 `YYYY-MM-DD`。                                                                                              |
| `outcomes_block`  | 对 `yes_no` 与 `binary_named` 为空，因为选项已嵌入 `output_format`。对 `multiple_choice` 为换行加 `A. <opt[0]>\nB. <opt[1]>\n...`，按 §4.8 格式化。     |
| `output_format`   | 三种 `question_type` 各对应一个模板。`binary_named` 模板包含 `<options[0]>` 与 `<options[1]>` 占位符，必须替换为实际实体名。                          |
| `guidance`        | 常量 `"Do not use any other format. Do not refuse to make a prediction. ..."`，原样插入。                                                              |

解析器最终匹配的输出格式形态：

* `yes_no` 要求 `\boxed{Yes}` 或 `\boxed{No}`。
* `binary_named` 渲染后形如 `\boxed{Golden Knights} or \boxed{Kings}`。
* `multiple_choice` 要求 `\boxed{A}` 或 `\boxed{B, C}`，并附带示例。

反思、预算感知与信念协议这些补充在对应开关打开时于运行时附加（§5.4）；
它们不进入 `dataset_metadata`，故 `prompt_templates_hash` 在这些开关
切换下保持不变。完整渲染后的 user 消息落入每条样本的
`s{i}_user_prompt` 字段。协议文本指纹独立持久化为
`run_meta.reflection_protocol_hash` 与
`run_meta.belief_protocol_hash`（§6.3）。

### 4.8 字母映射 $`\phi`$

DB 在答案上统一使用**字母**作为正则形式；LLM 的输出形式因
`question_type` 而异：

| question_type      | LLM 输出（位于 `\boxed{}` 内）                                       | 解析器归一化目标（$`\phi`$）                                          |
| ------------------ | -------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `yes_no`           | `Yes` / `No`（大小写不敏感）                                          | `frozenset({"A"})` / `frozenset({"B"})`，Yes=A、No=B                 |
| `binary_named`     | `options` 中某一项（trim + 大小写不敏感的精确匹配）                   | 在 `options` 中查索引，再映射到字母与 frozenset                      |
| `multiple_choice`  | 一个或多个字母，按逗号或空格分隔（`A`、`B, C`、`B,C`）                 | 切分为 token，再转 frozenset                                         |

字母→索引的规则（parser.py:L420–L429）支持至多 35 个选项：

```text
index = ord(letter) - ord('A')
A=0, B=1, ..., Z=25
[ =26, \ =27, ] =28, ^ =29, _ =30, ` =31, a =32, b =33, c =34, ...
```

> ⚠️ **兼容性警告。** 当一道题携带多于 26 个选项时，编码会落到非字母
> 符号上，例如 `[`、`\`、`]`、`^`、`_`、`` ` ``、`a`、`b`、`c`。这些
> ASCII 续接标签对 LLM 不友好：反引号与下划线会被 markdown 与代码块
> 吞掉，且小写 `a` 与大写 `A` 在内联渲染中容易混淆。保留该方案是因为
> 它对源数据字母编码维持了一一对应关系，便于按字母集合评分。三条强制
> 防御措施同时启用：当选项数 >26 时，`prompts.render_user_prompt` 在
> 生成 `outcomes_block` 时给标签加引号并转义；`parser.parse_answer`
> 对 >26 选项的 `multiple_choice` 设有往返单元测试（标签 → 字母 →
> 标签）；日志与报告同时记录字母与对应标签以便人工核查。

供展示或日志用的真值反查为：

```python
opts    = json.loads(row["options"])
letters = [t.strip() for t in row["answer"].split(",")]
labels  = [opts[ord(L) - ord('A')] for L in letters]
```

---

## 5. 预测系统 $`F_M`$

`react.run_react`（react.py:L248–L632）中的 ReAct 循环是 $`F_M`$ 的核心。
它在确定性的优先级链下交替进行 LLM 轮次与 Tavily 调用，该链支配着
预算逼近上限时框架如何介入。

### 5.1 循环骨架

下面的骨架保留了全部 v5.1 接线，同时短到可在一屏内读完。行内注释
标注循环必须遵守的四项契约。注释中编号引用的完整 Python 实现位于
`react.py`。

```python
async def run_react(q: Question, model: str, sample_idx: int, settings: Settings):
    # ① χ_i 对 LLM 不可见；在此处计算并贯穿到 Tavily。
    end_date = (date.fromisoformat(q.end_time)
                + timedelta(days=settings.TAVILY_END_DATE_OFFSET_DAYS)).isoformat()

    # ② m_0 = R(q^in)：一条 user 消息，附加全部启用的协议。
    user_prompt = prompts.render_user_prompt(q, settings.PROMPT_TEMPLATES,
                  budget_awareness=BUDGET_AWARENESS_TEXT_OR_NONE,
                  reflection_protocol=REFLECTION_TEXT_OR_NONE,
                  belief_protocol=BELIEF_TEXT_OR_NONE)
    messages = [{"role": "user", "content": user_prompt}]

    for step in range(settings.REACT_MAX_STEPS):
        # ③ 步前注入（优先级链见 §5.2）：四类机制中至多一种触发。
        injection = pick_injection(step, search_calls, pending_continuation, settings)
        if injection is not None:
            messages.append({"role": "user", "content": injection})

        # 工具 schema 决策：末步硬切断或预算用尽时 tools=[]。
        tools = [] if force_final_hard_cutoff or budget_dropped else [WEB_SEARCH_SCHEMA]

        resp = await llm.chat(model=model, messages=messages, tools=tools, ...)
        msg = resp.choices[0].message
        messages.append(msg.model_dump())

        if settings.BELIEF_PROTOCOL:
            beliefs_per_step.append(parser.parse_belief(msg.content or "", q))

        if not msg.tool_calls:
            # 无工具调用：可能轻推（软地板未达），或在 \boxed{} 出现时跳出，否则进入续接。
            if soft_floor_unmet_and_have_nudges():
                inject_nudge(); continue
            if "\\boxed{" not in (msg.content or ""):
                pending_continuation = True; continue
            final_raw = msg.content or ""
            break

        # 工具调用：先校验，再逐个分发。
        for tc in msg.tool_calls:
            err = _validate_tool_call(tc, settings, searches_done=len(search_calls))
            if err is not None:
                messages.append(prompts.tool_error_message(tc, err))
                continue
            # ④ χ_i 由此处注入而非 LLM。探测器随后审计结果。
            result = await search.tavily_search(query=extract_query(tc), end_date=end_date,
                                                settings=settings)
            search_calls.append(result.to_search_call_record())
            messages.append(prompts.tool_result_message(tc, result.to_llm_payload()))

    # v5.1 D1 兜底：若 final_raw 仍为空，用 tools=[] 重试一次扫尾。
    if final_raw == "" and settings.REACT_FINAL_ANSWER_RETRY:
        messages.append({"role": "user",
                         "content": "Time to commit. Output your final \\boxed{...} now."})
        resp = await llm.chat(model=model, messages=messages, tools=[], ...)
        final_raw = resp.choices[0].message.content or ""
        final_answer_retry_used = 1

    parsed = parser.parse_answer(final_raw, q)        # frozenset[str] | None
    correct = parser.is_correct(parsed, parser.parse_gt(q.answer))
    return SampleResult(...)
```

四项契约是：$`\chi_i`$ 对 LLM 始终不可见（①、④）；user 消息恰为渲染器
输出加上附加协议（②）；每步至多一种框架注入触发，按优先级顺序（③）；
循环以空结束时由后置兜底产生最终答案。

### 5.2 框架韧性优先级链

当循环逼近预算上限时，框架在 user 侧注入引导，推动模型提交答案。
共有四种机制参与，并按严格优先级触发，使行为确定且可被测试固定。
实现位于 react.py:L266，该处保留了原文注释 "Priority is (1) > (2) > (3) > (4)"。

| 优先级 | 触发条件                                                                | 对 `tools` 的影响        | 注入构造器（`prompts.py`）                       |
| ------ | ----------------------------------------------------------------------- | ------------------------ | -------------------------------------------------- |
| 1      | 处于 `LOOKAHEAD` 窗口内的最后一步：`T - step == 1`                      | `tools=[]`               | `build_last_step_force_finalisation`（L294）       |
| 2      | 倒数窗口：`1 < T - step <= LOOKAHEAD`                                    | `tools=[WEB_SEARCH]`     | `build_penultimate_step_warning`（L254）           |
| 3      | 搜索预算触顶：`len(search_calls) >= C`（每次运行触发一次）              | `tools=[]`               | `build_search_budget_exhausted_commit`（L325）     |
| 4      | 上一轮未产出 `\boxed{}`（续接标志）                                      | `tools=[WEB_SEARCH]`     | `build_continuation_after_unboxed_content`（L354） |

四者共享状态头构造器 `_build_status_header`
（prompts.py:L128–L164），该函数在前面拼上一行统一格式
`[Harness status] step k/N (R remaining) · web_search s/C used (M left).`

支配该链的七个旋钮汇总如下。`LOOKAHEAD` 在启动时被夹取到 $`[1, T]`$
（config.py:L696–L707）。

| 旋钮                                  | 默认值  | 作用域                  | 效果                                                                                          |
| ------------------------------------- | ------- | ----------------------- | --------------------------------------------------------------------------------------------- |
| `REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT` | True    | 循环内分级过渡          | 倒数窗口软警告，最后一步硬切断 `tools=[]`。                                                  |
| `REACT_FORCE_FINAL_ANSWER_LOOKAHEAD`  | 2       | 软窗口                  | 距上限多少步开始介入。                                                                        |
| `REACT_BUDGET_AWARENESS_PROTOCOL`     | True    | Prompt 层               | 把 `T` 与 `C` 附加到 user prompt，使模型可整体性规划。                                        |
| `REACT_BUDGET_EXCEEDED_DROP_TOOLS`    | True    | 循环内预算门控          | 一旦累计 `web_search >= C`，后续所有轮次丢弃工具。                                            |
| `REACT_FINAL_ANSWER_RETRY`            | False   | 循环后兜底              | 当循环以空 `final_raw` 结束时，再以 `tools=[]` 调一次 LLM 强制 `\boxed`。                     |
| `REACT_MIN_SEARCH_CALLS`              | 0       | 软地板                  | 若模型在到达 `MIN` 之前尝试提交答案，注入轻推。                                              |
| `REACT_MAX_NUDGES`                    | 2       | 软地板上限              | 每样本的轻推预算。                                                                            |

### 5.3 逐步信念处理

当 `BELIEF_PROTOCOL=True` 时，每个 assistant 轮次（含循环后的末次答案
重试）都被 `parser.parse_belief`（parser.py:L117–L213）解析。每步结果
落入 `beliefs_per_step`，再聚合为三个被持久化的字段：

* `belief_final` 是末步信念的概率 JSON，仅在末步信念解析成功时存在；
  否则为 NULL。
* `belief_trace` 是每步信念摘要的 JSON 数组，未能解析的步对应 `null`
  条目。
* `belief_parse_ok` 当且仅当末步信念合法解析时为 `1`；它独立于
  `parse_ok`。

信念 JSON 模式严格（prompts.py:L66–L105）：

```json
{
  "version": "v4.0",
  "probabilities": { "<letter>": <float in [0, 1]>, ... },
  "confidence": "low" | "medium" | "high",
  "key_evidence":     [ "<= 280 chars per bullet, 1-4 bullets" ],
  "counterevidence":  [ "<= 280 chars per bullet, 0-3 bullets" ],
  "open_questions":   [ "<= 280 chars per bullet, 0-3 bullets" ],
  "decision_rule": "argmax" | "multi-select@<threshold>"
}
```

`probabilities` 的键必须严格匹配预期字母集合（parser.py:L150）。
单选答案必须求和到 $`1.0 \pm 10^{-3}`$（parser.py:L167）；多选答案中
每一项独立处于 $`[0, 1]`$。`confidence` 必须为 `low`、`medium`、`high`
之一（parser.py:L173）。解析失败不影响 `parse_ok`，因为 `\boxed{}`
路径才是唯一的正确性信号。

### 5.4 反思协议

`prompts.REFLECTION_PROTOCOL`（prompts.py:L31–L53）是一份六步推理
脚手架，运行时附加到 user 消息：

1. **拆解**为若干子问题，其联合答案足以裁决预测。
2. **规划差异化角度**：在任何 `web_search` 之前，至少列出三种不同的
   调研角度。
3. **迭代搜索、每次结果后反思**：复述、标注相关性、识别矛盾，并据此
   选下一个最能填补缺口的查询。
4. **交叉验证**：在提交之前用至少两个独立来源加以印证。
5. **反向压力测试**：清晰地阐述对立结论的最强论证。
6. **校准后再提交**：陈述置信度、失败模式与决定性证据，然后才
   `\boxed{...}`。

完整文本被哈希为 `reflection_protocol_hash`（sha256[:16]），并原文存入
`run_meta.reflection_protocol_text`。该文本不进入
`prompt_templates_hash`，因此切换反思开关或编辑该文本不会改变模板
哈希（§6.3）。

---

## 6. 持久化

每一对 (run, model) 对应一个独立的 SQLite 文件，位于
`runs/{run_id}/db/<model_slug>.db`。每个文件自包含一份 `questions`
与 `prompt_templates` 的副本，因此单一文件即可独立重放，无需触碰源
数据库。聚合与统计不持久化；`forecast_eval.analysis` 在事后将其写入
`analysis/`。

### 6.1 模式（当前 = v5）

```sql
-- ⓪ 模式版本表
CREATE TABLE schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- ① 源题副本
CREATE TABLE questions (
    id            TEXT PRIMARY KEY,
    choice_type   TEXT NOT NULL CHECK (choice_type IN ('single','multi')),
    question_type TEXT NOT NULL CHECK (question_type IN ('yes_no','binary_named','multiple_choice')),
    event         TEXT NOT NULL,
    options       TEXT NOT NULL,             -- JSON 数组
    answer        TEXT NOT NULL,             -- 逗号分隔字母
    end_time      TEXT NOT NULL,             -- YYYY-MM-DD
    imported_at   TEXT NOT NULL
);
CREATE INDEX idx_questions_choice_type   ON questions(choice_type);
CREATE INDEX idx_questions_question_type ON questions(question_type);

-- ② prompt-templates 副本
CREATE TABLE prompt_templates (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    imported_at TEXT NOT NULL
);

-- ③ 唯一 (run, model) 元数据；单行
CREATE TABLE run_meta (
    run_id                    TEXT PRIMARY KEY,
    model                     TEXT NOT NULL,
    sampling_n                INTEGER NOT NULL,
    config_snapshot           TEXT NOT NULL,   -- 脱敏后的 .env JSON
    filters_snapshot          TEXT NOT NULL,
    source_db_hash            TEXT NOT NULL,
    metadata_hash             TEXT NOT NULL,
    prompt_templates_hash     TEXT NOT NULL,
    reflection_protocol_text  TEXT,            -- v3+
    reflection_protocol_hash  TEXT,            -- v3+
    belief_protocol_text      TEXT,            -- v4+
    belief_protocol_hash      TEXT,            -- v4+
    training_cutoff           TEXT,            -- κ_M (YYYY-MM-DD)
    started_at                TEXT NOT NULL,
    finished_at               TEXT
);

-- ④ 宽表：每题一行，每个样本一组 s{i}_* 列。
-- 24 字段 × SAMPLING_N 列由 db.init_schema 动态生成。
CREATE TABLE run_results (
    question_id TEXT PRIMARY KEY,
    user_prompt TEXT,                          -- COALESCE；首样本胜出

    -- v2 基础（14 列）
    s0_final_answer_letters TEXT,
    s0_final_answer_raw     TEXT,
    s0_correct              INTEGER,
    s0_parse_ok             INTEGER,
    s0_tool_calls_count     INTEGER,
    s0_react_steps          INTEGER,
    s0_prompt_tokens        INTEGER,
    s0_completion_tokens    INTEGER,
    s0_reasoning_tokens     INTEGER,
    s0_latency_ms           INTEGER,
    s0_messages_trace       TEXT,
    s0_search_calls         TEXT,
    s0_error                TEXT,
    s0_created_at           TEXT,
    -- v3 可观测性（6 列）
    s0_finish_reason        TEXT,
    s0_nudges_used          INTEGER,
    s0_step_metrics         TEXT,
    s0_response_id          TEXT,
    s0_system_fingerprint   TEXT,
    s0_service_tier         TEXT,
    -- v4 信念（3 列）
    s0_belief_final         TEXT,
    s0_belief_trace         TEXT,
    s0_belief_parse_ok      INTEGER,
    -- v5 框架韧性（1 列）
    s0_final_answer_retry_used INTEGER,

    -- ...同样的 s1_* / s2_* / ... 各组...

    FOREIGN KEY (question_id) REFERENCES questions(id)
);
CREATE INDEX idx_run_results_question ON run_results(question_id);
```

模式迁移（db.py:L222–L345）通过 `ALTER TABLE … ADD COLUMN` 完成，
SQLite 将其执行为元数据级操作，因此为 O(1)：

| 版本    | 变更                                                                | 迁移函数                     |
| ------- | ------------------------------------------------------------------- | --------------------------- |
| v2      | 基础 14 列每样本字段；`run_meta` 精简                                | `_init_v2_schema`            |
| v2 → v3 | 增加 6 列每样本可观测性字段及 2 列反思字段                            | `_migrate_v2_to_v3`（L222）  |
| v3 → v4 | 增加 3 列每样本信念字段及 `run_meta` 中的 2 列信念字段                | `_migrate_v3_to_v4`（L269）  |
| v4 → v5 | 增加 1 列每样本字段 `final_answer_retry_used`                         | `_migrate_v4_to_v5`（L312）  |

当 `Settings.BELIEF_PROTOCOL=False` 时，所有信念列写 NULL，且分析
流水线提前退出概率族。续跑路径首次打开旧 DB 时会自动迁移。

每次 `sqlite3.connect` 上执行的连接初始化 PRAGMA：

```sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;
```

### 6.2 每样本字段写入约定

| 字段                              | 来源                                                                                                                                                            |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `s{i}_final_answer_letters`       | `parser.parse_answer(final_raw, q)` 返回的 `frozenset[str]`，写为 `json.dumps(sorted(...))`。                                                                      |
| `s{i}_final_answer_raw`           | LLM 最末一条 assistant 消息的完整 `content` 文本。                                                                                                                |
| `s{i}_correct`                    | `frozenset == frozenset` 的 `int` 转换；当解析失败或样本不在 $`\mathcal{S}`$ 中时为 NULL。                                                                          |
| `s{i}_parse_ok`                   | `final_answer_letters is not None`，等价于有效性标志 $`v_{i,M}`$。                                                                                                  |
| `user_prompt`                     | `prompts.render_user_prompt(q, templates, …)` 的返回值，每题渲染一次并通过 COALESCE 保留。                                                                          |
| `s{i}_messages_trace`             | 完整 `messages` 列表的 JSON，或在 `WRITE_MESSAGES_TRACE=False` 时为 NULL。                                                                                          |
| `s{i}_search_calls`               | 每次调用的元数据：`query`、`end_date`、`n_results`、`published_dates`。启用泄漏过滤后还包括 `n_results_raw / n_results_kept / detector_verdicts / detector_latency_ms / detector_error_kind`。 |
| `s{i}_error`                      | 重试后的错误分类；正常完成（含拒绝或解析失败）为 NULL。                                                                                                            |
| `s{i}_created_at`                 | 写入时刻的 UTC ISO-8601；这是判定"该槽位已被填充"的唯一信号。                                                                                                      |
| `s{i}_finish_reason`              | 末轮的 `ChatCompletion.choices[0].finish_reason`；错误行为 NULL。                                                                                                  |
| `s{i}_nudges_used`                | 该样本内"严格地板未达 → 已注入提醒"的计数，由 `REACT_MAX_NUDGES` 封顶。                                                                                            |
| `s{i}_step_metrics`               | 每轮快照的 JSON 数组：`step / prompt / completion / reasoning / latency_ms / finish_reason / n_tool_calls`。                                                       |
| `s{i}_response_id`                | 末轮的 `ChatCompletion.id`。                                                                                                                                     |
| `s{i}_system_fingerprint`         | 末轮的 `ChatCompletion.system_fingerprint`（厂商提供时）；可用于侦测厂商侧的模型路由变化。                                                                            |
| `s{i}_service_tier`               | 末轮的 `ChatCompletion.service_tier`。                                                                                                                            |
| `s{i}_belief_final`               | v4。末步 `Belief.probabilities` 的 JSON 序列化；解析失败或 `BELIEF_PROTOCOL=False` 时为 NULL。                                                                      |
| `s{i}_belief_trace`               | v4。每个循环步的信念摘要 JSON 数组。                                                                                                                                |
| `s{i}_belief_parse_ok`            | v4。末步信念是否合法解析（0 或 1）；独立于 `parse_ok`。                                                                                                              |
| `s{i}_final_answer_retry_used`    | v5。0 或 1，当 `REACT_FINAL_ANSWER_RETRY` 兜底了空 `final_raw` 时（§5.1）置 1。                                                                                      |

### 6.3 三个独立的协议指纹

三个独立的 SHA-256 前缀描述本次运行 LLM 实际看到了什么，并将本会
冲撞到单一哈希上的多个消融轴解耦（设计依据见 DESIGN.md §5.6）。

* `prompt_templates_hash` 覆盖渲染器 $`R`$，对 §2.3 的八个模板键计算
  哈希。
* `reflection_protocol_hash` 覆盖搜索行为先验，仅对
  `prompts.REFLECTION_PROTOCOL` 文本计算哈希。切换反思或编辑文本会
  改变该哈希。
* `belief_protocol_hash` 覆盖概率族填充器，仅对
  `prompts.BELIEF_PROTOCOL` 文本计算哈希。

这三个哈希既存在于 `run_meta`，也出现在 `manifest.json` 顶层
（evaluation.py:L171–L178），因此"不打开 DB 即可 grep 协议指纹"
覆盖每条协议轴。

### 6.4 续跑查询

续跑契约见 §3.2；实现按样本槽逐一迭代：

```sql
-- 对 i ∈ 0..SAMPLING_N-1：
SELECT question_id FROM run_results
 WHERE s{i}_created_at IS NOT NULL
   AND (s{i}_error IS NULL OR s{i}_error = 'skipped_training_cutoff');
```

结果合并为 `set[(question_id, sample_idx)]`，并从任务队列中移除。
由于每个模型的 DB 仅含一次运行，`run_id` 不进入过滤；`run_meta` 中的
单一行即可裁决。

### 6.5 并发写入策略

每个 DB 连接在启动时执行 §6.1 的四条 PRAGMA。**每个模型只有一个
异步写出器任务**：`runner.run` 为每个模型 DB 打开一个
`db.AsyncWriter`（runner.py:L362），每个 Worker 的结果都入队到该
模型的写出器。写出器每攒满 `DB_COMMIT_BATCH` 条或每秒一次 flush
（`AsyncWriter.FLUSH_INTERVAL_S = 1.0`）；事务短小，且 SQLite 写入
通过 `await asyncio.to_thread(...)` 调用，使事件循环永不阻塞。
单模型 DB 因此始终是一个写者多个读者的格局，这在 WAL 下安全。

`asyncio.Queue` 不跨线程；如需跨线程消费，应改用 `queue.Queue` 或
`janus.Queue`。当前设计保持完全异步、单线程。

---

## 7. 配置

下面是最承重旋钮的精简视图；完整带注释的版本位于 `.env.example`。

### 7.1 `.env` 契约

```ini
# -------- LLM Endpoint（OpenAI 兼容） --------
LLM_API_KEY=REPLACE_ME
LLM_BASE_URL=https://openrouter.ai/api/v1
MODELS=openai/gpt-5,anthropic/claude-sonnet-4.5,google/gemini-2.5-pro,deepseek/deepseek-r1
# 每模型的 κ_M：为每个被评测模型声明
MODEL_TRAINING_CUTOFFS=openai/gpt-5=2024-10-01,anthropic/claude-sonnet-4.5=2025-03-01,...

# LLM 调用参数
LLM_MAX_TOKENS=12000
LLM_TIMEOUT_S=240
LLM_TEMPERATURE=0.7
LLM_TOP_P=1.0
# 推理模型：匹配的 slug 调用时不传 temperature / top_p
LLM_REASONING_MODEL_PATTERNS=o1,o3,o4,r1,qwq

# LLM 并发与重试
LLM_MAX_CONCURRENCY=5
LLM_RETRY_MAX=5
LLM_BACKOFF_NETWORK_S=2,5,15,30,60
LLM_BACKOFF_RATE_LIMIT_S=10,30,60,120,300
LLM_BACKOFF_SERVER_5XX_S=5,15,30,60,120

# -------- Web 搜索总开关 --------
ENABLE_WEB_SEARCH=true

# -------- Tavily 搜索 --------
TAVILY_API_KEY=tvly-REPLACE_ME           # 单值或 CSV（多 key 池）
TAVILY_KEY_COOLDOWN_S=60                 # 单 key 在 429 后的冷却
TAVILY_MAX_RESULTS=5                     # R_tav 轴；多值触发网格
TAVILY_SEARCH_DEPTH=basic                # basic（1 credit）| advanced（2 credits）
TAVILY_INCLUDE_RAW_CONTENT=markdown      # false | markdown（默认）| text
TAVILY_RAW_CONTENT_MAX_CHARS=8000        # 每条结果 raw_content 截断
TAVILY_INCLUDE_ANSWER=false              # 关闭（避免二级 LLM 污染）
TAVILY_END_DATE_OFFSET_DAYS=-1           # δ；项目默认 -1（严格）
SEARCH_MAX_CONCURRENCY=5
SEARCH_RETRY_MAX=3
SEARCH_BACKOFF_S=2,5,15

# -------- ReAct 循环 --------
REACT_MAX_STEPS=12                       # T（每样本 ReAct 步数上限）
REACT_MAX_SEARCH_CALLS=8                 # C 轴；多值触发网格
REACT_REFLECTION_PROTOCOL=true
REACT_BUDGET_AWARENESS_PROTOCOL=true
REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=true
REACT_FORCE_FINAL_ANSWER_LOOKAHEAD=2
REACT_MIN_SEARCH_CALLS=0                 # 软地板；可选启用
REACT_MAX_NUDGES=2

# v5.1 框架韧性
REACT_FINAL_ANSWER_RETRY=false           # 用 tools=[] 重试一次扫尾空 final_raw
REACT_BUDGET_EXCEEDED_DROP_TOOLS=true    # 触顶 C 后丢弃工具 schema

# -------- 搜索泄漏过滤（Stage-2 探测器） --------
ENABLE_SEARCH_LEAK_FILTER=true
LEAK_DETECTOR_API_KEY=REPLACE_ME
LEAK_DETECTOR_BASE_URL=                  # 空 → 回退到 LLM_BASE_URL
LEAK_DETECTOR_MODEL=anthropic/claude-sonnet-4.6
LEAK_DETECTOR_TIMEOUT_S=60
LEAK_DETECTOR_TEMPERATURE=0.0
LEAK_DETECTOR_MAX_TOKENS=512
LEAK_DETECTOR_RETRY_MAX=3
LEAK_DETECTOR_BACKOFF_S=2,5,15
LEAK_DETECTOR_FAIL_ACTION=drop           # drop（默认失败即丢弃）| keep（A/B 应急通道）
LEAK_DETECTOR_CONCURRENCY=5
LEAK_DETECTOR_PROMPT_VERSION=v1

# -------- Composite 评分权重 --------
COMPOSITE_WEIGHTS_QTYPE=yes_no=0.15,binary_named=0.15,multiple_choice=0.70
COMPOSITE_WEIGHTS_CTYPE=single=0.40,multi=0.60
COMPOSITE_WEIGHT_OVERRIDES_QTYPE=
COMPOSITE_WEIGHT_OVERRIDES_CTYPE=

# -------- 采样 --------
SAMPLING_N=5

# -------- 运行 / 续跑 --------
RUN_ID=
RESUME=true

# -------- 数据库 --------
SOURCE_DB=./forecast_eval_set_example.db
SOURCE_TABLE=forecast_eval_set_example
RUNS_ROOT=./runs
DB_COMMIT_BATCH=10
WRITE_MESSAGES_TRACE=true

# -------- 日志 --------
LOG_LEVEL=INFO
LOG_DIR=./logs

# -------- 信念协议（v4 概率族，默认关闭） --------
BELIEF_PROTOCOL=false

# -------- 网格搜索锚点（可选；仅当 R / C 多值时） --------
GRID_DEFAULT_R=
GRID_DEFAULT_C=
```

### 7.2 启动校验（fail-fast）

任何 LLM 或 Tavily 调用之前，`Settings()` 都会强制执行下表检查
（config.py:L577–L851）。任一检查失败都会抛出 `ValueError` 并在
任何 API 调用发出前中止运行。

| 检查                                                               | 位置（行）          | 失败模式                                            |
| ------------------------------------------------------------------ | ------------------- | --------------------------------------------------- |
| `RUN_ID` 非空时匹配 `^\d{8}-\d{6}-[0-9a-f]{4}$`                     | L577–L584           | ValueError                                          |
| `SOURCE_TABLE` 匹配 `^[A-Za-z_][A-Za-z0-9_]*$`                      | L586–L595           | ValueError，附 SQL 注入说明                         |
| `MODELS` 非空；不含 `:online`；不含 `::`                            | L599–L614           | ValueError                                          |
| `LLM_API_KEY` 非空；不含占位符                                       | L617–L622           | ValueError                                          |
| `TAVILY_API_KEY` 在 `ENABLE_WEB_SEARCH=True` 时非空                  | L623–L636           | 每个 key 一个 ValueError                            |
| `LLM_MAX_CONCURRENCY` ≥ 1；`SAMPLING_N` ≥ 1；`REACT_MAX_STEPS` ≥ 1   | L641–L646           | ValueError                                          |
| `REACT_MAX_SEARCH_CALLS` 各项 > 0；`TAVILY_MAX_RESULTS` 各项 > 0      | L455–L460           | 每个 cell 一个 ValueError                           |
| `REACT_MIN_SEARCH_CALLS` ≤ min($`C`$)                                 | L661–L671           | ValueError                                          |
| `REACT_FORCE_FINAL_ANSWER_LOOKAHEAD` ∈ $`[1, T]`$                     | L696–L707           | ValueError                                          |
| `GRID_DEFAULT_R` 设置时须 ∈ `TAVILY_MAX_RESULTS`                     | L711–L715           | ValueError                                          |
| `GRID_DEFAULT_C` 设置时须 ∈ `REACT_MAX_SEARCH_CALLS`                  | L716–L720           | ValueError                                          |
| 启用过滤时 `LEAK_DETECTOR_API_KEY` 与 `_MODEL` 非空                  | L758–L777           | ValueError；探测器 slug 不得含 `:online`             |
| `COMPOSITE_WEIGHTS_*` 桶在已知集合内；权重 ≥ 0；至少一个 > 0        | L781–L851           | ValueError                                          |
| `COMPOSITE_WEIGHT_OVERRIDES_*` 指标名在白名单内                      | L515–L535 + composite.py:L77–L127 | 错拼时 ValueError                       |

`test_config.py` 覆盖约 155 行的边界用例。

### 7.3 敏感字段脱敏

写入 `run_meta.config_snapshot` 之前，
`db.compute_redacted_config_snapshot` 会对每个敏感字段脱敏。脱敏格式
为前 4 字符加长度与 `sha256[:12]`。`TAVILY_API_KEY` 是 `list[str]`，
持久化为 `[{prefix, sha256_12, length, provider}, ...]`，使"本次运行
使用了哪些 key"可被审计。敏感明文绝不持久化。

---

## 8. 错误与可观测性

所有异常都被 `errors.py` 路由，分类为七层并按层选择退避序列。

### 8.1 错误层级

| 层级                              | 识别                                                                           | 处理                                                                                  |
| --------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| **网络 / 超时**                    | `httpx.ConnectError`、`httpx.ReadTimeout`、`asyncio.TimeoutError`、`RemoteProtocolError`、`WriteError`、`WriteTimeout`、`PoolTimeout` | 按 `LLM_BACKOFF_NETWORK_S` 退避；耗尽后 → `error="network"`                          |
| **速率限制（429）**                | HTTP 429                                                                       | 优先遵循 `Retry-After` 头部；否则 `LLM_BACKOFF_RATE_LIMIT_S`                            |
| **服务端 5xx**                     | HTTP 500/502/503/504                                                           | 按 `LLM_BACKOFF_SERVER_5XX_S` 退避；耗尽后 → `error="server_5xx"`                      |
| **鉴权（401/403）**                | HTTP 401/403                                                                   | 立即失败；通过 `AuthError` 中止整个运行                                                 |
| **错误请求（400）**                | HTTP 400 + `model_not_found` 或 `invalid_request`                              | 立即跳过，`error="bad_request"`                                                        |
| **内容策略**                       | HTTP 400，正文匹配 `errors.CONTENT_POLICY_NEEDLES`                              | 不重试；`error="content_policy"`、`parse_ok=0`、`correct=NULL`                          |
| **LLM 软拒绝**                     | 正常返回但找不到 `\boxed{...}`，或解析得到的 `frozenset` 为空                   | 不算错误；`parse_ok=0`、`correct=NULL`                                                  |
| **超过 `REACT_MAX_STEPS`**          | ReAct 循环耗尽仍无最终答案                                                      | 不算错误；除非 `REACT_FINAL_ANSWER_RETRY` 兜底，否则 `parse_ok=0`、`correct=NULL`        |
| **工具参数 JSON 解析失败**          | LLM `arguments` 不是合法 JSON                                                   | 把错误反馈给 LLM 并继续循环（非致命）                                                  |
| **Tavily 自身错误**                 | 通过 `SEARCH_BACKOFF_S` 独立重试；耗尽后作为 `tool_result` 反馈给 LLM            | LLM 可重试该查询或放弃                                                                  |
| **探测器错误（Stage 2）**          | 通过 `LEAK_DETECTOR_BACKOFF_S` 重试；AUTH 错误立即失败即丢弃                    | 在 `LEAK_DETECTOR_FAIL_ACTION=drop`（默认）下 → 丢弃条目；`keep` → 放行                |
| **训练数据污染过滤**                | 在任务计划阶段检测：`q.end_time <= κ_M`（§4.2）                                  | 不调用 LLM；直接写入 `error="skipped_training_cutoff"`                                  |

有六个边界值得强调。(i) AUTH 错误中止整个运行，因为继续在错误的 key
上烧预算毫无意义；`runner._run_task_with_retry` 重新抛出 `AuthError`
（runner.py:L245），外层循环取消所有任务、flush 写出器并退出。
(ii) 内容策略不重试，因为重发同一道题得到的结果一致；报告会统计每个
模型累计了多少次拒绝。(iii) 拒绝不是错误：合法的 LLM 返回未提交
boxed 答案属于模型能力的一部分，被计入统计而非 `error`。
(iv) Tavily 失败降级为 `tool_result` 错误，由 LLM 自行决定重试该查询
或放弃，整个样本不会因此中断。(v) 探测器失败默认失败即丢弃，因为
其抖动与条目内容无关。(vi) `skipped_training_cutoff` 不计入错误率，
因为它是主动数据清洗而非模型失败。

八个 content-policy 检测词为（errors.py:L39–L48）：

```python
CONTENT_POLICY_NEEDLES = (
    "content_policy", "content filter", "content_filter", "safety",
    "content_policy_violation", "data_inspection_failed",
    "inappropriate content", "sensitive",
)
```

该列表覆盖 OpenAI 风格与 Anthropic 风格的英文响应正文，外加 Aliyun
DashScope 的 `data_inspection_failed` 与 `inappropriate content`。
`_body_matches` 执行大小写不敏感的子串匹配，因此所有检测词必须为
小写 ASCII。

### 8.2 错误与解析的耦合规则

下表是 `react.py` 输出与分析分母之间的契约。

| 状态                                       | `parse_ok` | `correct` | 计入 $`\mathcal{S}`$？ | 计入 $`\mathcal{D}^{\mathrm{eval}}`$？      |
| ------------------------------------------ | ---------- | --------- | -------------------- | ----------------------------------------- |
| 截止排除                                    | 0          | NULL      | 否                    | 否（已排除）                               |
| 非截止类调用错误（network/5xx/policy）      | 0          | NULL      | 否                    | 是（计入分母）；否（不计入分子）            |
| 解析失败或软拒绝                             | 0          | 0         | 是                    | 是                                        |
| 严格相等命中                                 | 1          | 1         | 是                    | 是                                        |
| 严格相等未命中                               | 1          | 0         | 是                    | 是                                        |

### 8.3 日志

日志使用 `loguru`，stderr sink 使用 `LOG_LEVEL`（默认 `INFO`），并在
`LOG_DIR` 下设有 DEBUG 级别的滚动文件 sink：

```python
from loguru import logger
import sys, os

logger.remove()
logger.add(
    sys.stderr,
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
)
logger.add(
    f"{LOG_DIR}/{run_id}.log",
    level="DEBUG",
    rotation="100 MB",
    retention=5,
)
```

进度按每个样本完成打印一行：

```text
12:03:44 | INFO    | [run=20260424-120344-a7k3] [5/1610] q=69566c13 qt=binary_named ct=single model=openai/gpt-5 sample=2/5 correct=True steps=4 tool_calls=3 latency=8421ms
```

分母 `[5/1610]` 等于
`len(questions_after_filter) × len(MODELS) × SAMPLING_N` 减去已完成
的续跑任务。错误时该行降级为 `ERROR`，形如
`[x/xx] q=.. model=.. error=rate_limit retry_exhausted`。

---

## 9. 指标（$`\Gamma`$）

指标完全由 `forecast_eval.analysis` 在运行结束后计算，永不入库。
产物落入 `runs/{run_id}/analysis/`。下面每条定义都把一个数学对象
绑定到计算它的函数。

### 9.0 阅读指引

带着具体问题前来的读者可通过下表导航指标栈。表中给出应优先查阅的
小节；§X.Y 引用指向本节的小节。

| 问题                                                                     | 优先读                       | 然后                                  |
| ------------------------------------------------------------------------ | ---------------------------- | ------------------------------------- |
| "整体上谁最准？"                                                          | §9.6 Composite Accuracy      | §9.5 FSS、§9.10 配对自助               |
| "性价比最高的是谁？"                                                      | §9.7 Per-correct cost        | §9.6 Composite Accuracy                |
| "在重复采样下排名稳健吗？"                                                | §9.4 多试验一致性            | §9.10 配对自助、§9.3 pass@k             |
| "模型究竟有多大概率提交可解析答案？"                                      | §9.1 有效性                   | §9.7 Per-correct cost                  |
| "泄漏屏障守住了吗？"                                                      | §4.4 探测器审计              | §4.6 残余面                             |
| "模型把 token 与工具调用花在了哪里？"                                     | §9.11 行为诊断                | §9.12 输出产物                          |

### 9.1 有效性（$`\mathcal{E}^{\mathrm{valid}}`$）

有效性标志 $`v_{i,M} = \mathbb{1}[\Psi_i(o_{i,M}) \ne \bot]`$ 记录模型
原始输出能否产出可解析字母集。由其派生的 DB 列：

| 指标                            | 定义                                                                                    | DB 列来源                      |
| ------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------ |
| `parse_failure_rate`            | 在可评分集 $`\mathcal{S}`$ 上的 $`1 - \mathbb{E}[v_{i,M}]`$                                  | `s{i}_parse_ok = 0`            |
| `final_answer_retry_rate`       | v5.1 兜底扫尾空 `final_raw` 的样本占比                                                  | `s{i}_final_answer_retry_used = 1` |
| `error_rate`                    | 非截止类 `s{i}_error` 的样本占比                                                          | `s{i}_error NOT IN (NULL, 'skipped_training_cutoff')` |
| 各模型 `cutoff_skip_rate`       | 各模型 `count(error='skipped_training_cutoff') / count(*)`                                | `s{i}_error = 'skipped_training_cutoff'` |
| `error_breakdown`（CSV）        | 全样本（含截止）的 `Counter[error]`                                                       | `s{i}_error`                   |
| `finish_reason_breakdown`（CSV）| 在合规样本上的 `Counter[finish_reason]`；可发现异常 `length` 或 `content_filter`           | `s{i}_finish_reason`            |

### 9.2 题项级评分（$`\mathcal{E}^{\mathrm{item}}`$）

每个 `(question_id, model)` 对应 $`n`$ 个样本，$`n`$ 为 `SAMPLING_N`。统计
排除 `s{i}_error="skipped_training_cutoff"` 的行，因为这些是被排除的
题，并非模型答错。

**严格相等**：$`r_{i,M} = \mathbb{1}[\widehat{G}_{i,M} = G_i]`$ 对应
DB 中的 `s{i}_correct`。

**Exam 风格部分得分**是项目头条的每样本评分：

$$
\text{exam-score}(\hat S, G) = \begin{cases}
\dfrac{|\hat S \cap G|}{|G|}, & \hat S \setminus G = \varnothing \\\\
0, & \hat S \setminus G \ne \varnothing
\end{cases}
$$

单选题退化为严格 $`0/1`$。实现位于 `analysis.exam_score.exam_score`
（exam_score.py:L62），决策树如下（exam_score.py:L78–L91）：

```
is_cutoff           → None  （已排除）
error is not None   → None  （已排除）
parse_ok != 1       → 0.0   （解析失败计 0）
FP > 0              → 0.0   （任何假阳一票否决）
otherwise           → |TP| / |G|
```

**Tversky 相似度**用于 FSS：

$$
T(\hat S, G) = \frac{|\hat S \cap G|}{|\hat S \cap G| + \alpha\,|\hat S \setminus G| + \beta\,|G \setminus \hat S|}
$$

项目默认 $`(\alpha, \beta) = (2.0, 0.5)`$，使 FP 惩罚为 FN 的四倍。
实现是 `analysis.accuracy.tversky_score`（accuracy.py:L286）。

**Hamming 评分**仅用于多选，且对漏与错对称：

$$
\text{hamming}(\hat S, G, \mathcal{O}) = 1 - \frac{1}{k}\sum_{\ell\in\mathcal{O}}|\mathbb{1}[\ell\in\hat S] - \mathbb{1}[\ell\in G]|
$$

单选退化为 $`0/1`$。

### 9.3 题级聚合（$`\mathcal{E}^{\mathrm{question}}`$）

| 指标                                | 定义                                                                                                  | 实现                                          |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| `pass_at_1_avg`（$`\text{pass@1}`$）   | 每题内严格命中均值，再跨题等权                                                                         | `accuracy._aggregate`（accuracy.py:L124）      |
| `pass_any_at_n`（$`\text{pass-any@n}`$）| $`\mathbb{1}[\exists s: c_{q,s}=1]`$ 的跨题均值；这是标准 `pass@k`                                       | `accuracy._aggregate`（L134）                  |
| `at_least_all_at_n`（$`\text{pass-all@n}`$）| $`\prod_s c_{q,s}`$ 的跨题均值；是重复一致性下界                                                      | `accuracy._aggregate`（L141）                  |
| `at_least_majority_at_n`            | $`\mathbb{1}[\sum_s c_{q,s} \ge \lceil n/2 \rceil]`$ 的跨题均值                                          | `accuracy._aggregate`                          |
| `majority_vote_accuracy`            | 基于 Counter 的字母集合投票，单一胜出后再与 $`G_q`$ 严格相等                                              | `accuracy._aggregate`（L150–L164）              |
| `exam_score_at_n_avg`               | 在评分索引 $`\mathcal{J}_q^{\mathrm{cnt}}`$ 上两步（题内均值 → 题间均值）                                | `exam_score.exam_score_at_n_avg`（L94–L129）   |
| `cohen_kappa`                       | $`(\text{acc} - p_e)/(1 - p_e)`$，单选 $`p_e = 1/k_q`$，多选每标签 $`0.5`$                                  | `accuracy.cohen_kappa`（L493–L532）             |
| `hamming_score`                     | 各题 Hamming 的跨题均值（仅多选）                                                                       | `accuracy.hamming_score_per_question`（L535–L574） |

### 9.4 多试验一致性（$`n \ge 2`$）

| 指标            | 定义                                                                                                                    | 实现                                          |
| --------------- | ----------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| `fleiss_kappa`  | 在 $`K_q^{\mathrm{eff}}`$ 试验投票矩阵上的 $`(\bar{P} - \bar{P}_e)/(1 - \bar{P}_e)`$；单选按 $`k_q`$ 分层、多选按标签               | `consistency.fleiss_kappa`（L257–L297）        |
| `mean_entropy`  | 每题投票分布的平均 Shannon 熵；多选取每标签二元均值                                                                       | `consistency.prediction_entropy_*`（L305–L399） |
| `vci`           | $`\text{VCI}_q = \max_\ell n_{q,\ell}/K_q^{\mathrm{eff}}`$，跨题均值                                                       | `consistency.mean_vci`（L401–L425）             |
| `mvg`           | $`\text{MV-Acc} - \text{pass@1}`$；正值表示自一致性增益                                                                     | `consistency.mvg`（L427–L450）                  |

### 9.5 Format Skill Score（FSS）

头条的随机修正型技能指标。对题 $`q`$ 的第 $`j`$ 次试验：

$$
\bar{T}_q = \frac{1}{K_q^{\mathrm{eff}}}\sum_{j\in\mathcal{J}_q^{\mathrm{ok}}} T(P_{q,j}, G_q),
\qquad
\text{fss}_q = \frac{\bar{T}_q - T_q^{\mathrm{chance}}}{1 - T_q^{\mathrm{chance}}}
$$

随机基线的闭式表达为：

$$
T_q^{\mathrm{chance}} = \begin{cases}
\dfrac{1}{k_q}, & \text{单选} \\\\[6pt]
2^{-k_q}\sum_{tp=1}^{m_q}\sum_{fp=0}^{k_q-m_q}\binom{m_q}{tp}\binom{k_q-m_q}{fp}\cdot\dfrac{tp}{tp+\alpha\,fp+\beta(m_q-tp)}, & \text{多选}
\end{cases}
$$

数据集级取值为
$`\text{fss} = \frac{1}{|\mathcal{D}^{\mathrm{ok}}|}\sum_q \text{fss}_q`$，
其中 $`\mathcal{D}^{\mathrm{ok}} = \{q : \bar{T}_q \ne \text{None}\}`$。

实现是 `accuracy.fss`（accuracy.py:L386–L479），闭式随机基线由
`accuracy.tversky_baseline`（L316–L350）给出。返回
`{"fss", "n_valid", "mean_pe", "per_question"}`，便于下游按题分解。
由 `test_fss.py`（528 LOC）针对解析基线固定正确性，由
`test_fss_sensitivity.py` 固定 $`(\alpha, \beta)`$ 扫描。

### 9.6 Composite Accuracy（头条）

Composite Accuracy 是模型级别的汇总指标。代入
$`\text{exam}_{avg}^{(b)}`$ 作为每桶取值：

$$
\text{Composite Accuracy}_m = \frac{\sum_{b\in B_{\mathrm{valid}}(m)} w_b \cdot \text{exam}_{avg}^{(b),m}}{\sum_{b\in B_{\mathrm{valid}}(m)} w_b}
$$

其中
$`B_{\mathrm{valid}}(m) = \{b\in B : v_{m,b}\ne\text{None} \wedge w_b > 0\}`$。
缺失桶被剔除，剩余权重重新归一化。若
$`B_{\mathrm{valid}}(m) = \varnothing`$ 则 composite 为 `None`。

默认权重（config.py:L365–L368）：

```text
yes_no          = 0.15
binary_named    = 0.15
multiple_choice = 0.70
```

choice-type 权重：

```text
single = 0.40
multi  = 0.60
```

每指标的覆写通过 `COMPOSITE_WEIGHT_OVERRIDES_QTYPE` 与
`COMPOSITE_WEIGHT_OVERRIDES_CTYPE` 流入，二者均为
`metric=bucket=w,bucket=w;metric=...` 形态的 CSV。错拼的指标名会在
运行时通过已知指标白名单（composite.py:L77–L127）抛错。实现是
`composite.compute_composite`（composite.py:L18），加上
`composite.slice_v5_metrics_by_bucket`（L151–L198）。

### 9.7 单位正确成本

性价比标量将 OpenRouter 账单摊销到难度加权的名义正确数：

$$
C^{\mathrm{per\text{-}correct}}_m = \frac{C^{\mathrm{total}}_m}{|\mathcal{D}^{\mathrm{eval}}| \cdot n \cdot \text{Composite Accuracy}_m}
$$

分母
$`|\mathcal{D}^{\mathrm{eval}}| \cdot n \cdot \text{Composite Accuracy}_m`$
是难度加权后的名义正确样本数：当桶权重恰好与题型经验占比一致时，
等于原始正确数；否则它充当一个区分度感知的参考样本数，把更难的桶
上调权重。

$`C^{\mathrm{total}}_m`$ 直接来自 OpenRouter 账单接口。平台账单是唯一
可被第三方核验的财务事实，因而避免了由"标价 × token 用量"计算导致
的口径偏差，这类偏差通常源自推理 token 计费、prompt cache 折扣、
工具调用计费与厂商路由。

### 9.8 概率族（v4 配套，K=5 下被降级）

`forecast_eval/analysis/proper_score.py` 与 `probabilistic.py` 仅在
`BELIEF_PROTOCOL=True` 时启用。

| 指标                       | 公式                                                                                       | 适用范围         |
| -------------------------- | ------------------------------------------------------------------------------------------ | ---------------- |
| **Brier Index（BI）**      | $`100(1 - \sqrt{\overline{\text{BS}^{\mathrm{lab}}}})`$，先求均值再开方                       | 全部 qtype       |
| **BI_dec**                 | 决策粒度的 Brier index                                                                       | 仅单选           |
| **NLL**                    | 单选：$`-\log p_{q,l^*}`$；多选：每标签 BCE；裁剪 $`\epsilon = 10^{-3}`$                        | 全部 qtype       |
| **MBS**                    | $`100(\log_2 p_{q,l^*} + 1)`$，同样裁剪                                                        | 仅单选           |
| **ABI（crowd / uniform）** | 符号感知 $`100(1 \mp \sqrt{|\overline{\text{ABS}}|})`$，对照 LOO crowd 或均匀基线              | crowd：多模型     |
| **fallback share**         | 经 §9.8.1 兜底的题占比                                                                       | 任意运行         |

> **K=5 免责声明。** 当 `SAMPLING_N` 较小时，经验概率 $`\hat p = n/K`$
> 仅取六个离散值，使可靠性图、Murphy 三分解与 Platt LOO 校准在统计上
> 失去意义。v5 删除了 `calibration.py` 与其五个产物；概率列在
> `per_model_summary.md` 中保留 `†` 脚注。重新引入校准需要将 $`K`$
> 提高至至少 30。

#### 9.8.1 `belief_final IS NULL` 但 `parse_ok = 1` 时的信念兜底

旧的 v3 运行与 v4 信念解析失败仍可借助退化概率向量参与正分评分：

$$
p_l = \begin{cases} 1 - \epsilon, & \ell \in \widehat{G}_{i,M} \\\\ \dfrac{\epsilon}{k - |\widehat{G}_{i,M}|}, & \text{其他} \end{cases},\quad \epsilon = 0.05
$$

该样本以 `belief_parse_ok=0` 记录。完全失败的样本（`parse_ok=0`）
不允许进入概率均值，污染防御位于 flatten.py:L126–L152。

### 9.9 聚合策略（`aggregation.py`）

针对每题 $`K`$ 个样本的概率向量：

| 策略                   | 公式                                                                                       | 用途                                            |
| ---------------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| 算术均值               | $`\hat p_l = (1/K)\sum_k p_{k,l}`$                                                            | Phase 1 默认                                    |
| Logit 空间均值         | 单选：均值对数概率的 softmax；多选：均值 logit 的每标签 sigmoid                              | 贝叶斯模型平均                                   |
| LOO 收缩               | 在 logit 空间向均匀先验混合，扫描 $`\alpha \in \{0, 0.1, ..., 1.0\}`$                          | 自适应平滑（`aggregation.loo_shrinkage`，L145–L199） |

### 9.10 统计推断（`inference.py`）

| 函数                                       | 算法                                                                                       | 输出                                  |
| ------------------------------------------ | ------------------------------------------------------------------------------------------ | ------------------------------------- |
| `paired_bootstrap(bs_a, bs_b)`             | $`B=5000`$ 配对重采样；同一索引同时索引 A 与 B                                                | `delta_mean / ci_low / ci_high / p_two_sided` |
| `holm_bonferroni(p_values)`                | $`(n-i) \cdot p_{(i)}`$ 后取累计极大                                                          | 调整后 p 值                           |
| `difficulty_tertile(gammas)`               | 按题 $`\gamma_q`$ 排序后切三分位                                                               | `low / mid / high` 桶                  |
| `posterior_a_better_than_b(bs_a, bs_b)`    | 配对自助上的 Monte-Carlo $`\Pr(\overline{BS}_A < \overline{BS}_B)`$                          | $`\Pr(\mathrm{BI}_A > \mathrm{BI}_B) \in [0,1]`$ |
| `metric_paired_bootstrap(metric_fn, ...)`  | 任意指标（FSS、Acc、MV-Acc、Fleiss、EBI）上的通用配对自助                                   | `delta_mean / ci_low / ci_high / p_two_sided / cohens_d` |
| `pairwise_paired_bootstrap(...)`           | 跨模型对所有配对应用 `paired_bootstrap`                                                       | `list[ModelPairResult]`               |

多重比较控制采用 Holm-Bonferroni（FWER 级别）。配对自助是**同索引**：
同一次自助重采样将同一道题同时抽入 A 与 B 的数组，从而控制本评测中
通常占据总方差大头的题目级方差。

### 9.11 行为诊断（`behavior.py`）

在 `BELIEF_PROTOCOL=True` 时启用。四组诊断：

| 组别                            | 指标                                                                                       | 输出                              |
| ------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------- |
| 信念演化                        | 每试验波动率 $`V`$、试验间方差 $`\sigma`$、收敛步、证据效率 $`\eta`$、反证参与度                   | `belief_evolution.csv`             |
| 反思 A/B                        | 在匹配 `reflection_protocol_hash` 下 $`\Delta\text{BI}`$、$`\Delta\sigma`$、$`\Delta C`$、$`\Delta\eta`$ 的配对自助 95% CI | `reflection_ab.csv`                |
| 工具用法 PDP                    | 对 `tool_calls_count / react_steps / latency_ms / prompt_tokens / completion_tokens` 做 `Pr(correct \| x)` 与 `E[NLL \| x]` 的 logistic / linear 回归 | `tool_usage_pdp.csv`               |
| 置信度校准                      | 主观 3-bin（low/medium/high）与数值 max-$`p`$ 分箱命中率；冲突标志                           | `confidence_calibration_*.csv`     |

### 9.12 输出产物（`writers.py`）

每次运行的 `analysis/` 目录包含：

| 文件                                            | 模式                                            | 内容                                    |
| ----------------------------------------------- | ----------------------------------------------- | --------------------------------------- |
| `per_model_summary.csv` 与 `.md`                | 24 v3 + 4 FSS + 4 一致性 + 7 概率 = 39 列        | 每模型一行                              |
| `per_model_by_question_type.csv`                | 切片汇总                                         | 按 `question_type` 分桶                 |
| `per_model_by_choice_type.csv`                  | 切片汇总                                         | 按 `choice_type` 分桶                   |
| `per_model_composite_by_question_type.csv`      | composite 权重 + 每桶指标                        | 子类型权重下的 Composite Accuracy        |
| `per_model_composite_by_choice_type.csv`        | composite 权重 + 每桶指标                        | choice 权重下的 Composite Accuracy       |
| `error_breakdown.csv`                           | `Counter[error]`                                 | 全部样本（含截止）                       |
| `finish_reason_breakdown.csv`                   | `Counter[finish_reason]`                         | 仅合规样本                               |
| `paired_delta_bi.csv`                           | `ModelPairResult`                                | 配对自助 delta（BI 单位）               |
| `paired_delta_bi_by_difficulty.csv`             | 每三分位结果                                     | 难度分层配对检验                        |
| `metric_pairwise_bootstrap.csv`                 | 每指标 × 每对结果                                | v5 多指标两两                            |
| `belief_evolution.csv`                          | `BeliefEvolutionRow`                             | 波动率、方差、收敛                       |
| `reflection_ab.csv`                             | `ReflectionABRow`                                | 反思 A/B 配对 CI                         |
| `tool_usage_pdp.csv`                            | `PDPRow`                                         | 特征重要性                              |
| `confidence_calibration_subjective.csv`         | `ConfidenceCalibrationRow`                       | 3-bin 校准                              |
| `confidence_calibration_numeric.csv`            | `NumericConfidenceCalibrationRow`                | 按 max-$`p`$ 分箱                          |
| `entropy_accuracy_bins.csv`                     | 每桶熵 / acc / Fleiss                            | 每三分位诊断                             |
| `overall.json`                                  | 聚合指标 + 元数据                                | 单一 JSON 供下游工具                     |
| `grid_summary.csv`（启用网格时）                | 每 `(real_model, R, C)` 的 17 列主表             | 网格主表                                 |
| `grid_marginal_C.csv`、`grid_marginal_R.csv`     | 沿轴扫描，另一轴锚定                              | 饱和曲线                                 |
| `grid_pareto.csv`                               | 每 cell 一行 + `dominated_by`                    | 帕累托前沿                               |
| `grid_winrate.csv`                              | 每对真实模型 × 跨 (R,C) cell 胜负 + 显著性计数    | 胜率矩阵                                 |

默认舍入精度为四位小数（writers.py:L113–L116）；`avg_react_steps`
取两位小数，`avg_latency_ms` 取一位小数。

---

## 10. 网格搜索

`Settings.TAVILY_MAX_RESULTS`（$`R_{\mathrm{tav}}`$ 轴）与
`REACT_MAX_SEARCH_CALLS`（$`C`$ 轴）接受 CSV 列表。当任一长度大于 1 时，
运行变成 $`R \times C \times M`$ cell 上的笛卡尔网格，每个 cell 通过
*虚拟 slug* 产出自己的 DB 文件：

```text
{real_model}::r{R}::c{C}
```

组装由 `db.compose_virtual_slug(real_model, R, C)`（db.py:L477–L516）
完成；解析由 `db.parse_virtual_slug(slug)` 完成，返回
`(real_model, R, C)` 或对旧式单 cell 运行返回 `None`。`::` 分隔符
有意避开与厂商 slug 的冲撞，配置校验也额外拒绝 `MODELS` 中含 `::`。

`runner._resolve_settings(slug)`（runner.py:L160）读取 slug，通过
`model_copy(update=...)` 克隆 `Settings` 并在该 cell 上覆写 $`R`$ 与
$`C`$，把每个 cell 自己的 settings 视图交回。

`grid_summary.csv` 产物（§9.12）按 cell 输出 `real_model`、$`R`$、$`C`$、
`n_eligible`、`n_total`、`acc_mean`、`acc_ci_lo` / `acc_ci_hi`、
`bi_mean`、`bi_ci_lo` / `bi_ci_hi`、`nll_mean`、`ece`、
`mean_search_calls`、`mean_latency_ms`、`parse_ok_rate`、
`belief_parse_ok_rate`。cell 级自助 CI 由
`grid._bi_ci_from_bs_array` 与 `grid._acc_ci_for_samples`
（grid.py:L122–L142，$`B=5000`$，seed=42）计算。

`GRID_DEFAULT_R` 与 `GRID_DEFAULT_C`（config.py:L319–L322）在网格
多轴时锚定边际切片；未设置时使用 `r_list[0]` 与 `c_list[0]`
（须属于对应列表，见 §7.2）。

由 `test_grid_slug.py`、`test_grid_dispatcher.py`、
`test_grid_analysis.py`、`test_grid_settings_view.py` 与
`test_runner_grid_model.py` 固定。

---

## 11. 测试

仓库附带 33 个测试文件，约 560 个独立用例，全部离线运行。Tavily 与
OpenRouter 以 fixture 或 mock 替身存在，因为单次端到端评测代价昂贵，
而把测试稳住能省下显著的 API 开支。

### 11.1 CI 红线

五个测试一对一映射到 $`\mathcal{R}`$ 中若被破坏即让整个运行单元失效的
组件。它们必须在每次提交时保持绿色：

1. `test_prompts.py` 守护渲染器 $`R`$。
2. `test_parser.py` 守护 $`\Psi`$ 与 $`\phi`$。
3. `test_training_cutoff.py` 守护 $`\kappa_M`$ 可纳入性。
4. `test_llm_no_browsing.py` 守护信息屏障。
5. `test_analysis.py` 守护 $`\Gamma`$。

任何一项失败都意味着运行单元的契约破裂，下游结果不可信。

### 11.2 测试到不变量映射

测试充当 $`\mathcal{R}`$ 各组成"按宣传实现"的证明。

| 组件 / 主张                                                                                | 固定测试                                                                                                                                       |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| $`\mathcal{D}`$：数据集形态、模板契约、哈希确定性                                              | `test_db.py`、`test_evaluation.py`                                                                                                              |
| $`M`$：每模型独立 DB、虚拟 slug、每模型续跑                                                    | `test_runner_grid_model.py`、`test_runner_resume.py`                                                                                            |
| $`\kappa_M`$：可纳入性过滤、截止行写入契约                                                     | `test_training_cutoff.py`                                                                                                                       |
| $`\delta`$：工具层注入、LLM 看不到 `end_date`                                                  | `test_search.py`、`test_react.py`                                                                                                               |
| $`T`$、$`C`$：ReAct 循环有界、预算门控、框架优先级链、v5.1 开关                                  | `test_react.py`（1432 LOC）、`test_react_reflection.py`                                                                                          |
| $`R`$：渲染器在三种 qtype 下正确；协议补充不进入 `prompt_templates_hash`                        | `test_prompts.py`                                                                                                                              |
| $`\Psi`$ 与 $`\phi`$：解析正确性、严格相等、>26 选项往返                                          | `test_parser.py`、`test_parser_belief.py`                                                                                                       |
| $`\Gamma`$：端到端聚合正确性                                                                  | `test_analysis.py`（670 LOC）、`test_aggregation.py`、`test_consistency.py`、`test_inference.py`、`test_proper_score.py`                          |
| $`H_{\mathrm{aux}}`$：探测器白名单、失败即丢弃、AUTH 立即丢弃                                   | `test_leak_filter.py`                                                                                                                           |
| Composite：权重校验、按指标覆写、白名单                                                     | `test_composite_score.py`                                                                                                                       |
| FSS：闭式随机基线、$`(\alpha,\beta)`$ 敏感性                                                  | `test_fss.py`、`test_fss_sensitivity.py`                                                                                                        |
| Exam-score：边角用例（FP-veto、解析失败 = 0、截止 = None）                                   | `test_exam_score.py`                                                                                                                            |
| 行为指标：信念演化、反思 A/B、工具 PDP、置信度校准                                           | `test_behavior.py`                                                                                                                              |
| 网格：虚拟 slug 编码、每 cell settings 视图、分析流水线                                       | `test_grid_slug.py`、`test_grid_dispatcher.py`、`test_grid_analysis.py`、`test_grid_settings_view.py`、`test_plot_analysis_grid.py`             |
| DB schema 迁移：v2→v5 前向路径                                                              | `test_db_v4_migration.py`、`test_db_v5_migration.py`                                                                                            |
| 信息屏障：禁止厂商内置浏览                                                                   | `test_llm_no_browsing.py`                                                                                                                       |
| 错误分层：分类 + 退避查询                                                                   | `test_errors.py`                                                                                                                                |
| 配置：每个校验器、每个环境变量契约                                                           | `test_config.py`                                                                                                                                |
| 端到端：dry-run 用 stub 替换全部传输层                                                       | `test_smoke_dry_run.py`                                                                                                                         |

### 11.3 重型测试文件

实现的复杂度反映在测试文件大小上。最长的几个测试是：

| 文件                           | LOC   | 覆盖内容                                                                       |
| ------------------------------ | ----- | ------------------------------------------------------------------------------ |
| `test_react.py`                | 1432  | 完整 ReAct 循环：每个框架分支、优先级链、收尾、v5.1                             |
| `test_search.py`               |  830  | Tavily 包装、key 轮换、raw_content 截断、审计元数据                              |
| `test_behavior.py`             |  762  | 信念演化、反思 A/B、工具 PDP、置信度校准                                        |
| `test_analysis.py`             |  670  | 分析编排器的 Phase 0–6                                                          |
| `test_db.py`                   |  630  | 模式、迁移、AsyncWriter、哈希、脱敏                                              |
| `test_inference.py`            |  630  | 配对自助、Holm、后验、多指标                                                    |
| `test_grid_analysis.py`        |  605  | 端到端虚拟 slug 网格分析                                                        |
| `test_consistency.py`          |  595  | Fleiss κ 分层、熵、VCI、MVG                                                     |
| `test_leak_filter.py`          |  550  | 白名单、失败即丢弃、审计字段                                                    |
| `test_fss.py`                  |  528  | FSS Tversky、随机基线、边角用例                                                  |
| `test_composite_score.py`      |  509  | Composite 权重、白名单、覆写解析                                                |
| `test_prompts.py`              |  447  | 三种 qtype 渲染规则加协议开关                                                    |
| `test_exam_score.py`           |  426  | exam-score 边角用例（FP-veto、解析失败、截止排除）                                |

运行所有测试：

```bash
pytest tests/ -q
```

---

## 12. 安装、运维与重新实现

### 12.1 从零实现的顺序

如果要从零重新实现，下面这个顺序能保证每一步都本地可验证，并列出
进入下一步前应先通过的测试：

| 步骤 | 模块                            | 固定测试                          |
| ---: | ------------------------------- | --------------------------------- |
|  1   | `environment.yml` + `.env.example` + `.gitignore` | smoke：`python -c 'import forecast_eval'` |
|  2   | `forecast_eval/config.py`       | `test_config.py`                   |
|  3   | `forecast_eval/db.py`           | `test_db.py`、`test_db_v5_migration.py` |
|  4   | `forecast_eval/loader.py`       | 由 `test_db.py` 覆盖                |
|  5   | `forecast_eval/prompts.py`      | `test_prompts.py`                  |
|  6   | `forecast_eval/parser.py`       | `test_parser.py`、`test_parser_belief.py` |
|  7   | `forecast_eval/errors.py`       | `test_errors.py`                   |
|  8   | `forecast_eval/search.py`       | `test_search.py`                   |
|  9   | `forecast_eval/leak_filter.py`  | `test_leak_filter.py`              |
| 10   | `forecast_eval/tools.py`        | 由 `test_search.py` 覆盖           |
| 11   | `forecast_eval/llm.py`          | `test_llm_no_browsing.py`          |
| 12   | `forecast_eval/react.py`        | `test_react.py`、`test_react_reflection.py` |
| 13   | `forecast_eval/runner.py`       | `test_runner_resume.py`、`test_runner_grid_model.py`、`test_training_cutoff.py` |
| 14   | `forecast_eval/analysis/*`      | `test_analysis.py` 加各指标测试    |
| 15   | `evaluation.py`（主入口）        | `test_evaluation.py`、`test_smoke_dry_run.py` |

先用 `--question-type yes_no`、`MODELS=openai/gpt-4o-mini` 与
`SAMPLING_N=1` 跑通 smoke，验证渲染器输出与解析器归一化，然后再放开
到完整评测。

### 12.2 Conda 环境

```yaml
name: forecast
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - pip:
      - openai>=1.50            # OpenAI 兼容 SDK（主 LLM + 探测器）
      - tavily-python>=0.5
      - pydantic>=2.6
      - pydantic-settings>=2.2
      - python-dotenv>=1.0
      - loguru>=0.7
      - httpx>=0.27
      - tenacity>=9.0
      - pytest>=8.0
      - pytest-asyncio>=0.23
      - respx>=0.21
```

创建环境：

```bash
conda env create -f environment.yml
conda activate forecast
cp .env.example .env
# 编辑 .env：LLM_API_KEY、TAVILY_API_KEY、LEAK_DETECTOR_API_KEY、MODELS、MODEL_TRAINING_CUTOFFS
python evaluation.py --question-type yes_no
```

`matplotlib` 有意不放入 `environment.yml`，因为分析流水线刻意保持
轻依赖。仅在本地需要绘制 `scripts/plot_analysis.py` 中的按需绘图族
时才安装它。

### 12.3 命令行

主入口是 `evaluation.py`。三个 flag 控制输入：

```bash
# 跑整套数据集
python evaluation.py

# 按 question_type 过滤（可重复）
python evaluation.py --question-type yes_no --question-type binary_named

# 按 choice_type 过滤（可重复）
python evaluation.py --choice-type single

# 复合过滤（AND）：仅多选的 multiple_choice
python evaluation.py --question-type multiple_choice --choice-type multi

# 运行结束跳过分析（原始 DB 仍写入 db/）
python evaluation.py --skip-analysis

# 独立刷新 analysis/（不修改 DB）
python -m forecast_eval.analysis runs/{run_id}
```

`--question-type` 接受 `yes_no`、`binary_named` 或 `multiple_choice`，
可重复；`--choice-type` 接受 `single` 或 `multi`，可重复。任一 flag
未给出即不限制。其他可调项都在 `.env` 中。

`evaluation.py` 内部分步运行流程：

1. `argparse` 解析三个 flag，组装 `QFilter`。
2. `Settings()` 按 §7.2 加载并校验 `.env`。
3. 生成或复用 `run_id`，并创建 `run_dir = RUNS_ROOT/{run_id}`，下设
   `db/`、`analysis/` 与 `logs/`。
4. 计算四个可复现性哈希（evaluation.py:L46–L75）。
5. 对每个模型（或网格下的虚拟 slug）：
   * 打开 `conn = RUNS_ROOT/{run_id}/db/{safe_slug(model)}.db`，模型
     slug 字母表为 `[A-Za-z0-9._-]`，非法字符被替换。
   * `db.init_schema(conn, SAMPLING_N)` 动态创建 `s{i}_*` 列，并按需
     执行 v2→v5 迁移。
   * 从源 DB 同步 `prompt_templates` 与 `questions`。
   * `db.register_run_meta(conn, run_id, model, hashes, training_cutoff, ...)`。
6. `_write_manifest()` 写入 `manifest.json`（evaluation.py:L123–L192），
   包含 `run_id`、`schema_version`、`analysis_schema`、`sampling_n`、
   `models`、`model_files`、`model_training_cutoffs`、`filters`、
   `hashes`、`reflection_protocol_hash`、`belief_protocol_hash`、
   `grid`、`started_at` 与 `finished_at: null`。
7. `runner.run(...)` 启动 asyncio 事件循环，跑续跑基线、应用
   $`\kappa_M`$ 过滤、在三个信号量下分发任务、并按完成项写日志行。
8. `db.finish_run_meta(conn, run_id)` 与 `_finalise_manifest()` 按
   模型写入 `finished_at`。
9. 除非传入 `--skip-analysis`，否则
   `forecast_eval.analysis.run_analysis(run_dir)` 跑完整指标栈并写入
   §9.12 的产物。

---

## 附录 A. 模块目录

包结构镜像了运行单元的分解：每个模块拥有一个契约，并暴露下列符号。

### A.1 目录布局

```text
Forecast/
├── .env                           # 由用户填写，被 git 忽略
├── .env.example                   # 模板，受 git 管理
├── .gitignore
├── environment.yml                # conda 环境定义
├── README.md                      # 面向用户的入口
├── DESIGN.md                      # 设计依据（这个文件实现该设计）
├── FRAME.md                       # 本文档
├── evaluation.py                  # 主入口：CLI → runner.run → analysis.run_analysis
├── forecast_eval_set_example.db   # 源数据（只读，已纳入 Git）
├── runs/                          # 全部评测输出（被 git 忽略）
│   └── {run_id}/
│       ├── manifest.json
│       ├── db/{model_slug}.db     # 每模型一份 sqlite；自包含可重放
│       ├── analysis/
│       └── logs/{run_id}.log
├── forecast_eval/
│   ├── __init__.py
│   ├── config.py                  # pydantic-settings；Settings + 网格轴 + composite 权重
│   ├── db.py                      # 每模型宽表 schema + AsyncWriter + 哈希
│   ├── loader.py                  # 从 SOURCE_DB 同步 questions + prompt_templates
│   ├── prompts.py                 # 渲染器 R + 反思 / 预算感知 / 信念 / 框架
│   ├── llm.py                     # OpenAI 兼容客户端 + 分层重试；禁止厂商内置浏览
│   ├── search.py                  # Tavily 包装 + end_date 注入 + Stage-2 分发
│   ├── leak_filter.py             # Stage-2 探测器 H_aux（独立客户端、失败即丢弃）
│   ├── tavily_keys.py             # 多 key TavilyKeyPool（最少使用 + 401/403 黑名单 + 429 冷却）
│   ├── tools.py                   # web_search schema（LLM 可见；不含日期）
│   ├── react.py                   # ReAct 循环 F_M（单样本，4 旋钮框架韧性）
│   ├── parser.py                  # 解析器 Ψ + 标签归一化 φ + 信念解析器
│   ├── errors.py                  # 错误分类 + 退避策略
│   ├── runner.py                  # 任务编排 + 多模型写出器 + κ_M 过滤
│   ├── types.py                   # dataclass（Question / SampleResult / SearchResult / 等）
│   └── analysis/                  # 事后统计（Γ）；读 DB → CSV / MD / JSON
│       ├── __init__.py            # run_analysis(run_dir) 编排器
│       ├── accuracy.py            # 严格相等 + pass@k 族 + FSS / Cohen κ / Hamming
│       ├── exam_score.py          # exam 风格部分得分
│       ├── composite.py           # 子类型加权的 composite accuracy
│       ├── consistency.py         # Fleiss κ、平均熵、VCI、MVG（K-trial）
│       ├── proper_score.py        # BI / NLL / MBS / ABI（概率配套）
│       ├── aggregation.py         # 算术 / logit 空间均值 / LOO 收缩
│       ├── inference.py           # 配对自助、Holm-Bonferroni、后验、多指标
│       ├── grid.py                # 网格搜索分析（虚拟 slug 解码、边际 / 帕累托 / 胜率）
│       ├── behavior.py            # 反思 A/B、工具用法 PDP、置信度校准、信念演化
│       ├── probabilistic.py       # 概率族报告构建器
│       ├── flatten.py             # 宽表 → SampleRow + 按题分组
│       └── writers.py             # CSV / MD / JSON 序列化器；列舍入规则
├── scripts/                       # 运维脚本
│   ├── build_forecast_eval_set.py # 数据集构建（分层抽样 + 主题封顶）
│   ├── smoke_leak_filter.py       # Stage-2 探测器流水线 smoke 测试
│   ├── verify_leak_filter_e2e.py  # 端到端泄漏过滤审计复现器
│   ├── fss_sensitivity.py         # FSS α/β 敏感性扫描
│   ├── plot_analysis.py           # analysis/ 的 matplotlib 渲染
│   └── migrate_split_mc_output_format.py  # 一次性 dataset-metadata 迁移
└── tests/                         # 33 个单元 / 集成测试（约 13K LOC），全部离线（§11）
```

### A.2 模块职责

| 模块                    | 实现                                                                                                        | 关键接口                                                                                                       | 固定测试                               |
| ----------------------- | ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| `config.py`             | 通过 pydantic-settings 读 `.env`；类型校验；解析 CSV 列表；执行所有 §7.2 启动检查                            | `Settings` 类（单例）；`_parse_csv`、`_parse_int_list`、`_parse_cutoffs`                                       | `test_config.py`                        |
| `loader.py`             | 同步 `<SOURCE_TABLE>` → `questions`；`dataset_metadata.features_json.prompt_reconstruction` → `prompt_templates` | `sync_questions(source_db, conn, filters, table=...) -> list[Question]`；`sync_prompt_templates(...)`         | 由 `test_db.py` + `test_evaluation.py` 覆盖 |
| `prompts.py`            | 渲染器 $`R`$；反思 / 信念 / 预算感知协议正文；框架状态注入构造器                                                 | `render_user_prompt`、`REFLECTION_PROTOCOL`、`BELIEF_PROTOCOL`、`build_budget_awareness_protocol`、`build_*_warning`、`_build_status_header` | `test_prompts.py`                       |
| `tools.py`              | 定义 `web_search` 的 OpenAI schema；LLM 可见部分不含日期                                                       | `WEB_SEARCH_SCHEMA`、`parse_tool_arguments`、`extract_query`、`tool_error_message`、`tool_result_message`        | `test_search.py`                        |
| `search.py`             | Tavily 包装；注入 `end_date = q.end_time + δ`；截断 raw_content；分发 Stage-2 探测器                          | `tavily_search(query, end_date, settings) -> SearchResult`                                                      | `test_search.py`                        |
| `leak_filter.py`        | 探测器 $`H_{\mathrm{aux}}`$：每条结果 `keep` 或 `drop`；白名单；失败即丢弃                                      | `filter_search_result(result, cutoff_date, settings)`                                                           | `test_leak_filter.py`                   |
| `tavily_keys.py`        | 多 key 池：最少使用 + 401/403 黑名单 + 429 冷却                                                                | `TavilyKeyPool.acquire / report_failure`；`get_pool(keys, cooldown_s)`                                          | 由 `test_search.py` 覆盖                  |
| `llm.py`                | OpenAI 兼容客户端；按错误类型分层重试；拒绝 `:online`、`plugins`、非白名单工具                                  | `chat(model, messages, tools, ...) -> ChatResponse`；`_assert_no_browsing`                                      | `test_llm_no_browsing.py`               |
| `react.py`              | 预测系统 $`F_M`$：4 旋钮框架韧性的 ReAct 循环；逐步信念解析                                                       | `run_react(q, model, sample_idx, settings) -> SampleResult`                                                     | `test_react.py`（1432 LOC）、`test_react_reflection.py` |
| `parser.py`             | 解析器 $`\Psi`$ + 归一化 $`\phi`$：`\boxed{}` 抽取 → 字母 `frozenset[str]`；信念 JSON 校验器                       | `parse_answer(text, q)`、`parse_gt(answer)`、`is_correct(pred, gt)`、`parse_belief(text, q)`                    | `test_parser.py`、`test_parser_belief.py` |
| `errors.py`             | 错误分类 + 退避查询 + AuthError                                                                                 | `ErrorKind`、`classify(exc)`、`should_retry(kind)`、`backoff_seconds(kind, attempt, settings, retry_after)`     | `test_errors.py`                        |
| `db.py`                 | schema + AsyncWriter + 哈希 + 脱敏；v2→v5 迁移；续跑查询；模型 slug 安全化                                   | `init_schema`、`AsyncWriter.enqueue_result`、`load_completed_samples`、`register_run_meta`、`compute_*_hash`、`model_slug_safe` | `test_db.py`、`test_db_v4_migration.py`、`test_db_v5_migration.py` |
| `runner.py`             | 任务编排：笛卡尔去重 → $`\kappa_M`$ 过滤 → asyncio 并发 → 进度日志 → `finish_run_meta`                          | `run(settings, filters, questions, templates, run_id, conns) -> RunStats`；`build_task_plan`                  | `test_runner_resume.py`、`test_runner_grid_model.py`、`test_training_cutoff.py` |
| `analysis/__init__.py`  | 聚合 $`\Gamma`$ 编排器：遍历 DB → 跑指标栈 → 写 CSV/MD/JSON；自动调用或 `python -m forecast_eval.analysis runs/{run_id}` | `run_analysis(run_dir: Path) -> list[Path]`                                                                      | `test_analysis.py`                      |

`QFilter`（types.py:L26–L51）是一个 dataclass，含
`question_types: frozenset[str] | None` 与
`choice_types: frozenset[str] | None`；`None` 表示不过滤。
`apply_sql()` 返回 `(WHERE 子句, params)` 用于 SQLite 参数化执行；
`snapshot()` 返回供 `manifest.filters_snapshot` 使用的 dict。

### A.3 `prompts.render_user_prompt` 参考

```python
def render_user_prompt(
    q: Question,
    templates: dict[str, str],
    reflection_protocol: str | None = None,
    budget_awareness: str | None = None,
    belief_protocol: str | None = None,
) -> str:
    options = json.loads(q.options)

    if q.question_type == "yes_no":
        outcomes_block = ""
        output_format = templates["yes_no_output_format"]

    elif q.question_type == "binary_named":
        outcomes_block = ""
        output_format = (
            templates["binary_named_output_format"]
            .replace("<options[0]>", options[0])
            .replace("<options[1]>", options[1])
        )

    elif q.question_type == "multiple_choice":
        outcomes_block = "\n" + "\n".join(
            f"{index_to_letter(i)}. {label}" for i, label in enumerate(options)
        )
        output_format = (templates["multiple_choice_single_output_format"]
                         if q.choice_type == "single"
                         else templates["multiple_choice_multi_output_format"])

    else:
        raise ValueError(f"unknown question_type: {q.question_type}")

    body = templates["prompt_template"].format(
        agent_role=templates["agent_role"],
        event=q.event,
        end_time=q.end_time,
        outcomes_block=outcomes_block,
        output_format=output_format,
        guidance=templates["guidance"],
    )
    # 协议补充作为运行时槽位存在；templates_hash 不受影响。
    # 顺序：预算感知 → 反思 → 信念（与 react.py 接线一致）。
    for protocol in (budget_awareness, reflection_protocol, belief_protocol):
        if protocol:
            body += "\n\n" + protocol
    return body
```

### A.4 `parser.parse_answer` 参考

```python
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")

def index_to_letter(i: int) -> str:
    if i < 0:
        raise ValueError(f"index must be >= 0, got {i}")
    return chr(ord("A") + i)

def letter_to_index(letter: str) -> int:
    if len(letter) != 1:
        raise ValueError(f"letter must be a single character, got {letter!r}")
    return ord(letter) - ord("A")

def parse_answer(text: str, q: Question) -> frozenset[str] | None:
    matches = BOXED_RE.findall(text or "")
    if not matches:
        return None
    payload = matches[-1].strip()                       # 取最后一个 \boxed{...}

    if q.question_type == "yes_no":
        v = payload.lower()
        if v == "yes": return frozenset({"A"})
        if v == "no":  return frozenset({"B"})
        return None

    if q.question_type == "binary_named":
        opts = json.loads(q.options)
        norm = payload.strip().lower()
        for i, label in enumerate(opts):
            if label.strip().lower() == norm:
                return frozenset({index_to_letter(i)})
        return None

    if q.question_type == "multiple_choice":
        tokens = [t.strip() for t in re.split(r"[,\s]+", payload) if t.strip()]
        opts_n = len(json.loads(q.options))
        letters: set[str] = set()
        for t in tokens:
            if len(t) != 1:
                return None
            idx = letter_to_index(t)
            if not (0 <= idx < opts_n):
                return None
            letters.add(t)
        return frozenset(letters) if letters else None

    return None

def parse_gt(answer: str) -> frozenset[str]:
    return frozenset(t.strip() for t in answer.split(",") if t.strip())

def is_correct(pred: frozenset[str] | None, gt: frozenset[str]) -> bool | None:
    if pred is None:
        return None
    return pred == gt
```

---

## 附录 B. 符号索引

| 符号                       | 含义                                                | 定义于                                |
| -------------------------- | --------------------------------------------------- | ------------------------------------- |
| $`\mathcal{R}`$              | 运行单元                                             | §1.1                                   |
| $`\mathcal{D}`$              | 离散预测数据集                                       | §1.1、§2                                |
| $`\mathcal{D}^{\mathrm{eval}}`$ | $`\kappa_M`$ 过滤后的可评测子集                         | §4.2、§9.1                              |
| $`\mathcal{D}^{\mathrm{pred}}_M`$ | 每模型的可评测子集                                  | §4.2                                   |
| $`\mathcal{S}`$              | 可评分样本集                                          | §8.2、§9.1                              |
| $`M`$                        | 被评测模型 slug                                      | §1.1                                   |
| $`\kappa_M`$                 | $`M`$ 的知识截止                                       | §1.1、§4.2                              |
| $`\delta`$                   | 时间掩码偏移（天）                                    | §1.1、§4.3                              |
| $`\chi_i`$                   | 每题搜索截止 $`\tau_i + \delta`$                        | §4.3                                   |
| $`\tau_i`$                   | 题目解算时间                                          | §2.1                                   |
| $`T`$                        | ReAct 步数上限                                       | §1.1、§5                                |
| $`C`$                        | 每样本搜索调用上限                                    | §1.1                                   |
| $`R`$                        | 输入渲染器                                           | §1.1、§4.7                              |
| $`R_{\mathrm{tav}}`$         | Tavily 单次调用结果数（网格轴）                       | §1.1                                   |
| $`\Psi`$                     | 输出解析器                                           | §1.1、§4.8                              |
| $`\phi`$                     | 字母归一化映射                                        | §1.1、§4.8                              |
| $`\Gamma`$                   | 聚合规则                                             | §1.1、§9                                |
| $`H_{\mathrm{aux}}`$         | Stage-2 泄漏探测器                                    | §1.1、§4.4                              |
| $`\hat{p}_{q,j}`$            | 题 $`q`$ 第 $`j`$ 次试验的信念向量                         | §1.1、§5.3                              |
| $`G_q`$、$`\hat{S}_{q,j}`$     | 真值与预测字母集                                      | §9.2                                   |
| $`k_q`$、$`m_q`$               | 题 $`q`$ 的选项数与真值答案数                            | §9.2、§9.5                              |
| $`\text{exam}_{avg}^{(b)}`$           | 桶 $`b`$ 的 exam-score 均值                              | §9.6                                   |

---

> **一句话。** 本代码库把运行单元
> $`\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)`$
> 与辅助探测器 $`H_{\mathrm{aux}}`$ 实现为：每个符号对应一个 Python 模块、
> 每个观测对应一列 SQLite、每个指标对应一列 CSV、每个不变量对应一个
> 单元测试；报告中出现的每一个数字都可以追溯到宽表中的某一行、
> `run_meta` 中的某个哈希、`search_calls` 中的某条审计裁决，或
> `tests/` 中的某个绿色测试。
