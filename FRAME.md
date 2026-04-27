# Forecast Evaluation — 项目整体框架

## 1. 项目目标

基于 `forecast_eval_set_example.db` 数据集，评测 LLM 在**预测类单选/多选题**上的能力。

核心特色：通过自研 `web_search` Tool 限制 LLM 的信息获取边界——**只允许 LLM 搜索到每道题 `end_time`（事件解决日期）之前的信息**，以此模拟"在题目时间点预测未来"的真实场景，避免信息泄露。

> 重要限制：工具级时间截断只约束**工具搜索**这一条信息通路；模型参数记忆、provider 内置 browsing、搜索结果 snippet/缓存等泄漏源不可能被 Tool 层阻断。完整威胁模型与缓解手段见 §3.8。

- 评测 319 道题（`yes_no` 93 + `binary_named` 11 + `multiple_choice` 215），其中 285 道单选 + 34 道多选
- 通过 OpenRouter 的 OpenAI-compatible API 同时评测多个模型
- LLM 以 ReAct + Tool Use 模式与 `web_search` 工具交互
- 评测结果写入独立的 `results.db`，后续分析独立进行

---

## 2. 数据源

### 2.1 原数据库 `forecast_eval_set_example.db`（只读）

> 注：仓库自带的示例数据集文件名是 `forecast_eval_set_example.db`，主表名是
> `forecast_eval_set_example`。两者均通过 `.env` 的 `SOURCE_DB` / `SOURCE_TABLE`
> 参数可配置；自带数据集时只要保持 7 列 schema 与 `dataset_metadata` 结构一致即可。
> `SOURCE_TABLE` 仅接受 SQLite 合法标识符 `[A-Za-z_][A-Za-z0-9_]*`，启动时校验。

主表 `forecast_eval_set_example`，**319 行 × 7 列**：

| 字段            | 类型    | 说明                                                                                                                          |
| --------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `id`            | TEXT PK | 题目唯一 ID（来自 HuggingFace 源）                                                                                            |
| `choice_type`   | TEXT    | `single` \| `multi`，依据 `answer` 字母个数（1 个 → `single`，>1 → `multi`）                                                  |
| `question_type` | TEXT    | `yes_no` \| `binary_named` \| `multiple_choice`，决定走哪套 prompt 模板                                                       |
| `event`         | TEXT    | 事件描述（**不含**选项、**不含**角色设定、**不含**格式要求）                                                                  |
| `options`       | TEXT    | JSON array of strings。`yes_no`=`["Yes","No"]`；`binary_named`=两个实体名；`multiple_choice`=按 A/B/C... 顺序的标签            |
| `answer`        | TEXT    | 字母编码：单选 `'A'`；多选 `'A, B'`（逗号 + 空格分隔）。字母 ↔ 选项索引规则见 §3.7                                            |
| `end_time`      | TEXT    | 事件解决日期（Asia/Shanghai），`YYYY-MM-DD` 格式                                                                              |

索引（示例数据集自带；自带数据集请按 `idx_<table>_<column>` 命名以保持一致）：
`idx_forecast_eval_set_example_choice_type` / `idx_forecast_eval_set_example_question_type` / `idx_forecast_eval_set_example_end_time`。

辅表 `dataset_metadata`（一行），含 `features_json`，记录所有 prompt 模板、列说明、转换日志。

### 2.2 题量分布

| question_type / choice_type | single | multi | 合计 |
| --------------------------- | -----: | ----: | ---: |
| `yes_no`                    |     93 |     0 |   93 |
| `binary_named`              |     11 |     0 |   11 |
| `multiple_choice`           |    181 |    34 |  215 |
| **合计**                    |  **285** | **34** | **319** |

时间范围：`2026-01-15` ~ `2026-04-14`。
`multiple_choice` 选项数量范围：3 ~ 35（>26 时字母进入 ASCII 续接，详见 §3.7）。

### 2.3 样例

`yes_no`：
```
event:    "2026 a dream year for trump?"
options:  ["Yes","No"]
answer:   "B"           # B = No
end_time: "2026-01-31"
```

`binary_named`：
```
event:    "Golden Knights vs. Kings"
options:  ["Golden Knights","Kings"]
answer:   "A"           # A = Golden Knights
end_time: "2026-01-15"
```

`multiple_choice`（single）：
```
event:    "Bank of Brazil decision in January?"
options:  ["No change in the Selic rate ...",
           "the Bank of Brazil raise ...",
           "the Bank of Brazil lower ..."]
answer:   "A"
end_time: "2026-01-27"
```

`multiple_choice`（multi）：
```
event:    "Oscars 2026: Achievement in Casting Nominations"
options:  [<12 个候选名单条目>]
answer:   "A, B, D, E"
end_time: "2026-01-22"
```

> 关键约定：`event` 字段**不含**选项与格式要求；这些都在调用时由模板拼接（§3.6）。
> `dataset_metadata.features_json.prompt_reconstruction` 已经把所有模板和拼接规则保存好，loader 读出即可使用，**不要硬编码到代码里**。

---

## 3. 核心设计理念

### 3.1 LLM 看不到 `end_date`（最重要的安全边界）

`web_search` Tool 向 LLM 暴露的 schema **只有 `query` 一个参数**：

```python
{
  "type": "function",
  "function": {
    "name": "web_search",
    "description": "Search the web for information relevant to the question.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "Search query"}
      },
      "required": ["query"]
    }
  }
}
```

真正调用 Tavily 时，`end_date` 参数**由 Tool 实现层从当前题目的 `end_time` 硬编码注入**，LLM 无法感知、无法绕过。

```python
def web_search(query: str, question_end_time: str) -> dict:
    end_date = (date.fromisoformat(question_end_time)        # 已是 YYYY-MM-DD
                + timedelta(days=TAVILY_END_DATE_OFFSET_DAYS)).isoformat()
    return tavily_client.search(query=query, end_date=end_date, ...)
```

### 3.2 严格时间截断：`end_date = end_time - 1 day`

`end_time` 已经是 `YYYY-MM-DD` 粒度。为避免"事件当天信息泄露"（很多赛事/新闻在当天就出结果），默认 `TAVILY_END_DATE_OFFSET_DAYS=-1`（推荐的严格默认值；数值越小越保守）：

```
question.end_time = 2026-01-18
→ Tavily end_date = 2026-01-17
```

可在 `.env` 改为 `0`（当天可见，更宽松）或 `-2`、`-3`（更保守）。项目统一以 `-1` 为基准，所有报表默认在 `-1` 下比较。

### 3.3 原数据只读，每次 run 独立成目录、每个模型独立成 DB

`forecast_eval_set_example.db` 不动。每次 `python evaluation.py` 启动都会在 `RUNS_ROOT`
(默认 `./runs`) 下创建一个独立的 `{run_id}/` 子目录，内部结构：

```
{run_id}/
  manifest.json     # run 级元信息 (run_id, sampling_n, models, filters, hashes...)
  db/<model_slug>.db  # 每个参评模型一个 sqlite 文件, 内部自带 questions + prompt_templates 副本
  analysis/         # 跑完后由 forecast_eval.analysis 生成的 CSV / MD / JSON
  logs/{run_id}.log
```

DB 层只存**原始记录**，不做任何聚合/统计。pass@1、pass_any@N、majority 等指标
由后置的 `analysis/` 过程单独计算并写回磁盘（详见 §5 / §11）。

### 3.4 严格匹配评分（字母集合层面）

判分**完全在字母集合层面**进行，与各题型的输出形态无关：

- 数据库 `answer` 字段是字母逗号串（`'A'` / `'A, B'`），split 后变成 `frozenset({'A'})` / `frozenset({'A','B'})`
- LLM 输出的 `\boxed{...}` 经 parser 按 `question_type` 归一为同样的 `frozenset[str]`（详见 §3.7）
- `frozenset == frozenset` 即对，漏选/多选/顺序无关都按"严格相等"算

### 3.5 Parse 失败不是 error

LLM 没输出 `\boxed{...}`、或输出 "I cannot predict the future" 之类的软性拒绝，**不走 retry 路径**，而是记 `parse_ok=0`, `correct=NULL`，单独统计 refusal rate 作为模型能力的一个维度。

### 3.6 提示词拼接（user message assembly）

源数据库**只存原料**（`event` / `options` / `question_type` / `end_time`）。系统每次起 sample 时，从 `dataset_metadata.features_json.prompt_reconstruction` 读模板，按 `question_type` 拼一条完整 user message 喂给 LLM。

模板（`prompt_template`，存于 metadata）：
```
{agent_role} The event to be predicted: "{event} (resolved around {end_time} (GMT+8)).{outcomes_block}"

IMPORTANT: Your final answer MUST end with this exact format:
{output_format}
{guidance}
```

各 slot 渲染规则：

| slot              | 渲染逻辑                                                                                                       |
| ----------------- | -------------------------------------------------------------------------------------------------------------- |
| `agent_role`      | 常量 `"You are an agent that can predict future events."`，原样填入                                            |
| `event`           | `<SOURCE_TABLE>.event` 原文                                                                                    |
| `end_time`        | `<SOURCE_TABLE>.end_time` 原文（`YYYY-MM-DD`）                                                                 |
| `outcomes_block`  | `yes_no` / `binary_named` → **空字符串**（选项已隐含在 `output_format` 中）<br>`multiple_choice` → `"\n" + "A. <options[0]>\nB. <options[1]>\n..."`，字母按 §3.7 索引→字母规则生成 |
| `output_format`   | 三选一（按 `question_type`）：`yes_no_output_format` / `binary_named_output_format` / `multiple_choice_output_format`。**`binary_named` 模板含 `<options[0]>` / `<options[1]>` 占位符，拼接时必须替换为实际两个实体名** |
| `guidance`        | 常量 `"Do not use any other format. Do not refuse to make a prediction. ..."`，原样填入                       |

三种 `output_format` 长什么样：
- `yes_no` —— 要求 `\boxed{Yes}` 或 `\boxed{No}`
- `binary_named` —— 模板含占位符，渲染后形如 `\boxed{Golden Knights} or \boxed{Kings}`
- `multiple_choice` —— 要求 `\boxed{A}` 或 `\boxed{B, C}`，附带 example

`system` / `user` 角色怎么切由 runner 决定（参考 §10 的简化方案：整体作为单条 user message，最忠实模板）。

### 3.7 答案编码与解码（letter ↔ label）

数据库统一用**字母**作为 canonical answer，但 LLM 输出的形态因 `question_type` 而异：

| question_type      | LLM 输出（`\boxed{}` 内）                       | parser 归一目标                                                    |
| ------------------ | ------------------------------------------------ | ------------------------------------------------------------------ |
| `yes_no`           | `Yes` / `No`（大小写不敏感）                    | `frozenset({"A"})` / `frozenset({"B"})` —— `Yes`=A, `No`=B          |
| `binary_named`     | `options` 中的某一个（精确匹配，trim+大小写不敏感）| 在 `options` 列表中查 index → 字母 → frozenset                     |
| `multiple_choice`  | 一个或多个字母，逗号或空格分隔（`A` / `B, C` / `B,C`） | 直接 split → frozenset[str]                                        |

字母 ↔ index 规则（支持 multiple_choice 多达 35 个选项）：
```
index = ord(letter) - ord('A')
A=0, B=1, ..., Z=25
[ =26, \ =27, ] =28, ^ =29, _ =30, ` =31, a =32, b =33, c =34, ...
```

逆向（拼 prompt 时 index → letter）：`letter = chr(ord('A') + index)`。

> ⚠️ **源数据兼容模式警告**：数据库中共 4 道 `multiple_choice` 超过 26 个选项、且其中 3 道真值落在 `[ \ ] ^ _ ` ` ` a b c ...` 这类非字母符号上。这种 ASCII 续接标签对 LLM 非常不友好（反引号、下划线会被 markdown/代码块吞；小写 `a` 与大写 `A` 并存极易混淆）。**我们保留这一方案只为与源数据字母编码保持一一映射**，便于字母集合评分。
>
> 必做防护：
> 1. `prompts.render_user_prompt` 在生成 >26 选项 `outcomes_block` 时，显式在每条前加引号或转义标签（如 `` `[` ``、`` `\` `` 用反引号/引号包裹），避免在 markdown 中渲染丢失
> 2. `parser.parse_answer` 必须对 >26 选项的 `multiple_choice` 做 round-trip 单元测试（label→letter→label）
> 3. 在日志/报表中并行记录 letters 与对应 labels，便于人工复核
>
> 未来如确认 LLM 表现被标签方案拖累，再评估迁移到 `AA/AB` 或 `A01/A02` 的稳定标签方案。

真值反查（`answer` letters → labels，便于显示或日志）：
```python
opts    = json.loads(row["options"])
letters = [t.strip() for t in row["answer"].split(",")]
labels  = [opts[ord(L) - ord('A')] for L in letters]
```

### 3.8 泄漏边界与威胁模型

本项目只能严格控制**工具搜索**这一条信息通路。完整泄漏面与本项目的缓解策略：

| 泄漏源                            | 是否可控 | 缓解手段                                                                                          |
| --------------------------------- | -------- | ------------------------------------------------------------------------------------------------- |
| Tool 搜索内容（Tavily 返回正文）  | ✅ 可控   | `end_date = end_time + TAVILY_END_DATE_OFFSET_DAYS` 由 Tool 实现层注入，LLM 无法感知（§3.1 / §3.2） |
| Provider 内置 browsing / web tool | ✅ 可控   | **强制禁止**：`llm.chat` 只挂 `WEB_SEARCH_SCHEMA`，不开启任何 provider-native browsing / retrieval plugin；OpenRouter 路由时不传 `:online` 后缀、不传 `plugins` 字段 |
| 模型参数记忆（训练数据）          | ⚠️ 部分可控 | 详见 §3.9：按模型训练截止日期过滤早于截止日期的题                                                  |
| 搜索结果 snippet 里的"未来泄漏"   | ⚠️ 部分可控 | Tavily 的 `end_date` 过滤已在 publish date 层面截断；v5.2 起再叠一层 detector LLM 逐条审核内容（`search-leak-filter-v1`），verdict=drop 整条剔除 |
| 题目文字本身的时间线索（如年份）  | ❌ 不可控 | 属于题目固有信息，不干预                                                                          |
| LLM 训练后出现的外部知识回流      | ❌ 不可控 | 接受此偏差                                                                                        |

代码层强约束：
- `llm.chat` 调用中 `tools=[WEB_SEARCH_SCHEMA]` 是唯一允许的 tool schema，**不得**添加任何 provider-native browsing/online 开关
- 若某 provider 强制附加内置工具且无法禁用，在 README 与报表中显式标注"该模型不适用严格评测"

### 3.9 按模型训练截止日期过滤题目

**动机**：如果题目 `end_time` 早于某模型的训练截止日期，模型很可能已经在训练语料里"见过答案"，这类样本无法反映"预测未来"能力，必须从该模型的评测集中剔除。

**机制**：
- `.env` 中配置 `MODEL_TRAINING_CUTOFFS`，为每个模型声明训练截止日期（`YYYY-MM-DD`）
- 任务队列生成阶段，对每个 `(question, model)` 做过滤：
  ```
  cutoff = MODEL_TRAINING_CUTOFFS.get(model)   # None = 未声明, 不过滤
  if cutoff is not None and question.end_time <= cutoff:
      # 跳过该模型下所有 sample_idx
  ```
- 被过滤的 `(question, model, sample_idx)` **仍记一行**到 `run_results`，字段：
  - `error = "skipped_training_cutoff"`
  - `parse_ok = 0`, `correct = NULL`
  - `final_answer_raw = NULL`, `messages_trace = NULL`, `search_calls = NULL`
  - 数值字段置 0
- 目的：报表能清楚展示"每个模型被剔除了多少题、剩多少题可比"，且 resume 时不会重试该行

**Resume 语义细化**（见 §5.3）：
- `error IS NULL` → 正常完成
- `error = "skipped_training_cutoff"` → 主动剔除，**不重试**
- 其他 `error`（`network` / `server_5xx` / `bad_request` / `content_policy`）→ 按 §9 分别处理

用户未声明某模型的 cutoff 时，该模型不做过滤。建议在 `.env` 中对每个参评模型都显式给出 cutoff，以保证公平。

---

## 4. 整体流程

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Evaluation Pipeline                             │
└────────────────────────────────────────────────────────────────────────┘

[.env]  →  [python evaluation.py [--question-type ...] [--choice-type ...]]
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │  1. Load Config (.env) │
                          │  & Init run_id         │
                          └────────────────────────┘
                                      │
                                      ▼
                          ┌──────────────────────────────────┐
                          │  2. Sync Source                  │
                          │  forecast_eval_set_example.db            │
                          │    → results.db.questions        │
                          │    → results.db.prompt_templates │
                          │  (按 filters 过滤)               │
                          └──────────────────────────────────┘
                                      │
                                      ▼
                          ┌────────────────────────┐
                          │  3. Resume Check       │
                          │  已完成 (run_id,       │
                          │  question_id, model,   │
                          │  sample_idx) 跳过      │
                          └────────────────────────┘
                                      │
                                      ▼
                 ┌────────────────────────────────────────┐
                 │  4. Task Queue                         │
                 │  笛卡尔积: questions × models × N      │
                 │  asyncio.Semaphore 控制并发            │
                 └────────────────────────────────────────┘
                                      │
                      ┌───────────────┼───────────────┐
                      ▼               ▼               ▼
                ┌──────────┐    ┌──────────┐    ┌──────────┐
                │ Worker 1 │    │ Worker 2 │    │ Worker N │
                └────┬─────┘    └────┬─────┘    └────┬─────┘
                     │               │               │
                     └───────────────┼───────────────┘
                                     ▼
                      ┌────────────────────────────┐
                      │  ReAct Loop (per sample)   │
                      │                            │
                      │  ┌──────────────────────┐  │
                      │  │ prompts.render(q)    │  │
                      │  │ → user_message       │  │
                      │  └──────────┬───────────┘  │
                      │             ▼              │
                      │  ┌──────────────────────┐  │
                      │  │ LLM.chat(            │  │
                      │  │   model, messages,   │  │
                      │  │   tools=[web_search])│  │
                      │  └──────────┬───────────┘  │
                      │             │              │
                      │  tool_call? ┼── No → break │
                      │             │ Yes          │
                      │             ▼              │
                      │  ┌──────────────────────┐  │
                      │  │ web_search(query,    │  │
                      │  │   end_date = inject  │  │
                      │  │     from q.end_time) │  │
                      │  │ → Tavily API         │  │
                      │  └──────────────────────┘  │
                      │                            │
                      │  loop ≤ REACT_MAX_STEPS    │
                      │  且 ≤ REACT_MAX_SEARCH_CALLS│
                      └──────────┬─────────────────┘
                                 │
                                 ▼
                      ┌────────────────────────┐
                      │  5. Parse \boxed{...}  │
                      │  按 question_type 归一  │
                      │  → frozenset[str]      │
                      └──────────┬─────────────┘
                                 │
                                 ▼
                      ┌────────────────────────┐
                      │  6. Score (frozenset   │
                      │  字母集合严格相等)     │
                      └──────────┬─────────────┘
                                 │
                                 ▼
                      ┌────────────────────────┐
                      │  7. Enqueue → writer   │
                      │  Single writer thread  │
                      │  WAL + batch commit    │
                      └──────────┬─────────────┘
                                 │
                                 ▼
                      ┌────────────────────────┐
                      │  8. Done (results.db)  │
                      │  后续分析独立进行      │
                      └────────────────────────┘
```

---

## 5. 数据库设计 (`runs/{run_id}/db/<model_slug>.db`)

每个 run × model 对应**一个独立的 sqlite 文件**。文件内部自带
`questions` / `prompt_templates` 副本，便于单文件独立复盘。聚合/统计**不落库**，
跑完由 `forecast_eval.analysis` 另写到 `analysis/` 目录。

### 5.1 schema

```sql
-- ⓪ schema 版本表
CREATE TABLE schema_version (
    version    INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

-- ① 源题库副本 (每个 model DB 各存一份, 便于自包含分发)
CREATE TABLE questions (
    id            TEXT PRIMARY KEY,
    choice_type   TEXT NOT NULL CHECK (choice_type IN ('single','multi')),
    question_type TEXT NOT NULL CHECK (question_type IN ('yes_no','binary_named','multiple_choice')),
    event         TEXT NOT NULL,
    options       TEXT NOT NULL,             -- JSON array
    answer        TEXT NOT NULL,             -- 字母逗号串: 'A' / 'A, B'
    end_time      TEXT NOT NULL,             -- YYYY-MM-DD
    imported_at   TEXT NOT NULL
);
CREATE INDEX idx_questions_choice_type   ON questions(choice_type);
CREATE INDEX idx_questions_question_type ON questions(question_type);

-- ② prompt 模板副本
CREATE TABLE prompt_templates (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    imported_at TEXT NOT NULL
);

-- ③ 该 DB 对应的 (run, model) 唯一元信息, 只有一行
CREATE TABLE run_meta (
    run_id                TEXT PRIMARY KEY,
    model                 TEXT NOT NULL,
    sampling_n            INTEGER NOT NULL,
    config_snapshot       TEXT NOT NULL,   -- 脱敏后的 .env JSON
    filters_snapshot      TEXT NOT NULL,   -- {"question_types":..., "choice_types":..., "question_ids":[...], "question_count":N}
    source_db_hash        TEXT NOT NULL,
    metadata_hash         TEXT NOT NULL,
    prompt_templates_hash TEXT NOT NULL,
    reflection_protocol_text TEXT,         -- prompts.REFLECTION_PROTOCOL 全文; REACT_REFLECTION_PROTOCOL=false 时为 NULL
    reflection_protocol_hash TEXT,         -- sha256(reflection_protocol_text)[:16]; 同上, 关时 NULL
    belief_protocol_text   TEXT,           -- v4. prompts.BELIEF_PROTOCOL 全文; BELIEF_PROTOCOL=false 时为 NULL
    belief_protocol_hash   TEXT,           -- v4. sha256(belief_protocol_text)[:16]; 同上, 关时 NULL
    training_cutoff       TEXT,            -- 该模型的 cutoff (YYYY-MM-DD), 未声明时为 NULL
    started_at            TEXT NOT NULL,
    finished_at           TEXT
);

-- ④ 宽表: 每个问题一行, 每个 sample 一组 s{i}_* 列
-- 动态生成 14 × SAMPLING_N 列; 下方仅列示 SAMPLING_N=3 的形状
CREATE TABLE run_results (
    question_id TEXT PRIMARY KEY,
    user_prompt TEXT,                      -- 所有 sample 共用 (COALESCE 写入, 首样本胜出)

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
    -- v3 新增观测列 (schema_version=3): 单步指标 + 终态信封
    s0_finish_reason        TEXT,
    s0_nudges_used          INTEGER,
    s0_step_metrics         TEXT,          -- JSON 数组, 每元素一个 step 快照, 见 §5.2
    s0_response_id          TEXT,          -- ChatCompletion.id (最后一轮)
    s0_system_fingerprint   TEXT,          -- ChatCompletion.system_fingerprint (最后一轮)
    s0_service_tier         TEXT,          -- ChatCompletion.service_tier (最后一轮)
    -- v4 新增观测列 (schema_version=4): belief 协议结构化输出
    s0_belief_final         TEXT,          -- 末步 Belief.probabilities 的 JSON ({letter: float}); 解析失败为 NULL
    s0_belief_trace         TEXT,          -- 每步 belief 摘要 JSON 数组 [{step, p, confidence, delta_reason}|null, ...]
    s0_belief_parse_ok      INTEGER,       -- 末步 belief 是否合法解析 (0/1); 与 parse_ok 独立

    -- ...相同的 s1_* / s2_* 字段组...

    FOREIGN KEY (question_id) REFERENCES questions(id)
);
CREATE INDEX idx_run_results_question ON run_results(question_id);
```

> **schema_version 3 升级说明**：v2 → v3 由 `forecast_eval.db._migrate_v2_to_v3` 通过
> `ALTER TABLE … ADD COLUMN` 完成（`run_results` 加 6 × N 个 `s{i}_*` 列、`run_meta`
> 加 2 列、并 INSERT `(3, utcnow_iso())` 进 `schema_version`）。SQLite 的 ADD COLUMN
> 仅写表元数据，O(1) 完成；老行的新列默认 NULL。续跑路径上首次打开旧 DB 自动迁移。
>
> **schema_version 4 升级说明**：v3 → v4 由 `forecast_eval.db._migrate_v3_to_v4` 通过
> `ALTER TABLE … ADD COLUMN` 完成（`run_results` 加 3 × N 个 `s{i}_*` belief 列、
> `run_meta` 加 2 列、并 INSERT `(4, utcnow_iso())` 进 `schema_version`）。`init_schema`
> 链式调用 v2→v3→v4，幂等。`Settings.BELIEF_PROTOCOL=false` 时所有 belief 列写 NULL，
> 既有 accuracy 指标输出零变化。完整设计见 `ANALYSIS_DESIGN_v4.md`。

连接初始化 PRAGMA（所有 sqlite3 连接都执行一遍）：
```sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;     -- WAL 下足够安全且更快
PRAGMA busy_timeout = 5000;      -- 多 reader 场景下避免 SQLITE_BUSY
```

### 5.2 字段写入约定

| 字段                          | 来源                                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------------------- |
| `s{i}_final_answer_letters`   | `parser.parse_answer(final_raw, q)` 返回的 `frozenset[str]`，写入前 `sorted()` + `json.dumps`           |
| `s{i}_final_answer_raw`       | LLM 最后一条 assistant message 的 `content` 全文                                                        |
| `s{i}_correct`                | `frozenset == frozenset` → `int`；parse 失败或无法判分时 `NULL`                                         |
| `s{i}_parse_ok`               | `final_answer_letters is not None`                                                                      |
| `user_prompt`                 | `prompts.render_user_prompt(q, templates)` 的返回值；每个问题渲染一次，首样本写入后 COALESCE 保留       |
| `s{i}_messages_trace`         | 完整 `messages` 列表 JSON；`WRITE_MESSAGES_TRACE=false` 时 NULL                                         |
| `s{i}_search_calls`           | 每次 `web_search` 调用的元数据 list（query / end_date / n_results / published_dates；启用 leak filter 时再叠 `n_results_raw / n_results_kept / detector_verdicts / detector_latency_ms / detector_error_kind` 五字段，详见 `search-leak-filter-v1`） |
| `s{i}_error`                  | retry 用尽后的错误分类码；正常完成（含 refusal / parse fail）为 NULL                                    |
| `s{i}_created_at`             | 写入时刻的 UTC ISO-8601；作为"该 sample 槽是否被填过"的唯一信号                                         |
| `s{i}_finish_reason`          | 最后一轮 `ChatCompletion.choices[0].finish_reason`（`stop` / `tool_calls` / `length` / `content_filter` …）；error 行（never reached LLM）写 NULL |
| `s{i}_nudges_used`            | 该样本中"strict floor 未达标 → reminder 注入"的次数计数；上限受 `REACT_MAX_NUDGES` 限制；error 行写 0    |
| `s{i}_step_metrics`           | 每个 ReAct 轮的 JSON 数组；元素键 `step / prompt / completion / reasoning / latency_ms / finish_reason / n_tool_calls`，`latency_ms` 为该轮 `llm.chat` 的 `time.monotonic()` 墙时（仅 LLM 调用，不含 search） |
| `s{i}_response_id`            | 最后一轮 `ChatCompletion.id`（provider 唯一 ID，便于追溯 / 申诉）                                        |
| `s{i}_system_fingerprint`     | 最后一轮 `ChatCompletion.system_fingerprint`（provider 提供时；用于检测 provider 端模型路由变更）        |
| `s{i}_service_tier`           | 最后一轮 `ChatCompletion.service_tier`（OpenAI 等返回的实际 tier，例：`default` / `scale` / `flex`）     |
| `s{i}_belief_final`           | v4. 最末步 `parser.parse_belief(content, q)` 返回的 `Belief.probabilities` 序列化为 JSON（`{letter: float}`）；解析失败或 `BELIEF_PROTOCOL=false` 时 NULL |
| `s{i}_belief_trace`           | v4. 整个循环每步的 belief 摘要 JSON 数组，元素键 `step / p / confidence / delta_reason`；中间步解析失败的元素为 `null`；全部步骤都失败时整列 NULL |
| `s{i}_belief_parse_ok`        | v4. 末步 belief 是否合法解析（0/1）；与 `parse_ok` **独立**，belief 失败 MUST NOT 影响 boxed 路径的 `parse_ok` / `correct`；error / cutoff 行写 0 |

> 5 个新增字段（`finish_reason` / `response_id` / `system_fingerprint` / `service_tier`
> / `step_metrics`）只反映**最后一次** `llm.chat` 的封装；中间步骤的 finish_reason
> 进 `step_metrics`，envelope（response_id 等）按 OpenAI ChatCompletion 顶层语义、
> 每轮独立，目前不全部入库以控制宽表列爆炸。
>
> `run_meta.reflection_protocol_text` / `reflection_protocol_hash` 与
> `prompt_templates_hash` **独立分离**：前者只刻 `prompts.REFLECTION_PROTOCOL`
> 的内容指纹（开/关 + 文本变更皆敏感），便于跨 run 区分"反思协议是否启用 / 是否
> 改版"，而不会污染主模板的内容指纹。
>
> v4 新增 `run_meta.belief_protocol_text` / `belief_protocol_hash`：与 reflection 协议
> 字段**完全平行**，刻 `prompts.BELIEF_PROTOCOL` 的内容指纹；同样不污染
> `prompt_templates_hash`、不污染 `reflection_protocol_hash`，三个指纹彼此独立。
> v4 同时把 `belief_protocol_hash` 顶层写入 `manifest.json`（与 `reflection_protocol_hash`
> 同级），让"不开 DB 也能 grep 协议指纹"覆盖两个协议；并加 `analysis_schema: "v4"`
> 顶层字段，让分析模块按需分发概率族指标 / accuracy-only fallback。

### 5.3 断点续跑

对每个 sample slot 独立判定：
```sql
-- 对 i ∈ 0..N-1 各执行一次:
SELECT question_id FROM run_results
 WHERE s{i}_created_at IS NOT NULL
   AND (s{i}_error IS NULL OR s{i}_error = 'skipped_training_cutoff');
```
结果合并成 `set[(question_id, sample_idx)]`，从任务队列中剔除。因为每个模型
自己的 DB 里只有一个 run，`run_id` 不再进入筛选（`run_meta` 单行决定）。

状态分类：
| `error` 值                       | 含义                | 下次续跑是否重试 |
| -------------------------------- | ------------------- | ---------------- |
| `NULL`                           | 已正常完成          | 否               |
| `'skipped_training_cutoff'`      | §3.9 主动剔除       | 否               |
| `'network'` / `'server_5xx'`     | 退避用完仍失败      | 是               |
| `'bad_request'`                  | model_not_found 等  | 是（改配置后续跑） |
| `'content_policy'`               | provider 拒绝       | 可选：默认重试一次并覆盖原行 |

规则：
- 同 `run_id` 重跑 = 续跑，会写入既有的 `runs/{run_id}/db/<slug>.db`
- 换 `run_id` = 全新一跑，会创建新的 `runs/{new_run_id}/` 目录
- 覆盖语义由 `INSERT ... ON CONFLICT(question_id) DO UPDATE SET s{i}_* = excluded.s{i}_*`
  兜底，`user_prompt` 用 `COALESCE` 保留首样本值

### 5.4 并发写入策略

- 每个 DB 连接启动时执行 PRAGMA `journal_mode=WAL / foreign_keys=ON / synchronous=NORMAL / busy_timeout=5000`
- **每个模型一个 async writer task**：runner 为每个模型 DB 各开一个
  `forecast_eval.db.AsyncWriter`，所有 worker 的结果通过该模型对应的 writer 入队
- Writer task 每 `DB_COMMIT_BATCH` 条或 1 秒 flush 一次，短事务；sqlite 写入走
  `await asyncio.to_thread(...)` 避免阻塞 event loop
- 单模型 DB 只有一个 writer、多个 reader，WAL 下并发足够
- 若改为跨线程消费，必须换成 `queue.Queue` / `janus.Queue`；`asyncio.Queue` 不是跨线程安全

---

## 6. 目录结构

```
Forecast/
├── .env                           # gitignored, 用户填
├── .env.example                   # 模板, git 管理
├── .gitignore
├── environment.yml                # conda env 定义
├── README.md
├── FRAME.md                       # 本文档
├── evaluation.py                  # 主入口: parse CLI flags -> runner.run -> analysis.run_analysis
├── forecast_eval_set_example.db           # 原数据, 只读, **纳入 Git 管理** (确保 source_db_hash 可复现)
├── runs/                          # 所有评测输出根目录 (gitignored)
│   └── {run_id}/
│       ├── manifest.json          # run 级元信息 + model_files 映射
│       ├── db/
│       │   └── {model_slug}.db    # 每个模型一个 sqlite; 自带 questions + prompt_templates 副本
│       ├── analysis/              # 跑完后生成的统计产物
│       │   ├── per_model_summary.csv / .md
│       │   ├── per_model_by_question_type.csv
│       │   ├── per_model_by_choice_type.csv
│       │   ├── error_breakdown.csv
│       │   └── overall.json
│       └── logs/{run_id}.log
├── forecast_eval/
│   ├── __init__.py
│   ├── config.py                 # pydantic-settings; RUNS_ROOT + MODEL_TRAINING_CUTOFFS 解析
│   ├── db.py                     # per-model 宽表 schema + AsyncWriter + hash / 脱敏
│   ├── loader.py                 # 从 forecast_eval_set_example.db 同步 questions + prompt_templates 到每个 DB
│   ├── prompts.py                # 按 question_type 渲染 user message
│   ├── llm.py                    # OpenAI-compatible client + retry 分层 (明确禁用 provider-native browsing)
│   ├── search.py                 # Tavily + end_date 注入 + retry
│   ├── tools.py                  # web_search schema (LLM 可见部分, 不含日期)
│   ├── react.py                  # ReAct loop (一个 sample)
│   ├── parser.py                 # \boxed{} 解析 + 字母集合归一 + 严格匹配
│   ├── errors.py                 # 错误分类 + 退避策略 (含 skipped_training_cutoff)
│   ├── runner.py                 # 任务编排 + 多模型 writer + 训练截止过滤
│   └── analysis.py               # 后置统计 (读 DB -> CSV / MD / JSON), 可独立 `python -m` 调用
└── tests/                        # 单元测试 (§17)
    ├── test_prompts.py
    ├── test_parser.py
    ├── test_search.py
    ├── test_db.py
    ├── test_errors.py
    ├── test_llm_no_browsing.py
    ├── test_runner_resume.py
    ├── test_training_cutoff.py
    ├── test_analysis.py
    └── test_smoke_dry_run.py
```

---

## 7. `.env.example` 完整配置

```ini
# =============================================================
#  Forecast Evaluation — 环境变量配置
#  复制为 .env 后填入 API Key 即可运行: python evaluation.py
# =============================================================

# -------- LLM Endpoint (OpenAI-compatible) --------
# LLM_BASE_URL 示例: OpenRouter / 阿里百炼 / OpenAI / DeepSeek / SiliconFlow / 本地 vLLM
# 详见 .env.example 注释
LLM_API_KEY=REPLACE_ME
LLM_BASE_URL=https://openrouter.ai/api/v1

# 要评测的模型列表, 逗号分隔 (笛卡尔积: 每个模型都会跑所有题目 × 所有 sample)
# ⚠️ 不要在 model slug 里追加 ":online", 也不要启用任何 provider-native browsing (参见 §3.8)
MODELS=openai/gpt-5,anthropic/claude-sonnet-4.5,google/gemini-2.5-pro,deepseek/deepseek-r1

# 模型训练截止日期 (§3.9): 题目 end_time <= cutoff 的 (q, model) 将被跳过并标记 skipped_training_cutoff
# 格式: "<model_slug>=YYYY-MM-DD" 多组用逗号分隔. 未声明的模型不过滤
# 建议对每个参评模型都显式声明, 以保证评测公平
MODEL_TRAINING_CUTOFFS=openai/gpt-5=2024-10-01,anthropic/claude-sonnet-4.5=2025-03-01,google/gemini-2.5-pro=2025-01-01,deepseek/deepseek-r1=2024-07-01

# LLM 调用参数 (max_tokens 已给足 reasoning + output 预算)
LLM_MAX_TOKENS=12000
LLM_TIMEOUT_S=240
LLM_TEMPERATURE=0.7
LLM_TOP_P=1.0

# 推理模型 slug 子串列表: 命中后 **不传** temperature / top_p
# (o-series / deepseek-r1 / qwq 等推理模型对自定义采样参数会直接报 400)
LLM_REASONING_MODEL_PATTERNS=o1,o3,o4,r1,qwq

# LLM 并发 & 重试
LLM_MAX_CONCURRENCY=5
LLM_RETRY_MAX=5
# 不同错误类型的退避序列 (秒), 用完仍失败则跳过该 sample 并记 error
LLM_BACKOFF_NETWORK_S=2,5,15,30,60
LLM_BACKOFF_RATE_LIMIT_S=10,30,60,120,300
LLM_BACKOFF_SERVER_5XX_S=5,15,30,60,120

# -------- Tavily Search --------
# 单 key 或 CSV 多 key (`tvly-aaa,tvly-bbb`); 多 key 由 TavilyKeyPool 做 least-used
# 调度 + 401/403 永久拉黑 + 429 临时 cooldown, 详见 .env.example.
TAVILY_API_KEY=tvly-REPLACE_ME
# 单 key 命中 429 时临时拉黑秒数 (默认 60); 401/403 永久拉黑不受此参数影响.
TAVILY_KEY_COOLDOWN_S=60
TAVILY_MAX_RESULTS=5
# search_depth: basic (1 credit/call, 默认) | advanced (2 credits/call)
TAVILY_SEARCH_DEPTH=basic
# include_raw_content: false | markdown | text (旧值 true 兼容映射到 markdown)
# 体积大, 务必配合 TAVILY_RAW_CONTENT_MAX_CHARS 截断
TAVILY_INCLUDE_RAW_CONTENT=markdown
# 单结果 raw_content 截断长度. 0 = 不截断 (谨慎: 单结果可达 40k+ chars)
TAVILY_RAW_CONTENT_MAX_CHARS=8000
# include_answer: false | basic | advanced. 默认 false 避免 Tavily 内部 LLM 速答污染评测纯度
TAVILY_INCLUDE_ANSWER=false
# end_date = question.end_time + offset. 项目默认 -1 (前一天, 避免事件当天信息泄露).
# 数值越小越保守: -2/-3 更严格; 0 = 当天可见 (仅调试用, 不要在正式评测中使用)
TAVILY_END_DATE_OFFSET_DAYS=-1

# Tavily 并发 & 重试 (与 Tavily 配套)
SEARCH_MAX_CONCURRENCY=5
SEARCH_RETRY_MAX=3
SEARCH_BACKOFF_S=2,5,15

# -------- ReAct Loop --------
REACT_MAX_STEPS=12
REACT_MAX_SEARCH_CALLS=8
# 反思协议: 启用后在 user message 末尾附加多步推理脚手架, 显著抬升工具调用次数
# 与思考深度. 不写入 dataset_metadata (prompt_templates_hash 保持不变), 协议
# 文本通过 user_prompt 字段每条 sample 落库, 配置开关由 config_snapshot 记录.
REACT_REFLECTION_PROTOCOL=true
# 软性最低搜索次数 (默认 0=关). >0 时, LLM 试图给最终答案但搜索数 < 该值会被
# 注入一条 user nudge 让其继续检索. 受 REACT_MAX_SEARCH_CALLS 上限约束.
REACT_MIN_SEARCH_CALLS=0
# 单 sample nudge 注入次数上限 (默认 2), 防止 LLM 与系统反复 nudge 死循环.
REACT_MAX_NUDGES=2

# -------- Sampling --------
# 每道题每个模型采样几次 (pass@1 avg / pass_any@N / majority vote 都基于这 N 次)
SAMPLING_N=5

# -------- Run / Resume --------
# 留空则自动生成 YYYYMMDD-HHMMSS-{4位短uuid}. 填相同的 run_id 可断点续跑
RUN_ID=
RESUME=true

# -------- Database --------
SOURCE_DB=./forecast_eval_set_example.db
# 题库表名 (SOURCE_DB 内). 自带数据集请改成你自己的表名; 仅限 [A-Za-z_][A-Za-z0-9_]*.
SOURCE_TABLE=forecast_eval_set_example
# 每次评测会在 RUNS_ROOT 下创建独立 {run_id}/ 目录 (db/, analysis/, logs/)
RUNS_ROOT=./runs
DB_COMMIT_BATCH=10
# false 不存完整 messages trace, 可减小 db 80% 体积
WRITE_MESSAGES_TRACE=true

# -------- Logging --------
LOG_LEVEL=INFO
LOG_DIR=./logs
```

### 7.1 关键参数说明

- **`MODELS`**：逗号分隔，笛卡尔积展开。单跑一个模型就留一个。为空则报错退出。**禁止**在 slug 中拼接 `:online` 或启用 provider 内置 browsing（见 §3.8）。
- **`MODEL_TRAINING_CUTOFFS`**：`model=YYYY-MM-DD` 列表，逗号分隔。`config.py` 解析为 `dict[str, date]`。未声明的模型不过滤。过滤在 runner 任务生成阶段做，跳过的样本写一行 `error="skipped_training_cutoff"` 到 `run_results`。
- **`LLM_MAX_CONCURRENCY` vs `SEARCH_MAX_CONCURRENCY`**：分开控制，因为 Tavily 的 rate limit 通常比 LLM 紧。
- **`LLM_BACKOFF_*` 三条退避序列**：对应不同错误类型（见 §9），序列长度决定最大重试次数。
- **`TAVILY_SEARCH_DEPTH`**：`basic`（默认，1 credit）/ `advanced`（2 credits，召回更高）。一次预测平均 3-5 次搜索，`basic` 控制成本。
- **`TAVILY_INCLUDE_RAW_CONTENT`**：`false` / `markdown`（默认）/ `text`。控制 LLM 看到的页面正文形态。体积大时务必同时设 `TAVILY_RAW_CONTENT_MAX_CHARS`。旧版 `bool` 值仍兼容（`true → markdown`）。
- **`TAVILY_RAW_CONTENT_MAX_CHARS`**：单 result `raw_content` 截断阈值（chars），默认 `8000` ≈ 2k tokens。`0` = 不截断（谨慎：5 条结果总量可超 200k chars，极易塞爆 LLM context）。
- **`TAVILY_INCLUDE_ANSWER`**：`false`（默认）/ `basic` / `advanced`。默认关闭以避免引入 "第二个 LLM 判断" 污染评测纯度（启用后强弱模型间的差异会被压缩）。
- **`TAVILY_END_DATE_OFFSET_DAYS`**：项目默认 `-1`（前一天，推荐的严格默认值）。数值越小越保守；`0` 仅调试用。所有报表默认在 `-1` 下比较。
- **`RUN_ID` 自动生成格式**：`YYYYMMDD-HHMMSS-xxxx`，例如 `20260424-120344-a7k3`，`ls` 天然按时间排序；同时作为 `RUNS_ROOT/{run_id}/` 目录名。
- **`RUNS_ROOT`**：评测产物根目录（默认 `./runs`），每个 run 占一个子目录。
- **`WRITE_MESSAGES_TRACE`**：`true` 存完整 messages JSON（方便 debug 但 db 变大）；`false` 只存关键字段。
- **`REACT_REFLECTION_PROTOCOL`**：`true`（默认）在每条 sample 的 user message 末尾追加多步推理脚手架（拆题 / ≥3 检索角度 / 每次搜索后反思 / 交叉验证 / 反方向自检 / 置信度声明）。协议文本不进 `dataset_metadata`，因此 `prompt_templates_hash` 不受影响，但渲染后的完整 user message 会写入每条 sample 的 `user_prompt` 字段，开关同时由 `run_meta.config_snapshot` 记录，可事后比对开/关协议下的行为差异。
- **`REACT_MIN_SEARCH_CALLS` / `REACT_MAX_NUDGES`**：可选兜底机制。当 LLM 在 `web_search` 调用次数还不足 `REACT_MIN_SEARCH_CALLS` 时就试图给最终答案，系统会向消息序列注入一条 user nudge 提醒它再换角度检索；同一个 sample 最多 nudge `REACT_MAX_NUDGES` 次，整体仍受 `REACT_MAX_STEPS` / `REACT_MAX_SEARCH_CALLS` 硬上限约束。`REACT_MIN_SEARCH_CALLS=0`（默认）等价于关闭兜底，仅靠反思协议驱动；`ENABLE_WEB_SEARCH=false` 时 nudge 自动失效（无搜索可做）。Settings 校验会拒绝 `min > max`。
- **脱敏**：`run_meta.config_snapshot` 写入前 `config.py` 必须对 `LLM_API_KEY` / `TAVILY_API_KEY` 等敏感字段执行 redaction（只保留前 4 位 + 长度 + `sha256[:12]`），敏感明文一律不落库。`TAVILY_API_KEY` 现为 list[str]，每个 key 独立 redact，落盘形如 `[{prefix, sha256_12, length, provider}, ...]`，便于事后审计 "本 run 用了哪几把 key"。

---

## 8. 核心模块职责

| 模块         | 职责                                                                                                            | 关键接口                                                                                                       |
| ------------ | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `config.py`  | `pydantic-settings` 从 `.env` 读取，校验类型，逗号分隔列表解析                                                  | `Settings` 类（单例）                                                                                          |
| `loader.py`  | 从 `SOURCE_DB`（默认 `forecast_eval_set_example.db`）同步两张表到 `results.db`：① `<SOURCE_TABLE>`（默认 `forecast_eval_set_example`）→ `questions`（按 filters 过滤）；② `dataset_metadata.features_json.prompt_reconstruction` → `prompt_templates`（key/value 平铺） | `sync_questions(source_db, conn, filters, table=...) -> list[Question]`, `sync_prompt_templates(source_db, conn) -> dict[str,str]` |
| `prompts.py` | 按 `question_type` 渲染 user message：① 生成 `outcomes_block`（multiple_choice 用 §3.7 字母规则枚举选项）；② 选三套 `output_format` 之一，binary_named 时把 `<options[i]>` 占位符替换为实际实体名；③ 用 `prompt_template` 拼装最终文本 | `render_user_prompt(q: Question, templates: dict[str,str]) -> str`                                             |
| `tools.py`   | 定义 `web_search` OpenAI-schema；**LLM 可见部分不含日期**                                                       | `WEB_SEARCH_SCHEMA`, `execute_tool_call(tc, q, cfg)`                                                           |
| `search.py`  | 封装 Tavily `/search`，注入 `end_date = q.end_time + OFFSET`；按 `TAVILY_INCLUDE_RAW_CONTENT` 决定页面正文形态 + 按 `TAVILY_RAW_CONTENT_MAX_CHARS` 截断；retry | `tavily_search(query, end_date, settings) -> SearchResult`                                                     |
| `llm.py`     | OpenAI-compatible client (OpenRouter)，按错误类型分层 retry；**强制不启用 provider-native browsing**（不传 `plugins`、不加 `:online` 后缀、不发 provider 私有 web tool 字段） | `chat(model, messages, tools, ...) -> ChatResponse`                                                            |
| `react.py`   | 一次 ReAct 推理 = 一个 sample，循环到无 tool_call 或超限                                                        | `run_react(q, model, sample_idx, cfg) -> SampleResult`                                                         |
| `parser.py`  | 按 `question_type` 解析 `\boxed{...}` → 字母 `frozenset[str]`（yes_no: Yes/No→A/B；binary_named: label→letter；mc: split letters）；与 `q.answer` 解析出的字母集合做严格 frozenset 相等判对 | `parse_answer(text: str, q: Question) -> frozenset[str] \| None`, `parse_gt(answer: str) -> frozenset[str]`, `is_correct(pred, gt) -> bool` |
| `errors.py`  | 把 httpx/openai 异常映射到错误分类；给出等待秒数                                                                | `classify(exc) -> ErrorKind`, `backoff_seconds(kind, attempt)`                                                 |
| `db.py`      | 连接管理、WAL + PRAGMA、**per-model 宽表 schema 动态生成**（`init_schema(conn, sampling_n)` 建 `s{i}_*` 列）、`register_run_meta` / `finish_run_meta`、`AsyncWriter` 按 `(question_id, sample_idx)` UPSERT、`load_completed_samples`、source/metadata/templates hash 计算、config 脱敏、model slug 安全化 | `init_schema(conn, sampling_n)`, `AsyncWriter.enqueue_result`, `load_completed_samples`, `register_run_meta`, `upsert_sample_sync`, `model_slug_safe`, `compute_*_hash` |
| `runner.py`  | 任务编排：笛卡尔积 → 去重（per-model completed 集）→ **按 `MODEL_TRAINING_CUTOFFS` 过滤并落 skipped_training_cutoff 行到对应 model DB** → asyncio 并发 → 进度 log → 收尾 `finish_run_meta` | `run(settings, filters, questions, templates, run_id, conns: dict[model, sqlite3.Connection]) -> RunStats`, `build_task_plan(...)` |
| `analysis.py`| 后置统计：扫描 `runs/{run_id}/db/*.db` → 计算 §11 全部指标 → 写 `analysis/` CSV / MD / JSON。**不改 DB**。由 `evaluation.py` 自动调用，或独立 `python -m forecast_eval.analysis runs/{run_id}` 重刷 | `run_analysis(run_dir: Path) -> list[Path]` |

`QFilter` 是 dataclass，包含 `question_types: set[str] | None` 和 `choice_types: set[str] | None`，`None` 表示不过滤。

### 8.1 `prompts.render_user_prompt` 参考实现

```python
def render_user_prompt(q: Question, templates: dict[str, str]) -> str:
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
        lines = [f"{chr(ord('A') + i)}. {label}" for i, label in enumerate(options)]
        outcomes_block = "\n" + "\n".join(lines)
        output_format = templates["multiple_choice_output_format"]

    else:
        raise ValueError(f"unknown question_type: {q.question_type}")

    return templates["prompt_template"].format(
        agent_role=templates["agent_role"],
        event=q.event,
        end_time=q.end_time,
        outcomes_block=outcomes_block,
        output_format=output_format,
        guidance=templates["guidance"],
    )
```

### 8.2 `parser.parse_answer` 参考实现

```python
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")

def _index_to_letter(i: int) -> str:
    return chr(ord("A") + i)

def _letter_to_index(L: str) -> int:
    return ord(L) - ord("A")

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
                return frozenset({_index_to_letter(i)})
        return None

    if q.question_type == "multiple_choice":
        # split on comma or whitespace, drop empties
        tokens = [t.strip() for t in re.split(r"[,\s]+", payload) if t.strip()]
        opts_n = len(json.loads(q.options))
        letters = set()
        for t in tokens:
            if len(t) != 1:
                return None
            idx = _letter_to_index(t)
            if not (0 <= idx < opts_n):
                return None
            letters.add(t)
        return frozenset(letters) if letters else None

    return None

def parse_gt(answer: str) -> frozenset[str]:
    return frozenset(t.strip() for t in answer.split(",") if t.strip())

def is_correct(pred: frozenset[str], gt: frozenset[str]) -> bool:
    return pred == gt
```

---

## 9. 错误分层 & 退避策略

所有异常走下表决策：

| 错误类型                         | 识别方式                                                              | 处理策略                                                                 |
| -------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **Network / Timeout**            | `httpx.ConnectError`, `httpx.ReadTimeout`, `asyncio.TimeoutError`     | 走 `LLM_BACKOFF_NETWORK_S` 序列退避，用完仍失败 → `error="network"` 跳过 |
| **Rate Limit (429)**             | HTTP 429                                                              | 优先读 `Retry-After` header；否则走 `LLM_BACKOFF_RATE_LIMIT_S`           |
| **Server 5xx**                   | HTTP 500/502/503/504                                                  | 走 `LLM_BACKOFF_SERVER_5XX_S`，用完 → `error="server_5xx"` 跳过          |
| **Auth (401/403)**               | HTTP 401/403                                                          | **立即 fail，停止整个 run**（Key 错了继续跑没意义）                      |
| **Bad Request (400)**            | HTTP 400 + `model_not_found` / `invalid_request`                      | 立即跳过，`error="bad_request"`                                          |
| **Content Policy**               | HTTP 400 + `content_policy_violation` / provider 拒绝码               | **不重试**，`error="content_policy"`, `parse_ok=0`, `correct=NULL`       |
| **LLM 软性拒绝**                 | 正常返回但找不到 `\boxed{...}` 或解析后 `frozenset` 为空              | 不是 error，`parse_ok=0`, `correct=NULL`（计入 refusal rate）            |
| **超 `REACT_MAX_STEPS`**         | ReAct 循环耗尽没给最终答案                                            | 不是 error，`parse_ok=0`, `correct=NULL`                                 |
| **Tool arguments JSON 解析失败** | LLM 给的 arguments 不是合法 JSON                                      | 告诉 LLM 报错继续循环（非 fatal）                                        |
| **Tavily 自身错误**              | 独立走 `SEARCH_BACKOFF_S` 重试，用完则把错误作为 tool_result 喂给 LLM | LLM 可以选择重试或放弃                                                   |
| **训练数据污染剔除**             | 任务生成阶段检测 `q.end_time <= MODEL_TRAINING_CUTOFFS[model]`（见 §3.9） | **不调用 LLM**，直接写 `error="skipped_training_cutoff"`, `parse_ok=0`, `correct=NULL`；resume 不重试 |

### 9.1 关键边界

1. **Auth 错误停整个 run**：Key 错了继续烧没意义，早停省钱。
2. **Content policy 不重试**：同一题目再送一次结果一样。直接标记，最后统计每个模型被卡了多少。
3. **Refusal ≠ error**：LLM 返回了合法响应但没答（boxed 缺失 / 字母不在选项范围），是模型能力的一部分，进统计而不进 error 字段。
4. **Tavily 失败降级为 tool_result 错误**：让 LLM 自行决定是否重试 query 或放弃，不中断整个 sample。
5. **`skipped_training_cutoff` 不算 error rate**：这是主动数据清洗，不是模型失败，报表里单独统计"被剔除题数/占比"而不计入 `error rate by kind`。

---

## 10. ReAct Loop 伪代码

```python
async def run_react(q: Question, model: str, sample_idx: int, cfg: Settings) -> SampleResult:
    # ① 注入 end_date: LLM 永远看不到
    end_date = (date.fromisoformat(q.end_time)
                + timedelta(days=cfg.TAVILY_END_DATE_OFFSET_DAYS)).isoformat()

    # ② 拼接 user message: agent_role + event + outcomes_block + output_format + guidance
    #    全部从 prompt_templates 读, 与源数据保持解耦; 反思协议作为 addendum 在
    #    REACT_REFLECTION_PROTOCOL=true 时附加, 仍是单条 user message.
    user_prompt = prompts.render_user_prompt(
        q,
        cfg.PROMPT_TEMPLATES,
        reflection_protocol=prompts.REFLECTION_PROTOCOL if cfg.REACT_REFLECTION_PROTOCOL else None,
    )

    # ③ 整体作为单条 user message (最忠实模板; 不再拆 system/user)
    messages = [{"role": "user", "content": user_prompt}]
    search_calls: list[dict] = []
    final_raw = ""
    t0 = time.monotonic()
    tokens = {"prompt": 0, "completion": 0, "reasoning": 0}
    nudges_used = 0
    step = 0

    for step in range(cfg.REACT_MAX_STEPS):
        resp = await llm.chat(
            model=model,
            messages=messages,
            tools=[WEB_SEARCH_SCHEMA],
            temperature=cfg.LLM_TEMPERATURE,
            top_p=cfg.LLM_TOP_P,
            max_tokens=cfg.LLM_MAX_TOKENS,
            timeout=cfg.LLM_TIMEOUT_S,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_unset=True))
        _accumulate_tokens(tokens, resp.usage)

        # 没 tool_call = LLM 想给最终答案. 软性最低搜索次数兜底:
        # 不够就 nudge 一次让它继续检索 (受 REACT_MAX_NUDGES 与 REACT_MAX_STEPS 共同保护).
        if not msg.tool_calls:
            nudge_enabled = (
                cfg.ENABLE_WEB_SEARCH
                and cfg.REACT_MIN_SEARCH_CALLS > 0
                and cfg.REACT_MAX_NUDGES > 0
            )
            if (
                nudge_enabled
                and len(search_calls) < cfg.REACT_MIN_SEARCH_CALLS
                and nudges_used < cfg.REACT_MAX_NUDGES
                and step < cfg.REACT_MAX_STEPS - 1
            ):
                messages.append({
                    "role": "user",
                    "content": prompts._build_nudge_message(
                        searches_done=len(search_calls),
                        min_required=cfg.REACT_MIN_SEARCH_CALLS,
                    ),
                })
                nudges_used += 1
                continue
            final_raw = msg.content or ""
            break

        # 处理所有 tool_call (OpenAI 支持 parallel)
        for tc in msg.tool_calls:
            if tc.function.name != "web_search":
                messages.append(_tool_error(tc, "unknown tool"))
                continue
            if len(search_calls) >= cfg.REACT_MAX_SEARCH_CALLS:
                messages.append(_tool_error(tc, "search budget exceeded"))
                continue
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                messages.append(_tool_error(tc, f"invalid arguments JSON: {e}"))
                continue

            # 注入 end_date (LLM 看不到)
            result = await search.tavily_search(query=args["query"], end_date=end_date)
            search_calls.append({
                "query": args["query"],
                "end_date": end_date,
                "n_results": len(result.results),
                "published_dates": [r.published_date for r in result.results],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result.to_llm_payload()),
            })
    # 超步数了, final_raw 保持为空 → parser 会标记 parse_ok=0

    # ④ 解析与判分: 全部在字母集合层面
    parsed = parser.parse_answer(final_raw, q)              # frozenset[str] | None
    gt = parser.parse_gt(q.answer)                          # frozenset[str]
    correct = parser.is_correct(parsed, gt) if parsed is not None else None

    return SampleResult(
        run_id=cfg.RUN_ID,
        question_id=q.id,
        model=model,
        sample_idx=sample_idx,
        final_answer_letters=json.dumps(sorted(parsed)) if parsed is not None else None,
        final_answer_raw=final_raw,
        correct=int(correct) if correct is not None else None,
        parse_ok=1 if parsed is not None else 0,
        tool_calls_count=len(search_calls),
        react_steps=step + 1,
        prompt_tokens=tokens["prompt"],
        completion_tokens=tokens["completion"],
        reasoning_tokens=tokens["reasoning"],
        latency_ms=int((time.monotonic() - t0) * 1000),
        user_prompt=user_prompt,
        messages_trace=json.dumps(messages) if cfg.WRITE_MESSAGES_TRACE else None,
        search_calls=json.dumps(search_calls),
        error=None,
        created_at=utcnow_iso(),
    )
```

---

## 11. 评测指标定义

指标**完全由 `forecast_eval.analysis` 在 run 结束后计算**，不存 DB。产物落在
`runs/{run_id}/analysis/` 下（CSV / MD / JSON）。以下定义与源码实现一致。

一个 `(question_id, model)` 下有 N 个 sample（`N = SAMPLING_N`）。统计时**先排除** `s{i}_error="skipped_training_cutoff"` 的行（它们是被剔除的题，不是模型答错）：

| 指标                            | 定义                                                                              | 说明                    |
| ------------------------------- | --------------------------------------------------------------------------------- | ----------------------- |
| **pass@1 avg**                  | `mean(correct over N samples)`                                                    | 反映模型稳定能力        |
| **pass_any@N** (原 `pass@3`)    | `1 if any(correct) across N samples else 0`                                       | best-of-N 潜力（常见的 pass@k 含义） |
| **at_least_k_correct@N**        | `1 if sum(correct) ≥ k else 0`                                                    | 多次一致正确，适合做阈值分析 |
| **majority vote correct**       | N 个 `final_answer_letters`（作为 frozenset）做多数投票，再与 `q.answer` 比对     | self-consistency 指标   |
| **parse failure rate**          | `mean(1 - parse_ok)`                                                              | 反映格式遵循 / 拒答能力 |
| **avg tool_calls**              | `mean(tool_calls_count)`                                                          | 反映搜索使用策略        |
| **avg react_steps**             | `mean(react_steps)`                                                               | 反映推理深度            |
| **avg latency_ms / avg tokens** | 同名字段平均                                                                      | 反映成本                |
| **error rate by kind**          | 按 `error` 分类统计占比（不含 `skipped_training_cutoff`）                         | 反映稳定性              |
| **training_cutoff_skip rate**   | `count(error='skipped_training_cutoff') / count(*)` per model                     | 该模型被剔除多少题      |
| **avg_nudges_used**             | `mean(nudges_used)` over eligible samples（v3 起）                                 | 反映"strict floor 触发率"——值越大说明模型越频繁触发 reminder；为 0 时说明几乎都自发搜索 |
| **finish_reason_breakdown**     | per-model 的 `Counter[finish_reason]`，over eligible samples（v3 起）             | NULL 计入 `<missing>` 桶；用来甄别 `length`（输出截断）/ `content_filter`（被拒）等异常占比 |

> 指标命名变更：原文档里的 `pass@3 = sum(correct)≥3` 与业界通用的 `pass@k` 语义不一致（后者是 "any correct in k"），容易误读。现在明确用 `pass_any@N`（= any）与 `at_least_k_correct@N`（= 阈值）两个独立命名。

报表切片维度：`model × question_type × choice_type`。产出表：

| 文件                              | 内容                                                                              |
| --------------------------------- | --------------------------------------------------------------------------------- |
| `per_model_summary.csv` / `.md`   | 每模型一行，含上表全部指标                                                        |
| `per_model_by_question_type.csv`  | `model × question_type` 切片，同指标集                                            |
| `per_model_by_choice_type.csv`    | `model × choice_type` 切片，同指标集                                              |
| `error_breakdown.csv`             | `model × error_kind` 计数 + 样本占比（含 `<ok>` 与 `skipped_training_cutoff`）    |
| `overall.json`                    | 全部切片的结构化聚合，方便二次处理                                                |

### 11.5 Discrete-native 指标族（v5 主线）+ 概率族（companion）

**v5 重新定向**：本项目并行采样 `K=5`，每个 (question, label) 的经验概率
$\hat{p} = n/K$ 只有 6 个离散值 $\{0, 0.2, 0.4, 0.6, 0.8, 1.0\}$。这把 v4 的
Reliability Diagram / Murphy 三分解 / Platt scaling LOO 推到了"数学正确、
统计无意义"的位置（详见 archived `2026-04-26-probabilistic-analysis-v4`
和 `discrete-native-analysis-v5` 提案）。v5 把分析栈主体改到适合 K=5 的
**discrete-native** 指标族；BS / NLL / MBS / BI / ABI 降级为辅助列。

**v5 一等公民**（`forecast_eval/analysis/accuracy.py` + `consistency.py`）：

| 指标 | 公式 | 解读 |
| --- | --- | --- |
| **FSS** | Tversky α=2 / β=0.5 per-sample → per-question 均值 → chance correction $s_q = (c_q - p_e)/(1 - p_e)$ → 题间均值 | 主指标。多选错代价 = 漏选 4 倍；单选退化 strict 0/1 |
| Tversky baseline | 多选 $p_e$ 精确枚举 $O(m \times (k-m))$；单选 $p_e = 1/k$ | FSS 链条 chance correction 项 |
| Cohen's κ | $(\mathrm{acc} - p_e)/(1 - p_e)$，单选 $p_e = 1/k$ / 多选 $p_e = 0.5$ | strict 0/1 acc 的机会校正 |
| Hamming Score | $1 - \tfrac{1}{k}\sum_l|\hat{y}_l - o_l|$ | multi 题型 partial credit；纯单选 run 返 NULL |
| **Fleiss' κ** | $(\bar{P}-\bar{P}_e)/(1-\bar{P}_e)$ on $K$-trial vote matrix；single 按 letter argmax / multi 每 label 二元 Fleiss 取均 | 多评分者一致性，K-trial 独有 |
| 预测熵 $H_q$ | single：$-\sum_l \hat{p}_l \log_2 \hat{p}_l$；multi：per-label 二元熵均值 | 题级不确定度 |
| **熵-准确率联合** | per-model 三分位数桶 → 每桶 Acc / MV Acc / Fleiss κ | "高熵题模型表现如何 vs 低熵题"——v5 最具学术原创性的诊断维度 |
| VCI | $\max_l n_{q,l}/K$ 题间均值 | 投票集中度 |
| MVG | MV_Acc - Pass@1_Acc | majority vote 信号增益（K-trial 独有） |

**多指标 paired bootstrap**（`forecast_eval/analysis/inference.py`）：

`metric_paired_bootstrap(metric_fn, samples_a_by_q, samples_b_by_q, gt_map, ...)`
参数化 paired bootstrap，对 FSS / Acc / MV_Acc / Fleiss κ / EBI 同时跑 5000
次重采样，输出 95% CI / p-value / Cohen's d。`pairwise_bootstrap.csv` 是
模型对决长表；论文图 Figure 2 的 ΔFSS forest plot 就是它。

v4 BS-paired bootstrap (`paired_bootstrap` / `pairwise_paired_bootstrap`)
保留——`grid.py` 的 per-cell BI CI 与 `paired_delta_bi.csv` / 4 个 v4 产物
依赖它。

**Companion probabilistic family**（保留作辅助列，附 K=5 disclaimer）：

v4 在 accuracy 之外加了一阶 proper scoring rules 与难度调整指标。Phase 0 把
LLM 的隐式概率信号收成结构化字段（`s{i}_belief_final` / `belief_trace` /
`belief_parse_ok`），Phase 1 用这些字段加上 §2.4 fallback 在 `analysis/` 包里
计算 BS / NLL / MBS / BI / ABI 并写进 `per_model_summary.csv` 等。v5 保留这
些列作为 ForecastBench / BLF 论文的对标锚点；markdown 表加 `†` 脚注：
"Probabilistic metrics are computed from empirical vote frequencies over K=5
parallel trials, yielding only 6 discrete probability levels per label.
These values serve as ordinal companions to the primary discrete metrics."

**统一表示（per-option Bernoulli 标签向量）**：题目 $q$ 的真值
$\mathbf{o}_q \in \{0,1\}^{k_q}$ 与预测 $\mathbf{p}_q \in [0,1]^{k_q}$ 都按 letter
顺序排列；single 题型要求 $\sum_l p_l = 1$（容差 $10^{-3}$），multi 题型每个
$p_l$ 是该选项是否属于答案集的独立 Bernoulli 概率。

**一阶指标（`forecast_eval/analysis/proper_score.py`）**：

| 指标 | 公式 | 适用范围 | CSV 列 |
| --- | --- | --- | --- |
| Label-wise Brier | $\mathrm{BS}_q^{\text{lab}} = \tfrac{1}{k_q}\sum_l (p_{q,l}-o_{q,l})^2$ | 所有题型 | `bi`（聚合后） |
| Decision-wise Brier | $\mathrm{BS}_q^{\text{dec}} = \sum_l (p_{q,l}-o_{q,l})^2 = k_q\cdot\mathrm{BS}_q^{\text{lab}}$ | single only | `bi_dec` |
| Brier Index | $\mathrm{BI} = 100(1 - \sqrt{\overline{\mathrm{BS}^{\text{lab}}}})$，**先取均值再开方** | 所有题型 | `bi` |
| NLL | single：$-\log p_{q,l^*}$；multi：label-wise BCE；clip $\epsilon = 10^{-3}$ | 所有题型 | `nll` |
| MBS | $100(\log_2 p_{q,l^*} + 1)$，clip 同 NLL | single only；multi 写 NULL | `mbs` |
| ABI（crowd） | $\mathrm{ABI}^{(m_0)} = $ sign-aware $100(1\mp\sqrt{|\overline{\mathrm{ABS}^{(m_0)}}|})$，$\overline{\mathbf{p}}$ 排除 $m_0$ | 多模型 run | `abi_crowd` |
| ABI（uniform） | 同上，但 baseline 是 $\mathbf{p}=(1/k,\dots,1/k)$ | 所有 run；单模型时 `abi_crowd` 退化等于此列 | `abi_uniform` |
| fallback 占比 | 走 §2.4 fallback 的 question 数 / 该模型可评分 question 数 | 所有 run | `fallback_share` |

**ABI 的符号约定**：$\overline{\mathrm{ABS}} \ge 0$（模型不优于 baseline）→
$100(1 - \sqrt{\overline{\mathrm{ABS}}})$，落在 $[0, 100]$；
$\overline{\mathrm{ABS}} < 0$（模型优于 baseline）→ $100(1 + \sqrt{|\cdot|})$，
高过 100，保持"越好分越高"的单调。

**§2.4 fallback**：当 `s{i}_belief_final IS NULL` 但 `s{i}_parse_ok = 1`
（v3 老 run、或 v4 belief 解析失败但 boxed 解析成功）时，
$p_l = 1-\epsilon$（命中 boxed letter）/ $\epsilon/(k-|\text{boxed}|)$（其他），
$\epsilon = 0.05$。该 sample 走 fallback、`belief_parse_ok=0`。完全失败
（`parse_ok=0`）的 sample MUST NOT 进概率指标均值，避免污染。

**多试聚合**（`forecast_eval/analysis/aggregation.py`）：

Phase 1 的默认是 per (model, question) 取 K 个 sample probability vector 的
算术平均；Phase 2 加入两个论文 §C.9 同款的备选聚合器和一个诊断扫描：

| 函数 | 公式 | 用途 |
| --- | --- | --- |
| `arithmetic_mean(predictions)` | $\hat{\mathbf{p}} = \tfrac{1}{K}\sum_k \mathbf{p}^{(k)}$ | Phase 1 默认 |
| `logit_space_mean(predictions, ctype)` | single：$\mathrm{softmax}(\overline{\log p})$；multi：$\sigma(\overline{\mathrm{logit}\,p})$ | 论文默认；K 一致时与算术平均同 |
| `loo_shrinkage(...)` | 在 $\alpha \in \{0, 0.1, \dots, 1.0\}$ 网格上算 $\mathrm{softmax}(\alpha\overline{\log p})$ 的 BS，返回 $\alpha^*$ + 全曲线 | 诊断 dataset 是否需要朝 prior 收缩 |
| `majority_vote_accuracy_v4(...)` | logit-space mean 后 argmax；K 浮点 logit 几乎不可能 tie | 把 v3 majority\_vote 的 ~10% tie-unresolved 一次性回收 |

`majority_vote_accuracy_v4` 是 v3 letter-set vote 的升级版；当前 wired 为 unit-
testable function，未替换 v3 的 `majority_vote_accuracy` 列以避免破坏 byte
regression。`loo_shrinkage` 的 $\alpha$ 网格 BI 落到 `analysis/shrinkage_alpha_curve.csv`。

**分层校准（v5 已删除，因 K=5 离散分辨率约束）**：

v4 的 `calibration.py` 实现了 per-(question_type, choice_type) cell Platt /
Temperature scaling LOO + ECE / Murphy 三分解 / Reliability bins。在 K=5 工作
点上：

* Platt scaling 在 6 个独特概率值上拟 sigmoid 是教科书过拟合；
* Temperature scaling 单参数勉强稳定但"温度"在 6 级离散上语义存疑；
* ECE 用 15 bins 中 9+ bin 永远空，weighted-average 高方差不可比；
* Murphy 三分解 CAL/RES 项被空 bin 把方差吞掉。

v5 整文件删除 `calibration.py`，停产 `calibration_params.json` /
`per_model_summary_calibrated.csv` / `reliability_data*.json` /
`brier_decomposition.csv` 共 5 个产物。`per_model_summary.md` 删除
`BI_cal / NLL_cal / ECE_uncal / ECE_cal` 列与 `cal*` 哨兵。如未来 K 增加
到 ≥30，可在新 change 中重新引入校准。

**统计推断**（`forecast_eval/analysis/inference.py`）：

| 函数 | 算法 | 输出 |
| --- | --- | --- |
| `paired_bootstrap(bs_a, bs_b)` | $B=5000$ 配对重抽（同索引同时索引 A 和 B） | `delta_mean / ci_low / ci_high / p_two_sided` |
| `holm_bonferroni(p_values)` | $(n - i) \cdot p_{(i)}$ 后累积 max | adjusted p-values，原顺序返回 |
| `difficulty_tertile(gammas)` | per-question $\gamma_q$ 排序后切 tertile | `low / mid / high` 分桶 |
| `paired_bootstrap_by_difficulty(...)` | 每 tier 独立 paired bootstrap | `{tier: PairedBootstrapResult}` |
| `posterior_a_better_than_b(bs_a, bs_b)` | 在 paired bootstrap 上 Monte-Carlo $\Pr(\overline{BS}_A < \overline{BS}_B)$ | $\Pr(\mathrm{BI}_A > \mathrm{BI}_B) \in [0, 1]$ |
| `posterior_normal_fit(...)` | normal 闭式 $\Phi(-\bar\Delta / SE)$ | 同上的 sanity-check 通道 |

paired bootstrap 是同索引版本——同一次 bootstrap 抽到的 question id 同时索引
A 和 B 的 BS 数组——这控制 question-level 方差（论文 §G.2 量化为总方差的
62%）。多对比较通过 Holm-Bonferroni 控制 FWER。

**v5 产物清单**：

| 文件 | 内容 | 状态 |
| --- | --- | --- |
| `per_model_summary.csv` | v3 + v5 discrete (FSS / Cohen κ / Hamming / Fleiss κ / 均熵 / VCI / MVG) + v4 概率族 (companion) | v5 修订列序 |
| `per_model_summary.md` | 同上 markdown，v5 列在主区，v4 概率列加 `†` 脚注 | v5 修订 |
| `inter_trial_consistency.csv` | per-model Fleiss κ / mean entropy / VCI / MVG | **v5 新增** |
| `entropy_accuracy_bins.csv` | per-model × tertile (Acc / MV Acc / Fleiss κ)；per-model 桶边界不一致 | **v5 新增** |
| `pairwise_bootstrap.csv` | 多指标 paired bootstrap：FSS / Acc / MV_Acc / Fleiss κ / EBI × pairs × ΔMean / 95% CI / p / Cohen's d | **v5 新增** |
| `shrinkage_alpha_curve.csv` | per-(model, ctype) $\alpha$ 网格 mean BS / BI | v4 保留 |
| `paired_delta_bi.csv` | BS-paired 模型对决 ΔBS + 95% CI + Holm + posterior | v4 保留（grid.py 依赖） |
| `pairwise_significance.csv` | $\alpha = 0.05$ 显著性标记（raw + Holm） | v4 保留 |
| `posterior_pairwise.csv` | $\Pr(\mathrm{BI}_A > \mathrm{BI}_B)$ | v4 保留 |
| `per_model_by_difficulty.csv` | 按 difficulty tertile 分层的 BI / NLL / ABI | v4 保留 |
| `paired_delta_bi_by_difficulty.csv` | 每个 tier 独立 paired bootstrap | v4 保留 |
| ~~`calibration_params.json`~~ | ~~per-cell Platt / temperature~~ | **v5 删除** |
| ~~`per_model_summary_calibrated.csv`~~ | ~~校准后指标~~ | **v5 删除** |
| ~~`reliability_data.json` / `_calibrated.json`~~ | ~~per-(model, qtype) bins~~ | **v5 删除** |
| ~~`brier_decomposition.csv`~~ | ~~Murphy 三分解~~ | **v5 删除** |

`error_breakdown.csv` / `finish_reason_breakdown.csv` 的 byte-regression
保护从 Phase 1 延续——Phase 2 不改这两个文件。

**行为分析**（`forecast_eval/analysis/behavior.py`）：

Phase 3 把 `belief_trace` JSON 时间序列变成 5 个一等公民指标，并加上反思
协议 A/B、tool-usage PDP、confidence joint diagnosis 三组诊断：

| 指标 | 公式 | 解读 |
| --- | --- | --- |
| Trial-internal volatility | $V_{q,k} = \tfrac{1}{T-1}\sum_t \|b_t-b_{t-1}\|_2$ | 该 trial 内信念变化总幅度 |
| Inter-trial variance | $\sigma_q = \mathrm{std}_k\,b^{(q,k)}_T$ | 论文 §4 Figure 2 同款 |
| Convergence step | $C_{q,k} = \min\{t : \|b_T-b_t\|_2<0.05\}$ | 多少步达到最终信念 |
| Evidence efficiency | $\eta_{q,k} = (\mathrm{NLL}(b_0) - \mathrm{NLL}(b_T))/\max(1, \text{search\_calls})$ | 每次搜索带来的信息增益 |
| Counterevidence engagement | 至少一条 counterevidence 字符串中出现非最终选 letter（letter 匹配，无 NLP） | 是否做了反方向自检 |

反思协议 A/B（`find_paired_runs` + `reflection_ab_report`）扫描全部 run，
按"`reflection_protocol_hash` 不同、其他全部 hash 相同"配对；配对算 ΔBI /
Δσ / ΔC / Δη 的 paired bootstrap 95% CI，按 question_type 分层报告。指纹
不一致 MUST NOT 配对——这是 spec 26.5 的硬约束。

Tool-usage PDP（`tool_usage_pdp`）用纯 Python IRLS 拟合
$\Pr(\text{correct}\mid\mathbf{x})$（logistic）与 $\mathbb{E}[\mathrm{NLL}\mid\mathbf{x}]$
（ridge linear）在 5 个特征上的关系——
`tool_calls_count / react_steps / latency_ms / prompt_tokens / completion_tokens`，
对每特征做 quantile 网格 partial dependence。L2 正则 + 步长裁剪保证 IRLS
稳定（Phase 2 在饱和 sigmoid 上学到的教训）。

Confidence-calibration 联合诊断（`confidence_calibration` /
`numeric_confidence_calibration`）：把 `belief_trace` 最末步的
`confidence ∈ {low, medium, high}` 当主观置信，把 `max_l p_l` 当数值置信，
分别和命中率对照。`confidence_conflict_models` 哨兵：
（a）`low` 桶 `mean_max_p > 0.70` —— 语言保守 + 数值过度自信；
（b）`high` 桶 `mean_max_p < 0.55` —— 语言自信 + 数值不到位。命中其一就
在 `per_model_summary.md` 模型名后追加 `conflict*`。这是论文里**没有**的
诊断维度——论文只有二值 $p$，无法把 *language* 和 *numeric* confidence 解耦。

**Phase 3 产物清单**：

| 文件 | 内容 |
| --- | --- |
| `belief_evolution.csv` | per-(model, q, k) 5 指标行 |
| `reflection_ab.csv` | 配对 run 的 ΔBI / Δσ / ΔC / Δη paired bootstrap CI（含 per-qtype 切片） |
| `tool_usage_pdp.csv` | per-(model, feature, value) PDP 行 |
| `confidence_calibration.csv` | per-(model, low/medium/high) 主观置信 vs 命中率 |
| `numeric_confidence_calibration.csv` | per-(model, max_p bin) 数值置信 vs 命中率 |
| `per_model_summary.md` | 追加 `conflict*` 哨兵（与既有 `cal*` 并列） |

**可视化**：`scripts/plot_analysis.py` 是按需 CLI（matplotlib 仅在用户机
本地装），读 `analysis/*.csv` 出图：

* **v5 主图**：`fss_bar_with_ci.png` / `delta_fss_forest.png` /
  `entropy_accuracy_grid_<model>.png`（per-model 3 桶 × 3 指标）；
* **Companion / appendix**：`bi_bar_with_ci.png`（BLF 对标锚点）/
  `delta_bi_forest.png` / `difficulty_grid.png` / `belief_trajectory_*.png`
  / `tool_pdp_*.png`。

v5 删除了 `reliability_diagram_per_model.png` / `_calibrated.png` /
`brier_decomp_stacked.png` 三幅图（输入数据被删）。所有图落
`analysis/figs/`（`.gitignore` 隔离）。matplotlib 不进 `environment.yml`、
不影响 CI。

**FSS sensitivity（按需 CLI）**：`scripts/fss_sensitivity.py` 一次性脚本
跑 4 档 (α, β) 输出 `fss_sensitivity.csv`；不进 `run_analysis` 主流程
（Decision 12）。论文 appendix 用它回答"为什么是 (2, 0.5) 不是 (1, 1)"。

`Settings.BELIEF_PROTOCOL=false` 时旧 accuracy 列输出零变化、新概率列照样
通过 fallback 计算（虽然校准信号被削弱）；行为分析则降级到空交付物（v3 老
run 没有 belief_trace、`belief_evolution.csv` 不写）。这套向后兼容保证
v3→v4 单向迁移不重跑历史 run。

### 11.6 网格搜索分析（`react-tavily-grid-search`）

`Settings.TAVILY_MAX_RESULTS`（R）与 `REACT_MAX_SEARCH_CALLS`（C）支持
逗号分隔的多值列表；evaluation 入口对 `MODELS × R_list × C_list` 做
笛卡尔展开，每个 `(real_model, R, C)` cell 编码为虚拟 slug
`{real}::r{R}::c{C}`，runner / DB schema / 既有 analysis 主流程**一行
不动**。`forecast_eval/analysis/grid.py` 负责反解三元组、重聚合、出 paper
长表。详细决策见 `DESIGN.md` "grid search via virtual slug (C 方案)"。

| 文件 | 内容 |
| --- | --- |
| `grid_summary.csv` | per `(real_model, R, C)` 17 列主表：accuracy/BI/NLL + 95% CI + `mean_search_calls / mean_latency_ms` 等 cost 列 |
| `grid_marginal_C.csv` / `grid_marginal_R.csv` | 固定 `R = default_r` 沿 C 扫 / 固定 `C = default_c` 沿 R 扫 |
| `grid_pareto.csv` | 每个 cell 一行；frontier cell 的 `dominated_by` 列空，否则记字典序最小的支配者虚拟 slug |
| `grid_winrate.csv` | 每对 `(real_model_a, real_model_b)`：跨 (R, C) cell 的 wins/ties + paired bootstrap 显著 cell 计数 |

CI 全部走 `inference.paired_bootstrap`（5000 次重抽，seed=42）；BI 域
CI 通过 "BS 域 paired bootstrap + 单调变换 $\mathrm{BI}=100(1-\sqrt{\mathrm{BS}})$"
得到，不引入新的统计代码（`DESIGN.md` D8）。

可视化由 `scripts/plot_analysis.py` 在主流程检测到 `manifest.grid` 段
时按需输出：

| 图 | 内容 |
| --- | --- |
| `grid_pareto_C.png` | Fig 1 主图：固定 `R = default_r`，每个 real_model 一条 `BI vs mean_search_calls` 曲线 + 95% CI band，Pareto cell 标星 |
| `grid_pareto_C_R{R}.png` | 附录：每个非默认 R 的同款图 |
| `grid_heatmap_RC_<real_model>.png` | Fig 2 per real_model：(R, C) 平面 BI 热力图，与 best cell CI 重叠的格子 hatch |
| `grid_curve_C.png` / `grid_curve_R.png` | Fig 3：3 行（BI / NLL / Acc）× M 列 panel，CI shading + 饱和点（一阶差分 < 0.01）虚线竖标 |
| `grid_winrate_matrix.png` | Fig 4：`M × M` 行优于列的占比矩阵，sig\_cells_* ≥ 1 的 cell 加 `*` 标记 |

旧 v4 run（manifest 无 `grid` 段）下 `run_grid_analysis` 早退、不写
任何 `grid_*.csv`；plot 流程也跳过 grid 图族——单值 .env 在新代码下
解析为长度 1 列表 → 笛卡尔生成单一虚拟 slug，行为与本变更前字节级
等价（除 .db 文件名后缀 `__r{R}__c{C}`）。

---

## 12. CLI 与运行方式

### 12.1 命令

```bash
# 跑全部 319 道题
python evaluation.py

# 按 question_type 过滤 (可重复)
python evaluation.py --question-type yes_no --question-type binary_named

# 按 choice_type 过滤 (可重复)
python evaluation.py --choice-type single

# 组合过滤 (AND): 仅跑 multiple_choice 中的多选题 (34 道)
python evaluation.py --question-type multiple_choice --choice-type multi

# 不在 run 结束时生成 analysis/ (原始 DB 仍会落在 db/)
python evaluation.py --skip-analysis

# 独立重刷 analysis/ (不改 DB)
python -m forecast_eval.analysis runs/{run_id}
```

`--question-type` 取值：`yes_no` / `binary_named` / `multiple_choice`，可重复，不传 = 不限制。
`--choice-type`   取值：`single` / `multi`，可重复，不传 = 不限制。
除 `--skip-analysis` 以外，所有可调项仍走 `.env`。

### 12.2 流程

```
1. argparse 解析 --question-type / --choice-type / --skip-analysis, 组装为 QFilter
2. Settings() 加载并校验 .env (含 MODEL_TRAINING_CUTOFFS + RUNS_ROOT)
3. 生成或复用 run_id -> 确定 run_dir = RUNS_ROOT/{run_id}; 建 db/ / analysis/ / logs/
4. 计算 source_db_hash / metadata_hash / prompt_templates_hash
5. 对每个 MODELS[i]:
   a. open conn = RUNS_ROOT/{run_id}/db/{safe_slug(model)}.db
   b. db.init_schema(conn, SAMPLING_N)  # 动态建 s{i}_* 列
   c. loader.sync_prompt_templates(src, conn) / loader.sync_questions(src, conn, filter)
   d. db.register_run_meta(conn, run_id=..., model=..., hashes=..., training_cutoff=...)
6. 写 manifest.json (run_id, models, model_files, sampling_n, filters, hashes, started_at)
7. runner.run(..., conns={model: conn, ...}) 启动 asyncio event loop
   a. 对每个模型 db.load_completed_samples(conn, SAMPLING_N) 作为 resume 基准
   b. 生成笛卡尔积: questions × MODELS × range(SAMPLING_N); 扣除 resume 集
   c. §3.9 过滤: 把 q.end_time <= cutoff 的 (q, model, idx) 直接写 skipped_training_cutoff 行
      到对应 model 的 writer, 不入 LLM 任务队列
   d. 剩余任务: Semaphore 限流 (LLM / Search 各一个) 并发
   e. 每条完成 → 路由到该模型的 writer → 批量 UPSERT s{i}_* 列
   f. 每完成一条打一行 log: [x/xx] q=.. qt=.. ct=.. model=.. idx=.. correct=..
8. 每个模型 db.finish_run_meta(conn, run_id); 收尾 manifest.finished_at
9. 除非 --skip-analysis: 调用 forecast_eval.analysis.run_analysis(run_dir), 写 analysis/
```

---

## 13. 日志 (`loguru`)

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

### 13.1 进度打印

格式：
```
12:03:44 | INFO    | [run=20260424-120344-a7k3] [5/1610] q=69566c13 qt=binary_named ct=single model=openai/gpt-5 sample=2/5 correct=True steps=4 tool_calls=3 latency=8421ms
```

- `[5/1610]` 分母 = `len(questions_after_filter) × len(MODELS) × SAMPLING_N`（扣掉已完成的续跑任务）
- 每条 sample 完成就打一行
- 错误时打 `ERROR` 级：`[x/xx] q=.. model=.. error=rate_limit retry_exhausted`

---

## 14. Conda 环境 (`environment.yml`)

```yaml
name: forecast
channels:
  - conda-forge
dependencies:
  - python=3.12
  - pip
  - pip:
      - openai>=1.50            # OpenRouter 用 OpenAI-compatible SDK
      - tavily-python>=0.5
      - pydantic>=2.6
      - pydantic-settings>=2.2
      - python-dotenv>=1.0
      - loguru>=0.7
      - httpx>=0.27
      - tenacity>=9.0           # 重试装饰器, 实现分层退避
      - pytest>=8.0             # §17 测试
      - pytest-asyncio>=0.23    # async 测试支持
      - respx>=0.21             # mock httpx, 用于 LLM / Tavily dry-run
```

创建环境：
```bash
conda env create -f environment.yml
conda activate forecast
cp .env.example .env
# 编辑 .env 填入 LLM_API_KEY 和 TAVILY_API_KEY (LLM_BASE_URL 可指任意 OpenAI 兼容 endpoint)
python evaluation.py --question-type yes_no
```

---

## 15. 最终 Premise 汇总（供最后 review）

1. **源数据 7 字段**：`id / choice_type / question_type / event / options / answer / end_time`，统一字母编码答案
2. **源数据库 `forecast_eval_set_example.db` 纳入 Git 管理**（只读示例数据集，随仓库分发，保证 `source_db_hash` 可复现；`SOURCE_DB` / `SOURCE_TABLE` 可指向自带数据集）
3. **LLM 看不到 `end_date`**，注入在 Tool 实现层
4. **Tavily `end_date = end_time + TAVILY_END_DATE_OFFSET_DAYS`**，项目统一以 `-1` 为默认严格基准（所有报表默认在 `-1` 下比较）
5. **泄漏边界与威胁模型**（§3.8）：Tool 只能约束工具搜索；强制禁用 provider-native browsing / `:online`；参数记忆通过 §3.9 的训练截止过滤部分缓解
6. **按模型训练截止日期过滤题**（§3.9）：`.env` 的 `MODEL_TRAINING_CUTOFFS` 指定每个模型 cutoff，`q.end_time ≤ cutoff` 的样本被写为 `error="skipped_training_cutoff"`，不调用 LLM、resume 不重试
7. **Prompt 拼接由 `prompts.py` 完成**：从 `dataset_metadata` 拉模板 → 按 `question_type` 渲染 `outcomes_block` 与 `output_format`（binary_named 时替换 `<options[i]>` 占位符）；>26 选项走源数据 ASCII 续接兼容模式（§3.7 警告）
8. **评测 = 字母集合 frozenset 严格相等**，漏选/多选都算错
9. **Parse 失败 ≠ error**，单独统计 refusal / format_failure rate
10. **多模型一次 run 笛卡尔积**，通过 `run_id` 断点续跑；每个模型一个 DB，`run_meta`
    单行记录 `filters_snapshot` + `source_db_hash` + `metadata_hash` + `prompt_templates_hash`
    + `training_cutoff` + **脱敏**后的 `config_snapshot`（API Key 明文不落库）
11. **Auth 错误整个 run 停止**；其他错误按退避分层重试，retry 用完 skip + 记 `error`
12. **Content policy violation 不重试**，直接标记
13. **所有灵活参数在 `.env`**，CLI 仅 `--question-type` / `--choice-type` / `--skip-analysis`
14. **主入口 `evaluation.py`**：创建 `RUNS_ROOT/{run_id}/`、跑 runner、跑 analysis（除非 `--skip-analysis`）
15. **Conda + Python 3.12 + loguru**，进度 `[x/xx]` 打 log
16. **SQLite WAL + `PRAGMA foreign_keys=ON` + 每模型一个 async writer task**，避免并发写入锁竞争
17. **每个 model DB 自包含**：内置 `questions` + `prompt_templates` 副本 + `run_meta`，可独立分发与复盘
18. **指标命名**：业界 `pass@k` 对应本项目 `pass_any@N`；原阈值口径改名 `at_least_k_correct@N`
19. **记录与分析拆开**：DB 里只存原始 sample 记录；pass@1 / pass_any@N / majority / parse_failure / cutoff_skip 等全部由 `analysis.py` 后置计算，写到 `analysis/` 下的 CSV / MD / JSON

---

## 16. 待落地模块顺序（建议）

1. `environment.yml` + `.env.example` + `.gitignore`
2. `forecast_eval/config.py`（Settings 类，含 `RUNS_ROOT`）
3. `forecast_eval/db.py`（per-model 宽表 schema + `AsyncWriter` + resume 查询 + prompt_templates 表 + model_slug_safe）
4. `forecast_eval/loader.py`（同步 questions + prompt_templates）
5. `forecast_eval/prompts.py`（按 question_type 渲染 user message，**单元测试覆盖三种类型**）
6. `forecast_eval/parser.py`（`\boxed{}` 解析 + 字母集合归一 + 严格匹配，**单元测试覆盖三种类型 + edge case**）
7. `forecast_eval/errors.py`（错误分类 + 退避）
8. `forecast_eval/search.py`（Tavily + end_date 注入）
9. `forecast_eval/tools.py`（schema + execute_tool_call）
10. `forecast_eval/llm.py`（OpenRouter client + retry）
11. `forecast_eval/react.py`（单 sample ReAct loop）
12. `forecast_eval/runner.py`（编排 + 并发 + 多模型 writer + 进度）
13. `forecast_eval/analysis.py`（后置统计, 读 DB -> CSV / MD / JSON）
14. `evaluation.py`（main, 建目录 + register_run_meta + runner + analysis）

先用 `--question-type yes_no` + `MODELS=openai/gpt-4o-mini` + `SAMPLING_N=1` 跑通 smoke test（93 道，最便宜的题型），验证 `prompts.render_user_prompt` 输出与 `parser.parse_answer` 归一无误后，再放开完整评测。

---

## 17. 测试计划（`tests/`）

评测单次成本较高（319 题 × 模型数 × N samples），测试先站稳能省下大量 API 费用。所有测试 **不联网**、**不烧 API**：Tavily / OpenRouter 均以 fixture 或 mock 替身存在。

| 测试文件                    | 覆盖对象              | 关键用例                                                                                                                                                                                                                |
| --------------------------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_prompts.py`           | `prompts.py`          | ① `yes_no` / `binary_named` / `multiple_choice`（≤26 选项）三种模板渲染 snapshot；② `binary_named` 占位符替换正确；③ `multiple_choice` >26 选项（用数据库里真实的 4 道题做 fixture）outcomes_block 标签准确                  |
| `test_parser.py`            | `parser.py`           | ① 三种题型的 `\boxed{}` 正解路径；② 多 `\boxed{}` 取最后一个；③ 大小写、空格、逗号/空格分隔混排；④ 非法字母越界；⑤ >26 选项 label↔letter round-trip；⑥ `parse_gt` 对 `"A, B"` 解析；⑦ 软性拒绝 → None 不报错               |
| `test_search.py`            | `search.py` + `tools.py` | ① `web_search` schema LLM 可见字段 **不含** `end_date`；② `tavily_search` 注入的 `end_date = q.end_time + OFFSET`；③ Tavily 报错走 `SEARCH_BACKOFF_S` 重试；④ 重试用完后返回错误 payload 而不抛；⑤ `_build_request_payload` 把 `TAVILY_INCLUDE_RAW_CONTENT={false,markdown,text}` / `TAVILY_SEARCH_DEPTH` / `TAVILY_INCLUDE_ANSWER` 三档枚举映射到 Tavily 协议形式；⑥ 超长 `raw_content` 在 `_truncate_raw_content` 处截断到 `TAVILY_RAW_CONTENT_MAX_CHARS` 并追加省略标记；⑦ `to_llm_payload` 缺失字段（`score` / `raw_content` / `published_date` / `answer`）不输出 `null` 占位 |
| `test_db.py`                | `db.py`               | ① per-model schema 按 `sampling_n` 动态建 `s{i}_*` 列 + PRAGMA；② schema `N` mismatch 时 fail-fast；③ `model_slug_safe` 规则；④ hash 计算稳定；⑤ `config_snapshot` 脱敏；⑥ UPSERT 按 `(qid, sample_idx)` 覆盖；⑦ `AsyncWriter` 分桶批量提交 |
| `test_runner_resume.py`     | `runner.py`           | ① `load_completed_samples` 排除 retryable error；② `build_task_plan` 按 per-model completed 去重；③ 未在 `completed` 中声明的模型默认空集（全部入队）                                                                    |
| `test_training_cutoff.py`   | §3.9 过滤逻辑         | ① `q.end_time <= cutoff` 全部 N samples 都写 skipped_training_cutoff；② 未声明 cutoff 的模型不过滤；③ resume 优先于 cutoff；④ 写入后 `load_completed_samples` 命中                                                       |
| `test_llm_no_browsing.py`   | `llm.py`              | mock 客户端断言请求 payload 里**没有** `plugins`、`tools` 里没有 provider-native web_search、model 名字不以 `:online` 结尾                                                                                              |
| `test_errors.py`            | `errors.py`           | 各类 `httpx` / OpenAI 异常 → 正确 `ErrorKind`；`Retry-After` header 优先于默认退避                                                                                                                                      |
| `test_analysis.py`          | `analysis.py`         | ① 手工造宽表 fixture；② pass@1 / pass_any@N / ≥majority / ≥all / majority_vote / parse_failure / error_rate / cutoff_skip 数值正确；③ `overall.json` 与 CSV 对齐；④ `error_breakdown.csv` 汇总                           |
| `test_smoke_dry_run.py`     | 端到端 dry-run        | 用 httpx stub 替换 OpenRouter + Tavily，跑 3 道题 × 1 模型 × 1 sample，验证宽表 `s0_*` 字段齐全、`messages_trace` 合法 JSON、`search_calls` 记录 `end_date`                                                               |

运行：
```bash
pytest tests/ -q
```
CI 最低要求：`test_prompts.py` / `test_parser.py` / `test_training_cutoff.py` / `test_llm_no_browsing.py` / `test_analysis.py` 五项必须绿灯（核心语义 + 安全边界 + 统计正确性）。
