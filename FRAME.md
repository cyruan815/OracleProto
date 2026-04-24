# Forecast Evaluation — 项目整体框架

## 1. 项目目标

基于 `forecast_eval_set.db` 数据集，评测 LLM 在**预测类单选/多选题**上的能力。

核心特色：通过自研 `web_search` Tool 限制 LLM 的信息获取边界——**只允许 LLM 搜索到每道题 `end_time`（事件解决日期）之前的信息**，以此模拟"在题目时间点预测未来"的真实场景，避免信息泄露。

> 重要限制：工具级时间截断只约束**工具搜索**这一条信息通路；模型参数记忆、provider 内置 browsing、搜索结果 snippet/缓存等泄漏源不可能被 Tool 层阻断。完整威胁模型与缓解手段见 §3.8。

- 评测 322 道题（`yes_no` 93 + `binary_named` 11 + `multiple_choice` 218），其中 285 道单选 + 37 道多选
- 通过 OpenRouter 的 OpenAI-compatible API 同时评测多个模型
- LLM 以 ReAct + Tool Use 模式与 `web_search` 工具交互
- 评测结果写入独立的 `results.db`，后续分析独立进行

---

## 2. 数据源

### 2.1 原数据库 `forecast_eval_set.db`（只读）

主表 `forecast_eval_set`，**322 行 × 7 列**：

| 字段            | 类型    | 说明                                                                                                                          |
| --------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `id`            | TEXT PK | 题目唯一 ID（来自 HuggingFace 源）                                                                                            |
| `choice_type`   | TEXT    | `single` \| `multi`，依据 `answer` 字母个数（1 个 → `single`，>1 → `multi`）                                                  |
| `question_type` | TEXT    | `yes_no` \| `binary_named` \| `multiple_choice`，决定走哪套 prompt 模板                                                       |
| `event`         | TEXT    | 事件描述（**不含**选项、**不含**角色设定、**不含**格式要求）                                                                  |
| `options`       | TEXT    | JSON array of strings。`yes_no`=`["Yes","No"]`；`binary_named`=两个实体名；`multiple_choice`=按 A/B/C... 顺序的标签            |
| `answer`        | TEXT    | 字母编码：单选 `'A'`；多选 `'A, B'`（逗号 + 空格分隔）。字母 ↔ 选项索引规则见 §3.7                                            |
| `end_time`      | TEXT    | 事件解决日期（Asia/Shanghai），`YYYY-MM-DD` 格式                                                                              |

索引：`idx_forecast_eval_set_choice_type` / `idx_forecast_eval_set_question_type` / `idx_forecast_eval_set_end_time`。

辅表 `dataset_metadata`（一行），含 `features_json`，记录所有 prompt 模板、列说明、转换日志。

### 2.2 题量分布

| question_type / choice_type | single | multi | 合计 |
| --------------------------- | -----: | ----: | ---: |
| `yes_no`                    |     93 |     0 |   93 |
| `binary_named`              |     11 |     0 |   11 |
| `multiple_choice`           |    181 |    37 |  218 |
| **合计**                    |  **285** | **37** | **322** |

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

### 3.3 原数据只读，结果自包含

`forecast_eval_set.db` 不动。`results.db` 独立文件，内部**复制 questions 表 + prompt_templates 表**，保证 `results.db` 单文件即可完整复盘和分析（详见 §5）。

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
| `event`           | `forecast_eval_set.event` 原文                                                                                 |
| `end_time`        | `forecast_eval_set.end_time` 原文（`YYYY-MM-DD`）                                                              |
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
| 搜索结果 snippet 里的"未来泄漏"   | ⚠️ 部分可控 | Tavily 的 `end_date` 过滤已在 publish date 层面截断；极少数索引错日期的页面仍可能漏出，不做额外处理 |
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
                          │  forecast_eval_set.db            │
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

## 5. 数据库设计 (`results.db`)

### 5.1 schema

```sql
-- ① 复制源题库（让 results.db 自包含）
-- 字段名/含义与 forecast_eval_set 表一致, 加 imported_at 便于追溯
CREATE TABLE questions (
    id            TEXT PRIMARY KEY,
    choice_type   TEXT NOT NULL CHECK (choice_type IN ('single','multi')),
    question_type TEXT NOT NULL CHECK (question_type IN ('yes_no','binary_named','multiple_choice')),
    event         TEXT NOT NULL,
    options       TEXT NOT NULL,             -- JSON array, e.g. ["Yes","No"] / ["Golden Knights","Kings"] / [...]
    answer        TEXT NOT NULL,             -- 字母逗号串: 'A' / 'A, B'
    end_time      TEXT NOT NULL,             -- YYYY-MM-DD
    imported_at   TEXT NOT NULL
);
CREATE INDEX idx_questions_choice_type   ON questions(choice_type);
CREATE INDEX idx_questions_question_type ON questions(question_type);

-- ② 复制 prompt 模板（不依赖源 metadata 文件）
-- 至少包含: agent_role, guidance, prompt_template, outcomes_block_rule,
--           yes_no_output_format, binary_named_output_format, multiple_choice_output_format
CREATE TABLE prompt_templates (
    key          TEXT PRIMARY KEY,
    value        TEXT NOT NULL,
    imported_at  TEXT NOT NULL
);

-- ③ 一次跑的元信息
CREATE TABLE runs (
    run_id                 TEXT PRIMARY KEY,
    config_snapshot        TEXT NOT NULL,    -- .env 解析后的 JSON, **脱敏后**: API Key 类字段只存 provider 标识 + 前 4 位 + 长度 + sha256[:12], 不落明文
    filters_snapshot       TEXT NOT NULL,    -- CLI 过滤结果: {"question_types":[...]|null, "choice_types":[...]|null, "question_ids":[...], "question_count": N}
    source_db_hash         TEXT NOT NULL,    -- sha256(forecast_eval_set.db 文件二进制), 用于确认源数据未变
    metadata_hash          TEXT NOT NULL,    -- sha256(dataset_metadata.features_json 规范化字符串), prompt 模板源版本
    prompt_templates_hash  TEXT NOT NULL,    -- sha256(本次 run 实际使用的 prompt_templates 表 key/value 规范化串)
    started_at             TEXT NOT NULL,
    finished_at            TEXT
);

-- ⓪ 简单的 schema 版本表, 用于未来迁移
CREATE TABLE schema_version (
    version      INTEGER PRIMARY KEY,
    applied_at   TEXT NOT NULL
);

-- ④ 每个 sample 一行
CREATE TABLE run_results (
    run_id      TEXT NOT NULL,
    question_id TEXT NOT NULL,
    model       TEXT NOT NULL,
    sample_idx  INTEGER NOT NULL,            -- 0..N-1

    -- 答题结果（统一以"字母集合"视角存储, 与 question_type 无关）
    final_answer_letters TEXT,               -- JSON sorted list: ["A","B"]; NULL if parse_ok=0
    final_answer_raw     TEXT,               -- LLM 最终消息原文（含 \boxed{...}）
    correct              INTEGER,            -- 0/1; NULL if parse_ok=0
    parse_ok             INTEGER NOT NULL,   -- 0/1

    -- 过程指标
    tool_calls_count   INTEGER NOT NULL,
    react_steps        INTEGER NOT NULL,
    prompt_tokens      INTEGER,
    completion_tokens  INTEGER,
    reasoning_tokens   INTEGER,
    latency_ms         INTEGER NOT NULL,

    -- 调试 / 溯源
    user_prompt    TEXT,                     -- 拼接后传给 LLM 的 user message（便于复盘渲染结果）
    messages_trace TEXT,                     -- 完整 messages JSON; WRITE_MESSAGES_TRACE=false 时 NULL
    search_calls   TEXT,                     -- JSON list: [{query, end_date, n_results, published_dates}]
    error          TEXT,                     -- 非 NULL 表示这次 sample 失败; 分类见 §9
    created_at     TEXT NOT NULL,

    PRIMARY KEY (run_id, question_id, model, sample_idx),
    FOREIGN KEY (question_id) REFERENCES questions(id),
    FOREIGN KEY (run_id)      REFERENCES runs(run_id)
);
CREATE INDEX idx_run_results_lookup ON run_results(run_id, model, question_id);
```

连接初始化 PRAGMA（所有 sqlite3 连接都执行一遍）：
```sql
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;     -- WAL 下足够安全且更快
PRAGMA busy_timeout = 5000;      -- 多 reader 场景下避免 SQLITE_BUSY
```

### 5.2 字段写入约定

| 字段                      | 来源                                                                                                  |
| ------------------------- | ----------------------------------------------------------------------------------------------------- |
| `final_answer_letters`    | `parser.parse_answer(final_raw, q)` 返回的 `frozenset[str]`，写入前 `sorted()` + `json.dumps`         |
| `final_answer_raw`        | LLM 最后一条 assistant message 的 `content` 全文                                                      |
| `correct`                 | `frozenset(final_answer_letters) == frozenset(answer_letters_from_q)` → `int`；parse 失败时 `NULL`    |
| `parse_ok`                | `final_answer_letters is not None`                                                                    |
| `user_prompt`             | `prompts.render_user_prompt(q, templates)` 的返回值（每个 sample 同 q 渲染一致，存便于直接重放）      |
| `messages_trace`          | 完整 `messages` 列表（含 system/user/assistant/tool）的 JSON。开 `WRITE_MESSAGES_TRACE=false` 时 NULL |
| `search_calls`            | 每次 `web_search` 调用的元数据 list（query / end_date / n_results / published_dates）                 |
| `error`                   | retry 用尽后的错误分类码；正常完成（含 refusal / parse fail）为 NULL                                  |

### 5.3 断点续跑

跑之前执行：
```sql
SELECT question_id, model, sample_idx
FROM run_results
WHERE run_id = ?
  AND (error IS NULL OR error = 'skipped_training_cutoff');
```
将查到的 `(question_id, model, sample_idx)` 集合从任务队列中剔除。

状态分类：
| `error` 值                       | 含义                | 下次续跑是否重试 |
| -------------------------------- | ------------------- | ---------------- |
| `NULL`                           | 已正常完成          | 否               |
| `'skipped_training_cutoff'`      | §3.9 主动剔除       | 否               |
| `'network'` / `'server_5xx'`     | 退避用完仍失败      | 是               |
| `'bad_request'`                  | model_not_found 等  | 是（改配置后续跑） |
| `'content_policy'`               | provider 拒绝       | 可选：默认重试一次并覆盖原行 |

规则：
- 同 `run_id` 重跑 = 续跑
- 换 `run_id` = 全新一跑
- 需要覆盖（比如 content_policy 重试成功）时走 `INSERT OR REPLACE`，由 `(run_id, question_id, model, sample_idx)` 主键天然兜底

### 5.4 并发写入策略

- 启动时执行 §5.1 末尾的 PRAGMA 一组
- **单 async writer task**（不是 thread）：所有 async worker 通过 `asyncio.Queue` 把结果塞给同一个 writer task
- Writer task 每 `DB_COMMIT_BATCH` 条或 1 秒 flush 一次，短事务；单条 sqlite 写入用 `await asyncio.to_thread(conn.execute, ...)` 避免阻塞事件循环
- 若坚持使用真正的 writer thread，必须改用 `queue.Queue`（线程安全）或 `janus.Queue`；`asyncio.Queue` 不是跨线程安全的，不要在文档基础上直接拿它跨线程消费
- SQLite 单写入者 + WAL 已能并发 reader，无需多 writer

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
├── evaluation.py                  # 主入口: parse CLI flags, 调 runner.run()
├── forecast_eval_set.db           # 原数据, 只读, **纳入 Git 管理**(方便随仓库分发 + 保证 source_db_hash 在 CI 可复现)
├── results.db                     # 评测结果, 运行时创建 (gitignored)
├── logs/                          # loguru 落盘
│   └── {run_id}.log
├── forecast_eval/
│   ├── __init__.py
│   ├── config.py                 # pydantic-settings 从 .env 读 (含 MODEL_TRAINING_CUTOFFS 解析)
│   ├── db.py                     # SQLite WAL + PRAGMA + 单 async writer task + schema migration + hash 计算
│   ├── loader.py                 # 从 forecast_eval_set.db 同步 questions + prompt_templates
│   ├── prompts.py                # 按 question_type 渲染 user message
│   ├── llm.py                    # OpenRouter client + retry 分层 (明确禁用 provider-native browsing)
│   ├── search.py                 # Tavily + end_date 注入 + retry
│   ├── tools.py                  # web_search schema (LLM 可见部分, 不含日期)
│   ├── react.py                  # ReAct loop (一个 sample)
│   ├── parser.py                 # \boxed{} 解析 + 字母集合归一 + 严格匹配
│   ├── errors.py                 # 错误分类 + 退避策略 (含 skipped_training_cutoff)
│   └── runner.py                 # 任务编排 + 并发 + 进度 + 训练截止过滤
└── tests/                        # 单元测试 (§17)
    ├── test_prompts.py
    ├── test_parser.py
    ├── test_search.py
    ├── test_runner_resume.py
    └── test_training_cutoff.py
```

---

## 7. `.env.example` 完整配置

```ini
# =============================================================
#  Forecast Evaluation — 环境变量配置
#  复制为 .env 后填入 API Key 即可运行: python evaluation.py
# =============================================================

# -------- LLM Endpoint (OpenAI-compatible) --------
LLM_API_KEY=sk-or-v1-REPLACE_ME
LLM_BASE_URL=https://openrouter.ai/api/v1

# 要评测的模型列表, 逗号分隔 (笛卡尔积: 每个模型都会跑所有题目 × 所有 sample)
# ⚠️ 不要在 model slug 里追加 ":online", 也不要启用任何 provider-native browsing (参见 §3.8)
MODELS=openai/gpt-5,anthropic/claude-sonnet-4.5,google/gemini-2.5-pro,deepseek/deepseek-r1

# 模型训练截止日期 (§3.9): 题目 end_time <= cutoff 的 (q, model) 将被跳过并标记 skipped_training_cutoff
# 格式: "<model_slug>=YYYY-MM-DD" 多组用逗号分隔. 未声明的模型不过滤
# 建议对每个参评模型都显式声明, 以保证评测公平
MODEL_TRAINING_CUTOFFS=openai/gpt-5=2024-10-01,anthropic/claude-sonnet-4.5=2025-03-01,google/gemini-2.5-pro=2025-01-01,deepseek/deepseek-r1=2024-07-01

# LLM 调用参数
LLM_MAX_TOKENS=4096
LLM_TIMEOUT_S=120
LLM_TEMPERATURE=0.7
LLM_TOP_P=1.0

# LLM 并发 & 重试 (与 OpenRouter 配套)
LLM_MAX_CONCURRENCY=10
LLM_RETRY_MAX=5
# 不同错误类型的退避序列 (秒), 用完仍失败则跳过该 sample 并记 error
LLM_BACKOFF_NETWORK_S=2,5,15,30,60
LLM_BACKOFF_RATE_LIMIT_S=10,30,60,120,300
LLM_BACKOFF_SERVER_5XX_S=5,15,30,60,120

# -------- Tavily Search --------
TAVILY_API_KEY=tvly-REPLACE_ME
TAVILY_MAX_RESULTS=5
TAVILY_INCLUDE_RAW_CONTENT=false
# end_date = question.end_time + offset. 项目默认 -1 (前一天, 避免事件当天信息泄露).
# 数值越小越保守: -2/-3 更严格; 0 = 当天可见 (仅调试用, 不要在正式评测中使用)
TAVILY_END_DATE_OFFSET_DAYS=-1

# Tavily 并发 & 重试 (与 Tavily 配套)
SEARCH_MAX_CONCURRENCY=5
SEARCH_RETRY_MAX=3
SEARCH_BACKOFF_S=2,5,15

# -------- ReAct Loop --------
REACT_MAX_STEPS=10
REACT_MAX_SEARCH_CALLS=8

# -------- Sampling --------
# 每道题每个模型采样几次 (pass@1 avg / pass_any@N / majority vote 都基于这 N 次)
SAMPLING_N=5

# -------- Run / Resume --------
# 留空则自动生成 YYYYMMDD-HHMMSS-{4位短uuid}. 填相同的 run_id 可断点续跑
RUN_ID=
RESUME=true

# -------- Database --------
SOURCE_DB=./forecast_eval_set.db
RESULTS_DB=./results.db
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
- **`TAVILY_END_DATE_OFFSET_DAYS`**：项目默认 `-1`（前一天，推荐的严格默认值）。数值越小越保守；`0` 仅调试用。所有报表默认在 `-1` 下比较。
- **`RUN_ID` 自动生成格式**：`YYYYMMDD-HHMMSS-xxxx`，例如 `20260424-120344-a7k3`，`ls` 天然按时间排序。
- **`WRITE_MESSAGES_TRACE`**：`true` 存完整 messages JSON（方便 debug 但 db 变大）；`false` 只存关键字段。
- **脱敏**：`runs.config_snapshot` 写入前 `config.py` 必须对 `LLM_API_KEY` / `TAVILY_API_KEY` 等敏感字段执行 redaction（只保留前 4 位 + 长度 + `sha256[:12]`），敏感明文一律不落库。

---

## 8. 核心模块职责

| 模块         | 职责                                                                                                            | 关键接口                                                                                                       |
| ------------ | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `config.py`  | `pydantic-settings` 从 `.env` 读取，校验类型，逗号分隔列表解析                                                  | `Settings` 类（单例）                                                                                          |
| `loader.py`  | 从 `forecast_eval_set.db` 同步两张表到 `results.db`：① `forecast_eval_set` → `questions`（按 filters 过滤）；② `dataset_metadata.features_json.prompt_reconstruction` → `prompt_templates`（key/value 平铺） | `sync_questions(filters: QFilter) -> list[Question]`, `sync_prompt_templates() -> dict[str,str]`               |
| `prompts.py` | 按 `question_type` 渲染 user message：① 生成 `outcomes_block`（multiple_choice 用 §3.7 字母规则枚举选项）；② 选三套 `output_format` 之一，binary_named 时把 `<options[i]>` 占位符替换为实际实体名；③ 用 `prompt_template` 拼装最终文本 | `render_user_prompt(q: Question, templates: dict[str,str]) -> str`                                             |
| `tools.py`   | 定义 `web_search` OpenAI-schema；**LLM 可见部分不含日期**                                                       | `WEB_SEARCH_SCHEMA`, `execute_tool_call(tc, q, cfg)`                                                           |
| `search.py`  | 封装 Tavily `/search`，注入 `end_date = q.end_time + OFFSET`，retry                                             | `search(query, end_date) -> SearchResult`                                                                      |
| `llm.py`     | OpenAI-compatible client (OpenRouter)，按错误类型分层 retry；**强制不启用 provider-native browsing**（不传 `plugins`、不加 `:online` 后缀、不发 provider 私有 web tool 字段） | `chat(model, messages, tools, ...) -> ChatResponse`                                                            |
| `react.py`   | 一次 ReAct 推理 = 一个 sample，循环到无 tool_call 或超限                                                        | `run_react(q, model, sample_idx, cfg) -> SampleResult`                                                         |
| `parser.py`  | 按 `question_type` 解析 `\boxed{...}` → 字母 `frozenset[str]`（yes_no: Yes/No→A/B；binary_named: label→letter；mc: split letters）；与 `q.answer` 解析出的字母集合做严格 frozenset 相等判对 | `parse_answer(text: str, q: Question) -> frozenset[str] \| None`, `parse_gt(answer: str) -> frozenset[str]`, `is_correct(pred, gt) -> bool` |
| `errors.py`  | 把 httpx/openai 异常映射到错误分类；给出等待秒数                                                                | `classify(exc) -> ErrorKind`, `backoff_seconds(kind, attempt)`                                                 |
| `db.py`      | 连接管理、WAL + PRAGMA、schema migration、单 async writer task、断点续跑查询、prompt_templates 写入、source/metadata/templates hash 计算、config 脱敏 | `DB.enqueue_result(row)`, `DB.load_completed(run_id)`, `DB.register_run(...)`, `DB.upsert_prompt_templates(d)`, `DB.compute_hashes()` |
| `runner.py`  | 任务编排：笛卡尔积 → 去重 → **按 `MODEL_TRAINING_CUTOFFS` 过滤并落 skipped_training_cutoff 行** → asyncio 并发 → 进度 log | `run(cfg, filters: QFilter)`                                                                                   |

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
    #    全部从 prompt_templates 读, 与源数据保持解耦
    user_prompt = prompts.render_user_prompt(q, cfg.PROMPT_TEMPLATES)

    # ③ 整体作为单条 user message (最忠实模板; 不再拆 system/user)
    messages = [{"role": "user", "content": user_prompt}]
    search_calls: list[dict] = []
    final_raw = ""
    t0 = time.monotonic()
    tokens = {"prompt": 0, "completion": 0, "reasoning": 0}
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

        # 没 tool_call = LLM 给最终答案
        if not msg.tool_calls:
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

一个 `(run_id, question_id, model)` 下有 N 个 sample（`N = SAMPLING_N`）。统计时**先排除** `error="skipped_training_cutoff"` 的行（它们是被剔除的题，不是模型答错）：

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

> 指标命名变更：原文档里的 `pass@3 = sum(correct)≥3` 与业界通用的 `pass@k` 语义不一致（后者是 "any correct in k"），容易误读。现在明确用 `pass_any@N`（= any）与 `at_least_k_correct@N`（= 阈值）两个独立命名。

报表切片维度：`model × question_type × choice_type`。具体报表生成由用户后续独立设计。

---

## 12. CLI 与运行方式

### 12.1 命令

```bash
# 跑全部 322 道题
python evaluation.py

# 按 question_type 过滤 (可重复)
python evaluation.py --question-type yes_no --question-type binary_named

# 按 choice_type 过滤 (可重复)
python evaluation.py --choice-type single

# 组合过滤 (AND): 仅跑 multiple_choice 中的多选题 (37 道)
python evaluation.py --question-type multiple_choice --choice-type multi
```

**没有其他参数**。所有可调项全部走 `.env`。

`--question-type` 取值：`yes_no` / `binary_named` / `multiple_choice`，可重复，不传 = 不限制。
`--choice-type`   取值：`single` / `multi`，可重复，不传 = 不限制。

### 12.2 流程

```
1. argparse 解析 --question-type / --choice-type, 组装为 QFilter
2. Settings.from_env() 加载并校验 .env (含 MODEL_TRAINING_CUTOFFS)
3. 计算 source_db_hash / metadata_hash, 生成或复用 run_id
4. loader.sync_prompt_templates() 把模板平铺到 results.db.prompt_templates
   → 计算 prompt_templates_hash
5. loader.sync_questions(filter) 从 forecast_eval_set.db 同步到 results.db.questions
6. register_run: 写 runs 行 (含 filters_snapshot / 三个 hash / 脱敏后的 config_snapshot)
7. db.load_completed(run_id) 查出已完成集合 (含 skipped_training_cutoff)
8. runner.run(cfg, filter) 启动 asyncio event loop
   a. 生成笛卡尔积: questions × MODELS × range(SAMPLING_N)
   b. 剔除步骤 7 已完成的 (question_id, model, sample_idx)
   c. §3.9 过滤: 对 MODEL_TRAINING_CUTOFFS 中声明的模型, 将 q.end_time <= cutoff 的
      (q, model, idx) 直接写 skipped_training_cutoff 行, 不入 LLM 任务队列
   d. 剩余任务: Semaphore 限流 (LLM / Search 各一个) 并发
   e. 每条完成 → writer queue → 批量 commit
   f. 每完成一条打一行 log: [x/xx] q=.. qt=.. ct=.. model=.. idx=.. correct=..
9. runs.finished_at 更新, 退出
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
# 编辑 .env 填入 LLM_API_KEY 和 TAVILY_API_KEY
python evaluation.py --question-type yes_no
```

---

## 15. 最终 Premise 汇总（供最后 review）

1. **源数据 7 字段**：`id / choice_type / question_type / event / options / answer / end_time`，统一字母编码答案
2. **源数据库 `forecast_eval_set.db` 纳入 Git 管理**（只读，随仓库分发，保证 `source_db_hash` 可复现）
3. **LLM 看不到 `end_date`**，注入在 Tool 实现层
4. **Tavily `end_date = end_time + TAVILY_END_DATE_OFFSET_DAYS`**，项目统一以 `-1` 为默认严格基准（所有报表默认在 `-1` 下比较）
5. **泄漏边界与威胁模型**（§3.8）：Tool 只能约束工具搜索；强制禁用 provider-native browsing / `:online`；参数记忆通过 §3.9 的训练截止过滤部分缓解
6. **按模型训练截止日期过滤题**（§3.9）：`.env` 的 `MODEL_TRAINING_CUTOFFS` 指定每个模型 cutoff，`q.end_time ≤ cutoff` 的样本被写为 `error="skipped_training_cutoff"`，不调用 LLM、resume 不重试
7. **Prompt 拼接由 `prompts.py` 完成**：从 `dataset_metadata` 拉模板 → 按 `question_type` 渲染 `outcomes_block` 与 `output_format`（binary_named 时替换 `<options[i]>` 占位符）；>26 选项走源数据 ASCII 续接兼容模式（§3.7 警告）
8. **评测 = 字母集合 frozenset 严格相等**，漏选/多选都算错
9. **Parse 失败 ≠ error**，单独统计 refusal / format_failure rate
10. **多模型一次 run 笛卡尔积**，通过 `run_id` 断点续跑；`runs` 表记录 `filters_snapshot` + `source_db_hash` + `metadata_hash` + `prompt_templates_hash` + **脱敏**后的 `config_snapshot`（API Key 明文不落库）
11. **Auth 错误整个 run 停止**；其他错误按退避分层重试，retry 用完 skip + 记 `error`
12. **Content policy violation 不重试**，直接标记
13. **所有灵活参数在 `.env`**，CLI 仅 `--question-type` / `--choice-type` 两个过滤 flag
14. **主入口 `evaluation.py`**，跑完进 `results.db` 结束；不做报表（后续独立设计）
15. **Conda + Python 3.12 + loguru**，进度 `[x/xx]` 打 log
16. **SQLite WAL + `PRAGMA foreign_keys=ON` + 单 async writer task**，避免并发写入锁竞争；不用跨线程 `asyncio.Queue`
17. **`results.db` 自包含**：内置 `questions` + `prompt_templates` 副本，可独立分发与复盘渲染
18. **指标命名**：业界 `pass@k` 对应本项目 `pass_any@N`；原阈值口径改名 `at_least_k_correct@N`

---

## 16. 待落地模块顺序（建议）

1. `environment.yml` + `.env.example` + `.gitignore`
2. `forecast_eval/config.py`（Settings 类）
3. `forecast_eval/db.py`（schema + writer thread + resume 查询 + prompt_templates 表）
4. `forecast_eval/loader.py`（同步 questions + prompt_templates）
5. `forecast_eval/prompts.py`（按 question_type 渲染 user message，**单元测试覆盖三种类型**）
6. `forecast_eval/parser.py`（`\boxed{}` 解析 + 字母集合归一 + 严格匹配，**单元测试覆盖三种类型 + edge case**）
7. `forecast_eval/errors.py`（错误分类 + 退避）
8. `forecast_eval/search.py`（Tavily + end_date 注入）
9. `forecast_eval/tools.py`（schema + execute_tool_call）
10. `forecast_eval/llm.py`（OpenRouter client + retry）
11. `forecast_eval/react.py`（单 sample ReAct loop）
12. `forecast_eval/runner.py`（编排 + 并发 + 进度）
13. `evaluation.py`（main）

先用 `--question-type yes_no` + `MODELS=openai/gpt-4o-mini` + `SAMPLING_N=1` 跑通 smoke test（93 道，最便宜的题型），验证 `prompts.render_user_prompt` 输出与 `parser.parse_answer` 归一无误后，再放开完整评测。

---

## 17. 测试计划（`tests/`）

评测单次成本较高（322 题 × 模型数 × N samples），测试先站稳能省下大量 API 费用。所有测试 **不联网**、**不烧 API**：Tavily / OpenRouter 均以 fixture 或 mock 替身存在。

| 测试文件                    | 覆盖对象              | 关键用例                                                                                      |
| --------------------------- | --------------------- | --------------------------------------------------------------------------------------------- |
| `test_prompts.py`           | `prompts.py`          | ① `yes_no` / `binary_named` / `multiple_choice`（≤26 选项）三种模板渲染 snapshot；② `binary_named` 占位符替换正确；③ `multiple_choice` >26 选项（用数据库里真实的 4 道题做 fixture）outcomes_block 标签准确且可在 markdown 中保留 |
| `test_parser.py`            | `parser.py`           | ① 三种题型的 `\boxed{}` 正解路径；② 多 `\boxed{}` 取最后一个；③ 大小写、空格、逗号/空格分隔混排；④ 非法字母越界；⑤ >26 选项 label↔letter round-trip；⑥ `parse_gt` 对 `"A, B"` 解析；⑦ 软性拒绝 → None 不报错 |
| `test_search.py`            | `search.py` + `tools.py` | ① `web_search` schema LLM 可见字段 **不含** `end_date`；② `tavily_search` 注入的 `end_date = q.end_time + OFFSET`（mock httpx）；③ Tavily 报错走 `SEARCH_BACKOFF_S` 重试；④ 重试用完后返回错误 payload 而不抛 |
| `test_db.py`                | `db.py`               | ① schema 建表 + PRAGMA 生效（含 `foreign_keys=ON`）；② `source_db_hash` / `metadata_hash` / `prompt_templates_hash` 计算稳定；③ `config_snapshot` 脱敏 API Key 明文不出现；④ `INSERT OR REPLACE` 主键覆盖语义 |
| `test_runner_resume.py`     | `runner.py`           | ① 同 `run_id` 已完成行被剔除；② `error='network'` 的行会被重试；③ `error='skipped_training_cutoff'` 的行 **不**被重试；④ 换 `run_id` = 全新跑 |
| `test_training_cutoff.py`   | §3.9 过滤逻辑         | ① `q.end_time <= cutoff` 的 `(q, model)` 全部 N samples 都写 skipped_training_cutoff；② 未在 `MODEL_TRAINING_CUTOFFS` 声明的模型不过滤；③ 跨日期边界（`end_time == cutoff`）判定正确（当前规则：`≤` 即跳） |
| `test_llm_no_browsing.py`   | `llm.py`              | mock OpenRouter 客户端，断言 `chat(...)` 发出的 payload 里**没有** `plugins`、`tools` 里没有 provider-native web_search、model 名字不以 `:online` 结尾 |
| `test_errors.py`            | `errors.py`           | 各类 `httpx` / OpenAI 异常 → 正确 `ErrorKind`；`Retry-After` header 优先于默认退避                                                                  |
| `test_smoke_dry_run.py`     | 端到端 dry-run        | 用 httpx stub 替换 OpenRouter + Tavily，跑 3 道题 × 1 模型 × 1 sample，验证 `results.db` 字段齐全、`messages_trace` 格式正确、`search_calls` 记录 `end_date` |

运行：
```bash
pytest tests/ -q
```
CI 最低要求：`test_prompts.py` / `test_parser.py` / `test_training_cutoff.py` / `test_llm_no_browsing.py` 四项必须绿灯（核心语义 + 安全边界）。
