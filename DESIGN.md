# Forecast Evaluation — 设计理念详解

> 本文档面向第一次接触本项目、想理解"为什么这么做"而非"具体怎么做"的读者。
> 所有具体接口、字段定义、参数列表请配合 `FRAME.md` 阅读，本文聚焦在**设计取舍背后的动机**。

---

## 0. 写在最前：本项目想回答的问题

> **如果不让 LLM 看见事件解决之后的信息，它在 319 道真实预测题上的能力到底有多强？**

围绕这一句，项目给自己加了三条几乎"宗教式"的硬约束：

1. **信息边界必须严格**：模型只能通过我们写的 `web_search` 工具获取外部信息，且工具只能搜到事件解决日期之前的内容。
2. **结果必须可复现**：同一份数据集 + 同一份配置 + 同一组模型 → 任何人都能复跑出可比的数据。
3. **过程必须可复盘**：每一次模型调用、每一次搜索、每一次解析都要能事后追溯到字段级。

整套设计——从 prompt 拼接、Tool schema、DB schema 到 resume 语义——都是这三条约束的具体落地。理解这一点之后，下面所有"看起来过度严格"的设计都会变得自然。

---

## 1. 信息隔离：项目的"第一性原理"

### 1.1 LLM 永远看不到 `end_date`

`web_search` 工具向模型暴露的 schema 只有一个 `query` 参数。
真正调 Tavily 时，`end_date` 由 Tool 实现层从当前题目的 `end_time` **硬编码注入**，模型既感知不到、也无法绕过。

这背后的设计哲学有两层：

* **能力边界与 Tool 边界对齐**：能力（"知道某天之前的世界")是由系统配置决定的，不应该是 prompt 工程或模型选择行为能影响到的；让模型连"我被截断到哪天"都看不到，才能避免它通过 prompt 构造、参数注入等方式推断或绕过。
* **failure mode 唯一可控**：如果把 `end_date` 暴露成 LLM 工具参数，就必须假设它会"忘记填"或者"故意填一个未来日期"。把决定权握在工具实现手里，failure mode 从"模型可能犯错"压缩为"我们的代码可能犯错"——后者是可测、可审计、可单元覆盖的。

### 1.2 默认偏严格：`end_date = end_time - 1 day`

`TAVILY_END_DATE_OFFSET_DAYS=-1` 是项目默认值。理由很直接：很多题（赛事、央行决议、奥斯卡提名）当天就会出结果，如果用 `end_time` 当天作为搜索截止，新闻摘要里很可能已经包含答案。把搜索时间往前挪一天，是**用一点信息粒度换严格性**。

报表也都默认在 `-1` 下比较——这本身就是设计上的约束："不同 offset 下的数字不可直接对照"。

### 1.3 强制禁用 provider-native browsing

OpenRouter / OpenAI / Anthropic 都有自己的 web 工具或 `:online` 后缀。一旦走那条路，时间截断就完全失控。
项目在两层强约束：

* **代码层**：`llm.chat` 只挂自家定义的 `WEB_SEARCH_SCHEMA`，禁止任何 `plugins` / `:online` / provider-native retrieval。
* **测试层**：`test_llm_no_browsing.py` 直接 mock 客户端，断言 outbound payload 里没有这些字段。

设计哲学：**对外部工具的"诱惑"要在最早的阶段就被拒绝掉**。否则一旦在某个 release 中"为了方便"开了一次，整套数据的可比性就废了。

### 1.4 训练数据污染：用过滤而不是欺骗

工具截断管不到模型参数里已经记住的事实。对此项目采取了一个非常朴素的策略：

> 对每个模型声明它的 **训练截止日期**；如果题目 `end_time ≤ cutoff`，这道题对该模型直接跳过。

跳过的样本仍然会在 DB 里写一行 `error="skipped_training_cutoff"`：

* 报表能清楚展示"每个模型被剔除了多少题、剩多少题可比"。
* `resume` 不会重试这一行（与 `network` 这类瞬态错误区分开）。
* 不计入 `error rate by kind`——它不是失败，是**主动数据清洗**。

这背后有一个项目反复用到的设计准则：**"被剔除"和"失败"是两种语义，必须在数据层就分开**。如果只用一个布尔 `skipped` 字段，未来想做按 cutoff 的分层报表就会丢失信息。

### 1.5 我们能控制什么、控制不了什么

`FRAME.md §3.8` 里有一张"威胁模型"表，本质是一份**诚实自白**：

| 泄漏源                       | 是否可控 |
| ---------------------------- | -------- |
| Tavily 返回正文              | ✅       |
| Provider 内置 browsing       | ✅       |
| 模型参数记忆                 | ⚠️ 部分（靠 §1.4 缓解） |
| Tavily snippet 的未来泄漏     | ⚠️ 部分 |
| 题目本身含年份等时间线索     | ❌       |
| 训练后的外部知识回流         | ❌       |

设计哲学：**承认控制不了的部分，比假装能控制更重要**。不可控的部分会作为评测偏差的一部分被接受；可控的部分则用代码 + 测试守死。

---

## 2. 可复现性：每一次 run 都是一个独立时空

### 2.1 源数据库纳入 Git 管理

`forecast_eval_set_example.db` 直接进了仓库。它是评测的"金标准"示例数据集，必须随仓库分发；任何人都能用 `git clone` 拿到完全一致的 319 道题。文件名 (`SOURCE_DB`) 和内部题库表名 (`SOURCE_TABLE`，默认 `forecast_eval_set_example`) 都暴露成 `.env` 参数，自带数据集时改这两个变量即可，loader 在运行时把 `<SOURCE_TABLE>` 拼进 SQL 的 `FROM` 子句；表名在 Settings 阶段就被白名单 `[A-Za-z_][A-Za-z0-9_]*` 校验，杜绝注入。

每次 run 还会计算 `source_db_hash` 并写到 `run_meta`，配合 `metadata_hash` 与 `prompt_templates_hash`，构成"该 run 究竟基于哪份输入"的三段指纹。

### 2.2 每次 run 一个独立目录

```
runs/{run_id}/
  manifest.json          # run 级元信息
  db/{model_slug}.db     # 每个模型一个 sqlite
  analysis/              # 后置统计产物
  logs/{run_id}.log
```

设计选择有几个细节：

* **目录化而不是单库化**：早期的"`results.db` 单库"被替换掉了。原因是单库时多次 run 的边界全靠 `run_id` 字段切分，难以独立分发，也很容易在分析时混入其他 run 的数据。
* **`run_id` 默认 `YYYYMMDD-HHMMSS-xxxx`**：`ls` 天然按时间排序；同时是目录名，肉眼一眼能看出"什么时候跑的"。
* **`RUN_ID` 留空 → 新建；填相同的 → 续跑**。一个变量管两件事，CLI 不再多一个 `--resume` flag。

### 2.3 每个模型一个 SQLite

为什么不是"一个 run 一个大 DB"？三条理由，按重要性排序：

1. **可独立分发**：把 `runs/{run_id}/db/openai__gpt-5.db` 单独发给别人，对方就能复盘这一个模型，不需要拿到其他模型的结果。
2. **写入路径互不干扰**：每个模型一个 async writer task，单 writer 多 reader 的 WAL 模式下并发足够，且不会因为某个模型卡住而阻塞别人。
3. **schema 演进容易隔离**：未来某个模型需要存一个特殊字段（比如 reasoning trace），可以单独扩它的 schema，不影响别人。

代价是分析层要扫多个文件，但 `analysis.py` 已经把这件事封装好了。

### 2.4 每个 DB 自包含 `questions` + `prompt_templates` 副本

每个模型 DB 内部都嵌入了源题库与 prompt 模板的副本。乍看冗余，实际是为"独立复盘"服务的：

> 拿到 `openai__gpt-5.db` 的人，不需要再去找 `forecast_eval_set_example.db`，也不需要找当时 metadata 是哪一版——所有评测当时的输入都在这个 DB 里。

副本之间的一致性靠 hash 校验保证：`run_meta.source_db_hash` / `metadata_hash` / `prompt_templates_hash` 三个字段把"当时的源数据"钉死。

### 2.5 配置脱敏后写进 DB

`run_meta.config_snapshot` 存的是脱敏后的 `.env` JSON。`LLM_API_KEY` 这种敏感字段只保留前 4 位 + 长度 + `sha256[:12]`。

设计哲学：

* **想知道这次 run 用了什么参数（temperature、并发数、retry 序列）**——存。
* **想知道用的是哪个 key 的明文**——绝不存。
* 把"可审计"和"不可外泄"在同一个字段里同时满足。

---

## 3. 评分系统：朴素到极致

### 3.1 字母集合 frozenset 严格相等

整个判分逻辑就一行：

```python
predicted_letters == ground_truth_letters  # 都是 frozenset[str]
```

漏选、多选、顺序——都按"严格相等"算错。
这是项目最有"洁癖"的一处设计。

#### 为什么不用部分得分（Jaccard / F1）？

* **解释成本太高**：`pass@1=0.62` 比 `mean Jaccard=0.74` 直观得多；论文里写一个就够。
* **避免一题打半分掩盖问题**：如果一个模型每次都漏选一两个，平均得分仍然能 70+，但它本质上**没真正掌握**这道题。严格匹配把这种行为打成 0，强迫报表诚实。
* **统一三种题型的判分接口**：`yes_no` / `binary_named` / `multiple_choice` 在评分阶段都是 frozenset；写一次代码，三种题型都对。

#### 字母编码是 canonical answer

源数据库的 `answer` 字段统一用字母（`'A'` 或 `'A, B'`），而不是用选项文本：

* yes_no：`Yes=A, No=B`
* binary_named：第一个实体=A，第二个=B
* multiple_choice：按选项数组顺序 A/B/C/...

模型输出的形态因 question_type 而异（`Yes`、实体名、字母列表都可能），但 parser 一律归一为 `frozenset[str]`。这个设计让"模型怎么说"和"系统怎么打分"彻底解耦。

### 3.2 `parse_ok=0` 不是 error

LLM 没输出 `\boxed{...}`、或写了"I cannot predict the future"——**这不是系统失败，是模型能力的一部分**。

具体落地：

* **`parse_ok=0`, `correct=NULL`**：parse 失败/拒答会单独累计成"refusal rate"，进入报表。
* **不走 retry**：模型自己说不会答，再问一次结果还是不会——退避重试只是浪费 token。
* **`error` 字段保持 NULL**：`error rate by kind` 报表里不会被这种"软失败"污染。

设计哲学：**每一种行为都要在报表里有自己的格子**。"系统错误率"和"模型拒答率"是两件事，不能用一个总错误率混在一起。

### 3.3 ASCII 续接：不完美但保持映射的稳定

数据库里有 4 道 `multiple_choice` 选项数 > 26，且其中 3 道答案落在 `[ \ ] ^ _ ` ` ` a b c ...` 这种 ASCII 续接字符上。

这些字符对 LLM 极不友好（反引号会被 markdown 吞、大小写 a/A 混淆），但项目仍然保留——为了**保持字母 ↔ index 的一一映射**。

代价靠几道防线兜住：

1. `prompts.render_user_prompt` 在生成 >26 选项 `outcomes_block` 时显式加引号或转义。
2. `parser.parse_answer` 必须对 >26 选项做 round-trip 单元测试。
3. 日志/报表并行记录 letters 与 labels，便于人工复核。

未来如果发现 LLM 表现明显被标签拖累，再迁移到 `AA/AB` 或 `A01/A02` 等稳定方案。**这是一个被记录在案的"债务"，而不是一个被忽略的 bug**。

---

## 4. ReAct + Tool Use：把模型推理过程"摊开"

### 4.1 整段 prompt 作为单条 user message

模板就是一整块完整的 prompt（agent_role + event + outcomes + format + guidance），项目选择把它**整体作为单条 user message**喂进去，不再拆 system / user。

理由：

* **最忠实地复现源 metadata 的模板**：源数据 `dataset_metadata.features_json.prompt_reconstruction` 给的是一整块字符串，硬拆出 system 部分会丢失原始拼接的语义。
* **跨模型一致**：不同 provider 对 system 的处理不一样（OpenAI 会强 cache，Anthropic 有独立字段）。统一走 user 消息能拿到最稳定的可比性。
* **易于 hash 与对照**：整段 prompt 的内容直接写进 `user_prompt` 字段，未来任何模板变更都能通过 hash 一目了然。

### 4.2 ReAct loop 的硬天花板

每个 sample 有两道闸门：

* `REACT_MAX_STEPS=12`：LLM 总共最多与系统交互 12 轮（启用反思协议或 nudge 后比单步直答多 2-4 轮，因此默认略高于历史值）。
* `REACT_MAX_SEARCH_CALLS=8`：累计 8 次 `web_search` 之后，工具直接返回 `search budget exceeded` 给 LLM。

设计哲学是**用预算定义"模型自主搜索"的上限**：

* 没有上限，恶意/退化的模型可能调到天荒地老，把 API 账单烧穿。
* 有上限的同时返回错误而不是抛异常，让 LLM 还能基于已有信息给出一个"尽力答案"，把"超预算"和"系统挂了"两件事区分开。

超步数没给出 boxed 答案 → `parse_ok=0`，与拒答同样处理（§3.2）。

### 4.2.1 反思协议：用 prompt 而不是用规则把模型拉离"一步直答"

观察发现，部分模型平均只发 ~1.6 次搜索就给最终答案——这种"自信式一发"在面对长尾事件时会大幅拉低 `pass@1`。项目的回应分两层：

* **首选：反思协议（`REACT_REFLECTION_PROTOCOL=true`，默认开）。** 在每条 sample 的 user message 末尾追加一段 *Forecasting Protocol*：拆题 → 列 ≥3 个不同检索角度 → 每次搜索后反思 → 交叉验证 → 反方向自检 → 给出置信度。这段附录**不写入 `dataset_metadata`**，因此 `prompt_templates_hash` 不变；它的存在通过 `run_meta.config_snapshot` 与每条 sample 的 `user_prompt` 字段一起持久化，可事后逐题对照。
* **兜底：软性最低搜索次数（`REACT_MIN_SEARCH_CALLS`，默认 `0`=关）。** 当 LLM 在搜索次数不足时仍试图给最终答案，系统注入一条 user nudge 让它换个角度再搜一次；同一 sample 的 nudge 次数受 `REACT_MAX_NUDGES` 上限保护，整体仍受 `REACT_MAX_STEPS` 硬天花板约束。`ENABLE_WEB_SEARCH=false` 时 nudge 自动失效（无搜索可做）。

设计哲学：

* **优先用 prompt，不用规则。** 反思协议是"指导"，nudge 才是"限制"。先尝试用更好的指导让模型自发地多走几步，只有当模型仍然一意孤行时才上软性下限——避免把"评测的能力"和"系统的强制"搅在一起。
* **可关、可对照。** 两个开关都默认值清晰，关闭后行为退化为旧版（同一份代码可以同时跑"开协议 vs 关协议"的对照实验）。
* **可复盘。** 协议文本与 nudge 都会出现在 `messages_trace` 里；启停由 `config_snapshot` 锚定，**不是隐式行为**。

### 4.3 工具调用错误的优雅降级

ReAct 循环中，几种工具相关错误都不会中断整个 sample：

| 情况                      | 处理                                                        |
| ------------------------- | ----------------------------------------------------------- |
| 未知 tool name            | 给 LLM 返回 `unknown tool`，让它自己换思路                  |
| `arguments` JSON 解析失败 | 把错误信息发回去当 tool_result，LLM 可以重试                |
| 搜索预算用尽              | tool_result 返回 `search budget exceeded`                   |
| Tavily 自身报错           | 走 `SEARCH_BACKOFF_S` 重试，用完仍失败 → 把错误塞进 tool_result |

设计哲学：**让 LLM 在系统视角下"看到"自己的失败，而不是替它兜底**。这样统计出来的能力数据更接近真实——一个不会处理工具失败的模型，本来就该被打分低一些。

### 4.4 推理模型的"禁词"

某些推理模型（o1 / o3 / r1 / qwq…）对 `temperature` / `top_p` 这种自定义采样参数会直接报 400。
项目在 `.env` 里维护 `LLM_REASONING_MODEL_PATTERNS=o1,o3,o4,r1,qwq` 子串列表；命中的模型在调用时**不传**这两个参数。

这是一个典型的"维护成本前置"设计：与其在 retry / error handling 里去识别 400，不如在请求构造前就把它处理掉。

---

## 5. 数据存储：宽表 + 单 writer + 后置分析

### 5.1 为什么是宽表

每道题一行，每个 sample 一组 `s{i}_*` 列（v3 起 20 个字段：原 14 个 + 新增 6 个观测列）。
对比"长表 + (question_id, sample_idx) 复合主键"，宽表的优势：

* **resume 查询天然简单**：`SELECT question_id WHERE s{i}_created_at IS NOT NULL` 直接扫一列，不需要 group by。
* **单行原子读**：分析脚本读一行，所有 sample 全在；不需要 join 或聚合。
* **schema 决定 N**：建表时就把 `SAMPLING_N` 钉死，后续无论何时打开 DB，结构都和当时一致。

代价：`SAMPLING_N` 必须在 run 开始前确定，不能中途扩；schema 也需要"动态生成 20 × N 列"。这个代价在**评测场景下是可接受的**——`SAMPLING_N` 本来就属于 run 配置的一部分，不应该在 run 过程中变。

#### 为什么 `step_metrics` 用 JSON 列而不是单独长表

ReAct 每轮的单步指标天然是 1-to-N（一个 sample → 多个 step），看起来很想拆出
`run_step_metrics(question_id, sample_idx, step, prompt, completion, ...)` 这种长表。
项目最终把它压成 `s{i}_step_metrics TEXT`（JSON 数组）有三条理由：

* **没有跨 step 的查询需求**：分析层永远是"按 sample 取整段轨迹然后处理"，从来不
  做 `SELECT * FROM steps WHERE finish_reason='length'` 这种行级聚合——所有过滤都
  落在 sample 粒度。把这种数据正规化进表，相当于为不存在的查询付索引/JOIN 成本。
* **保持单 writer per model 的简单性**：v3 把数据从 14 列扩到 20 列只是
  `ALTER TABLE ADD COLUMN`，零复杂度；如果改成长表，就要新增第二张表 + 第二条
  外键 + 第二条 INSERT 路径，writer 边界一下子从"一行 upsert"变成"多行事务"，
  和 §5.2 的"用编排消除竞争"原则冲突。
* **JSON 体量可控**：每 sample 的 step 数受 `REACT_MAX_STEPS`（默认 6）限制，单条
  JSON 通常 < 1 KB；v3 schema 在 SAMPLING_N=3 / 题量 ~100 下，DB 增量在 KB 级，
  WAL 完全吃得下。

代价：长表能做的"按 step 聚合"在这里只能在 Python 里 reload + parse。考虑到分析
脚本本来就是一次性脚本（`python -m forecast_eval.analysis`），这个代价可接受。

#### 为什么 `reflection_protocol_hash` 与 `prompt_templates_hash` 独立

直觉上，反思协议是 prompt 的一部分，似乎应该并入 `prompt_templates_hash`。但项目
故意把它单独拎出来，理由是**它们的变更节奏与解释维度不同**：

* `prompt_templates_hash` 反映的是"题目内容是怎么渲染给模型的"——题干、选项、
  指令、问题类型说明等的**模板**。模板一旦改动，所有题面都变了，这是一个粗粒度
  的 run 区分键。
* `reflection_protocol_hash` 反映的是"模型在 ReAct 主循环里被注入了哪段元认知
  指令"，本质上是**搜索行为先验的开关**。它的变体只有三个轴：开/关、文本是否被
  改、版本号。把它合进主模板 hash 会让"我只是关掉了反思"这种小变更看起来等价于
  "我重写了所有题模板"——丢失了对照实验的解释力。

把两个 hash 分离的好处：跨 run 跑 A/B 比较时，可以选择"只允许
`reflection_protocol_hash` 不同，其他全等"——这正是消融实验的常见诉求。同样地，
`reflection_protocol_text`（全文）也并存于 `run_meta`，便于在不依赖 prompts.py
源代码的情况下做事后比对（例如发布报告时对方拿到的是脱敏 DB 而不是 git repo）。

#### 为什么 `belief_protocol_hash` 也独立（v4）

v4 引入的 `BELIEF_PROTOCOL` 同样是 user message 末尾的尾部追加段（追加在
`REFLECTION_PROTOCOL` 之后），让 LLM 在 `\boxed{...}` 之前再输出一段
`<belief>{...}</belief>` 严格 JSON，承载概率族指标的原料。和反思协议**完全
平行**地处理：

* 协议正文 **不进** `dataset_metadata.prompt_reconstruction`、**不进**
  `prompt_templates_hash`；
* 在 `run_meta` 新增 `belief_protocol_text` / `belief_protocol_hash` 两列，
  与 `reflection_protocol_*` 同源同语义；
* `manifest.json` 顶层同时含 `reflection_protocol_hash` 与 `belief_protocol_hash`，
  让"不开 DB 也能 grep 协议指纹"的检索路径覆盖两个协议；
* 三个指纹（`prompt_templates_hash` / `reflection_protocol_hash` /
  `belief_protocol_hash`）相互**独立**：换协议不影响主模板哈希、开关一个协议不
  影响另一个协议的指纹，便于做 belief A/B、reflection A/B、模板 A/B 三路独立的
  消融实验。

belief 协议的解析路径（`parser.parse_belief`）也与 `parse_answer` 完全独立：
belief 解析失败 MUST NOT 影响 `parse_ok` / `correct` / `final_answer_letters`，
让 v3 那条已经验证过的 boxed 路径保持稳定，v4 只是**叠加**而非替换。

#### Phase 2 calibration 用 LOO 而非 holdout（v4）

`forecast_eval/analysis/calibration.py` 的 Platt scaling 与 temperature
scaling 都强制用 leave-one-out（每题 $q$ 的校准参数从 $\mathcal{Q}_t \setminus
\{q\}$ 学）。理由：

* **N 偏小**：本数据集 319 题、按 cell 分层后每个 cell 50-150 题量级。在
  这个量级上 holdout 划分会让校准参数高方差——同一 dataset 不同 split 出来
  的 (a, b) 差异能到 ±0.3，足以让 ECE 比较失去意义。LOO 把每题都用上、
  又没让该题污染自己的校准参数——这是论文 §C.11 的核心防过拟手段，本项目
  照搬。
* **算力够便宜**：Platt 用 IRLS 单次 ~10 次 Newton 迭代、每次 O(N) 算
  Hessian。LOO 是 N 次 refit，naive 实现总开销 O(N²) ≈ 100k 浮点操作，
  319 题 < 1s。Temperature 用黄金分割搜索 30 次评估、每次 O(N)，同样
  sub-second。
* **过拟哨兵已就位**：LOO 仍可能在小 cell 上过拟（特别是 multi 类的边角
  cell）。`ModelCalibrationReport.overfit_warning` 在 `cal BI - uncal BI > 5`
  时返回 True，`per_model_summary.md` 把模型名标 `cal*`——reviewer 一眼
  就能看出这一行的校准结果不可信。

`scipy` 没有引入：IRLS 与黄金分割搜索都在 ~30 行纯 Python 写完，无依赖
膨胀。如果未来 Phase 2.x 加 Dirichlet calibration 等需要更复杂数值方法的
变种，再视情况按 design.md Open Q1 引入 scipy。

### 5.2 单 writer per model + WAL

并发写入 SQLite 是经典坑。项目的策略：

* **每个模型 DB 一个 async writer task**：所有 worker 的结果通过 `asyncio.Queue` 发给该模型对应的 writer。
* **`PRAGMA journal_mode=WAL` + `synchronous=NORMAL` + `busy_timeout=5000`**：单 writer 多 reader 下吞吐充足，且崩溃恢复仍然安全。
* **批量提交**：每 `DB_COMMIT_BATCH=10` 条或 1 秒触发一次 flush。

这套设计的核心思想：**用编排消除竞争，而不是用锁解决竞争**。一旦确定每个 DB 只有一个 writer，并发问题就退化成普通的单线程批量插入。

### 5.3 DB 只存原始观测，聚合全在 `analysis/` 后置算

```
DB:        raw observations only
├── correct (bool, NULL)
├── parse_ok (bool)
├── tool_calls_count
├── react_steps
├── tokens / latency
└── error / created_at

analysis/: aggregations
├── pass@1
├── pass_any@N
├── majority_vote
├── parse_failure_rate
└── error_breakdown
```

这是项目最重要的一条架构决策之一。

#### 为什么不在 DB 里预聚合？

* **指标定义会演进**：今天 `pass@3` 是 "3 中 1 算对"，明天可能改成 "至少 3 个对"。如果聚合落了库，每次改定义都要回填历史。把所有指标推迟到分析层算，随时可重刷。
* **`analysis.py` 是纯函数**：输入 = `runs/{run_id}/db/*.db`，输出 = `analysis/*.csv|md|json`。可以独立 `python -m forecast_eval.analysis` 重跑。
* **DB 与论文/报表解耦**：原始记录是工程产出，统计是产品/学术产出，二者节奏完全不同。

#### `pass@k` 命名的重新校准

业界 `pass@k` 一般指 "k 次里至少一次对"。项目早期用了 `pass@3 = sum(correct)≥3`（阈值口径），引发歧义。现在明确：

* `pass_any@N` ≡ 业界 `pass@k`：N 次里至少一次对
* `at_least_k_correct@N`：N 次里至少 k 次对（阈值分析）
* `pass@1 avg`：N 次里平均的正确率（稳定能力）
* `majority vote correct`：N 次的 frozenset 多数投票后是否正确（self-consistency）

设计哲学：**命名要么不歧义，要么显式声明自己的口径**。

---

## 6. 错误处理：把"失败"切成 8 种语义

错误处理表是 `FRAME.md §9`，但精神可以浓缩成几条原则。

### 6.1 不是所有错误都该重试

| 错误         | 重试？               | 理由                              |
| ------------ | -------------------- | --------------------------------- |
| Network/5xx  | 是（按退避序列）     | 多半是瞬态                        |
| Rate limit   | 是（优先 Retry-After）| Provider 已经告诉你等多久         |
| Auth 401/403 | **整个 run 停止**    | Key 错了重试无意义，越早停越省钱  |
| Bad request  | 否                   | model_not_found 这种改了配置才能跑 |
| Content policy | 否                 | 同样 prompt 再发一次结果一样      |
| Refusal / parse fail | 否           | 不是错误，是模型行为              |
| Tavily 自身 | 单独有 retry 序列    | 用完后把错误返回给 LLM            |
| 训练截止过滤 | 不调用              | 直接写 `skipped_training_cutoff`  |

### 6.2 三条独立退避序列

```
LLM_BACKOFF_NETWORK_S=2,5,15,30,60
LLM_BACKOFF_RATE_LIMIT_S=10,30,60,120,300
LLM_BACKOFF_SERVER_5XX_S=5,15,30,60,120
```

不同错误类型走不同退避——rate limit 比 network 慢得多，因为前者通常需要分钟级冷却，后者多半几秒就好。退避序列长度同时决定了"最大重试次数"，配置统一在 `.env`。

### 6.3 错误分类码是报表的一等公民

`error` 字段不是"出错就填 string"，而是固定的有限分类：
`network` / `server_5xx` / `bad_request` / `content_policy` / `skipped_training_cutoff`

`error_breakdown.csv` 直接按这个分类切片。设计哲学：**所有失败行为都要能在报表里被分类汇总**——一个 `error="something went wrong"` 是没用的。

---

## 7. 配置即合约：`.env` 是单一事实来源

### 7.1 几乎所有可调项都在 `.env`

CLI 只暴露 `--question-type` / `--choice-type` / `--skip-analysis` 三个 flag，其余全部走 `.env`。

理由：

* **复跑容易**：`.env` 一份就能完整复现配置；CLI flag 散在 shell history 里很容易丢。
* **CI/调度兼容**：脚本执行时通常更愿意管文件而不是命令行。
* **配置 ↔ DB 自洽**：`config_snapshot` 写进 `run_meta`，未来重看一次 run 就能知道"它当时的 `.env` 长什么样"（脱敏后）。

### 7.2 OpenAI-compatible 端点：水平兼容

`LLM_BASE_URL` 接受任何 OpenAI-compatible endpoint：OpenRouter、阿里百炼、OpenAI、DeepSeek、SiliconFlow、本地 vLLM 都行。

设计哲学：**对接面要小、要标准**。OpenAI 的 chat completion + function calling 协议已经成了事实标准，本项目不去做 provider 适配层，而是把适配的责任外推给 endpoint。

### 7.3 训练截止配置就是质量配置

`MODEL_TRAINING_CUTOFFS=openai/gpt-5=2024-10-01,...` 不是可选项，是**评测公平性的一部分**。文档中明确建议每个参评模型都显式声明，未声明则不过滤（带警示）。

---

## 8. 测试：把昂贵的失败前置到便宜的本地

### 8.1 测试不能联网，不能烧 API

319 题 × 模型数 × N samples 的完整 run 是**几十到几百美金的事情**。一旦在那种规模上踩到 prompt / parser / schema bug，钱就白花了。

测试设计的核心约束：

* `tavily-python` 不能真的发请求 → `respx` mock httpx
* OpenAI 客户端不能真的发请求 → fixture 替换
* SQLite 走临时目录 → `tmp_path` fixture
* 数据集要小但要"长得像真的" → 用源数据库里真实的几道题做 fixture

### 8.2 五条 CI 红线

```
test_prompts / test_parser / test_training_cutoff /
test_llm_no_browsing / test_analysis
```

这五条必须始终绿。它们覆盖的是项目最容易"悄悄坏掉"的部分：

| 测试                         | 守护的不变量                                      |
| ---------------------------- | ------------------------------------------------- |
| `test_prompts`               | prompt 模板渲染对三种 question_type 都正确        |
| `test_parser`                | 字母解析与严格相等判分                            |
| `test_training_cutoff`       | 训练截止过滤的语义和 resume 优先级                |
| `test_llm_no_browsing`       | provider-native browsing 永远不会被偷偷打开       |
| `test_analysis`              | 报表数值与原始 DB 对得上                          |

设计哲学：**把"破坏会很贵"的不变量挑出来，用单测当哨兵**。

### 8.3 dry-run smoke test

`test_smoke_dry_run.py` 用 httpx stub 替换 OpenRouter + Tavily，跑 3 题 × 1 模型 × 1 sample 的端到端流程。它不验证逻辑细节，验证的是"管道还通不通"——schema、宽表、`messages_trace` JSON、`search_calls` 字段是否都齐。

这是把 e2e 与 unit 测试拆开的体现：单测验证"局部正确"，smoke 验证"集成不爆"。

---

## 9. 可观测性：让每一次 sample 都可追溯

### 9.1 进度日志

```
12:03:44 | INFO | [run=20260424-120344-a7k3] [5/1610] q=69566c13 qt=binary_named ct=single model=openai/gpt-5 sample=2/5 correct=True steps=4 tool_calls=3 latency=8421ms
```

每条日志都带：question id、question_type、choice_type、model、sample_idx、是否正确、步数、工具调用次数、延迟。

设计哲学：**一条日志能完整描述一次 sample 在系统里走了什么路径**。看 log 等于看 trace，不需要再去 DB join 一遍。

### 9.2 `messages_trace` 与 `search_calls`

DB 里直接落两个 JSON：

* `messages_trace`：完整的 ReAct 消息序列（LLM 回复、tool_call、tool_result）。
* `search_calls`：每次 `web_search` 调用的 query、end_date、结果数、各结果的 published_date。

体积大（占 DB 的 ~80%），所以提供 `WRITE_MESSAGES_TRACE=false` 开关。
但默认开着——理由：**调试一次失败的复盘价值远高于多占的几十 MB 磁盘**。

### 9.3 `loguru` + 结构化 + 双输出

stderr（人看）+ rotating file（机器看）双通道；rotation 100 MB / retention 5。
设计哲学：**人和机器读 log 的需求不一样，分开伺候**。

---

## 10. 演进路径：openspec 驱动的 spec-first 变更

项目根目录有 `openspec/changes/`，里面用 spec 形式记录变更。`bootstrap-forecast-eval` 是初始 bootstrap 记录。

设计哲学：

* **变更先写 spec、再写代码**：避免"代码 merge 后才发现设计有问题"。
* **变更档案与代码 diff 并存**：将来回看一次架构演进时，能看到"为什么改"，而不只是"改了什么"。

---

## 11. 设计一致性原则汇总

把全文的设计哲学浓缩成一组原则，放在最后供 review 时参照：

1. **隔离 > 信任**：能在 Tool 层管住的边界，绝不交给 prompt。
2. **诚实 > 漂亮**：威胁模型里管不住的部分，明文写在文档里。
3. **跳过 ≠ 失败**：主动剔除的样本独立分类，不污染错误率。
4. **原始 > 聚合**：DB 只存观测，统计推迟到 `analysis/` 算。
5. **严格 > 慷慨**：评分用 frozenset 严格相等；offset 默认偏严格 `-1`。
6. **可复现 > 方便**：源数据进 Git，每个 DB 自包含，hash 钉死指纹。
7. **可观测 > 优雅**：完整 messages_trace 默认开；进度日志一行一 sample。
8. **失败要分类**：错误用有限分类码，报表里每种都有自己的格子。
9. **配置即合约**：`.env` 一份决定一切，CLI flag 极少。
10. **测试守住贵的**：五条 CI 红线 + dry-run smoke，把昂贵的失败前置到本地。

---

## 12. 阅读路线图

如果你刚接触本项目，建议按这个顺序读：

1. `README.md` —— 5 分钟搞清楚这是什么、怎么跑。
2. 本文（`DESIGN.md`）—— 理解每一处取舍背后的动机。
3. `FRAME.md` —— 字段级、接口级、伪代码级的完整规范。
4. `forecast_eval/prompts.py` + `forecast_eval/parser.py` —— 评分核心，两文件几乎是项目的"心脏"。
5. `forecast_eval/runner.py` + `forecast_eval/react.py` —— 编排与 ReAct 循环。
6. `tests/` —— 看测试反向理解契约。
7. `openspec/changes/archive/` —— 想知道为什么变成今天这样，回这里看。

---

> **总结成一句话**：
> 这是一个"用工程纪律守住科学严谨"的项目——所有看似过度的约束（信息隔离、严格匹配、字母 canonical 编码、宽表 + 单 writer、错误分类、CI 红线）都是为了让最终报表里那个数字真的有意义。
