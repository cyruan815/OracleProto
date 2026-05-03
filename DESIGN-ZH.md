# OracleProto — 设计原理

> *本文档解释代码库为何呈现现在的样子。如需字段级别的机制（schema、签名、错误码、行号），
> 请与 `FRAME.md` 配合阅读。建议的阅读顺序是 DESIGN → FRAME → 源码。本代码库中几乎每一个
> "看似怪异"的选择，都是两条框架级约束与一项经验观察相交的唯一不动点；脱离这一上下文阅
> 读源码，最容易把工程决策误读为过度设计或敷衍懒散。*

## 如何阅读本文档

每一节先给出它要消化的约束，再走通满足该约束的工程选择，最后陈述被否决的替代方案以及固
定该决策的文件路径。文件锚点采用相对于 `forecast_eval/` 的 `module.py:Lnnn` 形式，行号
跟随当前的 main 分支。

---

## 1. 问题与框架

### 1.1 OracleProto 要回答的问题

> 如果一个模型从不允许查看事件已经被裁决之后才发布的信息，那么在一个泄漏受控的数据集
> 上，它的原生预测能力到底有多强？

现有评测实践处于一个不稳定的中间地带，左右两个极端各以自己的方式失败。**前瞻式实时评
测**（如 ForecastBench、FutureX）只接收"提交预测时答案尚未存在"的事件，这是污染控制的
金标准，但排行榜是一条单向的时间流，记录在事件裁决后即消失，因此评测结果是不持久、不可
复用的。**回顾式评测**（如 FutureX-Past 或任何已裁决的实时问题归档）可审计、可比较，但
极易把"事实记忆"误判为"预测能力"；FutureX-Past 数据集卡片本身就提示，历史结局可能已经
进入更新模型的训练数据。

诊断侧的文献已在经验上证明，*伪装无知*与*真无知*存在系统性差异：以推理为优化目标的模型
尤其不擅长伪装；仅 1–5% 的标签噪声就足以击溃严格评分规则。BLF 从推理一侧得出相同结论，
即单次推理防线无法跨运行泛化，纪律必须落到数据集本身。

OracleProto 的回应是把纪律推进到数据集 schema 与运行单元
$`\mathcal{R}=(\mathcal{D}, M, \kappa_M, \delta, T, C, R, \Psi, \phi, \Gamma)`$。三条几乎
信仰式的硬性约束随之产生，本文其余每一条决策都是它们的下游。

1. **信息边界存在于数据集层与工具层，而非提示层。** 样本准入（$`\kappa_M \le \chi_i <
   \tau_i`$）是上游过滤，而非对模型的指令；工具层的时间掩码由工具实现注入，而非模型可填
   写的参数。模型可以提议查询，但无法改动 cutoff。
2. **结果是字节可复现的。** 一次 `git clone` 加上 `.env` 就足以让第三方重新运行评测并
   得到可比较的数字。源 DB 已被纳入仓库，六部分哈希指纹固定每一次运行所基于的输入。
3. **每一条我们能控制的泄漏路径都被控制了，每一条我们无法控制的都被声明了。** 威胁模
   型对自己无法修复的事坦诚相告。

一旦把这三条约束内化，本文后续每一个"看似过严"的选择都显得自然。

### 1.2 运行单元 $`\mathcal{R}`$ 作为契约

有一种心智模型把本文每一节串在一起：**$`\mathcal{R}`$ 是契约，不是配置**。两次运行是同一
次评测，当且仅当 $`\mathcal{R}`$ 的每个字段都匹配且每个指纹都匹配；否则严格地说，它们是
*两次不同的*评测，而不是同一次评测的更嘈杂的估计。

具体地，$`\delta`$ 不同的两次运行不可比较，因为可纳入性边界 $`\chi_i`$ 不同，可见的检索集合
$`\mathcal{T}_{\le\chi_i}`$ 也不同。同一模型 slug 下 $`\kappa_M`$ 不同的两次运行不可比较，
因为可纳入问题子集 $`\mathcal{D}^{\mathrm{pred}}_M = \{q_i : \kappa_M \le \chi_i <
\tau_i\}`$ 不同。`prompt_templates_hash` 不同的两次运行，作为对同一渲染器 $`R`$ 的评测，并
不可比较；它们评测的是两个不同的对象，唯一共享的标签是被测模型。

本代码库中的每个指纹、每个审计字段、每份配置快照，存在的目的都是让上述"不同评测"可观
测，避免报告把不等价契约下的数字平均到一起。

### 1.3 把 $`\mathcal{R}`$ 映射到代码库

$`\mathcal{R}`$ 的每个分量都精确映射到代码库中的一个对象，一旦这些对象固定下来，整条流水
线（样本准入、输入构造、工具掩码、输出解析、指标聚合）就拥有唯一的审计-重放路径。

| 符号               | 对象                                  | 实现                                                                              | 固定测试                    |
| ------------------ | ----------------------------------- | ------------------------------------------------------------------------------- | --------------------------- |
| $`\mathcal{D}`$      | 离散预测数据集                        | `SOURCE_DB` / `SOURCE_TABLE`；`loader.sync_questions`                            | `test_db.py`                |
| $`M`$                | 被测模型                              | `MODELS` 中的一个条目；`runs/{run_id}/db/` 下每个 $`M`$ 一个 SQLite 文件          | `test_runner_grid_model.py` |
| $`\kappa_M`$         | 知识截止                              | `MODEL_TRAINING_CUTOFFS[M]`（config.py:L224）；`runner.build_task_plan`           | `test_training_cutoff.py`   |
| $`\delta`$           | 时间掩码偏移                          | `TAVILY_END_DATE_OFFSET_DAYS`（默认 $`-1`$，config.py:L273）；`search.tavily_search` | `test_search.py`        |
| $`T`$                | ReAct 最大步数                        | `REACT_MAX_STEPS`（默认 12，config.py:L279）；`react.run_react` 循环上限          | `test_react.py`             |
| $`C`$                | 最大检索调用数                        | `REACT_MAX_SEARCH_CALLS`（默认 `[8]`，config.py:L283）；`react.py` 预算闸门        | `test_react.py`             |
| $`R`$                | 输入渲染器                            | `prompts.render_user_prompt`                                                    | `test_prompts.py`           |
| $`\Psi`$             | 输出解析与有效性                      | `parser.parse_answer`（parser.py:L40）                                          | `test_parser.py`           |
| $`\phi`$             | 答案归一化映射                        | 按 question_type 的字母编码 `A` / `A,B`；`parser.parse_gt`（parser.py:L92）      | `test_parser.py`           |
| $`\Gamma`$           | 聚合规则                              | `analysis/*`（综合准确率、FSS、$`\kappa`$、BI 等）                                 | `test_analysis.py`         |
| $`H_{\mathrm{aux}}`$ | 辅助泄漏探测器                        | `leak_filter.filter_search_result`；记录于 `run_meta.config_snapshot`           | `test_leak_filter.py`      |

框架有意把 $`H_{\mathrm{aux}}`$ 留在形式化元组之外，并通过 SHA-256 指纹绑定到运行元数据，
因为该探测器是支撑边界的可替换经验工程层，而不是预测系统本身的原始组件。代码库镜像了这
一区分：`MODELS` / `MODEL_TRAINING_CUTOFFS` / `REACT_MAX_*` 直接进入 `run_meta`，而
`LEAK_DETECTOR_*` 通过 `run_meta.config_snapshot.detector_*` 加 `run_meta.leak_detector_prompt_hash`
进入。

模型 $`M`$ 在问题 $`q_i`$ 上可见的信息为

$$\mathcal{I}_{i,M}^{\mathrm{vis}} = \mathcal{K}^{M}_{\le\kappa_M} \cup \mathcal{T}_{\le\chi_i},$$

其中 $`\mathcal{K}^{M}_{\le\kappa_M}`$ 是模型训练截止之前的参数化知识，
$`\mathcal{T}_{\le\chi_i}`$ 是经过时间掩码的外部信息。预测系统 $`F_M`$ 然后产出

$$\widehat{Y}_{i,M} = F_M(q_i^{\mathrm{in}}; \mathcal{I}_{i,M}^{\mathrm{vis}}), \quad \widehat{Y}_{i,M} \subseteq \mathcal{A}_i.$$

本代码库中的一切都是这条等式的执行者：要求 LLM 在受限信息下从有限候选集 $`\mathcal{A}_i`$
中作出选择，每一个工程决策都按"是否加强或削弱该边界"接受裁判。

### 1.4 框架未明确规定的部分

若干工程选择并非由 $`\mathcal{R}`$ 规定，在另一份替代实现中可以合理地不同。

| 工程选择                       | 是否由 $`\mathcal{R}`$ 规定 | 留给实现者的空间                                                                |
| ------------------------------ | ------------------------- | ----------------------------------------------------------------------------- |
| 存储后端                       | 否                        | 我们选择 SQLite（每个模型一个文件）；行存储如 Postgres 也可以                      |
| 检索后端                       | 否                        | 我们选择 Tavily；任何支持时间过滤的检索都允许                                     |
| 探测器模型                     | 否                        | 我们默认审计采用 `Qwen3.5-Flash`；任何足够严格的模型都可以                         |
| 并发模型                       | 否                        | 我们选择 `asyncio` 并为每个模型分配一个 writer 任务；线程或进程也可以               |
| 退避序列                       | 否                        | 三套序列在 `LLM_BACKOFF_*` 中，针对 OpenRouter 调优，并非框架固定                  |
| 日志栈                         | 否                        | 我们选择 `loguru`；任何结构化日志库都可以                                          |

这一列的决策会改变*这是哪一份实现*，但不会改变*这是哪一次评测*。$`R`$、$`\Psi`$、$`\phi`$、
$`\Gamma`$ 的指纹覆盖评测；探测器的指纹覆盖辅助泄漏屏障；其余皆为工程。我们显式记录这一区
分，因为它告诉你把 OracleProto 移植到不同技术栈时哪些旋钮可以安全替换。

---

## 2. 信息边界

边界沿三条受控通道加一条不可控残余进行组织。

| 泄漏来源                                       | 是否可控    | 缓解措施                                          | 是否审计                       |
| ---------------------------------------------- | ---------- | ------------------------------------------------ | ------------------------------ |
| Tavily 返回的内容（日期过滤）                  | 是         | 在工具层注入 $`\chi_i`$（§2.2–2.3）                | 是（§2.6）                     |
| 厂商原生 browsing                              | 是         | 代码与测试双重禁令（§2.4）                        | 是（`test_llm_no_browsing`）   |
| 模型参数化记忆                                 | 部分       | $`\kappa_M`$ 可纳入性过滤（§2.5）                   | 部分（model card 披露）         |
| 页面正文提及 $`\chi_i`$ 之后的事件               | 部分       | 第二阶段 LLM 探测器（§2.6）；审计后残余约 1.1%      | 是（§2.6 审计）                 |
| 问题文本本身的时间线索                         | 否         | 接受为评测偏差                                    | 否                             |
| 训练后流入的外部知识回流                       | 否         | 接受为评测偏差                                    | 否                             |

上表两行❌式条目是我们的主张未涵盖的残余来源。我们不假装它们不存在；任何针对 OracleProto
主张的攻击都应针对这两行，而非上面四行。§2.2 至 §2.6 按从"严格可执行"到"诚实声明"的顺
序镜像这些通道。

### 2.2 模型永远看不到 $`\chi_i`$

向 LLM 暴露的 `web_search` schema 仅有一个参数 `query`（tools.py:L7）。当 Tavily 实际被
调用时，$`\chi_i = \tau_i + \delta`$ 被工具实现硬编码并注入：偏移 $`\delta`$ 来自
`TAVILY_END_DATE_OFFSET_DAYS`（默认 $`-1`$ 天，config.py:L273），cutoff 由
`react._compute_end_date`（react.py:L39）计算，请求体由 `search._build_request_payload`
（search.py:L133）组装。模型既感知不到也绕不过它。

底下有两条设计哲学。**能力边界对齐工具边界。** "了解世界到某一日"的能力由系统配置决
定，不应受到提示工程或模型行为的影响。让模型连"我被截断在哪一天"都看不到，就阻止了它
通过提示构造或参数注入推断或绕过该边界。**失败单一且可控。** 倘若 `end_date` 作为工具参
数暴露，我们就必须假设模型可能忘记填写或故意填一个未来日期；把决策保留在工具实现内部，
失败模式从"模型可能犯错"塌缩为"我们的代码可能犯错"，后者可测、可审计、可单元测试。

一种自然的替代方案是把 `end_date` 暴露为工具参数，让模型对 cutoff 进行推理。我们拒绝
了，因为它要求信任模型不会扩大 cutoff；固定测试就必须对每一次发出的工具调用进行断言，
测试面从 $`O(1)`$ 膨胀到 $`O(N \cdot n)`$。第二种替代方案是改写 query 字符串以注入日期过滤，
但这把边界执行推入了一个脆弱的字符串处理环节，厂商可以忽略它，因为某些搜索引擎会悄悄丢
弃内联的 `before:2026-04-01` 操作符。

由 `test_search.py`（请求体始终包含从 `q.end_time` 派生的 `end_date`，从不来自任何 LLM
提供字段）和 `test_react.py`（LLM 在任何步骤看到的 schema 都没有日期参数）固定。

### 2.3 默认严格的原因：$`\delta = -1`$ 天

`TAVILY_END_DATE_OFFSET_DAYS = -1` 是项目默认值。示例数据集中许多问题（体育赛事、央行
决议、奥斯卡提名）当日裁决，把问题的 `end_time` 当成检索 cutoff 会浮出已经包含答案的新
闻摘要。把检索视野后撤一天，以微小的信息粒度损失换取严格性。

报告也默认在 $`\delta = -1`$ 下比较，这本身就是一项设计约束：不同偏移下的数字不可直接比
较，因为 $`\chi_i`$ 为每个 $`\delta`$ 取值定义了不同的可纳入信息状态（§1.2）。第二阶段探测
器在分类"泄漏 / 非泄漏"时锚定 $`\chi_i`$ 而非 $`\tau_i`$，正是因为审计定义必须与工具层实
际执行的运行 cutoff 一致；任何落在半开区间 $`(\chi_i, \tau_i]`$ 中的事实因此既被系统过
滤、又被审计归类为泄漏，从而消除了原本模糊的边界带。

我们拒绝了两种替代方案。$`\delta = 0`$ 时检索 cutoff 即问题裁决日；在示例 DB 上经验显
示，这能让大约 30–50% 的当日新闻泄漏漏网（取决于时区与事件类型），对我们想要的严格可
纳入性而言过于宽松，因此 $`\delta = +1`$ 仅作为消融旋钮存在。每问题 $`\delta_i`$ 可让"日内
事件"采用比"月度裁决事件"更严格的偏移。我们出于两点拒绝了它：第一，引入每问题旋钮违
反"一个 $`\delta`$ 定义一次评测"的契约；第二，把单个事件搞错的代价是非对称的：一次假阴
性泄漏远比一次假阳性的过严切割糟糕，单一保守默认胜过脆弱的每问题启发式。

### 2.4 强制禁用厂商原生 browsing

OpenRouter、OpenAI、Anthropic 各自暴露自己的 web 工具或 `:online` 后缀。一旦走上那条
路，时间 cutoff 就完全失控。项目在三层强制禁用，没有一层可以悄悄绕过。

在**启动层**，`Settings._post_validate`（config.py:L599）拒绝任何含 `:online` 或 `::`
的模型 slug，并在任何 LLM 或 Tavily 调用之前中止。在**单次调用层**，`llm.chat` 只附加
我们自己的 `WEB_SEARCH_SCHEMA`；kwargs 中任何 `plugins` / `:online` / 厂商原生检索关
键字都会被 `_assert_no_browsing`（llm.py:L74）截获，探测器路径通过
`_assert_detector_safe`（leak_filter.py:L139）复制此断言。在**测试层**，
`test_llm_no_browsing.py` 直接 mock 客户端，断言出站 payload 中不含上述任何字段，对主
LLM 与探测器路径都生效。

三联是有意为之。任何一层都足以阻止当下的回归，但只有三联能扛过绕过其中一层的重构。如
果只在启动时强制，测试 fixture 或调度器代码中通过 `model_copy(update={...})` 的局部
配置漂移就能绕过启动时校验；在 `llm.chat` 发送时复检防的正是这种失败模式。如果把强制
退化为告警而非拒绝，告警最终会被日志等级、配置模板或 CI 噪声过滤掉。拒绝会停止运行，
不可被悄悄忽略。

### 2.5 参数化记忆：过滤而不撒谎

工具 cutoff 无法约束模型已经在参数中记忆的事实。项目采取一个非常朴素的策略：声明每个
模型的训练截止 $`\kappa_M`$，若问题的 $`\tau_i \le \kappa_M`$（在 $`\delta = -1`$、按日取整
的时间戳下等价于 $`\chi_i < \kappa_M`$），则该模型直接跳过该问题。

被跳过的样本仍会写入一行 `error="skipped_training_cutoff"`，由 `_skipped_cutoff_row`
（runner.py:L94）生成、由 runner 主循环（runner.py:L181）持久化。把跳过显式保留有三项
属性。报告可以清晰地展示每个模型被过滤掉了多少题、还剩多少可比较的题。`resume` 不会重
试该行，从而把它与 `network` 等瞬态错误区分开。该行不计入按种类统计的错误率，因为它不
是失败，而是*主动的数据净化*。

这背后是项目反复援引的一条原则：**"被过滤"与"失败"是两种不同的语义，必须在数据层分
开。** 若用一个布尔 `skipped` 字段，未来按 cutoff 分层报告时就会丢信息。过滤在任务生成
阶段、即任何 LLM 调用之前应用，所以 cutoff 排除消耗零 API 预算；`test_training_cutoff.py`
通过断言 cutoff 跳过的样本上 `llm.chat` mock 一次也未被命中来固定这一点。

当 model card 仅以月级粒度披露 cutoff 时，推荐约定是把已披露月份的*最后一日*作为
$`\kappa_M`$，这是最保守的选择。代码库在 `_parse_training_cutoffs`（config.py:L181）中将
该日期加载为 `datetime.date`，故 `q_end <= cutoff` 的比较是与该约定对齐的、严格按日取整
的等值匹配。

我们考虑过两种替代方案。*加权排除*——靠近 cutoff 的样本被折扣而非剔除——增加了分析复杂
度，却没有消除底层的污染担忧；要么问题处于模型训练视野之内，要么不在，二元判定让可纳
入集 $`\mathcal{D}^{\mathrm{pred}}_M`$ 保持为 $`\mathcal{D}`$ 的一个干净子集。*数据集级跳
过*——任何模型不可纳入则全数据集丢弃该问题——会把 $`\mathcal{D}`$ 收缩为
$`\mathcal{D}^{\mathrm{pred}}_{\bigcap_m M}`$（所有模型的交集），在异质模型组上丢掉 10–
20% 的语料。每模型可纳入既保证比较公平，又不焚烧共享语料。

### 2.6 第二阶段 LLM 内容审计

§2.2 至 §2.5 的防线属于协议层（schema、`end_date` 注入、`:online` 禁令、cutoff 跳
过）。它们覆盖不到的一类泄漏是**Tavily 返回页面的正文里描述了 $`\chi_i`$ 之后发生的事件**：
Tavily 的 `end_date` 过滤作用于页面的爬取/索引时间，而非页面正文所描述事件的时间。一
份在 $`\chi_i`$ 之前已被索引的 wiki、聚合页或长文，其正文完全可以引用未来事件。Tavily-only
基线的残余泄漏率被观察到在个位到中两位百分比区间，足够高到使模型间个位百分点的准确率
差异在没有进一步过滤时变得统计上无意义。

`search-leak-filter-v1` 方案在 `tavily_search` 尾部、主 LLM 看到 `tool_result` 之前，
追加一道独立的 LLM 审计层（"探测器"）。每个 `SearchResultItem` 通过
`leak_filter.filter_search_result` 单独发送给探测器，裁决落在 `{keep, drop, failed:*}`
之中。裁决为 `drop` 的条目被完全移除；主 LLM 永远看不到被丢弃条目的任何字段，包括
`title`、`url`、`content`、`raw_content`。探测器裁决仅通过 `SearchResult.audit` 浮
出、由 `react._record_search_call` 消费，从不通过任何 LLM 可见的 payload 暴露。

| 维度               | 实现                                                                                                                  |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| 切入点             | `forecast_eval/search.py:tavily_search` 末尾、`return` 之前                                                              |
| 客户端             | `_detector_client: AsyncOpenAI`（leak_filter.py:L109），独立的模块级单例，不与主 LLM 共享                                |
| 输入字段           | 白名单：`title / url / published_date / content / raw_content / cutoff_date`；探测器禁止看到 `Question` 的任何字段        |
| 提示严格度         | `LEAK_DETECTOR_PROMPT_TEMPLATE`（leak_filter.py:L55）中的 6 条原则：cutoff_date 占位符、把具体/已排期/推测性未来事件等同看待、"模糊则丢弃"、禁用参数化知识、严格 JSON 输出、对问题无感知 |
| 参数               | temperature `0.0`、max_tokens `512`、timeout `60s`、并发 `5`（config.py:L345）                                            |
| 失败模式           | FAIL-RETRY → CLOSED：$`K`$ 次重试仍失败则丢弃；AUTH 错误本地捕获并立即丢弃，不向上传播也不中止整次运行                       |
| 可观测性           | `search_calls.detector_*` 五字段加 `run_meta.config_snapshot` 探测器三键指纹                                              |
| 总开关             | `ENABLE_SEARCH_LEAK_FILTER`（config.py:L337），默认 True；关闭时探测器路径在字节级被禁用，行为与无探测器路径完全一致         |

三个设计选择值得强调。

**探测器看不到问题。** 知道问题的探测器会蜕变为"答案审计员"，丢弃一切支持反方向论据的
内容，从而生成与具体问题相关的二阶泄漏。白名单强制了探测器的角色：分类事实的时间泄漏
（这一页是否提到 $`\chi_i`$ 之后的事件？），而非与答案的相关性。这一原理被编码为不可违
反的输入契约；由 `test_leak_filter.py` 断言探测器的 user 消息从不包含问题字段来固定。

**默认失败即丢弃。** 探测器抖动（超时、网络）与条目内容无关；把残余偏向"不确定即丢
弃"是保守选择。仅用于与未过滤基线对比的 A/B 应急口（`LEAK_DETECTOR_FAIL_ACTION=keep`）
存在，作为罕见情形下的逃生口。鉴权失败（401/403）跳过重试并立即丢弃，因为重试鉴权失
败既无意义又是计费深坑。

**独立的客户端单例。** 复用 `forecast_eval/llm.py:_client` 会让两套错误预算池耦合，主
LLM 重试会膨胀探测器配额并扰乱日志归因。两个单例、两套退避序列、两个日志命名空间。独
立客户端也允许探测器使用比被测模型*更强*的模型，这严格更可取，因为更强的探测器是流水
线中花额外能力最划算的位置。

探测器的参考日期是问题的 $`\chi_i = \tau_i + \delta`$，与 Tavily 的 `end_date` 同源，独立
于 §2.5 的 `MODEL_TRAINING_CUTOFFS`（即按模型索引的训练截止 $`\kappa_M`$）。即使一个问题
通过了可纳入性过滤进入执行，探测器仍会审计其检索结果；两道过滤不互相替代。

此前一项基于"日期过滤的检索后端 + LLM 泄漏分类器"的基线证明，运行时过滤可以捕获绝大
部分实际泄漏并把残余泄漏压到低个位数。第二阶段就是项目对"算法层 + 语义层双重保险"这一
工程实践的实例化。完整规约见
`openspec/changes/search-leak-filter-v1/specs/`（能力 `search-leak-filter`、
`search-tool`、`information-barrier`、`results-persistence`）。

---

## 3. 数据集重构

框架的核心概念动作是把一个已裁决事件 $`z_i`$ 重写为一道有时间界的预测：

$$(x_i, \mathcal{A}_i, Y_i, \tau_i, \rho_i) \quad \Rightarrow \quad q_i^{\mathrm{in}} = (x_i, \mathcal{A}_i, \chi_i, \rho_i).$$

真值 $`Y_i`$ 仅供评分使用；可见提示由结构化字段确定性地渲染。这一构造赋予数据集四项属
性，使评测对象成为数据集级而非事件级。*时间可复现*意味着同一源行加同一 $`\delta`$ 始终
得到同一 $`\chi_i`$。*模型相关的可纳入性*意味着可纳入问题集随 $`\kappa_M`$ 变化，而底层语
料共享。*可离散评分*意味着 $`\widehat{G}_{i,M}, G_i \subseteq \mathcal{L}_i`$ 是有限基数
集合而非自由文本。*跨年的审计可复现*意味着同一数据集可针对 cutoff 更晚的模型重放，且
可纳入集自动平移。

### 3.1 离散答案空间，而非自由生成

每个问题必须有一个有限答案空间 $`2 \le K_i < \infty`$ 与一个已核实答案集
$`Y_i \subseteq \mathcal{A}_i`$。这阻止开放生成绕过评测约束。三种问题类型 `yes_no`、
`binary_named`、`multiple_choice` 在评分层都坍缩为字母集，单选问题满足 $`|Y_i|=1`$，多
选问题满足 $`|Y_i| \ge 1`$。结构约束 $`\rho_i`$（单选 vs 多选）记录在 `choice_type` 中，由
解析器消费以校验模型输出基数是否合法。

我们拒绝了两种替代方案。由 LLM 评判的开放式自然语言输出会把第二个 LLM 引入评分路径并
重新带回污染担忧，因为评判器自己可能存在训练数据交集；严格字母集评分使评分路径确定、
可审计-重放且与评判器无关。仅由 Brier 或 NLL 评分的数值概率输出在 K=5 采样制度下会失
败，此时经验 $`\hat p`$ 只能取六个离散值，校准参数变得统计上无意义；可选的信念协议确实
采集概率，但作为字母集输出的伴随而非替代（§6.6）。

### 3.2 字母编码作为规范答案

源 DB 的 `answer` 字段统一使用字母（`'A'` 或 `'A, B'`）而非选项文本。对 `yes_no` 问
题，Yes 映射到 A、No 映射到 B。对 `binary_named` 问题，第一个实体映射到 A，第二个映
射到 B。对 `multiple_choice` 问题，A/B/C/… 按 `options` JSON 数组的顺序。

模型的输出形式因 question_type 而异（`Yes`、实体名或字母列表都可能），但
`parser.parse_answer`（parser.py:L40）一致地归一化为 `frozenset[str]`。这把*模型如何说*
与*系统如何评分*解耦：同一份评分代码覆盖三种问题类型，因为它们都归约为字母集上的集合相
等。由 `test_parser.py` 在每种类型的代表性输入上的往返用例固定，包括大小写变化与空白容
忍。

### 3.3 严格 frozenset 相等是评分原语

整个评分逻辑就是 `parser.py:L102` 一行：

```python
predicted_letters == ground_truth_letters  # 两侧都是 frozenset[str]
```

漏选、多选、顺序错乱：在严格相等下都被评为错。这是项目最一丝不苟的设计。

我们在严格层不使用 Jaccard、F1 或部分得分，原因有三。第一，*解释成本太高*：`pass@1=0.62`
比 `mean Jaccard=0.74` 直观得多，一个数字足以撑起一份报告。第二，*避免半分掩盖真问
题*：每次都漏选一两个的模型在软评分器上仍能均分 70+，根本没掌握题；严格匹配把这
种行为评为 0，强迫报告诚实。第三，*统一三种问题类型的评分接口*：三者在评分阶段都归约
为 frozenset 相等，代码写一次就能全部覆盖。

对多选问题，项目在严格相等之外配上两条软惩罚伴随，从不替换严格相等。

* **考试式部分得分**（`analysis/exam_score.py:L62`）遵循"任何 FP 一票否决为 0；否则
  得分为 $`|TP|/|G|`$"，即零-FP 闸门下的召回率。它使头条综合准确率在严格相等接近零方差
  的多选桶上更具层次。
* **格式技能分（FSS）**（`analysis/accuracy.py:L386`）是 $`(\alpha, \beta) = (2.0, 0.5)`$
  下的随机校正 Tversky 相似度，对假阳性的惩罚是假阴性的 4 倍。直觉是*声明事件会发生*
  比*漏报事件*更危险。单选问题退化为严格 0/1；非对称只在多选桶上生效。

两条软惩罚与严格相等共存，正是为了把指标选择留给分析者而非系统的偏置。审计轨迹
（`composite_meta.json`）确切地记录每个综合数值采用了哪些桶。

Tversky 不对称 $`(\alpha, \beta) = (2.0, 0.5)`$ 编码了一条预测域先验：在预测中，声明事件
会发生承担更多下游风险（被据以行动的虚假信号），而漏报事件承担机会成本。4× 的 FP-vs-FN
惩罚相当于对数尺度上一个八度，足以在多选桶上翻转跨模型排名，又足够保守以保留 $`\alpha =
\beta = 1`$（Jaccard）作为消融旋钮可恢复。该值通过 `tversky_score` 与 `tversky_baseline`
的 `alpha` 与 `beta` 关键字参数（accuracy.py:L289 与 L320）可配置，但分析流水线把
`(2.0, 0.5)` 默认值硬编码；修改它需要改代码而非改 `.env`，因为 FSS 的*解释*取决于这种
不对称在跨运行间被固定。

### 3.4 ASCII 续接标签：被记录的债务

当 `multiple_choice` 题携带超过 26 个选项时，编码会落在 ASCII 续接字符上，例如 `[`、
`\`、`]`、`^`、`_`、`` ` ``、`a`、`b`、`c`。这些字符对 LLM 极不友好（反引号会被
markdown 吞掉，小写与大写 `a`/`A` 容易混淆），但项目仍保留它们，因为这能保住字母 ↔
索引的一一映射。

成本由若干防线缓解：`prompts.render_user_prompt` 在为 > 26 个选项生成 `outcomes_block`
时显式引用或转义标签；`parser.parse_answer`（parser.py:L74）只迭代长度为 1 的
`tokens` 并使用 `letter_to_index` 往返校验，由 `test_parser.py` 在 > 26 个选项上的往
返用例固定；日志与报告并行记录字母与标签以供人工复核。如果未来确认标签方案确实拖
累 LLM 性能，我们将迁移到 `AA/AB` 或 `A01/A02` 等稳定方案。**这是被记录的债务，不是被
忽视的 bug。** 直接跳过 > 26 选项问题这一替代方案会在包含此类问题的任何自定义数据集上
丢失数据集级覆盖；我们更倾向往返测试加上选项稳定的编码。

### 3.5 解析失败不是错误

当 LLM 不输出 `\boxed{...}` 或写"我无法预测未来"时，这不是系统失败而是模型能力的一部
分。三项机械后果随之产生。

`parse_ok=0` 且 `correct=NULL`：解析失败与拒答分别累入"拒答率"并在报告中浮出。*不重
试*：当模型自己说它无法回答时，再问一次得到的还是同一答案，退避重试只会浪费 token。
`error` 字段保持 NULL：按种类的错误率报告不会被这种软失败污染。

同一耦合规则被编码为 4 状态矩阵；`exam_score`（exam_score.py:L62）以 7 行决策树实现该
矩阵，由 `test_exam_score.py` 与 `test_aggregation.py` 固定。原则是**每一种行为都必须
在报告中有自己的格子**：*系统错误率*与*模型拒答率*是两件不同的事，不能塞进单一总错误
率。

*解析失败时重试*这一替代方案双重失败。第一，能力遮蔽：拒答的模型是缺乏预测能力的模
型，重试只是粉饰。第二，成本：在 10K 量级评测上重试乘倍 API 开销，却不会改变总体均
值。

---

## 4. 分层评测

评测系统是

$$\mathcal{E}_M = (\mathcal{E}^{\mathrm{valid}}_M, \mathcal{E}^{\mathrm{item}}_M, \mathcal{E}^{\mathrm{question}}_M, \mathcal{E}^{\mathrm{model}}_M).$$

四层，每层独立语义，每层都从同一份归一化的离散答案空间计算。

| 层级                      | 对象                                                                                | 此处所驻信息                                          | 代码                                                |
| ------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------- | --------------------------------------------------- |
| **有效性**                | $`v_{i,M} = \mathbb{1}[\Psi_i(o_{i,M}) \ne \bot]`$                                    | parse_ok / parse_failure_rate                         | `parser.parse_answer` / `analysis/aggregation.py`   |
| **条目**                  | $`r_{i,M} = \mathbb{1}[\widehat{G}_{i,M} = G_i]`$ 单次试验                            | 严格相等 / 考试得分                                    | `parser.is_correct` / `analysis/exam_score.py`      |
| **题目**                  | $`\{\widehat{G}_{i,M}^{(s)}\}_{s=1}^{S}`$ 跨 $`S`$ 次试验                               | pass_any@N / pass_all@N / Fleiss $`\kappa`$ / VCI / MV  | `analysis/accuracy.py` / `analysis/consistency.py`  |
| **模型**                  | $`\Gamma(\{\mathcal{E}^{\mathrm{question}}_{i,M} \mid q_i \in \mathcal{D}^{\mathrm{pred}}_M\})`$ | 综合准确率 / FSS / BI / 每正确成本 | `analysis/composite.py` / `analysis/__init__.py`    |

每一层都被分析输出中独立的列族捕获，模型层 $`\Gamma`$ 的选择是分析者的杠杆而非系统的杠
杆。同一份原始观测同时支持平直均值、加权综合、配对自助 CI 和后验比较，因为 DB 中没有
任何东西被预聚合。

### 4.1 为什么 DB 不预聚合

三项后果使其成为项目最重要的架构决策之一。*指标定义会变化*：今天的 `pass@3` 意味"3 次
中 1 次算通过"，明天可能改为"至少 3 次正确"；若聚合落到 DB，每次重定义都需要回填，把
所有指标推到分析层意味着随时可重算。*`analysis.py` 是纯函数*，输入是
`runs/{run_id}/db/*.db`，输出是 `analysis/*.csv|md|json`，可以通过
`python -m forecast_eval.analysis` 独立重跑。*DB 与报告解耦*：原始记录是运维产物，统计
是分析产物，二者节奏完全不同。

这是框架层面"指标无关"设计的运维体现。由 `test_analysis.py` 通过手工 DB fixture 构造
并确认 `run_analysis` 既不回写也不变更来固定。

### 4.2 重新校准 `pass@k` 命名

在更广社区里，`pass@k` 一般指"$`k`$ 次中至少一次正确"。为避免与阈值语义
`pass@3 = sum(correct) ≥ 3` 混淆，项目采用显式命名：

* `pass_any@N` 是标准 `pass@k`：$`N`$ 次中至少一次正确。
* `at_least_k_correct@N`：$`N`$ 次中至少 $`k`$ 次正确（阈值分析）。
* `pass@1 avg`：跨 $`N`$ 的平均准确率（稳定能力）。
* `majority vote correct`：跨 $`N`$ 的多数票 frozenset 是否正确（自一致性）。

四列由 `analysis/accuracy.py::Aggregate.as_ordered_dict`（accuracy.py:L66）发出；它们
的定义恒等式 $`\mathrm{pass\_all} \le \mathrm{pass@1}_{\mathrm{avg}} \le
\mathrm{pass\_any}`$ 按构造成立，由 `test_aggregation.py` 断言。原则简单：一个名字要么
不歧义，要么必须显式声明语义。

### 4.3 题目级信号：稳定不等于正确

跨模型评测反复浮现一种富有教益的模式：两个模型在 $`\mathrm{pass\_any}@N`$（best-of-N 命
中天花板）上打平，却在 $`\mathrm{pass\_all}@N`$ 与 Fleiss' $`\kappa`$ 上急剧分歧；一个模
型可以在一致性上排名靠前，却在 $`\mathrm{pass}@1`$ 上排末位——也就是*一致地给出错误答
案*。

这正是题目级信号设计要暴露的诊断：高一致性不蕴含正确，而高 best-of-N 天花板可能来自
"三个不同答案中恰好一个命中"而非"每次都一致正确"。项目因此并排报告两个轴而非把它们坍
缩为一个。

Fleiss' $`\kappa`$ 实现（consistency.py:L176）覆盖单选问题的逐分层分解（每个 $`k_q`$ 分层
是自己的 $`\kappa`$，按题数加权）与多选问题的逐标签二元分解。`test_consistency.py` 在手
工票表上同时固定两种分解。

### 4.4 按子题型加权的综合准确率

`per_model_summary.csv` 报告平直混合均值以保证向后兼容。头条评分采用综合准确率：每
桶考试得分后按子型加权平均。两个维度独立计算，因为 `multiple_choice` 自身又包含
single 与 multi，二者非正交。

`question_type` 维度（yes_no / binary_named / multiple_choice）落在
`per_model_composite_by_question_type.csv`，`choice_type` 维度（single / multi）落在
`per_model_composite_by_choice_type.csv`。对每个 (model, dimension, metric)：

$$\text{composite}_m = \frac{\sum_{b \in B_{\text{valid}}} w_{m,b} \cdot v_{m,b}}{\sum_{b \in B_{\text{valid}}} w_{m,b}}.$$

缺失桶（切片不可用或权重为 0）被丢弃，剩余权重按比例归一化，*不*被视为 0。全部为
None 时综合也为 None。同一公式与重归一化规则出现在 `composite.py:L18`（公式）与
`composite.py:L77`（白名单加按指标覆盖解析）。

#### 默认权重：更难的题目区分度更高

| 维度             | 桶                | 默认权重 | 难度依据                                                       |
| --------------- | ----------------- | -------- | ----------------------------------------------------------- |
| `question_type` | `yes_no`          | 0.15     | $`k=2`$，瞎猜 50%，模型间几乎无区分度                              |
| `question_type` | `binary_named`    | 0.15     | $`k=2`$ 同上，加上实体识别                                        |
| `question_type` | `multiple_choice` | 0.70     | $`k=2..N`$ 跨度大，包含多选，区分度最高                            |
| `choice_type`   | `single`          | 0.40     | 整体更易（包含 yes_no 与 binary_named）                          |
| `choice_type`   | `multi`           | 0.60     | 真正的多选，几乎每个模型都吃力，区分度高                          |

默认值在 `config.py:L365`。原则一句话：**让能区分模型能力的桶贡献更多**。切到"我更关
心简单题"只需在 `.env` 中翻动 `COMPOSITE_WEIGHTS_QTYPE` 或 `COMPOSITE_WEIGHTS_CTYPE`。
这是一种有立场的默认而非中立默认；我们认为上述取向对绝大多数评测场景更合理，不同意者
通过覆盖一行 `.env` 解决。

两种诱人但错误的替代方案。*跨桶等权重*听起来中立其实不是，因为示例 DB 上经验问题型分
布大致为 $`\{\text{yes\_no}: 37/80, \text{binary}: 3/80, \text{mc}: 40/80\}`$；等权重会
让 yes_no 相对 mc 重复计入，并抹掉最具区分度的桶。*经验流行度权重*等价于平直无加权均
值，由于同样的区分度原因失败：当 50% 题接近随机基线时，按流行度加权综合会淹没承载信
号的桶。

#### 为什么考试视角下随机基线很重要

在考试视角下，多选多答桶的随机基线为 $`T^{\text{chance}}_q = 2^{-(k_q - m_q + 1)}`$，对
典型 $`(k_q, m_q)`$ 落在 $`[0.06, 0.25]`$。与之相比，严格相等基线 $`0.5^{k_q}`$ 在 $`k_q \ge 5`$
时基本为零。考试视角把多选答列在绝对值上拉到与单选桶同一量级，所以真正区分模型的多选答
信号不再被它的近零严格视图方差淹没。这就是为头条综合选 `exam_score_at_n_avg` 而非严格
$`\mathrm{pass@1}`$ 的核心理由。

#### 配置入口

`COMPOSITE_WEIGHTS_QTYPE` 与 `COMPOSITE_WEIGHTS_CTYPE`（config.py:L365 与 L372）保存全
局默认权重，由所有未被显式覆盖的指标共享。`COMPOSITE_WEIGHT_OVERRIDES_QTYPE` 与
`COMPOSITE_WEIGHT_OVERRIDES_CTYPE` 以 `"fss=yes_no=0.05,multiple_choice=0.95"` 形式
保存按指标的独立覆盖，多指标用分号分隔。拼写错误的指标名在分析阶段从 `compute_composite`
抛出，而非悄悄回退到默认；这是有意为之，因为当你配错时我们要确保你知道。启动时校验
（config.py:L515）要求桶名属于合法集合、权重 $`\ge 0`$、且至少有一个权重 $`> 0`$。

### 4.5 每正确成本作为 Pareto 轴

成本-效益标量为

$$C^{\text{per-correct}}_m = \frac{C^{\text{total}}_m}{|\mathcal{D}^{\text{eval}}| \cdot n \cdot \text{Composite\,Accuracy}_m}.$$

采用 OpenRouter 实际账单而非"公布单价 × token 用量"计算，绕开了诸如推理 token 是否
计费、提示缓存折扣如何记账、工具调用是否计费、不同 provider 路由价格是否不同等灰
区。平台账单是可被第三方核实的唯一财务事实。

除以**难度加权的名义正确数**而非除以原始正确数很重要，因为它把"贵但准"与"便宜但鲁
莽"的模型置于同一成本-效益尺度上，避免"低样本单价但高错误率"产生的虚假低成本错觉。语
义上，$`C^{\text{per-correct}}`$ 是"一美元买到多少难度加权的正确预测"的倒数。

实践中，(准确率, 每正确成本) 联合 Pareto 前沿是唯一有意义的对比面；仅按准确率排或仅按
成本排都具误导性。比领先者准确率落后零点几分却只花一小部分成本的模型并不是"更差"，它
在前沿上来自不同方向。

---

## 5. ReAct 循环与工具使用

### 5.1 整段提示作为单条 user 消息

模板是一整块提示（`agent_role + event + outcomes + format + guidance`）；项目选择以
单条 user 消息送入而非拆分为 system / user。三个理由驱动这一选择。

该选择**最忠实地复现源元数据模板**：源数据 `dataset_metadata.features_json.prompt_reconstruction`
是单一字符串，强行抽出 system 部分会丢掉原始拼接的语义。它**保留跨模型一致性**：不同
provider 处理 system 消息的方式不同（OpenAI 硬缓存它，Anthropic 用独立字段），统一走
user 消息得到最稳定的可比性，这正是框架对 $`R`$ 作为确定性渲染器的要求。它也**易于哈希
与 diff**，因为整段提示直接写入 `user_prompt` 字段，未来任何模板变化都能通过哈希一
眼看到。

拆分 system / user 这一替代方案丢失跨 provider 一致性；同一 $`R`$ 因 provider 不同而事
实上变成两个不同渲染器。

### 5.2 循环的硬上限

每个样本有两个闸。`REACT_MAX_STEPS = 12`（config.py:L279）是每样本 LLM-系统交互的最
大轮数；启用反思协议或推动会在单步直答之上再加 2–4 轮，所以默认值为两者都留出余量。
`REACT_MAX_SEARCH_CALLS = [8]`（config.py:L283）是每样本 `web_search` 的累计预算；
一旦用尽，工具直接向 LLM 返回 `search budget exceeded`。

通过预算定义模型自主搜索的上限，服务两个目的。无封顶时，恶意或退化模型可无限调用并烧
穿 API 账单。封顶但返回错误而非抛异常，让 LLM 仍能基于现有信息提供"尽力答案"，并把
"超预算"与"系统崩溃"分开。

超步数而未出 boxed 答案的处理与拒答相同：`parse_ok=0`（§3.5）。

代码库默认 $`C = 8`$ 对行为分析有意宽松。更紧的预算如 $`C = 4`$——理由是
$`R_{\mathrm{tav}} \cdot C = 5 \cdot 4 = 20`$，约两页 Google 检索结果——把配置推入有意
紧绷的区分制度；切换是一行 `.env` 覆盖。

### 5.3 反思协议把模型从"单次直答"中拉出

某些模型平均仅 ~1.6 次搜索后就给出最终答案；这种自信的单发行为在长尾事件上严重压低
`pass@1`。项目以三件套协议响应。

**反思协议**（`REACT_REFLECTION_PROTOCOL=true`，默认开，config.py:L288）在每条样本
user 消息末尾追加*Forecasting Protocol*：分解问题、列举 ≥3 条不同检索角度、每次搜索
后反思、交叉验证、检视反方向、声明置信度。**预算感知协议**
（`REACT_BUDGET_AWARENESS_PROTOCOL=true`，默认开，config.py:L313）在提示前部前置"总步
数 + 总搜索数"，使模型可整体规划并为发出 `\boxed{...}` 保留最后一步。**接近上限时强
制收尾**（`REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=true` 且 `LOOKAHEAD=2`，默认开，
config.py:L314 与 L315）在循环接近上限时主动注入 user 消息：倒数第二步软提醒，工具调
用仍允许；最后一步硬切换为空工具列表，强制纯文本输出 `\boxed{...}`。

这些协议追加*不写入* `dataset_metadata`，因此 `prompt_templates_hash` 不变；它们的存
在通过每条样本 `user_prompt` 字段旁的 `run_meta.config_snapshot` 持久化，事后可按问题
diff。

补充性的 `REACT_MIN_SEARCH_CALLS`（搜索次数软下限，默认 `0` 即关闭，config.py:L292）
是罕见情形下的回退方案：当仅靠提示引导无法把模型从单发直答中拉出时，系统注入一次 user
推动，要求它换角度再搜，每样本推动次数由 `REACT_MAX_NUDGES` 封顶。

设计哲学是**先提示，后规则**。协议家族是引导；推动是限制。我们先尝试更好的引导让模型
自发多走几步，只有在模型仍坚持时才施加软下限，以避免把"被评测的能力"与"系统的强制"混
在一起。所有开关都有清晰默认，关闭后退化为裸循环行为，所以同一份代码可以做"协议开 vs
关"的对照实验。协议文本与推动都出现在 `messages_trace` 中，开/关状态由 `config_snapshot`
锚定，因此这*不是*隐式行为。

默认硬下限 `REACT_MIN_SEARCH_CALLS` 这一替代方案会把能力（"模型搜得够吗？"）与强制
（"我们让它搜了"）混淆，这就是默认 0 让下限保持 opt-in、由反思协议驱动自然搜索深度的
原因。

### 5.4 四旋钮优先链

跨模型比较中，**`parse_failure_rate` 必须仅反映模型自身的格式失败，而非框架上游资源
耗尽。** 若不加干预，`REACT_MAX_SEARCH_CALLS` 用尽后 `web_search` schema 仍会向 LLM
暴露；模型会继续请求该工具直到撞上 `REACT_MAX_STEPS` 上限，`final_raw=""` 直接变成
`parse_ok=0`，把"工具饥荒"伪装成"格式失败"。

`react.run_react` 加入四个正交开关，优先决策逻辑在 react.py:L266，每个防御不同失败
模式。

| # | 开关                                      | 默认    | 作用                                                                       |
| - | ----------------------------------------- | ------- | -------------------------------------------------------------------------- |
| 1 | `REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT`     | True    | 最后一步 → `tools=[]` 加强制收尾文本（按 LOOKAHEAD 分级）                   |
| 2 | `REACT_BUDGET_EXCEEDED_DROP_TOOLS`        | True    | 一旦完成 $`\ge C`$ 次搜索 → 后续轮次接收 `tools=[]`（在循环内）                |
| 3 | `REACT_FINAL_ANSWER_RETRY`                | False   | 循环正常结束但 `final_raw=""` → 额外一次 `tools=[]` 的 LLM 调用             |
| 4 | `REACT_MIN_SEARCH_CALLS` / `MAX_NUDGES`   | 0 / 2   | Opt-in 软下限；低于下限时推动 user 消息                                      |

四者每轮都按**严格优先序**评估，被 react.py:L266 一字不漏地编码为
`Priority is (1) > (2) > (3) > (4)`。

1. **最后一步硬切换。** 通过 `(REACT_MAX_STEPS - step) <=
   REACT_FORCE_FINAL_ANSWER_LOOKAHEAD` 与 `remaining == 1`（react.py:L272）检测。压
   倒其余一切；模型本轮只能输出内容。
2. **倒数第二步软警告。** 同一 `force_final_active` 标志加 `remaining ∈ [2,
   LOOKAHEAD]`（react.py:L297）。工具仍暴露；警告文本在内部按搜索预算是否已用尽分
   支。
3. **预算耗尽承诺通知。** 当 `searches_done_now >= REACT_MAX_SEARCH_CALLS` 且
   `REACT_BUDGET_EXCEEDED_DROP_TOOLS=True`（react.py:L310）时一次性触发。一旦触发，无
   论其他分支是否触发，后续每轮都接收 `tools=[]`。
4. **续接提醒。** 优先级最低；上一轮是无 `\boxed{...}` 的内容且无其他需要触发
   （react.py:L320）。每步至多注入一次内联"Harness: step N complete"，由优先级链
   保证不会跨分支重复注入。

分析列 `final_answer_retry_rate` 落在 `per_model_summary.csv`，让分析者单独看到该回
退捕获了多少，并在必要时决定是否从 `pass_at_1` 分母扣减。`run_results` 中每条样本
slot 携带 `s{i}_final_answer_retry_used INTEGER`；缺该列的 DB 由 `init_schema` 在打开
时通过 ALTER ADD 自动补加（NULL 兼容）。

`REACT_FINAL_ANSWER_RETRY` 默认为 **False**，被开关 #1（接近上限的强制收尾，循环
内）压倒，作为可选的循环外应急回退保留。开启它代价是多一步 LLM（`react_steps + 1`），
但*不*计入 `nudges_used`，因为语义不同：推动关乎搜索深度，这关乎格式合规。

原则是**让样本未能落定的每个原因都可见、可分**。四旋钮正交是因为它们针对的失败模式正
交；把它们固定到优先链，防止两个开关互相打架。单一"框架救援"开关这一替代方案会丢消融
区分度：四个开关里有两个是纯救援（1 与 3），两个是纯塑形（2 与 4），合并就抹掉了 A/B
测试"哪个干预真正推动了 `parse_failure_rate`"的能力。

### 5.5 工具调用错误的优雅降级

ReAct 循环内若干工具相关错误不会中断整个样本。

| 情形                            | 处理                                                                            |
| ------------------------------ | ------------------------------------------------------------------------------- |
| 未知工具名                      | 向 LLM 返回 `unknown tool`，让它换思路                                           |
| `arguments` JSON 解析失败       | 把错误作为 `tool_result` 回送；LLM 可重试                                         |
| 搜索预算耗尽                    | `tool_result` 返回 `search budget exceeded`                                     |
| Tavily 自身报错                 | 走 `SEARCH_BACKOFF_S` 重试；仍失败则把错误塞入 `tool_result`                     |

原则是**让 LLM 从系统视角看到自己的失败，而非粉饰它**。这样产生的能力数字更接近真
实，因为不能处理工具失败的模型本就该得更低分。

### 5.6 推理模型参数排除

某些推理模型（`o1`、`o3`、`r1`、`qwq`）对 `temperature` 或 `top_p` 等自定义采样参数
直接返回 400。项目在 `.env` 中维护子串列表 `LLM_REASONING_MODEL_PATTERNS=o1,o3,o4,r1,qwq`
（默认值在 config.py:L241）；对匹配模型，调用时不传这两个参数。这是"把维护成本前移"
的设计：与其在重试或错误处理里识别 400，不如在请求构造时处理。

---

## 6. 存储与 writer

### 6.1 每运行目录与每模型 SQLite

```text
runs/{run_id}/
  manifest.json          # 运行级元数据
  db/{model_slug}.db     # 每模型一个 sqlite（grid 下每个虚拟 slug 一个文件）
  analysis/              # 后期统计产物
  logs/{run_id}.log
```

单一共享 `results.db` 的方案被否决。单 DB 时运行间边界将完全依赖 `run_id` 列，独立
分发困难，且容易把其他运行的数据混入分析。`run_id` 默认 `YYYYMMDD-HHMMSS-xxxx`，所以
`ls`
按时间自然排序，目录名一眼告诉你"何时跑的"。空 `RUN_ID` 启动新运行；同值续跑既有运
行，所以一个变量同时处理两种模式而无需额外 `--resume` CLI 标志。

我们选择每模型一个 SQLite 文件而非每运行一个，按重要性排序的三个理由。

*可独立分发。* 把 `runs/{run_id}/db/openai__gpt-5.db` 交给别人，他可以仅重放该模
型，无需获取其他模型的结果。*写路径互不干扰。* 每模型一个 async writer 任务，
single-writer-multi-reader WAL 模式提供充分并发，一个模型卡住不会阻塞另一个。
*Schema 扩展隔离。* 若某模型需要存特殊字段（如推理轨迹），其 schema 可独立扩展而不
影响其他。

代价是分析层必须扫多个文件，但 `analysis.py` 已封装该逻辑。"一个 DB 加 `model` 列"这
一替代方案在三项属性上都翻车；具体来说，单 writer 抢占会让一个慢 provider 在 WAL 下
持有同一 DB 级锁，造成全局停滞。

### 6.2 每个 DB 自含问题与 prompt_templates

每个模型 DB 内嵌源问题集与提示模板的副本。乍看冗余，但服务于*独立重放*：拿到
`openai__gpt-5.db` 的人无需追查 `forecast_eval_set_example.db`，也无需考证当时使用的
是哪个版本元数据，因为评测所需的每一项输入都在这一个 DB 里。

副本一致性由哈希校验保证：`run_meta.source_db_hash`、`metadata_hash`、
`prompt_templates_hash` 三字段固定"当时的源数据"。存路径不存副本这一替代方案破坏自
含性；路径一旦移动，DB 就不可重放。

### 6.3 宽表，N 在建表时固定

每题一行，每条样本一组 `s{i}_*` 列。schema 每条样本 24 字段，按功能分组为 14 项核心、
6 项可观测性、3 项信念与 1 项 final-answer-retry 列。与"长表加 (question_id,
sample_idx) 复合主键"相比，宽表有三项优势。

*Resume 查询自然简单。* `SELECT question_id WHERE s{i}_created_at IS NOT NULL` 仅扫一
列即可，无需 group-by。*单行原子读。* 分析脚本读一行就拥有所有样本，无需 join 或聚
合。*Schema 固定 $`N`$。* `SAMPLING_N` 在建表时固定，未来无论何时重新打开 DB，结构都与
当时一致。

代价是 `SAMPLING_N` 必须在运行前确定且不能中途扩展；schema 也需动态生成 `20 × N`
列。这一代价在评测场景可接受，因为 `SAMPLING_N` 本就是运行配置的一部分，不应中途变
化。长表加复合主键这一替代方案在可变 $`N`$ 上胜出，在简单 resume 查询上输，在无 JOIN
分析上输：长表更适合生产遥测，宽表更适合评测产物。

### 6.4 `step_metrics` 是 JSON 列而非独立长表

ReAct 的逐轮步指标天然是 1-对-N（一个样本产生多步），乍看应该拆成长表。项目最终把它
压缩为 `s{i}_step_metrics TEXT`（JSON 数组），三个理由。

*无跨步查询需求。* 分析层总是按样本取整条轨迹再处理，从不做 `SELECT * FROM steps
WHERE finish_reason='length'` 这样的行级聚合。每次过滤都在样本粒度，把这份数据归一化
入表意味着为不存在的查询付出索引/JOIN 成本。*保留每模型一个 writer 的简洁。* 改成长
表需要第二张表、第二个外键、第二条 INSERT 路径；writer 边界从"单行 upsert"跳到"多行
事务"，与 §6.5 的"靠编排消除竞态"原则冲突。*JSON 体积可控。* 每样本步数受
`REACT_MAX_STEPS` 限制（默认 12）；一份 JSON 通常 < 1 KB；在宽表 schema、
`SAMPLING_N=3`、~100 题的设置下 DB 增量在 KB 量级，WAL 轻松处理。

代价是长表本可做的步级聚合得在 Python 中重新加载 + 解析。由于分析脚本是一次性工具
（`python -m forecast_eval.analysis`），这一代价可接受。

### 6.5 每模型一个 writer 加 WAL

并发写 SQLite 是经典坑。项目策略：

*每模型 DB 一个 async writer 任务。* 每个 worker 的结果通过 `asyncio.Queue` 发往该模
型的 writer。*`PRAGMA journal_mode=WAL` 加 `synchronous=NORMAL` 加
`busy_timeout=5000`。* 在 single-writer-multi-reader 下吞吐充裕，崩溃恢复仍安全。*批
量 commit。* 每 `DB_COMMIT_BATCH=10` 条或每 1 秒刷一次。

核心思路是**靠编排消除竞态，不要用锁解决竞态**。一旦固定"每 DB 一个 writer"，并发问
题就退化为普通的单线程批量插入。每 INSERT 加锁这一替代方案能跑，但在抢占上烧 CPU 而
无摊销；编排路线把成本移到入队（便宜）而非写入（昂贵）。

### 6.6 DB 只存原始观测

```text
DB:        仅原始观测
├── correct (bool, NULL)
├── parse_ok (bool)
├── tool_calls_count
├── react_steps
├── tokens / latency
├── belief_final / belief_trace / belief_parse_ok  (当 BELIEF_PROTOCOL=true)
├── search_calls (开启泄漏过滤时附带探测器裁决)
└── error / created_at

analysis/: 聚合
├── pass@1 / pass_any@N / majority_vote
├── FSS / Cohen κ / Hamming / Fleiss κ
├── BI / NLL / MBS / ABI (概率族，存在信念数据时)
├── parse_failure_rate / error_breakdown
└── 每正确成本 / Pareto 前沿 / 配对自助
```

这是项目最重要的架构决策之一，也是框架层面"指标无关"设计的运维体现。

### 6.7 K=5 下概率族指标的降级

在 $`K = 5`$ 并行采样下，每个 (题, 标签) 的经验概率 $`\hat{p} = n / K`$ 只能取六个离散值
$`\{0, 0.2, 0.4, 0.6, 0.8, 1.0\}`$。这使得 Reliability Diagram、Murphy 三分解、
Platt-scaling LOO 数学正确但统计无意义。分析栈因此使用 $`K=5`$ 适宜的**离散原生**
指标族：BS / NLL / MBS / BI / ABI 以辅助列形式存在，附 `†` 脚注与
`per_model_summary.md` 中的 $`K`$ 免责声明。

`calibration.py` 不在分析栈中，对应的 5 项产物（`calibration_params.json`、
`per_model_summary_calibrated.csv`、`reliability_data*.json`、`brier_decomposition.csv`）
不会产出。离散族（FSS、Cohen $`\kappa`$、Hamming、Fleiss $`\kappa`$、平均熵、VCI、MVG）
是头条主线，配合 `entropy_accuracy_bins.csv` 与 `inter_trial_consistency.csv`。把
$`K`$ 提升至 $`\ge 30`$ 即可重新引入校准族。

这一决定**不是**框架层面的约束；它是由样本量统计驱动的*工程*选择。设置 $`K = 30`$
将重新启用概率线；降级出于分析约定而非硬编码闸门。

---

## 7. 可复现与审计

### 7.1 源数据库纳入 Git

`forecast_eval_set_example.db` 直接进入仓库。它是评测的金标准示例数据集，必须随仓库一
同发布；任何人 `git clone` 都能拿到完全相同的题目。文件名（`SOURCE_DB`）与内部题表名
（`SOURCE_TABLE`，默认 `forecast_eval_set_example`）都暴露为 `.env` 参数；用自定义数据
集时只需改这两个变量，loader 在运行时把 `<SOURCE_TABLE>` 拼入 SQL `FROM` 子句。表名在
Settings 阶段以 `^[A-Za-z_][A-Za-z0-9_]*$` 白名单校验（config.py:L586），堵死 SQL 注入，
因为这是我们唯一会被注入的位置，因此校验只需在此处进行。

把 DB 放在独立 registry 这一替代方案给复现增添了网络依赖；纳入 Git 让 `git clone` 成
为唯一的安装步骤。

### 7.2 六部分指纹固定输入

每次运行计算 `source_db_hash` 并写入 `run_meta`；连同 `metadata_hash`、
`prompt_templates_hash`，以及在适用时的 `reflection_protocol_hash`、
`belief_protocol_hash`、`leak_detector_prompt_hash`，构成"该次运行确切基于的输入"的
多部分指纹。三个核心哈希在 db.py:L385 计算；完整集合在 `evaluation.py` 中组装。

### 7.3 三个独立的协议指纹

`prompt_templates_hash`、`reflection_protocol_hash`、`belief_protocol_hash` 是**三个
互相独立**的 SHA-256 指纹，并列保存在 `run_meta` 与 manifest 中。这种独立是有意为之
且对消融研究承重，因为反思 A/B 配对自助分析要求"除一项哈希外完全相同"的不变量。

`prompt_templates_hash` 反映"问题内容如何渲染给模型"：题干、选项、指令、问题型描述等
模板。模板一变，每个问题文本都会变，因此这是粗粒度的运行区分键。由
`compute_prompt_templates_hash`（db.py:L397）计算。`reflection_protocol_hash` 反映
"在 ReAct 主循环中向模型注入了哪条元认知指令"，本质是搜索行为先验的开关。其变化只有
三轴：开/关、文本是否被改、版本号。`belief_protocol_hash` 反映"模型是否在
`\boxed{...}` 之前发出结构化信念向量"，是概率族指标是否被填入的开关。

三套独立哈希的好处是：跨运行 A/B 对比可以要求"仅 `reflection_protocol_hash` 不同，其
余相等"，这正是消融研究所要的。`analysis/behavior.py::find_paired_runs` 中的反思 A/B
配对*要求*这一"除一项哈希外完全相同"不变量，无关运行被自动过滤。

每条协议的全文与 `run_meta`（`reflection_protocol_text`、`belief_protocol_text`）共
存，便于不依赖 `prompts.py` 源码做事后 diff；例如发布报告时，接收者拿到的是经过编辑
的 DB 而非 git 仓库。

单一组合哈希这一替代方案丢失消融区分度：三者中任一变化都让一切"看起来不同"，再也无
法沿一轴配对运行。"探测器哈希加源 DB 哈希"的 6 路复合键也被考虑过并被否决；框架把
$`H_{\mathrm{aux}}`$ 留在 $`\mathcal{R}`$ 元组之外，正是因为探测器是辅助工程层，把它作为
独立轴（`leak_detector_prompt_hash`）携带，保留了消融层"严格相等除一轴外"的配对模
式。

### 7.4 配置快照编辑

`run_meta.config_snapshot` 通过 `db.snapshot_settings`（db.py:L429）以 JSON 存储编辑
后的 `.env`。`LLM_API_KEY` 等敏感字段仅保留前 4 字符加长度加 `sha256[:12]`；
`TAVILY_API_KEY` 现在是 `list[str]`，每个 key 独立编辑并持久化为
`[{prefix, sha256_12, length, provider}, ...]`。

编辑形态平衡两个相互冲突的需求。*想知道这次运行用了哪些参数？* 已存。*想知道 key 的
明文？* 永不存。要点是"可审计"与"不可泄漏"在同一字段中共存。由 `test_db.py` 通过
`snapshot_settings` 往返并断言前缀长度与摘要长度而非原始值来固定。

### 7.5 进度日志与 `messages_trace`

每条进度日志都携带题目 id、question_type、choice_type、模型、sample_idx、正确性、步
数、工具调用数、延迟：

```text
12:03:44 | INFO | [run=20260424-120344-a7k3] [5/1610] q=69566c13 qt=binary_named ct=single model=openai/gpt-5 sample=2/5 correct=True steps=4 tool_calls=3 latency=8421ms
```

单条日志应完整描述一个样本走过系统的路径，使得读日志即读追踪，无需 DB join。

DB 直接存两个大 JSON blob。`messages_trace` 保存完整的 ReAct 消息序列（LLM 回复、
`tool_call`、`tool_result`）。`search_calls` 保存每次 `web_search` 的 query、end_date、
结果数、每条结果的 published_date，并在开启泄漏过滤时附加 `n_results_raw`、
`n_results_kept`、`detector_verdicts`、`detector_latency_ms`、`detector_error_kind`。

它们体积大（约占 DB 的 80%），故提供 `WRITE_MESSAGES_TRACE=false` 开关。但默认是开
的，因为调试一次失败的价值远超几 MB 额外磁盘；没有 trace，泄漏审计无法事后重做。

探测器审计字段进入 `search_calls`，从不进入 `messages_trace`，由 `leak_filter.py:L25`
固定并由 `test_leak_filter.py` 断言。这是有意为之：探测器裁决是*审计元数据*，不是 LLM
可见内容；若它进入 `messages_trace`，下游不知情的模型可能看到它并偏置自己的行为。

`loguru` 写两条通道：stderr 给人看，滚动文件给机器读，rotation 100 MB、retention 5。
人与机器读日志需求不同，所以分开服务。

---

## 8. 错误处理

### 8.1 为何某些错误不应重试

| 错误                         | 重试？                       | 原因                                                                              |
| --------------------------- | ---------------------------- | -------------------------------------------------------------------------------- |
| 网络 / 5xx                   | 是，按退避序列                | 多为瞬态                                                                           |
| 限速                         | 是，优先 Retry-After          | provider 已经告诉你等多久                                                            |
| 鉴权 401/403                 | **停止整次运行**              | key 错了；重试无意义，提前停止省钱                                                    |
| Bad request                 | 否                           | `model_not_found` 等只在配置变化后触发                                                |
| 内容策略                     | 否                           | 同样的提示再发一次结果相同                                                            |
| 拒答 / 解析失败              | 否                           | 不是错误，而是模型行为                                                                |
| Tavily 自身                  | 自带重试序列                  | 用尽则把错误返回给 LLM                                                                |
| 训练 cutoff 过滤             | 不触发                        | 直接写 `skipped_training_cutoff`                                                    |

### 8.2 三套独立退避序列

```bash
LLM_BACKOFF_NETWORK_S=2,5,15,30,60         # config.py:L236
LLM_BACKOFF_RATE_LIMIT_S=10,30,60,120,300   # config.py:L237
LLM_BACKOFF_SERVER_5XX_S=5,15,30,60,120     # config.py:L238
```

不同错误类型用不同退避，因为限速比网络慢得多：前者通常需要分钟级冷却而后者一般几秒就
清理。序列长度同时决定了最大重试次数；配置在 `.env` 中统一。

三套序列针对 OpenRouter 的行为模式调优：网络错误几秒内清（如瞬态 TCP reset），限速
分钟级清（provider 端冷却），5xx 浪潮几分钟内清（provider 端恢复）。不同 provider 可
能需要不同序列，这就是每条都是独立 `.env` 旋钮而非单一 `LLM_BACKOFF_S` 的原因。

### 8.3 错误分类码是报告中的一等公民

`error` 字段不是"出错时填一段字符串"，而是固定有限枚举：`network`、`server_5xx`、
`bad_request`、`content_policy`、`skipped_training_cutoff`（见 `errors.ErrorKind`）。

`error_breakdown.csv` 直接按该分类切片。原则是**每一种失败行为都必须可在报告中被分类
和聚合**：`error="something went wrong"` 是没用的。

### 8.4 跨 provider 分类覆盖

跨 provider 评测中存在两类典型失败模式，需要超出英文针线的显式覆盖。

*阿里云内容审核（`data_inspection_failed`）必须不落入 `bad_request`。* 仅识别英文针
线 `content_policy / content_filter / safety` 会让 DashScope
（`https://dashscope.aliyuncs.com`）返回的 `code=data_inspection_failed` 落入兜底
`bad_request`。`errors.CONTENT_POLICY_NEEDLES` 统一针线表，包含
`data_inspection_failed`、`inappropriate content`、`sensitive`；命中即归类为
`content_policy`，从而保留"不得重试"的语义。

*远端断连 `RemoteProtocolError` 必须不落入 `unknown`。* 仅列出 `ConnectError`、
`ReadTimeout`、`ConnectTimeout`、`WriteTimeout` 的网络异常元组会让
`httpx.RemoteProtocolError`（"Server disconnected without sending a response."）落入
`UNKNOWN`，整个样本未重试就失败。网络异常族对齐 httpx 的 `NetworkError` 子集，包含
`RemoteProtocolError`、`WriteError`、`PoolTimeout`，LLM 侧（`errors.classify`）与
Tavily 侧（`search._single_request`）并行覆盖。

原则是**误分类的错误会在报告中悄悄被错算**。两种模式在跨 provider 运行中各自每 ~2K
样本浮现一次，数字上很小但对诚实报告致命，因为它们会撬动 `bad_request` 对
`content_policy` 的比例。

---

## 9. 配置作为契约

### 9.1 几乎每个可调项都活在 `.env`

CLI 仅暴露三个标志：`--question-type`、`--choice-type`、`--skip-analysis`；其余全部
经 `.env`。三个理由。

*易于重跑。* 单一 `.env` 足以复现整个配置；散落在 shell 历史中的 CLI 标志容易丢失。
*友好于 CI 与调度器。* 脚本化执行通常更愿意管理一份文件而非一行命令。*配置 / DB 自一
致。* `config_snapshot` 写入 `run_meta`，所以事后检查一次运行时能看到当时编辑过的
`.env` 长什么样。

### 9.2 OpenAI 兼容端点作为集成面

`LLM_BASE_URL` 接受任何 OpenAI 兼容端点：OpenRouter、阿里云百炼、OpenAI、DeepSeek、
SiliconFlow、本地 vLLM 都行。集成面有意保持小且标准。OpenAI 的 chat completion 加
function calling 协议已成事实标准，所以本项目不构建 provider 适配层而把适配责任推给
端点。这种中立性使得单一评测流水线可以在不同 provider 托管的模型之间比较，而无需
provider 专属代码路径。

### 9.3 训练 cutoff 配置是质量配置

`MODEL_TRAINING_CUTOFFS=openai/gpt-5=2024-10-01,...`（config.py:L224）不是可选的；它
是**评测公平的一部分**。文档明确建议为每个被评测模型声明显式 cutoff；未指定的模型不
被过滤（带告警）。推荐约定是最保守的一个：当 model card 仅以月级粒度披露 cutoff 时，
采用披露月份的*最后一日*作为 $`\kappa_M`$。

### 9.4 启动时校验作为执行

`Settings._post_validate` 在进程启动时运行，若以下任意一项失败则在任何 LLM/Tavily 调
用之前中止。

* `MODELS` 或 `LEAK_DETECTOR_MODEL` 中含 `:online` slug 或 `::` 子串
  （config.py:L599）；
* `MODEL_TRAINING_CUTOFFS` 解析失败（config.py:L181）；
* `REACT_MAX_SEARCH_CALLS` 为空列表（config.py:L580）；
* `SOURCE_TABLE` 不通过 `^[A-Za-z_][A-Za-z0-9_]*$` 白名单（config.py:L586）；
* `ENABLE_WEB_SEARCH=True` 时 `TAVILY_API_KEY` 为空；
* `ENABLE_SEARCH_LEAK_FILTER=True` 时 `LEAK_DETECTOR_API_KEY` 或
  `LEAK_DETECTOR_MODEL` 为空；
* `COMPOSITE_WEIGHTS_QTYPE` 或 `COMPOSITE_WEIGHTS_CTYPE` 含未知桶名或全零权重
  （config.py:L515）；
* `GRID_DEFAULT_R` 或 `GRID_DEFAULT_C` 不在各自列表中。

完整集合在 FRAME §7.2 中列出。原则是**早失败，先于任何可计费调用**：被悄悄错算的评测
比未启动的评测昂贵得多。由 `test_config.py` 固定，每条规则至少有一份 fixture 同时演
练 accept 与 reject 路径。

---

## 10. 测试作为哨兵

### 10.1 测试不得触网或烧 API

完整跑一遍示例数据集 × 模型数 × $`N`$ 个样本是**几十到几百美元**。在那种规模上撞到提
示、解析器或 schema bug 完全是浪费钱。测试设计的核心约束：

* `tavily-python` 不得真的发请求；`respx` mock `httpx`。
* OpenAI 客户端不得真的发请求；fixture 替换。
* SQLite 用临时目录；`tmp_path` fixture。
* 数据集要小但"看起来真"；用源 DB 中几道真实题作为 fixture。

### 10.2 五条 CI 红线

```text
test_prompts / test_parser / test_training_cutoff /
test_llm_no_browsing / test_analysis
```

这五条必须始终为绿。它们覆盖项目最容易悄悄崩坏的部分，恰好就是实现框架 $`\mathcal{R}`$
的组件。

| 测试                    | 守护的不变量                                                   | 框架组件                  | 一旦崩溃                                                                       |
| ----------------------- | -------------------------------------------------------------- | ------------------------- | ----------------------------------------------------------------------------- |
| `test_prompts`          | 三种 question_type 的提示模板渲染正确                            | $`R`$                       | `user_prompt` 文本漂移；`prompt_templates_hash` 不再固定输入                  |
| `test_parser`           | 字母解析与严格相等评分                                          | $`\Psi`$、$`\phi`$            | 字母实际匹配却被标"错"，反之亦然                                                |
| `test_training_cutoff`  | 训练 cutoff 过滤语义与 resume 优先级                            | $`\kappa_M`$ 可纳入性        | cutoff 跳过的题被计费，或已完成的行被重新计费                                    |
| `test_llm_no_browsing`  | 厂商原生 browsing 永不被悄悄打开                                | 信息屏障                  | 整次评测契约失效                                                                |
| `test_analysis`         | 报告数字与原始 DB 对账一致                                      | $`\Gamma`$                  | CSV / MD 数字偏离原始观测所支持的结论                                           |

原则是**挑选崩溃代价高的不变量，把单元测试当哨兵**。每条红线对应 $`\mathcal{R}`$ 的一
个组件，崩溃则使整次运行单元失效。"一旦崩溃"列刻意写得直白；这是省略测试的代价，而
非运行测试的代价。完整映射（33 个测试对 11 个框架组件）见 FRAME §11.2。

### 10.3 dry-run 烟雾测试

`test_smoke_dry_run.py` 用 `httpx` stub 替换 OpenRouter 与 Tavily，端到端跑 3 题 ×
1 模型 × 1 样本。它不验证逻辑细节，验证"管道是否还通"，检查 schema、宽表、
`messages_trace` JSON、`search_calls` 字段是否齐全。这表达了 e2e/单元测试的分工：单
元测试验证局部正确，烟雾测试验证集成不爆。

### 10.4 测试即文档

每条框架级不变量都有测试。当 FRAME 或 DESIGN 的散文与代码漂移时，测试是裁决者。
`tests/` 下的 33 个测试文件（约 13K LOC，全部离线）是"本代码库实际做什么"最权威的形
式；如果你读到的与某条测试断言对不上，以测试为准。

---

## 11. openspec 变更档案

仓库根目录有 `openspec/changes/`，每一项规约驱动的变更都落在此处。每个变更目录都
附带 `proposal.md`（动机）、`design.md`（决策档案）、`specs/.../spec.md`（能力
delta）、`tasks.md`（实现清单）。已命名的变更包括 `bootstrap-forecast-eval`、
`react-tavily-grid-search`、`harness-resilience-v1`、`search-leak-filter-v1`、
`add-exam-score-metric`、`composite-score-by-subtype`、`discrete-native-analysis-v5`。

两条原则驱动这一布局。**规约先于代码**，使错误的设计在合并前被发现。**变更档案与
代码 diff 并存**，使日后审计能恢复某项决定*为何*成立，而不只是看到*什么*代码实现了
它。

### 11.1 通过虚拟 slug 进行网格搜索

`react-tavily-grid-search` 把 $`(Q \times M \times N)`$ 三轴空间扩展到
$`(Q \times M \times R \times C \times N)`$，**不**升级 schema、**不**触动 runner 核心
循环。方法：在评测入口把每个 $`(\text{real\_model}, R, C)`$ 三元组编码为**虚拟模型 slug**
`{real}::r{R}::c{C}`（`db.compose_virtual_slug` 在 db.py:L477；反向解析
`parse_virtual_slug` 在 db.py:L500）；runner、DB、分析主流水线把它当作不透明字符串。
现有产物自然按虚拟 slug 扩展为多行，新模块 `forecast_eval/analysis/grid.py` 解码三元
组、再聚合，并发出长格 grid 表与每格图族。完整决策档案在
`openspec/changes/react-tavily-grid-search/design.md`；10 条关键决策：

| ID  | 决策                                                                                                                                                |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | 选方案 C（虚拟 slug + 每任务 settings 视图）；否决 A（单运行、多 (R, C) DB schema 重写）与 B（每 cell 一个 run_dir，`runs/` 膨胀且跨运行聚合复杂） |
| D2  | 虚拟 slug 用 `::r{R}::c{C}` 后缀；`db.model_slug_safe` 把 `::` 替换为 `_` 落到 fs 安全文件名 `openai__gpt-5__r5__c3.db`；正则 `^(?P<real>.+?)::r(?P<R>\d+)::c(?P<C>\d+)$` 非贪婪捕获 real_model |
| D3  | `runner.Task` 携带 cell 局部的 `settings: Settings`；调度器经 `model_copy(update={...})` 派生不可变子视图；`react.py` 与 `search.py` 字节不变       |
| D4  | 仅当 `REACT_MIN_SEARCH_CALLS > min(C_list)` 时报错；对 `C < MIN` 的 cell，静默 clamp `effective_min = min(MIN, C)` 并记入 `run_meta.config_snapshot.grid_origin` 供审计 |
| D5  | `run_meta.config_snapshot` 写**单值** R/C；新增 `grid_origin = {real_model, R, C, effective_min_search_calls}` 子键；manifest 顶层加 `grid` 块（`r_list / c_list / default_r / default_c / real_models / n_cells`），分析层就不必逐 .db 解码三元组 |
| D6  | `manifest.models` 与 `manifest.model_files` 字段语义保持"虚拟 slug 列表"；`grid.real_models` 是去重后的 real-slug 便利字段，从而保留分析主路径"读 `manifest.models` 作为 db 文件列表"的契约 |
| D7  | `analysis/__init__.py::run_analysis` 主路径**零侵入**；末尾追加一个 `try/except` 包裹的 `grid.run_grid_analysis(...)`（与反思 A/B 同样的 best-effort 模式）；失败不打断现有流水线 |
| D8  | grid CI 全部走 `inference.paired_bootstrap`（5000 次重采，seed=42）；BI 域 CI 通过 "BS 域配对自助 + 单调变换 $`\mathrm{BI}=100(1-\sqrt{\mathrm{BS}})`$" 获得；**不**引入新统计代码 |
| D9  | Pareto 前沿的成本维默认 `mean_search_calls`（实际平均搜索数，比 C 上限更诚实），允许 `mean_latency_ms / C` 回退；y 轴默认 `bi_mean`，`nll_mean`（最小化方向）作为可选 |
| D10 | 主图 Fig 1 固定 $`R = \texttt{GRID\_DEFAULT\_R}`$，每个 real_model 一条曲线；其他 R 值各得一张同格式附录图，避免叠加 $`M \cdot |R|`$ 条曲线后主图不可读 |

三个 PR Phase 0 / 1 / 2 顺序发布；每阶段都通过 `pytest -q` 与
`openspec validate --strict`，删除该阶段自身代码后系统等价于上一阶段完成态（回滚策
略）。新代码下，单值 `.env` 解析为长度 1 的列表，所以笛卡儿积只产生单一虚拟 slug，唯
一可见差异是 .db 文件名上的 `__r{R}__c{C}` 后缀。对 manifest 不含 `grid` 块的运行，
grid 分析与 grid 图族整体早退，零侵入。

### 11.2 阶段化分批发布

每次 openspec 变更以阶段序列发布（典型为 `Phase 0 schema → Phase 1 code → Phase 2
docs/CSV columns`）。纪律：

* **每阶段独立通过 CI。** 不做"一个大 PR 全包"；CI 在每个闸的错侧捕获 schema/代码不
  匹配。
* **每阶段可逆。** 删掉该阶段自身代码即回到上一阶段，假想回滚后重跑测试以验证。
* **变更档案累积。** `openspec/changes/archive/` 是项目自己的"为何走到今天"账本；它
  比 git log 更耐久，因为每条都带有 spec / design / tasks 分离，能在 squash-merge 后
  幸存。

---

## 12. 知道自己在拧哪个旋钮

跨团队对话中反复出现一种困惑：哪些旋钮属于"评测契约的一部分"，哪些属于"不影响科学主
张的工程调优"。这个区分很重要，因为改契约旋钮会使跨运行可比性失效，改工程旋钮则不
会。

### 12.1 契约旋钮（改它们使运行不可比较）

下表每条都对应一个指纹或 $`\mathcal{R}`$ 元组字段；全部写入 `run_meta`，任何变化在配对
比较中都会以哈希不匹配呈现。

| 旋钮                              | 为何是契约                                                            |
| --------------------------------- | --------------------------------------------------------------------- |
| `SOURCE_DB` + `SOURCE_TABLE`      | 定义 $`\mathcal{D}`$；指纹经 `source_db_hash`                            |
| `MODEL_TRAINING_CUTOFFS`          | 定义 $`\kappa_M`$；按模型可纳入性                                          |
| `TAVILY_END_DATE_OFFSET_DAYS`     | 定义 $`\delta`$，进而 $`\chi_i`$                                           |
| `REACT_MAX_STEPS`                 | 定义 $`T`$                                                              |
| `REACT_MAX_SEARCH_CALLS`          | 定义 $`C`$（grid 轴）                                                     |
| `TAVILY_MAX_RESULTS`              | 定义 $`R_{\mathrm{tav}}`$（grid 轴）                                      |
| 提示模板（8 键）                   | 定义 $`R`$；指纹经 `prompt_templates_hash`                                |
| 反思协议文本 + 开关                | 定义 $`F_M`$ 的一部分；指纹经 `reflection_protocol_hash`                   |
| 信念协议文本 + 开关                | 定义 $`F_M`$ 的一部分；指纹经 `belief_protocol_hash`                       |
| 探测器提示 + 版本                  | 定义 $`H_{\mathrm{aux}}`$；指纹经 `leak_detector_prompt_hash`              |
| 综合权重                          | 定义 $`\Gamma`$（默认子型加权形态）；记入 `run_meta`                       |
| `SAMPLING_N`                      | 定义 $`S`$；建表时固定                                                    |

### 12.2 工程旋钮（在一次比较内可安全更改）

下表每条仅关乎吞吐、成本或鲁棒性；都不影响 $`\mathcal{R}`$。

| 旋钮                                     | 为何是工程                                                            |
| ---------------------------------------- | -------------------------------------------------------------------- |
| `LLM_MAX_CONCURRENCY`                    | 吞吐；与结果无数值相关                                                 |
| `LLM_BACKOFF_*`                          | provider 端噪声下的鲁棒性；序列足够长即收敛                              |
| `SEARCH_RETRY_MAX` / `_BACKOFF_S`        | 同上，针对 Tavily                                                      |
| `LEAK_DETECTOR_RETRY_MAX` / `_BACKOFF_S` | 同上，针对探测器                                                        |
| `LEAK_DETECTOR_CONCURRENCY`              | 探测器阶段吞吐                                                         |
| `DB_COMMIT_BATCH`                        | 磁盘写入批次；与数值无关                                                |
| `WRITE_MESSAGES_TRACE`                   | 磁盘体积旋钮；影响事后审计，不影响数字                                    |
| `LOG_LEVEL` / `LOG_DIR`                  | 日志详细度 / 位置                                                      |
| `RUNS_ROOT`                              | 产物落点；不改变它们是什么                                               |

契约 / 工程的二分使审查 PR 或 `.env` 变更变得机械：§12.1 中的任一变化是*一次新评
测*，§12.2 中的任一变化是*bug 修复或调优*。运行时指纹集（§7.2）使这一区分在任何一对
运行上都可观测。

---

## 收尾原则

将整篇文档的设计哲学浓缩为单一原则集，供审查参考：

1. **边界在数据层而非提示层。** 样本准入（$`\kappa_M`$）、时间掩码（$`\delta`$、$`\chi_i`$）、
   探测器（$`H_{\mathrm{aux}}`$）都活在评测者控制的代码里，从不在模型的指令里。
2. **诚实胜于漂亮。** 威胁模型用平实语言声明自己控制不了什么；泄漏审计发布 Wilson 上
   界而不只是点估计。
3. **跳过不是失败。** 主动排除的样本独立归类，不污染错误率。
4. **原始胜于聚合。** DB 仅存观测；统计推迟到 `analysis/`。指标定义比 DB schema 变化
   得更快。
5. **默认严格，分得分按需。** 头条指标是严格 frozenset 相等；综合用考试式部分得分；
   FSS 增加随机校正。三者在同一份原始样本上共存。
6. **可复现胜于便利。** 源数据进 Git；每个 DB 自含；六部分哈希固定指纹。
7. **可观测胜于优雅。** 完整 `messages_trace` 默认开；进度日志一行一样本；逐调用审计
   字段持久化探测器裁决。
8. **失败要分类。** 错误用有限枚举；每种都在报告中有自己的格子。
9. **配置即契约。** `.env` 一份决定一切；CLI 标志极简；`config_snapshot` 持久化前编
   辑；契约 / 工程二分（§12）告诉你哪种 `.env` 变化使比较失效。
10. **测试守护昂贵之物。** 五条 CI 红线一对一映射运行单元 $`\mathcal{R}`$ 的组件；dry-
    run 烟雾测试验证集成；昂贵失败被前移到本地。
11. **框架就是契约。** 每个工程决策都按"加强或削弱 $`\mathcal{R}`$"接受裁判；只有清晰
    划界的逃生口（`LEAK_DETECTOR_FAIL_ACTION=keep`、`WRITE_MESSAGES_TRACE=false`、
    A/B 开关）才允许便利胜过框架，它们自身也留下审计轨迹。
12. **早失败，先于可计费调用。** 启动校验在任何 LLM/Tavily 接触之前拒绝错配；未启
    动的评测比悄悄被错算的评测更便宜。
13. **靠编排消除竞态，不靠锁。** 每 DB 一个 writer，每 writer 一个 queue，批量 commit。
14. **阶段化变更。** 每次 openspec 变更以可逆阶段发布；变更档案是项目"为何走到今天"
    的账本。
15. **三个独立指纹，不是一个。** 模板、反思协议、信念协议各有独立哈希，使消融研究可
    沿单一轴配对运行。

---

## 附录 A. 被否决方案索引

本代码库中考虑过并被否决的所有替代方案的合并列表，按决策领域排序。每一行引用详细论证
所在的小节。

| 决策领域                    | 被否决的替代方案                                            | 否决原因                                                       | §       |
| ---------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------- | ------- |
| 工具中介边界                 | 把 `end_date` 暴露为 LLM 工具参数                           | 信任模型不扩大 $`\chi_i`$；固定测试面爆炸                         | 2.2     |
| 工具中介边界                 | 改写 query 字符串注入日期过滤                                | 字符串处理脆弱；provider 可忽略内联日期操作符                   | 2.2     |
| $`\delta`$ 默认                | $`\delta = 0`$（用裁决日）                                    | 在示例 DB 上漏掉 30–50% 的当日新闻泄漏；过松                    | 2.3     |
| $`\delta`$ 默认                | 每问题 $`\delta_i`$                                           | 违反"一个 $`\delta`$ 定义一次评测"契约                            | 2.3     |
| 厂商原生 browsing             | 警告而非拒绝                                                | 警告会被过滤；拒绝才能停止运行                                  | 2.4     |
| 厂商原生 browsing             | 仅在启动时强制                                              | 经 `model_copy(update={...})` 可绕过；需逐调用复检               | 2.4     |
| Cutoff 过滤                   | 加权排除（折扣靠近 cutoff 的样本）                          | 增加分析复杂度；二元进/出更干净                                 | 2.5     |
| Cutoff 过滤                   | 任一模型不可纳入则数据集级跳过                              | 在异质模型组上丢掉 10–20% 语料                                  | 2.5     |
| 离散答案空间                 | 由 LLM 评判的开放式 NL 输出                                 | 重新引入污染风险；评分非确定                                    | 3.1     |
| 离散答案空间                 | 仅由 Brier/NLL 评分的数值概率输出                           | 在 $`K=5`$ 下经验 $`\hat p`$ 太离散（见 §6.7）                      | 3.1     |
| 严格评分                     | 严格层用 Jaccard / F1 / 部分得分                            | 削弱头条数字；软伴随在综合层加入                                | 3.3     |
| > 26 选项标签                 | 跳过 > 26 选项的题                                          | 丢失数据集覆盖；往返测试缓解                                     | 3.4     |
| 解析失败处理                 | 解析失败时重试                                              | 能力遮蔽 + 成本；重试预期同答案                                  | 3.5     |
| 系统消息                     | 拆分 system / user                                          | 丢失跨 provider 一致性                                          | 5.1     |
| 反思                         | 默认硬下限 `REACT_MIN_SEARCH_CALLS`                          | 把能力与强制混淆                                                | 5.3     |
| 框架韧性                     | 单一"框架救援"开关                                          | 丢失四种正交模式的消融区分度                                    | 5.4     |
| DB 布局                       | 单 DB 加 `model` 列                                         | 单 writer 抢占；一个慢 provider 拖死所有                         | 6.1     |
| DB 布局                       | 存路径而非副本                                              | 破坏自含属性                                                    | 6.2     |
| DB 布局                       | 长表 + 复合主键                                             | 丢失简单 resume 查询；多行写打破编排                            | 6.3     |
| 步指标                       | 用长表存逐步行                                              | 为不存在的查询付出索引/JOIN 成本                                | 6.4     |
| 并发                         | 每 INSERT 加锁                                              | 抢占烧 CPU；编排更便宜                                          | 6.5     |
| 可复现                       | 把 DB 放独立 registry                                       | 给复现增添网络依赖                                              | 7.1     |
| 可复现                       | 单一组合哈希                                                | 在 {模板、反思、信念} 上丢失消融区分度                          | 7.3     |
| 综合权重                     | 跨桶等权重                                                  | 抹掉区分度桶；让接近随机的桶重复计入                            | 4.4     |
| 综合权重                     | 经验流行度权重                                              | 淹没承载信号的桶                                                | 4.4     |

---

## 附录 B. 阅读路线

如果你是项目新人，建议按以下顺序阅读。这一顺序经过设计：每一步给你下一步的语言。1–3
步建立概念模型；4–7 步把它落到代码里；8–9 步给你测试契约与规约档案。

1. `README.md`：用 10 分钟弄清 OracleProto 是什么以及如何运行。
2. 本文（`DESIGN.md`）：理解每一项权衡背后的动机。
3. `FRAME.md`：字段、接口、伪代码层的完整规约。每当 DESIGN 说"见 file:line"时，把
   §1.1 的总图作为符号到代码的查表使用。
4. `forecast_eval/prompts.py` 与 `forecast_eval/parser.py`：渲染器 $`R`$ 与解析器
   $`\Psi`$；这两个文件是项目的心脏。
5. `forecast_eval/runner.py` 与 `forecast_eval/react.py`：编排与 ReAct 循环。特别留
   意 react.py:L266，四旋钮优先链就在那里。
6. `forecast_eval/leak_filter.py` 与 `forecast_eval/search.py`：时间掩码实现与第二阶
   段探测器。leak_filter.py:L55 的提示模板是强制"无问题字段"白名单的所在。
7. `forecast_eval/analysis/`：指标层。从 `exam_score.py` 开始，单文件、自含、五分钟
   读完；然后 `accuracy.py` 看 Tversky、FSS、Cohen $`\kappa`$；然后 `composite.py` 看
   子型加权聚合；然后 `consistency.py` 看 Fleiss $`\kappa`$、VCI、MVG。
8. `tests/`：通过读测试反向工程契约。五条 CI 红线（§10.2）是优先级最高的入口。
9. `openspec/changes/archive/`：要弄清某项决定*为何*成立，来这里。每次变更都有
   `proposal.md`（动机）、`design.md`（决策档案）、`specs/.../spec.md`（能力
   delta）、`tasks.md`（实现清单）。

---

> **一句话总结：** OracleProto 用工程纪律守护科学严谨。每一道看似过分的约束之所以存
> 在，是因为放弃它就会得到一份在最终报告中没有真实意义的数字。信息边界是数据的一部
> 分，不是提示；运行单元 $`\mathcal{R}`$ 是契约，不是配置；审计轨迹是报告的根基，不是
> 它的附录。
