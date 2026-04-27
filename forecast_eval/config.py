from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


_PLACEHOLDER_TOKENS = {"REPLACE_ME", "CHANGEME", "PUT_YOUR_KEY_HERE"}
_RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")
_SQL_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# composite-score-by-subtype: 子题型分桶常量。`question_type` 来自 questions
# 表 CHECK 约束 (`yes_no` / `binary_named` / `multiple_choice`); `choice_type`
# 同表 (`single` / `multi`)。校验使用 frozenset 即可，不需要枚举类。
COMPOSITE_QTYPE_BUCKETS: frozenset[str] = frozenset(
    {"yes_no", "binary_named", "multiple_choice"}
)
COMPOSITE_CTYPE_BUCKETS: frozenset[str] = frozenset({"single", "multi"})


def _parse_csv(raw: str | list[Any] | None) -> list[str]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _parse_csv_int(raw: str | list[Any] | None) -> list[int]:
    return [int(x) for x in _parse_csv(raw)]


def _parse_weights_dict(
    raw: str | dict[str, Any] | None, *, allowed_buckets: frozenset[str], field_name: str
) -> dict[str, float]:
    """解析 ``"yes_no=0.15,binary_named=0.15,multiple_choice=0.70"`` 形式的权重映射。

    校验：桶名必须 ∈ ``allowed_buckets``、权重必须可解析为 float、权重 ≥ 0。
    缺失的桶视为 0（即"该桶不参与合成"），但要求至少一个桶权重 > 0。
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        items = [(str(k).strip(), v) for k, v in raw.items()]
    else:
        items = []
        for pair in str(raw).split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(
                    f"{field_name} entry must be 'bucket=weight', got: {pair!r}"
                )
            bucket, value = pair.split("=", 1)
            items.append((bucket.strip(), value.strip()))
    out: dict[str, float] = {}
    for bucket, value in items:
        if not bucket:
            raise ValueError(f"{field_name} has an empty bucket name")
        if bucket not in allowed_buckets:
            raise ValueError(
                f"{field_name} bucket {bucket!r} not in {sorted(allowed_buckets)}"
            )
        try:
            w = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"{field_name}[{bucket}] is not a valid float: {value!r}"
            ) from e
        if w < 0:
            raise ValueError(
                f"{field_name}[{bucket}] must be >= 0; got {w}"
            )
        out[bucket] = w
    if out and not any(w > 0 for w in out.values()):
        raise ValueError(
            f"{field_name} requires at least one weight > 0 (otherwise composite "
            "score has no defined denominator)"
        )
    return out


def _parse_overrides_dict(
    raw: str | dict[str, Any] | None,
    *,
    allowed_buckets: frozenset[str],
    field_name: str,
) -> dict[str, dict[str, float]]:
    """解析 ``"fss=yes_no=0.05,binary_named=0.05,multiple_choice=0.90;cohen_kappa=..."``
    形式的按指标权重覆盖。

    分号分隔不同指标，逗号分隔同指标内的桶；每段沿用
    :func:`_parse_weights_dict` 的语义（含至少一个 > 0 校验）。指标名拼写错误
    （不在已知指标白名单内）由 ``forecast_eval.analysis.composite`` 在运行时
    raise，本函数不做该层校验，避免 config 模块反向 import analysis。
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        # 直接传入 dict：值可以是 "k=v,k=v" 字符串或已解析的 dict
        out: dict[str, dict[str, float]] = {}
        for metric, sub in raw.items():
            metric_name = str(metric).strip()
            if not metric_name:
                raise ValueError(f"{field_name} has an empty metric name")
            sub_field = f"{field_name}[{metric_name}]"
            out[metric_name] = _parse_weights_dict(
                sub, allowed_buckets=allowed_buckets, field_name=sub_field
            )
        return out
    out = {}
    for segment in str(raw).split(";"):
        segment = segment.strip()
        if not segment:
            continue
        if "=" not in segment:
            raise ValueError(
                f"{field_name} segment must be 'metric=bucket=w,bucket=w'; got: {segment!r}"
            )
        metric_name, rest = segment.split("=", 1)
        metric_name = metric_name.strip()
        if not metric_name:
            raise ValueError(f"{field_name} has an empty metric name in segment: {segment!r}")
        sub_field = f"{field_name}[{metric_name}]"
        out[metric_name] = _parse_weights_dict(
            rest, allowed_buckets=allowed_buckets, field_name=sub_field
        )
    return out


_MAX_TOKENS_PARAM_ALLOWED = {"max_tokens", "max_completion_tokens"}


def _parse_max_tokens_param(raw: str | dict[str, Any] | None) -> dict[str, str]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        items = [(str(k), str(v)) for k, v in raw.items()]
    else:
        items = []
        for pair in str(raw).split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(
                    f"MODEL_MAX_TOKENS_PARAM entry must be 'model=param_name', got: {pair!r}"
                )
            model_slug, name = pair.split("=", 1)
            items.append((model_slug.strip(), name.strip()))
    out: dict[str, str] = {}
    for model_slug, name in items:
        if not model_slug:
            raise ValueError("MODEL_MAX_TOKENS_PARAM has an empty model slug")
        if name not in _MAX_TOKENS_PARAM_ALLOWED:
            raise ValueError(
                f"MODEL_MAX_TOKENS_PARAM[{model_slug}]={name!r} must be one of "
                f"{sorted(_MAX_TOKENS_PARAM_ALLOWED)}"
            )
        out[model_slug] = name
    return out


def _parse_cutoffs(raw: str | dict[str, Any] | None) -> dict[str, date]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        items = [(str(k), v) for k, v in raw.items()]
    else:
        items = []
        for pair in str(raw).split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(
                    f"MODEL_TRAINING_CUTOFFS entry must be 'model=YYYY-MM-DD', got: {pair!r}"
                )
            model_slug, d = pair.split("=", 1)
            items.append((model_slug.strip(), d.strip()))
    out: dict[str, date] = {}
    for model_slug, value in items:
        if not model_slug:
            raise ValueError("MODEL_TRAINING_CUTOFFS has an empty model slug")
        if isinstance(value, date):
            out[model_slug] = value
        else:
            try:
                out[model_slug] = date.fromisoformat(str(value))
            except ValueError as e:
                raise ValueError(
                    f"MODEL_TRAINING_CUTOFFS[{model_slug}] is not a valid YYYY-MM-DD: {value!r}"
                ) from e
    return out


class Settings(BaseSettings):
    """Runtime configuration loaded once from `.env`.

    All tunable parameters live here; CLI only owns --question-type/--choice-type.
    Instances MUST be constructed at process start; failures raise ValueError to
    abort before any LLM / Tavily call is made.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # LLM (任意 OpenAI-compatible endpoint)
    LLM_API_KEY: str
    LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
    MODELS: Annotated[list[str], NoDecode] = Field(default_factory=list)
    MODEL_TRAINING_CUTOFFS: Annotated[dict[str, date], NoDecode] = Field(default_factory=dict)
    LLM_MAX_TOKENS: int = 12000
    # 部分 provider (例如 OpenAI 官方 o-series / GPT-5 的 /v1/chat/completions)
    # 已弃用 `max_tokens`, 改用 `max_completion_tokens`. 这里按 model slug 覆盖
    # 实际请求体里使用的字段名; 未声明的模型默认仍使用 `max_tokens`.
    # 格式: "<model_slug>=max_completion_tokens" 多组用逗号分隔.
    MODEL_MAX_TOKENS_PARAM: Annotated[dict[str, str], NoDecode] = Field(default_factory=dict)
    LLM_TIMEOUT_S: int = 240
    LLM_TEMPERATURE: float = 0.7
    LLM_TOP_P: float = 1.0
    LLM_MAX_CONCURRENCY: int = 5
    LLM_RETRY_MAX: int = 5
    LLM_BACKOFF_NETWORK_S: Annotated[list[int], NoDecode] = Field(default_factory=lambda: [2, 5, 15, 30, 60])
    LLM_BACKOFF_RATE_LIMIT_S: Annotated[list[int], NoDecode] = Field(default_factory=lambda: [10, 30, 60, 120, 300])
    LLM_BACKOFF_SERVER_5XX_S: Annotated[list[int], NoDecode] = Field(default_factory=lambda: [5, 15, 30, 60, 120])
    # 推理类模型 slug 子串列表: 匹配到的模型调用时将不传 temperature / top_p
    # (o-series / deepseek-r1 / qwq 等推理模型不接受自定义采样参数, 会直接报 400)
    LLM_REASONING_MODEL_PATTERNS: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["o1", "o3", "o4", "r1", "qwq"]
    )

    # Web search master switch: when False, the ReAct loop runs with no tool
    # schema at all — Tavily is never hit and TAVILY_API_KEY becomes optional.
    ENABLE_WEB_SEARCH: bool = True

    # Tavily — 详见 .env.example 中各字段注释
    # 支持单 key 或 CSV 多 key (`TAVILY_API_KEY=tvly-aaa,tvly-bbb`). 多 key 时
    # 由 forecast_eval.tavily_keys.TavilyKeyPool 做 least-used 调度 + 失败熔断,
    # 同一 process 内所有 grid cell 通过模块级 cache 共享同一个池实例 (cache
    # key = tuple(TAVILY_API_KEY)), 因此用量计数跨 cell 累加而非按 cell 分开.
    TAVILY_API_KEY: Annotated[list[str], NoDecode] = Field(default_factory=list)
    # 单个 key 命中 429 / 配额超限时, 临时拉黑的秒数. 0 = 不拉黑 (仅靠 acquire
    # 顺序避开). 401/403 走永久拉黑, 不受此参数影响.
    TAVILY_KEY_COOLDOWN_S: float = 60.0
    # Grid-scannable: list of cell-local R values, parsed from CSV in .env.
    # Single-value envs (`TAVILY_MAX_RESULTS=5`) parse to `[5]`; multi-value
    # (`TAVILY_MAX_RESULTS=5,10`) drives the (R, C) cartesian dispatcher in
    # evaluation.py. Per-cell sub-views downcast this to a single int via
    # `model_copy(update={"TAVILY_MAX_RESULTS": R})` before being handed to
    # `tavily_search`, so the Tavily request body always sees a single int.
    TAVILY_MAX_RESULTS: Annotated[list[int], NoDecode] = Field(default_factory=lambda: [5])
    # basic | advanced (Tavily 官方 search_depth)
    TAVILY_SEARCH_DEPTH: str = "basic"
    # false | markdown | text (旧版 bool 'true' 兼容映射到 'markdown')
    TAVILY_INCLUDE_RAW_CONTENT: str = "markdown"
    # 单结果 raw_content 截断长度; 0 = 不截断
    TAVILY_RAW_CONTENT_MAX_CHARS: int = 8000
    # false | basic | advanced (Tavily 内部 LLM 速答, 默认关闭以免污染评测)
    TAVILY_INCLUDE_ANSWER: str = "false"
    TAVILY_END_DATE_OFFSET_DAYS: int = -1
    SEARCH_MAX_CONCURRENCY: int = 5
    SEARCH_RETRY_MAX: int = 3
    SEARCH_BACKOFF_S: Annotated[list[int], NoDecode] = Field(default_factory=lambda: [2, 5, 15])

    # ReAct
    REACT_MAX_STEPS: int = 12
    # Grid-scannable: list of cell-local C values, parsed from CSV. Same shape
    # contract as TAVILY_MAX_RESULTS — dispatcher derives per-cell sub-views
    # carrying a single int; runner / react never see the list form.
    REACT_MAX_SEARCH_CALLS: Annotated[list[int], NoDecode] = Field(default_factory=lambda: [8])
    # 反思协议总开关: 启用后 user prompt 末尾追加多步推理脚手架, 显著提升搜索 + 反思深度.
    # 不会写入 prompt_templates (因此 prompt_templates_hash 保持不变), 但会作为 user message
    # 实际文本落入每个 sample 的 user_prompt 字段, 同时通过 config_snapshot 记入 run_meta.
    REACT_REFLECTION_PROTOCOL: bool = True
    # 软性最低搜索次数: LLM 试图给最终答案但累计 web_search < min 时, 注入一条 user nudge
    # 让它继续检索. 0 = 关闭 (默认; 主要靠反思协议自然驱动). >0 = 开启兜底 floor.
    REACT_MIN_SEARCH_CALLS: int = 0
    # nudge 最多注入几次, 防止 LLM 与系统互相 nudge 死循环. REACT_MAX_STEPS 仍是硬天花板.
    REACT_MAX_NUDGES: int = 2
    # v5.1 harness-resilience 开关. 见 openspec/changes/harness-resilience-v1.
    # REACT_FINAL_ANSWER_RETRY=True: 循环正常结束但 final_raw=="" 时, 用 tools=[] 再调一次 LLM.
    #   默认 False — 已被 force-final-answer-near-limit-v1 (循环内最后一步硬切 tools=[]) 覆盖,
    #   保留为可选的循环外应急兜底. 启用会多花一步 LLM 预算 (react_steps + 1).
    # REACT_BUDGET_EXCEEDED_DROP_TOOLS=True (默认): 一旦累计 web_search >= REACT_MAX_SEARCH_CALLS,
    #   之后每轮都以 tools=[] 调用 LLM, 让模型只能输出 content (而非反复撞 budget exceeded).
    REACT_FINAL_ANSWER_RETRY: bool = False
    REACT_BUDGET_EXCEEDED_DROP_TOOLS: bool = True

    # force-final-answer-near-limit-v1: prompt 层面把"步数即将耗尽"的事实前置, 与
    # REACT_FINAL_ANSWER_RETRY (事后兜底) 互补.
    # REACT_BUDGET_AWARENESS_PROTOCOL=True: 在 user prompt 末尾追加预算意识段, 告知模型
    #   总步数 + 总搜索次数, 引导其留最后一步给 \boxed{...} 答案.
    # REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=True: 在循环临近上限时主动注入 user message:
    #   - 倒数第 (REACT_FORCE_FINAL_ANSWER_LOOKAHEAD) 步至倒数第二步: 注入软提醒, 仍允许 tool_calls.
    #   - 最后一步: 注入硬收尾文案 + tools=[], 强制 content-only 输出 \boxed{...}.
    # REACT_FORCE_FINAL_ANSWER_LOOKAHEAD: int = 多少步开始干预 (>= 1, <= REACT_MAX_STEPS).
    #   = 1: 只在最后一步硬切, 无软提醒. = 2 (默认): 倒数第二步软提醒 + 最后一步硬切.
    REACT_BUDGET_AWARENESS_PROTOCOL: bool = True
    REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT: bool = True
    REACT_FORCE_FINAL_ANSWER_LOOKAHEAD: int = 2

    # 网格搜索锚点 (可选): 当 .env 配置多值 R / C 时, paper 主图固定 R = GRID_DEFAULT_R
    # 画 BI vs C 曲线; 类似地 GRID_DEFAULT_C 控制 BI vs R 曲线的 C 锚点. 未设置时
    # plot_analysis.py 默认取列表第一个值. 必须 ∈ 对应列表 (启动期校验).
    GRID_DEFAULT_R: int | None = None
    GRID_DEFAULT_C: int | None = None

    # 结构化置信度协议总开关 (v4). 启用后在 user prompt 末尾追加 belief 协议段
    # (位置在 reflection 协议之后), 要求 LLM 在 \\boxed{...} 之前输出一段
    # <belief>...</belief> JSON, 携带 probabilities / confidence / key_evidence /
    # counterevidence / decision_rule. 与 reflection 协议互相独立, 不进
    # prompt_templates_hash, 由 run_meta.belief_protocol_text/_hash 记录指纹.
    # 默认关闭以保 v3 行为字节级一致; 启用前先在候选模型上 pilot 解析率.
    BELIEF_PROTOCOL: bool = False

    # -------- Search leak filter (Stage 2 detector, search-leak-filter-v1) --------
    # 总开关: True 时 tavily_search 返回前每条 result 经独立 LLM (detector) 审核,
    # verdict=drop 整条剔除; False 时字节级回退到本提案前. 默认开启 (默认严苛).
    # 启用要求 LEAK_DETECTOR_API_KEY / LEAK_DETECTOR_MODEL 必须配齐, 且
    # ENABLE_WEB_SEARCH 必须同时为 True (否则 detector 路径死代码).
    ENABLE_SEARCH_LEAK_FILTER: bool = True
    # detector 自有 OpenAI-compatible endpoint 配置, 与 LLM_* 完全解耦, 允许
    # detector 选用比被评测模型更高级的模型做严苛判断.
    LEAK_DETECTOR_API_KEY: str = ""
    # 留空时由 leak_filter 在 client 初始化阶段回填 LLM_BASE_URL.
    LEAK_DETECTOR_BASE_URL: str = ""
    # detector 模型 slug. 不允许 :online 后缀 (provider-native browsing 防护).
    LEAK_DETECTOR_MODEL: str = ""
    LEAK_DETECTOR_TIMEOUT_S: int = 60
    LEAK_DETECTOR_TEMPERATURE: float = 0.0
    LEAK_DETECTOR_MAX_TOKENS: int = 512
    LEAK_DETECTOR_RETRY_MAX: int = 3
    LEAK_DETECTOR_BACKOFF_S: Annotated[list[int], NoDecode] = Field(default_factory=lambda: [2, 5, 15])
    # drop = 失败条目剔除 (默认, fail-closed); keep = 失败条目透传 (escape hatch).
    LEAK_DETECTOR_FAIL_ACTION: str = "drop"
    LEAK_DETECTOR_CONCURRENCY: int = 5
    # 人工版本号; sha256(prompt_template) 自动算 hash, 这个是给人读的标签.
    LEAK_DETECTOR_PROMPT_VERSION: str = "v1"

    # -------- Composite score weights (composite-score-by-subtype) --------
    # 综合得分按子题型加权: 对所有指标 (per_model_summary 全部列) 在两个维度
    # 上独立做"先按桶分别算 → 再按权重合成 → 缺失桶剔除并归一化"的合成。
    # 写出 runs/{run_id}/analysis/per_model_composite_by_question_type.csv
    # 与 .../per_model_composite_by_choice_type.csv. 旧 per_model_summary.csv
    # 与 per_model_by_question_type.csv 保持原"全部题混算"语义不变。
    #
    # 默认权重按"难题区分度高"原则: yes_no/binary_named 各 0.15、
    # multiple_choice 0.70; single 0.40、multi 0.60.
    COMPOSITE_WEIGHTS_QTYPE: Annotated[dict[str, float], NoDecode] = Field(
        default_factory=lambda: {
            "yes_no": 0.15,
            "binary_named": 0.15,
            "multiple_choice": 0.70,
        }
    )
    COMPOSITE_WEIGHTS_CTYPE: Annotated[dict[str, float], NoDecode] = Field(
        default_factory=lambda: {"single": 0.40, "multi": 0.60}
    )
    # 按指标覆盖: 形如 ``"fss=yes_no=0.05,binary_named=0.05,multiple_choice=0.90;
    # cohen_kappa=..."``. 缺省走全局默认; 指标名拼写错误 (不在 _SUMMARY_FIELDS
    # + 概率族白名单内) 由 analysis.composite 运行时 raise.
    COMPOSITE_WEIGHT_OVERRIDES_QTYPE: Annotated[
        dict[str, dict[str, float]], NoDecode
    ] = Field(default_factory=dict)
    COMPOSITE_WEIGHT_OVERRIDES_CTYPE: Annotated[
        dict[str, dict[str, float]], NoDecode
    ] = Field(default_factory=dict)

    # Sampling / Run
    SAMPLING_N: int = 5
    RUN_ID: str = ""
    RESUME: bool = True

    # Database
    SOURCE_DB: str = "./forecast_eval_set_example.db"
    # Question table inside SOURCE_DB. The bundled example DB ships with
    # `forecast_eval_set_example`; bring-your-own datasets can point at any other
    # table name as long as it has the same 7-column schema (see FRAME.md §2.1).
    SOURCE_TABLE: str = "forecast_eval_set_example"
    # Every evaluation gets its own folder at RUNS_ROOT/{run_id}/, containing one
    # SQLite file per model under db/, plus analysis/ (post-run statistics) and
    # logs/. The old single-file RESULTS_DB layout is gone — see FRAME.md §5/§6.
    RUNS_ROOT: str = "./runs"
    DB_COMMIT_BATCH: int = 10
    WRITE_MESSAGES_TRACE: bool = True

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "./logs"

    @field_validator("MODELS", mode="before")
    @classmethod
    def _parse_models(cls, v: Any) -> list[str]:
        return _parse_csv(v)

    @field_validator("TAVILY_API_KEY", mode="before")
    @classmethod
    def _parse_tavily_keys(cls, v: Any) -> list[str]:
        # 单值 (`tvly-xxx`) 解析为 length-1 list, CSV 多值展开. 空值返回 []
        # 让 _post_validate 统一根据 ENABLE_WEB_SEARCH 决定是否要求非空.
        return _parse_csv(v)

    @field_validator(
        "LLM_BACKOFF_NETWORK_S",
        "LLM_BACKOFF_RATE_LIMIT_S",
        "LLM_BACKOFF_SERVER_5XX_S",
        "SEARCH_BACKOFF_S",
        "LEAK_DETECTOR_BACKOFF_S",
        mode="before",
    )
    @classmethod
    def _parse_backoffs(cls, v: Any) -> list[int]:
        if isinstance(v, list):
            return [int(x) for x in v]
        return _parse_csv_int(v)

    @field_validator(
        "TAVILY_MAX_RESULTS",
        "REACT_MAX_SEARCH_CALLS",
        mode="before",
    )
    @classmethod
    def _parse_grid_int_list(cls, v: Any) -> list[int]:
        # Multi-value grid axis: CSV in .env, list/int passthrough for tests.
        # Single-value envs degrade to a length-1 list (back-compat with v4
        # `TAVILY_MAX_RESULTS=5`); empty values raise so we never silently
        # swallow a misconfigured .env into a no-op dispatcher.
        if isinstance(v, list):
            parsed = [int(x) for x in v]
        elif isinstance(v, int) and not isinstance(v, bool):
            parsed = [int(v)]
        else:
            parsed = _parse_csv_int(v)
        if not parsed:
            raise ValueError(
                "TAVILY_MAX_RESULTS / REACT_MAX_SEARCH_CALLS must be a non-empty CSV "
                "of positive integers (e.g. '5' or '5,10')"
            )
        for n in parsed:
            if n <= 0:
                raise ValueError(
                    f"TAVILY_MAX_RESULTS / REACT_MAX_SEARCH_CALLS values must be > 0; got {n}"
                )
        return parsed

    @field_validator("LLM_REASONING_MODEL_PATTERNS", mode="before")
    @classmethod
    def _parse_reasoning_patterns(cls, v: Any) -> list[str]:
        return _parse_csv(v)

    @field_validator("GRID_DEFAULT_R", "GRID_DEFAULT_C", mode="before")
    @classmethod
    def _parse_grid_default(cls, v: Any) -> Any:
        # `.env.example` documents that leaving these blank means "take the
        # first list element". A literal empty string from the env file would
        # otherwise fail pydantic's int parser, so coerce "" → None here.
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @field_validator("MODEL_TRAINING_CUTOFFS", mode="before")
    @classmethod
    def _parse_training_cutoffs(cls, v: Any) -> dict[str, date]:
        return _parse_cutoffs(v)

    @field_validator("MODEL_MAX_TOKENS_PARAM", mode="before")
    @classmethod
    def _parse_max_tokens_param_field(cls, v: Any) -> dict[str, str]:
        return _parse_max_tokens_param(v)

    @field_validator("COMPOSITE_WEIGHTS_QTYPE", mode="before")
    @classmethod
    def _parse_composite_weights_qtype(cls, v: Any) -> dict[str, float]:
        # 空字符串视为 "走默认 dict" — pydantic field default_factory 已经
        # 提供了完整的默认权重，这里只在用户显式设置时解析。
        if v is None or v == "":
            return {
                "yes_no": 0.15,
                "binary_named": 0.15,
                "multiple_choice": 0.70,
            }
        return _parse_weights_dict(
            v,
            allowed_buckets=COMPOSITE_QTYPE_BUCKETS,
            field_name="COMPOSITE_WEIGHTS_QTYPE",
        )

    @field_validator("COMPOSITE_WEIGHTS_CTYPE", mode="before")
    @classmethod
    def _parse_composite_weights_ctype(cls, v: Any) -> dict[str, float]:
        if v is None or v == "":
            return {"single": 0.40, "multi": 0.60}
        return _parse_weights_dict(
            v,
            allowed_buckets=COMPOSITE_CTYPE_BUCKETS,
            field_name="COMPOSITE_WEIGHTS_CTYPE",
        )

    @field_validator("COMPOSITE_WEIGHT_OVERRIDES_QTYPE", mode="before")
    @classmethod
    def _parse_composite_overrides_qtype(
        cls, v: Any
    ) -> dict[str, dict[str, float]]:
        return _parse_overrides_dict(
            v,
            allowed_buckets=COMPOSITE_QTYPE_BUCKETS,
            field_name="COMPOSITE_WEIGHT_OVERRIDES_QTYPE",
        )

    @field_validator("COMPOSITE_WEIGHT_OVERRIDES_CTYPE", mode="before")
    @classmethod
    def _parse_composite_overrides_ctype(
        cls, v: Any
    ) -> dict[str, dict[str, float]]:
        return _parse_overrides_dict(
            v,
            allowed_buckets=COMPOSITE_CTYPE_BUCKETS,
            field_name="COMPOSITE_WEIGHT_OVERRIDES_CTYPE",
        )

    @field_validator("TAVILY_INCLUDE_RAW_CONTENT", mode="before")
    @classmethod
    def _parse_include_raw_content(cls, v: Any) -> str:
        # 兼容旧布尔值: True → "markdown", False → "false"
        if isinstance(v, bool):
            return "markdown" if v else "false"
        s = str(v).strip().lower()
        if s == "true":
            return "markdown"
        if s in ("false", "markdown", "text"):
            return s
        raise ValueError(
            f"TAVILY_INCLUDE_RAW_CONTENT must be one of false|markdown|text "
            f"(or legacy bool); got {v!r}"
        )

    @field_validator("TAVILY_INCLUDE_ANSWER", mode="before")
    @classmethod
    def _parse_include_answer(cls, v: Any) -> str:
        if isinstance(v, bool):
            return "basic" if v else "false"
        s = str(v).strip().lower()
        if s == "true":
            return "basic"
        if s in ("false", "basic", "advanced"):
            return s
        raise ValueError(
            f"TAVILY_INCLUDE_ANSWER must be one of false|basic|advanced; got {v!r}"
        )

    @field_validator("TAVILY_SEARCH_DEPTH", mode="before")
    @classmethod
    def _parse_search_depth(cls, v: Any) -> str:
        s = str(v).strip().lower()
        if s in ("basic", "advanced"):
            return s
        raise ValueError(
            f"TAVILY_SEARCH_DEPTH must be one of basic|advanced; got {v!r}"
        )

    @field_validator("RUN_ID")
    @classmethod
    def _validate_run_id(cls, v: str) -> str:
        if v and not _RUN_ID_RE.match(v):
            raise ValueError(
                f"RUN_ID {v!r} does not match YYYYMMDD-HHMMSS-xxxx (4 hex) format"
            )
        return v

    @field_validator("SOURCE_TABLE")
    @classmethod
    def _validate_source_table(cls, v: str) -> str:
        # SOURCE_TABLE is interpolated into SQL — restrict to a safe identifier
        # so misconfiguration cannot turn into injection.
        if not _SQL_IDENT_RE.match(v):
            raise ValueError(
                f"SOURCE_TABLE {v!r} must match [A-Za-z_][A-Za-z0-9_]* (SQLite identifier)"
            )
        return v

    @model_validator(mode="after")
    def _post_validate(self) -> "Settings":
        if not self.MODELS:
            raise ValueError("MODELS must not be empty")
        for slug in self.MODELS:
            if slug.endswith(":online"):
                raise ValueError(
                    f"MODELS entry {slug!r} must not end with ':online' — "
                    "provider-native browsing is not allowed (see information-barrier spec)"
                )
            # Defensive: real_model slugs containing `::` would collide with the
            # virtual slug encoding `{real}::r{R}::c{C}` and break round-tripping
            # in `db.parse_virtual_slug`. Reject early with a clear message.
            if "::" in slug:
                raise ValueError(
                    f"MODELS entry {slug!r} must not contain '::' — that delimiter is "
                    "reserved for grid-search virtual slugs (compose_virtual_slug)"
                )
        # LLM_API_KEY 是普通字符串; TAVILY_API_KEY 现已升级为 list[str]
        # (CSV 多 key 支持), 需要逐个 placeholder 校验.
        if not self.LLM_API_KEY:
            raise ValueError("LLM_API_KEY must not be empty")
        if any(tok in self.LLM_API_KEY for tok in _PLACEHOLDER_TOKENS):
            raise ValueError(
                "LLM_API_KEY still holds a placeholder token; fill your real key in .env"
            )
        if self.ENABLE_WEB_SEARCH:
            if not self.TAVILY_API_KEY:
                raise ValueError(
                    "TAVILY_API_KEY must not be empty when ENABLE_WEB_SEARCH=true "
                    "(provide one or more comma-separated keys)"
                )
            for idx, key in enumerate(self.TAVILY_API_KEY):
                if not key:
                    raise ValueError(f"TAVILY_API_KEY[{idx}] is empty")
                if any(tok in key for tok in _PLACEHOLDER_TOKENS):
                    raise ValueError(
                        f"TAVILY_API_KEY[{idx}] still holds a placeholder token; "
                        "fill real keys in .env"
                    )
        if self.TAVILY_KEY_COOLDOWN_S < 0:
            raise ValueError(
                f"TAVILY_KEY_COOLDOWN_S must be >= 0; got {self.TAVILY_KEY_COOLDOWN_S}"
            )
        if self.SAMPLING_N < 1:
            raise ValueError("SAMPLING_N must be >= 1")
        if self.LLM_MAX_CONCURRENCY < 1 or self.SEARCH_MAX_CONCURRENCY < 1:
            raise ValueError("concurrency settings must be >= 1")
        if self.REACT_MAX_STEPS < 1:
            raise ValueError("REACT_MAX_STEPS must be >= 1")
        # REACT_MAX_SEARCH_CALLS is a list (multi-value grid axis); each entry
        # must be a non-negative C value. The field_validator already requires
        # > 0 at parse time, but keep this as a belt-and-braces guard for tests
        # that construct Settings via overrides.
        for c in self.REACT_MAX_SEARCH_CALLS:
            if c < 0:
                raise ValueError(
                    f"REACT_MAX_SEARCH_CALLS contains a negative value: {c}"
                )
        for r in self.TAVILY_MAX_RESULTS:
            if r <= 0:
                raise ValueError(
                    f"TAVILY_MAX_RESULTS must be > 0 per cell; got {r}"
                )
        if self.REACT_MIN_SEARCH_CALLS < 0:
            raise ValueError("REACT_MIN_SEARCH_CALLS must be >= 0 (0 = disabled)")
        # MIN is a single int but C is a list. Hard error only when MIN exceeds
        # the smallest C in the grid (then no cell could honor the floor). When
        # MIN exceeds *some* but not all cells, the dispatcher silently clamps
        # `effective_min = min(MIN, C)` per cell — see DESIGN.md decision 4.
        if self.REACT_MAX_SEARCH_CALLS and self.REACT_MIN_SEARCH_CALLS > min(self.REACT_MAX_SEARCH_CALLS):
            raise ValueError(
                "REACT_MIN_SEARCH_CALLS must not exceed min(REACT_MAX_SEARCH_CALLS) "
                f"(min_floor={self.REACT_MIN_SEARCH_CALLS}, c_list={self.REACT_MAX_SEARCH_CALLS})"
            )
        if self.REACT_MAX_NUDGES < 0:
            raise ValueError("REACT_MAX_NUDGES must be >= 0")
        # v5.1 harness-resilience 开关: pydantic 已校验 bool 类型; 这里只做防御性 type guard,
        # 防止测试通过 model_copy(update={...}) 把任意值塞进来.
        if not isinstance(self.REACT_FINAL_ANSWER_RETRY, bool):
            raise ValueError(
                f"REACT_FINAL_ANSWER_RETRY must be bool; got {type(self.REACT_FINAL_ANSWER_RETRY).__name__}"
            )
        if not isinstance(self.REACT_BUDGET_EXCEEDED_DROP_TOOLS, bool):
            raise ValueError(
                f"REACT_BUDGET_EXCEEDED_DROP_TOOLS must be bool; got "
                f"{type(self.REACT_BUDGET_EXCEEDED_DROP_TOOLS).__name__}"
            )
        # force-final-answer-near-limit-v1 三件套校验.
        if not isinstance(self.REACT_BUDGET_AWARENESS_PROTOCOL, bool):
            raise ValueError(
                f"REACT_BUDGET_AWARENESS_PROTOCOL must be bool; got "
                f"{type(self.REACT_BUDGET_AWARENESS_PROTOCOL).__name__}"
            )
        if not isinstance(self.REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT, bool):
            raise ValueError(
                f"REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT must be bool; got "
                f"{type(self.REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT).__name__}"
            )
        if self.REACT_FORCE_FINAL_ANSWER_LOOKAHEAD < 1:
            raise ValueError(
                "REACT_FORCE_FINAL_ANSWER_LOOKAHEAD must be >= 1 "
                f"(0 = no near-limit window; disable via REACT_FORCE_FINAL_ANSWER_NEAR_LIMIT=false instead); "
                f"got {self.REACT_FORCE_FINAL_ANSWER_LOOKAHEAD}"
            )
        if self.REACT_FORCE_FINAL_ANSWER_LOOKAHEAD > self.REACT_MAX_STEPS:
            raise ValueError(
                f"REACT_FORCE_FINAL_ANSWER_LOOKAHEAD={self.REACT_FORCE_FINAL_ANSWER_LOOKAHEAD} "
                f"must not exceed REACT_MAX_STEPS={self.REACT_MAX_STEPS} "
                "(otherwise every step would be treated as near-limit, defeating the purpose)"
            )
        if self.TAVILY_RAW_CONTENT_MAX_CHARS < 0:
            raise ValueError("TAVILY_RAW_CONTENT_MAX_CHARS must be >= 0 (0 = no truncation)")
        # Optional grid anchors: when set, must be one of the configured cells.
        if self.GRID_DEFAULT_R is not None and self.GRID_DEFAULT_R not in self.TAVILY_MAX_RESULTS:
            raise ValueError(
                f"GRID_DEFAULT_R={self.GRID_DEFAULT_R} not in TAVILY_MAX_RESULTS "
                f"={self.TAVILY_MAX_RESULTS}; pick one of the configured cells"
            )
        if self.GRID_DEFAULT_C is not None and self.GRID_DEFAULT_C not in self.REACT_MAX_SEARCH_CALLS:
            raise ValueError(
                f"GRID_DEFAULT_C={self.GRID_DEFAULT_C} not in REACT_MAX_SEARCH_CALLS "
                f"={self.REACT_MAX_SEARCH_CALLS}; pick one of the configured cells"
            )
        # search-leak-filter-v1 启动校验
        if self.LEAK_DETECTOR_FAIL_ACTION not in ("drop", "keep"):
            raise ValueError(
                f"LEAK_DETECTOR_FAIL_ACTION must be one of {{drop, keep}}; "
                f"got {self.LEAK_DETECTOR_FAIL_ACTION!r}"
            )
        if self.LEAK_DETECTOR_CONCURRENCY < 1:
            raise ValueError(
                f"LEAK_DETECTOR_CONCURRENCY must be >= 1; got {self.LEAK_DETECTOR_CONCURRENCY}"
            )
        if self.LEAK_DETECTOR_RETRY_MAX < 0:
            raise ValueError(
                f"LEAK_DETECTOR_RETRY_MAX must be >= 0; got {self.LEAK_DETECTOR_RETRY_MAX}"
            )
        if self.LEAK_DETECTOR_TIMEOUT_S <= 0:
            raise ValueError(
                f"LEAK_DETECTOR_TIMEOUT_S must be > 0; got {self.LEAK_DETECTOR_TIMEOUT_S}"
            )
        if self.LEAK_DETECTOR_TEMPERATURE < 0:
            raise ValueError(
                f"LEAK_DETECTOR_TEMPERATURE must be >= 0; got {self.LEAK_DETECTOR_TEMPERATURE}"
            )
        if self.LEAK_DETECTOR_MAX_TOKENS < 1:
            raise ValueError(
                f"LEAK_DETECTOR_MAX_TOKENS must be >= 1; got {self.LEAK_DETECTOR_MAX_TOKENS}"
            )
        if self.LEAK_DETECTOR_MODEL.endswith(":online"):
            raise ValueError(
                f"LEAK_DETECTOR_MODEL {self.LEAK_DETECTOR_MODEL!r} must not end with "
                "':online' — provider-native browsing is not allowed"
            )
        if self.ENABLE_SEARCH_LEAK_FILTER:
            if not self.ENABLE_WEB_SEARCH:
                raise ValueError(
                    "ENABLE_SEARCH_LEAK_FILTER requires ENABLE_WEB_SEARCH=true; "
                    "the leak filter only runs against tavily_search results"
                )
            if not self.LEAK_DETECTOR_API_KEY:
                raise ValueError(
                    "LEAK_DETECTOR_API_KEY must not be empty when "
                    "ENABLE_SEARCH_LEAK_FILTER=true"
                )
            if any(tok in self.LEAK_DETECTOR_API_KEY for tok in _PLACEHOLDER_TOKENS):
                raise ValueError(
                    "LEAK_DETECTOR_API_KEY still holds a placeholder token; "
                    "fill your real detector key in .env"
                )
            if not self.LEAK_DETECTOR_MODEL:
                raise ValueError(
                    "LEAK_DETECTOR_MODEL must not be empty when "
                    "ENABLE_SEARCH_LEAK_FILTER=true"
                )
            if any(tok in self.LEAK_DETECTOR_MODEL for tok in _PLACEHOLDER_TOKENS):
                raise ValueError(
                    "LEAK_DETECTOR_MODEL still holds a placeholder token; "
                    "fill a real model slug in .env"
                )
        # composite-score-by-subtype 后置校验: field_validator (mode=before) 不会
        # 处理 default_factory 直接产生的 dict, 也不会处理 model_copy(update=...)
        # 注入的值, 这里再走一遍以挡住代码内构造的 Settings.
        for bucket, weight in self.COMPOSITE_WEIGHTS_QTYPE.items():
            if bucket not in COMPOSITE_QTYPE_BUCKETS:
                raise ValueError(
                    f"COMPOSITE_WEIGHTS_QTYPE bucket {bucket!r} not in "
                    f"{sorted(COMPOSITE_QTYPE_BUCKETS)}"
                )
            if weight < 0:
                raise ValueError(
                    f"COMPOSITE_WEIGHTS_QTYPE[{bucket}] must be >= 0; got {weight}"
                )
        if not any(w > 0 for w in self.COMPOSITE_WEIGHTS_QTYPE.values()):
            raise ValueError(
                "COMPOSITE_WEIGHTS_QTYPE requires at least one weight > 0 "
                "(otherwise composite score has no defined denominator)"
            )
        for bucket, weight in self.COMPOSITE_WEIGHTS_CTYPE.items():
            if bucket not in COMPOSITE_CTYPE_BUCKETS:
                raise ValueError(
                    f"COMPOSITE_WEIGHTS_CTYPE bucket {bucket!r} not in "
                    f"{sorted(COMPOSITE_CTYPE_BUCKETS)}"
                )
            if weight < 0:
                raise ValueError(
                    f"COMPOSITE_WEIGHTS_CTYPE[{bucket}] must be >= 0; got {weight}"
                )
        if not any(w > 0 for w in self.COMPOSITE_WEIGHTS_CTYPE.values()):
            raise ValueError(
                "COMPOSITE_WEIGHTS_CTYPE requires at least one weight > 0"
            )
        for metric, sub in self.COMPOSITE_WEIGHT_OVERRIDES_QTYPE.items():
            if not metric:
                raise ValueError(
                    "COMPOSITE_WEIGHT_OVERRIDES_QTYPE has an empty metric name"
                )
            for bucket, weight in sub.items():
                if bucket not in COMPOSITE_QTYPE_BUCKETS:
                    raise ValueError(
                        f"COMPOSITE_WEIGHT_OVERRIDES_QTYPE[{metric}] bucket "
                        f"{bucket!r} not in {sorted(COMPOSITE_QTYPE_BUCKETS)}"
                    )
                if weight < 0:
                    raise ValueError(
                        f"COMPOSITE_WEIGHT_OVERRIDES_QTYPE[{metric}][{bucket}] "
                        f"must be >= 0; got {weight}"
                    )
            if sub and not any(w > 0 for w in sub.values()):
                raise ValueError(
                    f"COMPOSITE_WEIGHT_OVERRIDES_QTYPE[{metric}] requires at "
                    "least one weight > 0"
                )
        for metric, sub in self.COMPOSITE_WEIGHT_OVERRIDES_CTYPE.items():
            if not metric:
                raise ValueError(
                    "COMPOSITE_WEIGHT_OVERRIDES_CTYPE has an empty metric name"
                )
            for bucket, weight in sub.items():
                if bucket not in COMPOSITE_CTYPE_BUCKETS:
                    raise ValueError(
                        f"COMPOSITE_WEIGHT_OVERRIDES_CTYPE[{metric}] bucket "
                        f"{bucket!r} not in {sorted(COMPOSITE_CTYPE_BUCKETS)}"
                    )
                if weight < 0:
                    raise ValueError(
                        f"COMPOSITE_WEIGHT_OVERRIDES_CTYPE[{metric}][{bucket}] "
                        f"must be >= 0; got {weight}"
                    )
            if sub and not any(w > 0 for w in sub.values()):
                raise ValueError(
                    f"COMPOSITE_WEIGHT_OVERRIDES_CTYPE[{metric}] requires at "
                    "least one weight > 0"
                )
        return self

    def __repr__(self) -> str:
        redacted = {
            k: ("<redacted>" if k.endswith("_API_KEY") else v)
            for k, v in self.model_dump().items()
        }
        return f"Settings({redacted!r})"

    __str__ = __repr__

    def source_db_path(self) -> Path:
        return Path(self.SOURCE_DB).expanduser().resolve()

    def runs_root_path(self) -> Path:
        return Path(self.RUNS_ROOT).expanduser().resolve()

    def run_dir(self, run_id: str) -> Path:
        return self.runs_root_path() / run_id

    def log_dir_path(self) -> Path:
        return Path(self.LOG_DIR).expanduser().resolve()


def load_settings(**overrides: Any) -> Settings:
    """Construct a Settings instance; `overrides` take precedence over env."""
    return Settings(**overrides)
