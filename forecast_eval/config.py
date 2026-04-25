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


def _parse_csv(raw: str | list[Any] | None) -> list[str]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _parse_csv_int(raw: str | list[Any] | None) -> list[int]:
    return [int(x) for x in _parse_csv(raw)]


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
    TAVILY_API_KEY: str = ""
    TAVILY_MAX_RESULTS: int = 5
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
    REACT_MAX_SEARCH_CALLS: int = 8
    # 反思协议总开关: 启用后 user prompt 末尾追加多步推理脚手架, 显著提升搜索 + 反思深度.
    # 不会写入 prompt_templates (因此 prompt_templates_hash 保持不变), 但会作为 user message
    # 实际文本落入每个 sample 的 user_prompt 字段, 同时通过 config_snapshot 记入 run_meta.
    REACT_REFLECTION_PROTOCOL: bool = True
    # 软性最低搜索次数: LLM 试图给最终答案但累计 web_search < min 时, 注入一条 user nudge
    # 让它继续检索. 0 = 关闭 (默认; 主要靠反思协议自然驱动). >0 = 开启兜底 floor.
    REACT_MIN_SEARCH_CALLS: int = 0
    # nudge 最多注入几次, 防止 LLM 与系统互相 nudge 死循环. REACT_MAX_STEPS 仍是硬天花板.
    REACT_MAX_NUDGES: int = 2

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

    @field_validator(
        "LLM_BACKOFF_NETWORK_S",
        "LLM_BACKOFF_RATE_LIMIT_S",
        "LLM_BACKOFF_SERVER_5XX_S",
        "SEARCH_BACKOFF_S",
        mode="before",
    )
    @classmethod
    def _parse_backoffs(cls, v: Any) -> list[int]:
        if isinstance(v, list):
            return [int(x) for x in v]
        return _parse_csv_int(v)

    @field_validator("LLM_REASONING_MODEL_PATTERNS", mode="before")
    @classmethod
    def _parse_reasoning_patterns(cls, v: Any) -> list[str]:
        return _parse_csv(v)

    @field_validator("MODEL_TRAINING_CUTOFFS", mode="before")
    @classmethod
    def _parse_training_cutoffs(cls, v: Any) -> dict[str, date]:
        return _parse_cutoffs(v)

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
        required_keys = ["LLM_API_KEY"]
        if self.ENABLE_WEB_SEARCH:
            required_keys.append("TAVILY_API_KEY")
        for key_name in required_keys:
            value = getattr(self, key_name)
            if not value:
                raise ValueError(f"{key_name} must not be empty")
            if any(tok in value for tok in _PLACEHOLDER_TOKENS):
                raise ValueError(
                    f"{key_name} still holds a placeholder token; fill your real key in .env"
                )
        if self.SAMPLING_N < 1:
            raise ValueError("SAMPLING_N must be >= 1")
        if self.LLM_MAX_CONCURRENCY < 1 or self.SEARCH_MAX_CONCURRENCY < 1:
            raise ValueError("concurrency settings must be >= 1")
        if self.REACT_MAX_STEPS < 1 or self.REACT_MAX_SEARCH_CALLS < 0:
            raise ValueError("REACT_MAX_STEPS must be >= 1 and REACT_MAX_SEARCH_CALLS >= 0")
        if self.REACT_MIN_SEARCH_CALLS < 0:
            raise ValueError("REACT_MIN_SEARCH_CALLS must be >= 0 (0 = disabled)")
        if self.REACT_MIN_SEARCH_CALLS > self.REACT_MAX_SEARCH_CALLS:
            raise ValueError(
                "REACT_MIN_SEARCH_CALLS must not exceed REACT_MAX_SEARCH_CALLS "
                f"(min={self.REACT_MIN_SEARCH_CALLS}, max={self.REACT_MAX_SEARCH_CALLS})"
            )
        if self.REACT_MAX_NUDGES < 0:
            raise ValueError("REACT_MAX_NUDGES must be >= 0")
        if self.TAVILY_RAW_CONTENT_MAX_CHARS < 0:
            raise ValueError("TAVILY_RAW_CONTENT_MAX_CHARS must be >= 0 (0 = no truncation)")
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
