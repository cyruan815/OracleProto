from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger

from .config import Settings
from .tavily_keys import AllKeysExhausted, TavilyKeyPool, get_pool


TAVILY_ENDPOINT = "https://api.tavily.com/search"
# Tavily 用 401 / 403 表示 key 无效或权限问题 (永久拉黑); 429 / 配额相关返回
# 也走 429 (临时 cooldown). 其他状态码归到 "other" (网络/服务器错误).
_AUTH_STATUS = frozenset({401, 403})
_RATE_LIMIT_STATUS = frozenset({429})


@dataclass
class SearchResultItem:
    title: str
    url: str
    content: str
    published_date: str | None = None
    # Tavily 返回的相关性 score (0-1, 越大越相关). 缺失时为 None.
    score: float | None = None
    # 完整页面正文 (markdown / text). include_raw_content="false" 时 Tavily 不返回, 保持 None.
    # 长度上限由 settings.TAVILY_RAW_CONTENT_MAX_CHARS 在 _parse_tavily_response 处截断.
    raw_content: str | None = None


@dataclass
class SearchResult:
    query: str
    end_date: str
    answer: str | None = None
    results: list[SearchResultItem] = field(default_factory=list)
    error_kind: str | None = None
    error_message: str | None = None

    @property
    def ok(self) -> bool:
        return self.error_kind is None

    def to_llm_payload(self) -> dict[str, Any]:
        """Compact payload handed back to the LLM.

        可选字段 (published_date / score / raw_content / answer) 仅在非 None 时输出,
        避免 LLM 看到一堆 null 占位降低判断信号. error 路径保留 answer=None 以便
        消费方一眼区分成功/失败.
        """
        if self.error_kind is not None:
            return {
                "error": self.error_kind,
                "message": self.error_message or "",
                "results": [],
                "answer": None,
            }
        items: list[dict[str, Any]] = []
        for r in self.results:
            item: dict[str, Any] = {
                "title": r.title,
                "url": r.url,
                "content": r.content,
            }
            if r.published_date is not None:
                item["published_date"] = r.published_date
            if r.score is not None:
                item["score"] = r.score
            if r.raw_content is not None:
                item["raw_content"] = r.raw_content
            items.append(item)
        payload: dict[str, Any] = {"results": items}
        if self.answer is not None:
            payload["answer"] = self.answer
        return payload


def _truncate_raw_content(text: str | None, max_chars: int) -> str | None:
    """`max_chars=0` 表示不截断; 命中阈值时追加省略提示."""
    if text is None:
        return None
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n…[truncated to {max_chars} chars]"


def _parse_tavily_response(
    data: dict[str, Any],
    query: str,
    end_date: str,
    *,
    raw_content_max_chars: int,
) -> SearchResult:
    answer = data.get("answer")
    items: list[SearchResultItem] = []
    for raw in data.get("results", []) or []:
        if not isinstance(raw, dict):
            continue
        score_raw = raw.get("score")
        try:
            score = float(score_raw) if score_raw is not None else None
        except (TypeError, ValueError):
            score = None
        rc = raw.get("raw_content")
        rc_str = str(rc) if isinstance(rc, str) and rc else None
        rc_str = _truncate_raw_content(rc_str, raw_content_max_chars)
        items.append(
            SearchResultItem(
                title=str(raw.get("title") or ""),
                url=str(raw.get("url") or ""),
                content=str(raw.get("content") or ""),
                published_date=raw.get("published_date"),
                score=score,
                raw_content=rc_str,
            )
        )
    return SearchResult(query=query, end_date=end_date, answer=answer, results=items)


def _build_request_payload(
    *,
    query: str,
    end_date: str,
    settings: Settings,
    api_key: str,
) -> dict[str, Any]:
    """组装 Tavily 请求体. 把 enum 字符串映射到 Tavily 协议接受的类型,
    且 include_answer="false" 时整个字段不发送 (Tavily 默认即 false).

    `api_key` 由调用方从 TavilyKeyPool 取出 (而非读 settings.TAVILY_API_KEY,
    后者已升级为 list[str]); 这样每次请求都能换 key 实现轮换 + 熔断.
    """
    raw_setting = settings.TAVILY_INCLUDE_RAW_CONTENT
    # Tavily 的 include_raw_content 接受 bool | "markdown" | "text"
    raw_param: bool | str = False if raw_setting == "false" else raw_setting

    payload: dict[str, Any] = {
        "api_key": api_key,
        "query": query,
        "end_date": end_date,
        "max_results": settings.TAVILY_MAX_RESULTS,
        "search_depth": settings.TAVILY_SEARCH_DEPTH,
        "include_raw_content": raw_param,
    }
    if settings.TAVILY_INCLUDE_ANSWER != "false":
        payload["include_answer"] = settings.TAVILY_INCLUDE_ANSWER
    return payload


async def _single_request(
    client: httpx.AsyncClient,
    *,
    query: str,
    end_date: str,
    settings: Settings,
    api_key: str,
) -> httpx.Response:
    return await client.post(
        TAVILY_ENDPOINT,
        json=_build_request_payload(
            query=query, end_date=end_date, settings=settings, api_key=api_key
        ),
        timeout=settings.LLM_TIMEOUT_S,
    )


async def tavily_search(
    query: str,
    end_date: str,
    settings: Settings,
    *,
    client: httpx.AsyncClient | None = None,
    pool: TavilyKeyPool | None = None,
) -> SearchResult:
    """Call Tavily /search with the given `end_date` injected by the caller.

    Multi-key behaviour:
    - 每次尝试前从 `pool` (默认按 settings.TAVILY_API_KEY 走 process-wide cache)
      acquire 一把 least-used 健康 key.
    - 401/403 → 该 key 永久拉黑, **不算网络重试**, 立即换 key 再试 (no sleep).
    - 429 → 该 key 临时 cooldown, 同样立即换 key 再试.
    - 5xx / 网络 / bad-JSON → 计入 `SEARCH_RETRY_MAX` 网络重试配额, 按
      `SEARCH_BACKOFF_S` sleep 后再试 (key 不拉黑).
    - 所有 key 都不可用 (AllKeysExhausted) → 立即返回错误, 不再轮转.

    返回值仍是 `SearchResult` (不抛异常), 让 ReAct 循环能把 tool_result 喂回 LLM.
    """
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient()

    if pool is None:
        pool = get_pool(list(settings.TAVILY_API_KEY), float(settings.TAVILY_KEY_COOLDOWN_S))

    sequence = list(settings.SEARCH_BACKOFF_S)
    max_attempts = max(1, int(settings.SEARCH_RETRY_MAX))
    n_keys = max(1, len(pool.states))
    # 硬上限防 cooldown_s=0 / 全部 key 持续返回 429 时的死循环;
    # 正常路径靠 AllKeysExhausted 或 max_attempts 退出.
    hard_limit = max_attempts + n_keys
    last_error_kind: str | None = None
    last_error_message: str | None = None
    raw_content_max_chars = int(settings.TAVILY_RAW_CONTENT_MAX_CHARS)

    network_attempts = 0
    total_iterations = 0

    try:
        while total_iterations < hard_limit:
            total_iterations += 1
            try:
                api_key = await pool.acquire()
            except AllKeysExhausted as e:
                last_error_kind = "all_keys_exhausted"
                last_error_message = str(e)
                logger.warning("tavily {}", last_error_message)
                break

            network_class_error = False
            try:
                resp = await _single_request(
                    client, query=query, end_date=end_date, settings=settings, api_key=api_key
                )
            except (
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
                asyncio.TimeoutError,
            ) as e:
                await pool.report_failure(api_key, "other")
                last_error_kind = "network"
                last_error_message = f"{type(e).__name__}: {e}"
                logger.debug(
                    "tavily network error iter={} err={}", total_iterations, last_error_message
                )
                network_class_error = True
            else:
                status = resp.status_code
                if status == 200:
                    try:
                        data = resp.json()
                    except ValueError as e:
                        # 200 但非 JSON: 不是 key 问题, 走网络重试配额; 不拉黑 key.
                        await pool.report_ok(api_key)
                        last_error_kind = "bad_response"
                        last_error_message = f"Tavily returned non-JSON body: {e}"
                        network_class_error = True
                    else:
                        await pool.report_ok(api_key)
                        return _parse_tavily_response(
                            data,
                            query=query,
                            end_date=end_date,
                            raw_content_max_chars=raw_content_max_chars,
                        )
                elif status in _AUTH_STATUS:
                    await pool.report_failure(api_key, "auth")
                    last_error_kind = "auth"
                    last_error_message = f"HTTP {status}: {resp.text[:500]}"
                    logger.debug("tavily auth error iter={} status={}", total_iterations, status)
                    # 不算网络重试, 也不 sleep — 直接换 key 重试.
                elif status in _RATE_LIMIT_STATUS:
                    await pool.report_failure(api_key, "rate_limit")
                    last_error_kind = "rate_limit"
                    last_error_message = f"HTTP {status}: {resp.text[:500]}"
                    logger.debug(
                        "tavily rate limit iter={} status={}", total_iterations, status
                    )
                    # 同上: 不计 attempt, 不 sleep.
                else:
                    # 5xx / 其他 4xx: 视作服务端瞬时故障, 不拉黑 key, 走网络重试.
                    await pool.report_failure(api_key, "other")
                    last_error_kind = "http_error"
                    last_error_message = f"HTTP {status}: {resp.text[:500]}"
                    logger.debug(
                        "tavily http error iter={} status={}", total_iterations, status
                    )
                    network_class_error = True

            if network_class_error:
                network_attempts += 1
                if network_attempts >= max_attempts:
                    break
                idx = network_attempts - 1
                wait_s = float(sequence[idx]) if idx < len(sequence) else 0.0
                if wait_s > 0:
                    await asyncio.sleep(wait_s)
            # 否则 (auth / rate_limit): 直接进入下一次循环换 key, 不 sleep.
    finally:
        if owns_client:
            await client.aclose()

    return SearchResult(
        query=query,
        end_date=end_date,
        error_kind="tavily_error",
        error_message=last_error_message or last_error_kind or "Tavily failed",
    )
