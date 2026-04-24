from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger

from .config import Settings


TAVILY_ENDPOINT = "https://api.tavily.com/search"


@dataclass
class SearchResultItem:
    title: str
    url: str
    content: str
    published_date: str | None = None


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
        """Compact payload handed back to the LLM (no raw_content, no HTML)."""
        if self.error_kind is not None:
            return {
                "error": self.error_kind,
                "message": self.error_message or "",
                "results": [],
                "answer": None,
            }
        return {
            "answer": self.answer,
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "content": r.content,
                    "published_date": r.published_date,
                }
                for r in self.results
            ],
        }


def _parse_tavily_response(data: dict[str, Any], query: str, end_date: str) -> SearchResult:
    answer = data.get("answer")
    items = []
    for raw in data.get("results", []) or []:
        if not isinstance(raw, dict):
            continue
        items.append(
            SearchResultItem(
                title=str(raw.get("title") or ""),
                url=str(raw.get("url") or ""),
                content=str(raw.get("content") or ""),
                published_date=raw.get("published_date"),
            )
        )
    return SearchResult(query=query, end_date=end_date, answer=answer, results=items)


async def _single_request(
    client: httpx.AsyncClient,
    *,
    query: str,
    end_date: str,
    settings: Settings,
) -> httpx.Response:
    payload = {
        "api_key": settings.TAVILY_API_KEY,
        "query": query,
        "end_date": end_date,
        "max_results": settings.TAVILY_MAX_RESULTS,
        "include_raw_content": settings.TAVILY_INCLUDE_RAW_CONTENT,
    }
    return await client.post(
        TAVILY_ENDPOINT,
        json=payload,
        timeout=settings.LLM_TIMEOUT_S,
    )


async def tavily_search(
    query: str,
    end_date: str,
    settings: Settings,
    *,
    client: httpx.AsyncClient | None = None,
) -> SearchResult:
    """Call Tavily /search with the given `end_date` injected by the caller.

    Retries non-2xx / network errors using `SEARCH_BACKOFF_S`. If every attempt
    fails, returns a `SearchResult` with `error_kind` set instead of raising so
    the ReAct loop can still feed a tool_result back to the LLM.
    """
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient()

    sequence = list(settings.SEARCH_BACKOFF_S)
    max_attempts = max(1, int(settings.SEARCH_RETRY_MAX))
    last_error_kind: str | None = None
    last_error_message: str | None = None

    try:
        for attempt in range(1, max_attempts + 1):
            try:
                resp = await _single_request(client, query=query, end_date=end_date, settings=settings)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout, asyncio.TimeoutError) as e:
                last_error_kind = "network"
                last_error_message = f"{type(e).__name__}: {e}"
                logger.debug("tavily network error attempt={} err={}", attempt, last_error_message)
            else:
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except ValueError as e:
                        last_error_kind = "bad_response"
                        last_error_message = f"Tavily returned non-JSON body: {e}"
                    else:
                        return _parse_tavily_response(data, query=query, end_date=end_date)
                else:
                    last_error_kind = "http_error"
                    last_error_message = f"HTTP {resp.status_code}: {resp.text[:500]}"
                    logger.debug(
                        "tavily http error attempt={} status={}",
                        attempt,
                        resp.status_code,
                    )

            if attempt >= max_attempts:
                break
            idx = attempt - 1
            wait_s = float(sequence[idx]) if idx < len(sequence) else 0.0
            if wait_s > 0:
                await asyncio.sleep(wait_s)
    finally:
        if owns_client:
            await client.aclose()

    return SearchResult(
        query=query,
        end_date=end_date,
        error_kind="tavily_error",
        error_message=last_error_message or last_error_kind or "Tavily failed",
    )
