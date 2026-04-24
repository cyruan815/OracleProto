from __future__ import annotations

import json
from typing import Any


WEB_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information relevant to the question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


# Defensive assertion: any drift that adds an argument LLMs can control (like
# end_date) would break the information barrier, so fail loudly at import time.
assert set(WEB_SEARCH_SCHEMA["function"]["parameters"]["properties"].keys()) == {"query"}, (
    "web_search tool schema must only expose a single `query` parameter"
)
assert WEB_SEARCH_SCHEMA["function"]["parameters"]["required"] == ["query"]


def parse_tool_arguments(raw: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """Return `(args, error_message)` — exactly one is non-None."""
    if raw is None or raw == "":
        return {}, None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"invalid arguments JSON: {e.msg}"
    if not isinstance(value, dict):
        return None, "arguments must be a JSON object"
    return value, None


def extract_query(args: dict[str, Any]) -> tuple[str | None, str | None]:
    """Pull `query` out of tool arguments, ignoring any extras LLMs may inject."""
    q = args.get("query")
    if q is None:
        return None, "missing required argument: query"
    if not isinstance(q, str):
        return None, "query must be a string"
    if not q.strip():
        return None, "query must be non-empty"
    return q, None


def tool_error_message(tool_call_id: str, reason: str) -> dict[str, Any]:
    """Build a role=tool message carrying an error payload back to the LLM."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps({"error": reason}, ensure_ascii=False),
    }


def tool_result_message(tool_call_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps(payload, ensure_ascii=False),
    }
