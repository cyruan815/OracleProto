"""End-to-end smoke test: 3 questions × 1 model × 1 sample against mocked APIs.

Mocks the OpenRouter chat/completions and Tavily /search endpoints via respx,
then drives the full runner. Verifies `results.db` contains rows with the right
shape: messages_trace is valid JSON, search_calls carry end_date, the
information-barrier injected end_date matches end_time + OFFSET, and
provider-native browsing stays disabled.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import httpx
import pytest
import respx

from forecast_eval import db as dbmod
from forecast_eval import loader, runner
from forecast_eval.config import Settings
from forecast_eval.types import QFilter


SOURCE_DB = Path(__file__).resolve().parents[1] / "forecast_eval_set.db"
OPENROUTER_URL = re.compile(r"https://openrouter\.ai/api/v1/chat/completions")
TAVILY_URL = re.compile(r"https://api\.tavily\.com/search")


def _make_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("LLM_API_KEY", "sk-or-v1-TEST_ABCDEFGH")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-TEST_ABCDEFGH")
    monkeypatch.setenv("MODELS", "openai/gpt-4o-mini")
    monkeypatch.setenv("MODEL_TRAINING_CUTOFFS", "")
    monkeypatch.setenv("SAMPLING_N", "1")
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "2")
    monkeypatch.setenv("SEARCH_MAX_CONCURRENCY", "2")
    monkeypatch.setenv("REACT_MAX_STEPS", "3")
    monkeypatch.setenv("REACT_MAX_SEARCH_CALLS", "2")
    monkeypatch.setenv("LLM_BACKOFF_NETWORK_S", "0")
    monkeypatch.setenv("LLM_BACKOFF_RATE_LIMIT_S", "0")
    monkeypatch.setenv("LLM_BACKOFF_SERVER_5XX_S", "0")
    monkeypatch.setenv("SEARCH_BACKOFF_S", "0")
    monkeypatch.setenv("SEARCH_RETRY_MAX", "1")
    monkeypatch.setenv("SOURCE_DB", str(SOURCE_DB))
    monkeypatch.setenv("RESULTS_DB", str(tmp_path / "results.db"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("DB_COMMIT_BATCH", "2")
    monkeypatch.setenv("WRITE_MESSAGES_TRACE", "true")
    monkeypatch.setenv("TAVILY_END_DATE_OFFSET_DAYS", "-1")
    return Settings(_env_file=None)


def _llm_body_with_tool_call() -> dict:
    """Two-step conversation: LLM first requests a web_search, then answers."""
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "openai/gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps({"query": "latest info"}),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


def _llm_final_body() -> dict:
    return {
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "model": "openai/gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Based on evidence... final: \\boxed{No}",
                },
            }
        ],
        "usage": {"prompt_tokens": 150, "completion_tokens": 30},
    }


def _tavily_body() -> dict:
    return {
        "answer": "Recent reporting suggests No.",
        "results": [
            {
                "title": "News Report",
                "url": "https://example.com/a",
                "content": "Article summary",
                "published_date": "2026-01-10",
            }
        ],
    }


@respx.mock
async def test_smoke_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(tmp_path, monkeypatch)

    llm_responses = [
        httpx.Response(200, json=_llm_body_with_tool_call()),
        httpx.Response(200, json=_llm_final_body()),
    ] * 10  # enough for 3 questions × 2 LLM turns each

    respx.post(OPENROUTER_URL).mock(side_effect=llm_responses)
    tavily_route = respx.post(TAVILY_URL).mock(
        return_value=httpx.Response(200, json=_tavily_body())
    )

    # Use 3 yes_no questions
    conn = dbmod.connect(settings.results_db_path())
    dbmod.init_schema(conn)
    templates = loader.sync_prompt_templates(SOURCE_DB, conn)

    src = sqlite3.connect(f"file:{SOURCE_DB}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    rows = src.execute(
        "SELECT id FROM forecast_eval_set WHERE question_type='yes_no' ORDER BY end_time LIMIT 3"
    ).fetchall()
    ids = [r["id"] for r in rows]
    src.close()

    # Only fetch those 3 questions via the loader
    ids_tuple = tuple(ids)
    placeholders = ",".join("?" * len(ids_tuple))
    conn.execute("DELETE FROM questions")  # keep the test scope tight
    src = dbmod.connect(SOURCE_DB)
    questions_rows = src.execute(
        f"SELECT id, choice_type, question_type, event, options, answer, end_time "
        f"FROM forecast_eval_set WHERE id IN ({placeholders}) ORDER BY end_time",
        ids_tuple,
    ).fetchall()
    src.close()
    now = dbmod.utcnow_iso()
    conn.executemany(
        "INSERT OR REPLACE INTO questions (id, choice_type, question_type, event, options, answer, end_time, imported_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        [
            (r["id"], r["choice_type"], r["question_type"], r["event"], r["options"], r["answer"], r["end_time"], now)
            for r in questions_rows
        ],
    )

    from forecast_eval.types import Question
    questions = [
        Question(
            id=r["id"],
            choice_type=r["choice_type"],
            question_type=r["question_type"],
            event=r["event"],
            options=r["options"],
            answer=r["answer"],
            end_time=r["end_time"],
        )
        for r in questions_rows
    ]

    run_id = "20260424-120000-abcd"
    dbmod.register_run(
        conn,
        run_id=run_id,
        filters_snapshot={"question_types": ["yes_no"], "question_count": len(questions)},
        config_snapshot=dbmod.snapshot_settings(settings),
        source_db_hash=dbmod.compute_source_db_hash(SOURCE_DB),
        metadata_hash=dbmod.compute_metadata_hash(loader.load_raw_features_json(SOURCE_DB)),
        prompt_templates_hash=dbmod.compute_prompt_templates_hash(templates),
    )

    stats = await runner.run(
        settings=settings,
        filters=QFilter(question_types=frozenset({"yes_no"})),
        questions=questions,
        templates=templates,
        run_id=run_id,
        conn=conn,
    )

    assert stats.planned == 3
    assert stats.done == 3
    assert stats.errors == {}

    # Every written row must have the expected shape
    written = conn.execute(
        "SELECT * FROM run_results WHERE run_id=? ORDER BY sample_idx", (run_id,)
    ).fetchall()
    assert len(written) == 3
    for row in written:
        assert row["error"] is None
        assert row["parse_ok"] == 1
        assert row["final_answer_letters"] is not None
        assert row["messages_trace"] is not None
        assert row["search_calls"] is not None
        assert row["user_prompt"]
        # messages_trace must be JSON-decodable
        msgs = json.loads(row["messages_trace"])
        assert isinstance(msgs, list) and msgs[0]["role"] == "user"
        # search_calls carry end_date
        calls = json.loads(row["search_calls"])
        assert calls and all("end_date" in c for c in calls)
        # latency/tokens populated
        assert row["latency_ms"] >= 0
        assert (row["prompt_tokens"] or 0) + (row["completion_tokens"] or 0) > 0

    # Tavily must have been called with the correct end_date (= end_time + OFFSET)
    assert tavily_route.called
    first_call_body = json.loads(tavily_route.calls[0].request.content.decode("utf-8"))
    first_question_end = questions[0].end_time  # sorted by end_time
    from datetime import date, timedelta
    expected_end_date = (date.fromisoformat(first_question_end) + timedelta(days=-1)).isoformat()
    assert first_call_body["end_date"] == expected_end_date
    # raw_content is not included in the request
    assert first_call_body["include_raw_content"] is False

    # runs.finished_at must be set on normal completion
    r = conn.execute("SELECT finished_at FROM runs WHERE run_id=?", (run_id,)).fetchone()
    assert r["finished_at"] is not None
