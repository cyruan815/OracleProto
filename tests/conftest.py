"""Global pytest fixtures shared across the test suite."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _default_disable_leak_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """search-leak-filter-v1 默认 ENABLE_SEARCH_LEAK_FILTER=true 会要求
    LEAK_DETECTOR_API_KEY / LEAK_DETECTOR_MODEL 必填; 现有大量 fixture 不
    关心 leak filter, 这里把整个 suite 默认拨回 false 以维持本提案前的
    byte-level 行为. 专门测试 leak filter 的用例通过本地 monkeypatch.setenv
    再次开启即可 (见 test_leak_filter.py / test_search.py 中相关用例).
    """
    monkeypatch.setenv("ENABLE_SEARCH_LEAK_FILTER", "false")
