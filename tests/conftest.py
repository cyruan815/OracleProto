"""Global pytest fixtures shared across the test suite."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _default_disable_leak_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """search-leak-filter-v1 default ENABLE_SEARCH_LEAK_FILTER=true requires
    LEAK_DETECTOR_API_KEY / LEAK_DETECTOR_MODEL to be set; many existing fixtures
    don't care about the leak filter, so we default the whole suite back to false
    to preserve the byte-level behavior from before this proposal. Test cases
    specifically exercising the leak filter can re-enable it via a local
    monkeypatch.setenv (see related cases in test_leak_filter.py / test_search.py).
    """
    monkeypatch.setenv("ENABLE_SEARCH_LEAK_FILTER", "false")
