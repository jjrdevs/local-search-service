import os

from search_api import RequestMetrics, SearchBudgetGuard
from external_adapter import ExternalSearchAdapter


def test_budget_guard_blocks_after_limit():
    guard = SearchBudgetGuard(max_requests_per_minute=2)
    assert guard.allow_request() is True
    assert guard.allow_request() is True
    assert guard.allow_request() is False


def test_metrics_tracks_success_and_failure():
    metrics = RequestMetrics()
    metrics.record_success(0.1)
    metrics.record_failure("boom")
    snapshot = metrics.snapshot()
    assert snapshot["requests_total"] == 2
    assert snapshot["successes"] == 1
    assert snapshot["failures"] == 1
    assert snapshot["last_error"] == "boom"


def test_external_adapter_requires_configuration(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    adapter = ExternalSearchAdapter(enabled=False)
    assert adapter.search("hello") == []


def test_external_adapter_normalizes_response(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "dummy-key")

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "results": [
                    {"title": "Title", "url": "https://example.com", "snippet": "Hello world", "source": "firecrawl"}
                ]
            }

    def fake_post(url, json=None, headers=None, timeout=None):
        assert headers["Authorization"] == "Bearer dummy-key"
        assert json["q"] == "hello"
        assert json["limit"] == 1
        return DummyResponse()

    monkeypatch.setattr("external_adapter.requests.post", fake_post)
    adapter = ExternalSearchAdapter(enabled=True)

    results = adapter.search("hello", limit=1)
    assert len(results) == 1
    assert results[0]["title"] == "Title"
    assert results[0]["url"] == "https://example.com"
    assert results[0]["snippet"] == "Hello world"
    assert results[0]["source"] == "firecrawl"


def test_local_search_falls_back_to_external_when_index_unloaded(monkeypatch):
    monkeypatch.setattr("external_adapter.requests.post", lambda url, json=None, headers=None, timeout=None: type(
        "R", (), {"raise_for_status": lambda self: None, "json": lambda self: {"results": [{"title": "Fallback", "url": "https://ext.example.com", "snippet": "External snippet", "source": "firecrawl"}]}}
    )())
    monkeypatch.setenv("FIRECRAWL_API_KEY", "dummy-key")

    # Reload module to ensure _emb is unset for the test scenario.
    import importlib
    import search_api
    importlib.reload(search_api)

    # Simulate the local index not being loaded.
    search_api._emb = None
    result = search_api.local_search("query", k=3)

    assert len(result) == 1
    assert result[0].title == "Fallback"
    assert result[0].url == "https://ext.example.com"
    assert result[0].snippet == "External snippet"
