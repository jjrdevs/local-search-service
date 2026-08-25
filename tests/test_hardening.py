import pytest

from crawl_extract import CircuitBreaker, canonicalize_url, fetch_with_retry
from indexer import is_near_duplicate


def test_canonicalize_url_normalizes_common_variants():
    assert canonicalize_url("https://example.com/path?x=1&y=2#frag") == "https://example.com/path?x=1&y=2"
    assert canonicalize_url("https://example.com:443/path/") == "https://example.com/path"
    assert canonicalize_url("https://example.com/%7Euser") == "https://example.com/~user"


def test_near_duplicate_detection_uses_similarity_threshold():
    assert is_near_duplicate("hello world from the same text", "hello world from the same text") is True
    assert is_near_duplicate("hello world from the same text", "hello world from a slightly different text") is False


def test_fetch_with_retry_retries_transient_failures_and_stops_after_success(monkeypatch):
    calls = {"count": 0}

    def flaky(url):
        calls["count"] += 1
        if calls["count"] < 3:
            raise TimeoutError("temporary")
        return "ok"

    result = fetch_with_retry("https://example.com", flaky, max_retries=3, base_delay=0.0)

    assert result == "ok"
    assert calls["count"] == 3


def test_circuit_breaker_blocks_requests_after_threshold():
    breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=60)
    breaker.record_failure()
    breaker.record_failure()

    assert breaker.allow_request() is False
