import json
import logging
import os
from typing import Any, List

import requests


class SearchBudgetGuard:
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests_per_minute = max_requests_per_minute
        self._window_start = None
        self._count = 0

    def allow_request(self) -> bool:
        import time

        now = time.time()
        if self._window_start is None or now - self._window_start >= 60:
            self._window_start = now
            self._count = 0
        if self._count >= self.max_requests_per_minute:
            return False
        self._count += 1
        return True


class RequestMetrics:
    def __init__(self) -> None:
        self.requests_total = 0
        self.successes = 0
        self.failures = 0
        self.last_error = None
        self.last_latency_ms = None

    def record_success(self, latency_ms: float) -> None:
        self.requests_total += 1
        self.successes += 1
        self.last_latency_ms = latency_ms

    def record_failure(self, error: str) -> None:
        self.requests_total += 1
        self.failures += 1
        self.last_error = error

    def snapshot(self) -> dict:
        return {
            "requests_total": self.requests_total,
            "successes": self.successes,
            "failures": self.failures,
            "last_error": self.last_error,
            "last_latency_ms": self.last_latency_ms,
        }


logger = logging.getLogger("local_search.external_adapter")


class ExternalSearchAdapter:
    DEFAULT_BASE_URL = "https://api.firecrawl.dev/v1/search"

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.api_key = os.getenv("FIRECRAWL_API_KEY", "").strip()
        self.base_url = os.getenv("FIRECRAWL_API_URL", self.DEFAULT_BASE_URL).strip()

    def _normalize_result(self, item: Any) -> dict | None:
        if not isinstance(item, dict):
            return None
        snippet = item.get("snippet") or item.get("text") or item.get("description") or ""
        return {
            "title": item.get("title") or item.get("name") or None,
            "url": item.get("url") or item.get("link") or None,
            "snippet": snippet,
            "source": item.get("source", "external"),
        }

    def search(self, query: str, limit: int = 5) -> List[dict]:
        if not self.enabled:
            return []

        if not self.api_key:
            self.api_key = os.getenv("FIRECRAWL_API_KEY", "").strip()
        if not self.base_url:
            self.base_url = os.getenv("FIRECRAWL_API_URL", self.DEFAULT_BASE_URL).strip()
        if not self.api_key or not self.base_url:
            return []

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {"q": query, "limit": limit}
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            body = response.json()

            if isinstance(body, dict):
                items = body.get("results") or body.get("items") or body.get("data") or []
            else:
                items = body

            results: List[dict] = []
            for item in items:
                normalized = self._normalize_result(item)
                if normalized is None:
                    continue
                results.append(normalized)
                if len(results) >= limit:
                    break
            return results
        except Exception as exc:
            logger.debug("External search failed: %s", exc, exc_info=True)
            return []
