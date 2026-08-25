#!/usr/bin/env python3
"""Crawl and extract textual content from seed URLs.

Usage:
  python crawl_extract.py --seeds seeds.txt --out docs.jsonl [--use-playwright]

Produces a JSONL file with one object per document: {url, title, text, fetched_at}
"""
import argparse
import json
import logging
import re
import time
from datetime import datetime
from urllib.parse import urlparse, urlunparse, unquote

import requests
from readability import Document
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("local_search.crawler")

try:
    from playwright.sync_api import sync_playwright
    _has_playwright = True
except Exception:
    _has_playwright = False


class CircuitBreaker:
    def __init__(self, failure_threshold=3, cooldown_seconds=300):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.failures = 0
        self.opened_at = None

    def allow_request(self):
        if self.opened_at is not None and time.time() - self.opened_at < self.cooldown_seconds:
            return False
        if self.opened_at is not None:
            self.opened_at = None
            self.failures = 0
        return True

    def record_failure(self):
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = time.time()

    def record_success(self):
        self.failures = 0
        self.opened_at = None


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    path = unquote(parsed.path or "/")
    path = re.sub(r"//+", "/", path)
    if parsed.port in (80, 443) and parsed.port is not None:
        netloc = netloc.rsplit(":", 1)[0]
    if path.endswith("/") and path != "/":
        path = path.rstrip("/")
    if parsed.query:
        query = parsed.query
    else:
        query = ""
    return urlunparse((scheme, netloc, path, "", query, ""))


def fetch_with_retry(url, fetcher, max_retries=3, base_delay=0.2):
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            return fetcher(url)
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
            time.sleep(base_delay * (attempt + 1))
    raise last_error


def fetch_requests(url, timeout=15):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "HermesLocalRAG/1.0"})
    r.raise_for_status()
    return r.text


def extract_readability(html):
    doc = Document(html)
    title = doc.short_title() or ''
    summary = doc.summary()
    # strip tags simply
    text = _strip_tags(summary)
    return title, text


def _strip_tags(html: str) -> str:
    # very small and safe HTML -> text fallback
    import re
    text = re.sub(r'<script.*?>.*?</script>', '', html, flags=re.S)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def fetch_playwright(url, timeout=30):
    assert _has_playwright, 'playwright not installed or playwright browsers not installed'
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=timeout*1000)
        html = page.content()
        browser.close()
        return html


def crawl(seeds, out_path, use_playwright=False, max_docs=None):
    breaker = CircuitBreaker()
    written = 0
    with open(out_path, 'w', encoding='utf-8') as out:
        for url in tqdm(seeds):
            if max_docs is not None and written >= max_docs:
                logger.info('Reached max_docs=%s; stopping crawl', max_docs)
                break
            url = url.strip()
            if not url:
                continue
            canonical = canonicalize_url(url)
            if not breaker.allow_request():
                logger.warning('Skipping %s: circuit breaker open', canonical)
                continue
            try:
                if use_playwright:
                    html = fetch_with_retry(canonical, lambda u: fetch_playwright(u))
                else:
                    html = fetch_with_retry(canonical, lambda u: fetch_requests(u))
                breaker.record_success()
                title, text = extract_readability(html)
                doc = {
                    'url': canonical,
                    'title': title,
                    'text': text,
                    'fetched_at': time.time(),
                }
                out.write(json.dumps(doc, ensure_ascii=False) + '\n')
                written += 1
            except Exception as e:
                breaker.record_failure()
                logger.exception('Failed %s: %s', canonical, e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', required=True, help='File with seed URLs, one per line')
    ap.add_argument('--out', required=True, help='Output JSONL file')
    ap.add_argument('--use-playwright', action='store_true')
    ap.add_argument('--max-docs', type=int, default=None, help='Stop after writing this many documents')
    args = ap.parse_args()

    with open(args.seeds, 'r', encoding='utf-8') as f:
        seeds = [l.strip() for l in f if l.strip()]
    crawl(seeds, args.out, use_playwright=args.use_playwright, max_docs=args.max_docs)


if __name__ == '__main__':
    main()
