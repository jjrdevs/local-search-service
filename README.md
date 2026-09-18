# Local RAG & Semantic Search Service

A private, self-hosted research stack: **crawl → extract → embed → index →
semantic search**, served as a hardened HTTP API. No cloud, no telemetry — runs
on your box (or one container).

```
seed URLs
   |
   v
crawl  (Playwright for JS-heavy pages · requests for static)
   |    + readability extraction into clean text
   |    + respects robots.txt + per-host rate limit by default
   v
docs.jsonl     (chunked, deduped)
   |
   v
embed  (sentence-transformers — Hugging Face sentence embeddings)
   |
   v
vector index   (in-memory cosine similarity; FAISS/Chroma-ready at scale)
   |
   v
FastAPI search API
   /local_search?q=...&k=5    -> ranked results
   /health                    -> liveness
   /metrics                   -> request metrics
   + SearchBudgetGuard        (per-minute request budget)
   + RequestMetrics           (lightweight observability)
   + ExternalSearchAdapter    (pluggable Firecrawl-style integration hook)
   + Dockerfile + docker-compose.yml (containerized backend)
```

---

## Why it's an honest RAG/semantic-search project (not a stub)

- **Embeddings, not string search** — `sentence-transformers` maps documents and
  queries into a shared vector space; ranking is cosine similarity on embeddings.
- **Real ingestion pipeline** — Playwright-backed crawling (JS-heavy) *or* `requests`
  (static), with `readability-lxml` extraction, `robots.txt` + per-host rate limiting,
  and a `--max-docs` volume guard for safe incremental crawls.
- **Hardened serving** — `SearchBudgetGuard` (per-minute request budget),
  `RequestMetrics` (`/metrics`), a pluggable `ExternalSearchAdapter`, and an index
  **backup path** for recovery (`backup_dir` on index artifacts).
- **Containerized** — `Dockerfile` (python:3.11-slim) + `docker-compose.yml`.
- **Tested** — three hardening test suites (`tests/test_phase4.py`, `test_phase5.py`,
  `test_hardening.py`).

## Stack

| Concern | Tech |
|---|---|
| Embeddings / search | **sentence-transformers** (transformer sentence embeddings), cosine similarity index |
| Ingestion | Playwright (JS) / requests (static), readability-lxml, tqdm, python-magic |
| API / serving | **FastAPI** + uvicorn |
| Hardening | `SearchBudgetGuard`, `RequestMetrics`, `backup_dir` index recovery |
| Integration | `ExternalSearchAdapter` (Firecrawl-style hook, opt-in) |
| Packaging | `Dockerfile` + `docker-compose.yml` |
| Tests | pytest (phase4 / phase5 / hardening) |

---

## Quickstart

```bash
pip install -r requirements.txt

# 1) Crawl seeds into a clean JSONL corpus
python crawl_extract.py --seeds seeds.txt --out docs.jsonl

# 2) Build the embedding + cosine index
python indexer.py --docs docs.jsonl --index-dir ./index

# 3) Serve the search API
uvicorn search_api:app --host 127.0.0.1 --port 9100

# 4) Query it
curl 'http://127.0.0.1:9100/local_search?q=embedding&k=5'
```

### Docker

```bash
docker compose up --build
curl 'http://127.0.0.1:9100/health'
```

### Optional: external search adapter (Firecrawl-style)

```bash
export FIRECRAWL_API_KEY=your-key            # in the same shell that starts the service
# or set it in a .env file next to docker-compose.yml for the containerized run
```

---

## Design notes (stated plainly, on purpose)

- **In-memory cosine index** — deliberately simple and fast to iterate; swap the
  backend to **FAISS/Chroma** when the corpus outgrows RAM. The interface
  (`index → rank(query, k)`) is the only thing that changes.
- **Safety-by-default crawling** — `robots.txt` + per-host rate limit are on; the
  `--max-docs` guard caps crawl volume. Do not enable wide crawls on untrusted hosts.
- **Private by default** — binds to `127.0.0.1`; no telemetry, no external calls
  unless the opt-in adapter is configured.

## Files

- `crawl_extract.py` — fetch + extract text from seed URLs (Playwright / requests).
- `indexer.py` — chunk, embed (`sentence-transformers`), write a cosine index.
- `search_api.py` — FastAPI app: `/local_search`, `/health`, `/metrics` + budget guard.
- `external_adapter.py` — pluggable (Firecrawl-style) external search hook.
- `Dockerfile`, `docker-compose.yml` — containerized backend.
- `tests/` — `test_phase4.py`, `test_phase5.py`, `test_hardening.py`.
