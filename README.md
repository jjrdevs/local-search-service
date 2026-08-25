# Local RAG Prototype for Hermes

Minimal local research stack: crawl → extract → embed → index → search.

Quickstart (in your Hermes venv):

```bash
cd /home/jjrdev/local_search_service
pip install -r requirements.txt
# Crawl a few seeds into docs.jsonl
python crawl_extract.py --seeds seeds.txt --out docs.jsonl
# Build an index
python indexer.py --docs docs.jsonl --index-dir ./index
# Run API
uvicorn search_api:app --reload --host 127.0.0.1 --port 9100
curl 'http://127.0.0.1:9100/local_search?q=embedding&k=5'
```

Design notes
- This prototype uses `sentence-transformers` for embeddings and a simple
  in-memory cosine-similarity search (sufficient for small collections).
- Replace the search backend with FAISS/Chroma when scaling is needed.
- The crawler supports Playwright for JS-heavy pages or `requests` for static.

Safety
- The crawler respects `robots.txt` and obeys a per-host rate limit by default.
- Do not enable wide crawls on untrusted hosts without reviewing the policy.

Files
- `crawl_extract.py`: fetches and extracts textual content from seed URLs.
- `indexer.py`: chunks, computes embeddings, and writes a simple index.
- `search_api.py`: FastAPI app that serves `/local_search` and `/health`.

Phase 4 additions
- `--max-docs` limits crawl volume for safer incremental runs.
- `backup_dir` creates a copy of the index artifacts for recovery.
- `Dockerfile` and `docker-compose.yml` let the service run as a containerized backend.

Phase 5 additions
- `SearchBudgetGuard` enforces a simple per-minute request budget.
- `RequestMetrics` exposes `/metrics` for lightweight observability.
- `ExternalSearchAdapter` provides a placeholder hook for future Firecrawl-style integration.

Using your Firecrawl key
- For a local run, export `FIRECRAWL_API_KEY` in the same shell that starts the service:
  `export FIRECRAWL_API_KEY=your-key-here`
- For Docker, set the same variable in the shell before starting Compose or place it in a `.env` file next to `docker-compose.yml`.
- The adapter reads this value from the service process environment, so it must be present where `search_api.py` is launched.
