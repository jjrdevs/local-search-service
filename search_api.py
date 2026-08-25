#!/usr/bin/env python3
"""Simple FastAPI service to search the local index produced by indexer.py

Endpoints:
- GET /local_search?q=...&k=5
"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from typing import List
from pathlib import Path
import logging
import os
import time

from external_adapter import ExternalSearchAdapter, RequestMetrics, SearchBudgetGuard


class Result(BaseModel):
    score: float
    snippet: str
    url: str | None = None
    title: str | None = None


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("local_search.api")

_AGENT_ENV_PATH = Path.home() / ".hermes/hermes-agent/.env"
_AGENT_ENV_KEYS = {"FIRECRAWL_API_KEY", "FIRECRAWL_API_URL"}


def _load_agent_env_file() -> None:
    if not _AGENT_ENV_PATH.is_file():
        return
    try:
        raw = _AGENT_ENV_PATH.read_text(encoding="utf-8")
    except OSError:
        return

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name not in _AGENT_ENV_KEYS:
            continue
        value = value.strip()
        if (value.startswith("'") and value.endswith("'")) or (
            value.startswith('"') and value.endswith('"')
        ):
            value = value[1:-1]
        if os.getenv(name) is None:
            os.environ[name] = value


_load_agent_env_file()

app = FastAPI(title='Local Search API', version='0.2.0')
_model = None
_emb = None
_meta = None
_passages = None
_budget_guard = SearchBudgetGuard(max_requests_per_minute=60)
_metrics = RequestMetrics()
_external_adapter = ExternalSearchAdapter(enabled=True)


def load_index(index_dir='index'):
    global _model, _emb, _meta, _passages
    index_dir = Path(index_dir)
    emb_path = index_dir / 'embeddings.npy'
    meta_path = index_dir / 'metadata.json'
    pass_path = index_dir / 'passages.jsonl'
    if not emb_path.exists():
        raise FileNotFoundError('index not found; run indexer first')
    _emb = np.load(emb_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        _meta = json.load(f)
    _passages = [json.loads(l) for l in open(pass_path, 'r', encoding='utf-8')]
    _model = SentenceTransformer('all-MiniLM-L6-v2')


def _search(query: str, k: int = 5):
    qv = _model.encode([query], show_progress_bar=False)
    qv = qv / np.linalg.norm(qv, axis=1, keepdims=True)
    sims = (_emb @ qv[0])
    ids = np.argsort(-sims)[:k]
    results = []
    for i in ids:
        m = _meta[i]
        p = _passages[i]['text']
        results.append(Result(score=float(sims[i]), snippet=p, url=m.get('url'), title=m.get('title')))
    return results


@app.on_event('startup')
def startup_event():
    try:
        load_index('index')
        print('Loaded local index')
    except Exception as e:
        print('Index load failed:', e)


@app.get('/health')
def health():
    return {'status': 'ok' if _emb is not None else 'degraded', 'index_loaded': _emb is not None}


@app.get('/metrics')
def metrics():
    return _metrics.snapshot()


@app.get('/local_search', response_model=List[Result])
def local_search(q: str = Query(..., min_length=1), k: int = 5):
    if not _budget_guard.allow_request():
        raise HTTPException(status_code=429, detail='Request budget exceeded')

    start = time.time()
    try:
        external_results = []
        if _external_adapter.enabled and _external_adapter.api_key:
            external_results = _external_adapter.search(q, limit=max(2, k // 2))

        if _emb is None:
            if external_results:
                _metrics.record_success((time.time() - start) * 1000)
                return [Result(score=0.0, snippet=item['snippet'], url=item['url'], title=item['title']) for item in external_results]
            raise HTTPException(status_code=503, detail='Index not loaded')

        local_results = _search(q, k)
        if external_results:
            local_results.extend(
                [Result(score=0.0, snippet=item['snippet'], url=item['url'], title=item['title']) for item in external_results]
            )
        _metrics.record_success((time.time() - start) * 1000)
        return local_results
    except HTTPException:
        raise
    except Exception as exc:
        _metrics.record_failure(str(exc))
        raise


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('search_api:app', host='127.0.0.1', port=9100, reload=True)
