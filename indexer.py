#!/usr/bin/env python3
"""Index documents by embedding passages and saving a small vector store.

This prototype uses `sentence-transformers` to embed text and stores
embeddings+metadata as files under --index-dir. Search is a cosine-similarity
scan (fine for small collections).
"""
import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse, urlunparse

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    path = unquote(parsed.path or "/")
    path = re.sub(r"//+", "/", path)
    if parsed.query:
        query = parsed.query
    else:
        query = ""
    return urlunparse((scheme, netloc, path, "", query, ""))


def is_near_duplicate(text_a: str, text_b: str, threshold: float = 0.9) -> bool:
    if not text_a or not text_b:
        return text_a == text_b
    tokens_a = set(re.findall(r"\w+", text_a.lower()))
    tokens_b = set(re.findall(r"\w+", text_b.lower()))
    if not tokens_a or not tokens_b:
        return text_a == text_b
    overlap = len(tokens_a & tokens_b) / max(1, len(tokens_a | tokens_b))
    return overlap >= threshold


def chunk_text(text, chunk_size=800, overlap=100):
    # chunk by characters as a simple heuristic
    text = text.strip()
    if not text:
        return []
    chunks = []
    i = 0
    L = len(text)
    while i < L:
        end = min(L, i + chunk_size)
        chunks.append(text[i:end])
        i = max(i + chunk_size - overlap, end)
    return chunks


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def build_index(docs_path, index_dir, model_name='all-MiniLM-L6-v2', backup_dir=None):
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    if backup_dir is not None:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name)

    passages = []
    metadatas = []
    seen = []
    for line in tqdm(open(docs_path, 'r', encoding='utf-8')):
        doc = json.loads(line)
        url = canonicalize_url(doc.get('url') or '')
        title = doc.get('title') or ''
        text = doc.get('text') or ''
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            if any(is_near_duplicate(c, existing) for existing in seen):
                continue
            passages.append(c)
            metadatas.append({'url': url, 'title': title, 'chunk': i})
            seen.append(c)

    if not passages:
        print('No passages to index')
        return

    # embed in batches
    batch_size = 64
    embeddings = []
    for i in tqdm(range(0, len(passages), batch_size)):
        batch = passages[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)
    # normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms==0]=1.0
    embeddings = embeddings / norms

    np.save(index_dir / 'embeddings.npy', embeddings)
    with open(index_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadatas, f)
    with open(index_dir / 'passages.jsonl', 'w', encoding='utf-8') as f:
        for p in passages:
            f.write(json.dumps({'text': p}, ensure_ascii=False) + '\n')

    manifest = {
        'model_name': model_name,
        'passage_count': len(passages),
        'embedding_shape': list(embeddings.shape),
    }
    with open(index_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    if backup_dir is not None:
        shutil.copy2(index_dir / 'embeddings.npy', backup_dir / 'embeddings.npy')
        shutil.copy2(index_dir / 'metadata.json', backup_dir / 'metadata.json')
        shutil.copy2(index_dir / 'passages.jsonl', backup_dir / 'passages.jsonl')
        shutil.copy2(index_dir / 'manifest.json', backup_dir / 'manifest.json')

    print('Wrote index to', index_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--docs', required=True, help='JSONL docs file from crawl_extract')
    ap.add_argument('--index-dir', required=True)
    ap.add_argument('--model', default='all-MiniLM-L6-v2')
    args = ap.parse_args()
    build_index(args.docs, args.index_dir, args.model)


if __name__ == '__main__':
    main()
