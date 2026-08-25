import json
from pathlib import Path

import numpy as np

import crawl_extract
import indexer


class FakeModel:
    def encode(self, batch, show_progress_bar=False):
        return np.array([[1.0, 0.0] for _ in batch], dtype=float)


def test_crawl_honors_max_docs_limit(tmp_path, monkeypatch):
    out_path = tmp_path / "docs.jsonl"

    monkeypatch.setattr(crawl_extract, "fetch_with_retry", lambda url, fetcher, max_retries=3, base_delay=0.2: "<html></html>")
    monkeypatch.setattr(crawl_extract, "extract_readability", lambda html: ("title", "text"))

    crawl_extract.crawl(["https://example.com/1", "https://example.com/2", "https://example.com/3"], str(out_path), max_docs=2)

    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_build_index_creates_manifest_and_backup(tmp_path, monkeypatch):
    docs_path = tmp_path / "docs.jsonl"
    docs_path.write_text(json.dumps({"url": "https://example.com", "title": "Example", "text": "hello world"}) + "\n", encoding="utf-8")

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "embeddings.npy").write_bytes(b"old")
    (index_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (index_dir / "passages.jsonl").write_text("{}\n", encoding="utf-8")

    backup_dir = tmp_path / "backup"

    monkeypatch.setattr(indexer, "SentenceTransformer", lambda model_name: FakeModel())

    indexer.build_index(str(docs_path), str(index_dir), backup_dir=str(backup_dir))

    assert (index_dir / "manifest.json").exists()
    assert (backup_dir / "embeddings.npy").exists()
    assert (backup_dir / "metadata.json").exists()
    assert (backup_dir / "passages.jsonl").exists()
