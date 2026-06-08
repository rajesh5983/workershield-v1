"""
WorkerShield ingest orchestrator.

For each document in docs/corpus_registry.yaml:
  1. Extract text page-by-page via pypdf
  2. Chunk with the strategy registered for that doc_id
  3. Embed each chunk via Ollama nomic-embed-text
  4. Upsert to Qdrant collection 'workershield'

Environment variables:
  OLLAMA_HOST   — Ollama base URL (default: http://192.168.100.1:11434)
  QDRANT_HOST   — Qdrant base URL (default: http://localhost:6333)
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

import requests
import yaml
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm

from ingest.chunk_strategy import get_chunker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://192.168.100.1:11434")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "http://localhost:6333")
COLLECTION = "workershield"
EMBED_MODEL = "nomic-embed-text"
VECTOR_SIZE = 768
CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus" / "raw"
REGISTRY_PATH = Path(__file__).resolve().parents[1] / "docs" / "corpus_registry.yaml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_registry() -> list[dict]:
    with open(REGISTRY_PATH) as fh:
        return yaml.safe_load(fh)["documents"]


def _extract_text_by_page(pdf_path: Path) -> list[str]:
    """Return a list of page text strings (index 0 = page 1)."""
    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


_MAX_EMBED_CHARS = 6_000  # ~1 500 tokens — safe headroom under nomic-embed-text's 8 192-token limit


def _embed(text: str) -> list[float]:
    """Call Ollama embeddings endpoint and return the vector."""
    if len(text) > _MAX_EMBED_CHARS:
        text = text[:_MAX_EMBED_CHARS]
    resp = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def _ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"  Created collection '{COLLECTION}'")
    else:
        print(f"  Collection '{COLLECTION}' already exists")


def _upsert_batch(client: QdrantClient, points: list[PointStruct]) -> None:
    client.upsert(collection_name=COLLECTION, points=points, wait=True)


# ---------------------------------------------------------------------------
# Per-document ingestion
# ---------------------------------------------------------------------------


def ingest_document(doc: dict, client: QdrantClient) -> dict[str, Any]:
    """Ingest one document. Returns a result dict for the summary table."""
    doc_id = doc["id"]
    domain = doc["domain"]
    title = doc["title"]
    source = doc["source"]
    filename = doc["filename"]
    total_pages = doc.get("pages", 1)

    pdf_path = CORPUS_DIR / filename
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # --- Extract ---
    pages = _extract_text_by_page(pdf_path)
    full_text = "\n".join(pages)

    # --- Chunk ---
    chunker = get_chunker(doc_id)
    meta = {"pages": total_pages}
    chunks = chunker(full_text, meta)

    if not chunks:
        raise ValueError(f"No chunks produced for {doc_id}")

    # --- Embed & upsert ---
    points: list[PointStruct] = []
    bar = tqdm(chunks, desc=f"  {doc_id}", unit="chunk", leave=False)
    for chunk in bar:
        vector = _embed(chunk["text"])
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "doc_id": doc_id,
                    "domain": domain,
                    "title": title,
                    "source": source,
                    "chunk_type": chunk["chunk_type"],
                    "section": chunk.get("section", ""),
                    "page_estimate": chunk.get("page_estimate", 1),
                    "text": chunk["text"],
                },
            )
        )

    _upsert_batch(client, points)

    return {"doc_id": doc_id, "domain": domain, "chunks_created": len(points), "status": "ok"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_ingest() -> None:
    registry = _load_registry()
    client = QdrantClient(url=QDRANT_HOST)

    print(f"\nOllama : {OLLAMA_HOST}")
    print(f"Qdrant : {QDRANT_HOST}")
    print(f"Corpus : {CORPUS_DIR}\n")

    _ensure_collection(client)
    print()

    results: list[dict[str, Any]] = []

    for doc in registry:
        doc_id = doc["id"]
        print(f"[{doc_id}] {doc['title'][:60]}")
        try:
            result = ingest_document(doc, client)
            results.append(result)
            print(f"  ✓ {result['chunks_created']} chunks upserted\n")
        except Exception as exc:
            results.append({
                "doc_id": doc_id,
                "domain": doc.get("domain", ""),
                "chunks_created": 0,
                "status": f"ERROR: {exc}",
            })
            print(f"  ✗ {exc}\n", file=sys.stderr)

    # --- Summary table ---
    print("\n" + "=" * 70)
    print(f"{'doc_id':<10} {'domain':<12} {'chunks':>8}  status")
    print("-" * 70)
    total_chunks = 0
    for r in results:
        status_str = r["status"] if r["status"] == "ok" else r["status"][:45]
        print(f"{r['doc_id']:<10} {r['domain']:<12} {r['chunks_created']:>8}  {status_str}")
        total_chunks += r["chunks_created"]
    print("-" * 70)
    ok_count = sum(1 for r in results if r["status"] == "ok")
    print(f"{'TOTAL':<10} {'':<12} {total_chunks:>8}  {ok_count}/{len(results)} documents ok")
    print("=" * 70)


if __name__ == "__main__":
    run_ingest()
