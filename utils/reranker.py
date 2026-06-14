"""
Cross-encoder reranker — third-pass relevance filter after hybrid RRF fusion.

Uses ms-marco-MiniLM-L-6-v2 (CPU-only, zero cost) to re-score query–chunk
pairs and return the top-N most relevant chunks per domain.
"""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class WorkerShieldReranker:
    def __init__(self) -> None:
        logger.info("[reranker] loading cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
        )

    def rerank(self, query: str, chunks: list[dict], top_n: int = 3) -> list[dict]:
        """Rerank chunks using cross-encoder scores; return top_n by relevance.

        Adds ``rerank_score`` (float) to each returned chunk dict.
        """
        if not chunks:
            return chunks

        pairs = [[query, chunk["text"]] for chunk in chunks]
        scores = self.model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n]

    def rerank_by_domain(
        self,
        query: str,
        domain_chunks: dict[str, list[dict]],
        top_n: int = 3,
    ) -> dict[str, list[dict]]:
        """Rerank chunks per domain independently.

        domain_chunks: {"safeshift": [...], "fairdesk": [...], "healthnav": [...]}
        Returns same structure with reranked + trimmed chunks per domain.
        """
        return {
            domain: self.rerank(query, chunks, top_n)
            for domain, chunks in domain_chunks.items()
            if chunks
        }


# ---------------------------------------------------------------------------
# Module-level lazy singleton — model loads once per Python process
# ---------------------------------------------------------------------------

_reranker: WorkerShieldReranker | None = None


def get_reranker() -> WorkerShieldReranker:
    global _reranker
    if _reranker is None:
        _reranker = WorkerShieldReranker()
    return _reranker
