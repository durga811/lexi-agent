"""Cross-encoder reranking (I1).

The ensemble's BM25/dense scores come from bi-encoders — query and passage are
embedded separately, so the model never sees them together. A cross-encoder reads
(query, passage) jointly and scores true relevance; far more accurate, but too
slow to run over all 2819 chunks. So we use it as a SECOND stage: over-retrieve a
candidate pool with the cheap ensemble, then rerank that pool with the
cross-encoder and keep the top results.

Why this is the highest-leverage change here: the retriever-level eval (I8) showed
mean recall jumps from 0.32 @8 to 0.71 @40 — the gold docs are already in the
pool, just ranked too low. A reranker reorders them to the top.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document

# bge-reranker-base: strong open cross-encoder, runs locally, no API cost.
DEFAULT_RERANKER = "BAAI/bge-reranker-base"


@lru_cache(maxsize=2)
def _cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank(
    query: str,
    docs: list[Document],
    top_n: int | None = None,
    model_name: str = DEFAULT_RERANKER,
) -> list[Document]:
    """Reorder `docs` by cross-encoder relevance to `query`; return top_n (or all)."""
    if not docs:
        return []
    ce = _cross_encoder(model_name)
    scores = ce.predict([(query, d.page_content) for d in docs])
    ranked = [d for d, _ in sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)]
    return ranked if top_n is None else ranked[:top_n]
