"""Cross-encoder reranking — the retrieval pipeline's second stage.

The ensemble's BM25/dense scores come from bi-encoders that embed query and
passage separately. A cross-encoder reads (query, passage) jointly and scores
true relevance — more accurate, but too slow to run over all chunks. So we
over-retrieve a candidate pool with the cheap ensemble, then rerank that pool and
keep the top. Measured: the gold docs are usually already in the pool (recall
0.32@8 → 0.71@40), just ranked too low; reranking lifts them.
"""
from __future__ import annotations

import threading
from functools import lru_cache

from langchain_core.documents import Document

# Must match config.reranker_model — this default only guards a bare call, but if
# it diverged, warmup and queries could load two different models. MiniLM won the
# A/B over bge-reranker-base (+0.07 recall@8, ~10x faster).
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# The shared cross-encoder isn't thread-safe: both construction and concurrent
# predict() calls intermittently segfault under the eval harness. This lock
# serialises all cross-encoder work — nearly free, since reranking is CPU-bound
# (torch pinned to 1 thread) and the parallelism win is on the Gemini calls, which
# run outside this lock.
_ce_lock = threading.Lock()


@lru_cache(maxsize=2)
def _cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def warm_reranker(model_name: str = DEFAULT_RERANKER) -> None:
    """Load the cross-encoder on the calling thread (idempotent). Call once on the
    main thread before spawning worker threads to avoid the concurrent-init crash."""
    with _ce_lock:
        _cross_encoder(model_name)


def rerank(
    query: str,
    docs: list[Document],
    top_n: int | None = None,
    model_name: str = DEFAULT_RERANKER,
) -> list[Document]:
    """Reorder `docs` by cross-encoder relevance to `query`; return top_n (or all)."""
    if not docs:
        return []
    with _ce_lock:  # serialise construction AND inference (torch not thread-safe)
        ce = _cross_encoder(model_name)
        scores = ce.predict([(query, d.page_content) for d in docs])
    ordered = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    selected = ordered if top_n is None else ordered[:top_n]
    # Attach rank + score so the UI can show "how it ranked". Build fresh Documents
    # rather than mutating the shared, lru_cached chunks (reused across eval threads).
    ranked = [
        Document(
            page_content=d.page_content,
            metadata={**d.metadata, "rank": i, "rerank_score": round(float(s), 4)},
        )
        for i, (d, s) in enumerate(selected, start=1)
    ]
    # Mirror the verdict onto the active trace span.
    from src.tracing import add_trace_metadata

    add_trace_metadata(reranker_scores=[
        {"doc_id": d.metadata.get("doc_id"), "rank": d.metadata["rank"],
         "score": d.metadata["rerank_score"]}
        for d in ranked[:8]
    ])
    return ranked
