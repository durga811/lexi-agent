"""Hybrid retriever: BM25 (keyword) + dense (semantic) via an ensemble.

Why hybrid: legal queries hinge on exact terms (statute names, "Section 149",
"MV Act", "pay and recover") where keyword search shines, AND on conceptual
similarity ("policy void because driver unlicensed") where dense embeddings
shine. Blending both beats either alone.
"""
from __future__ import annotations

import threading
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

# EnsembleRetriever moved across langchain versions: langchain 1.x ships it in
# langchain_classic; older layouts had it under langchain / langchain_community.
try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:  # pragma: no cover
    try:
        from langchain.retrievers import EnsembleRetriever
    except ImportError:
        from langchain_community.retrievers import EnsembleRetriever

from src.config import settings
from src.ingest.chunk import get_chunks


@lru_cache(maxsize=1)
def _embeddings() -> HuggingFaceEmbeddings:
    """Embedding model — loaded once, shared across threads (inference is read-only)."""
    return HuggingFaceEmbeddings(model_name=settings.embed_model)


# One retriever per thread, not one shared: the Chroma client isn't thread-safe,
# and the eval harness runs agent invocations in a ThreadPoolExecutor. The costly
# embedding model + cross-encoder are still shared. In production (single-threaded
# Streamlit) this builds exactly one retriever.
#
# _build_lock: Chroma client construction races on a process-wide bindings
# singleton, so we serialise it; queries then run on independent per-thread
# clients. Call warmup() once on the main thread before spawning workers.
_local = threading.local()
_build_lock = threading.Lock()


def warmup() -> None:
    """Pre-load the shared models on the calling thread before workers spawn —
    Chroma's bindings singleton and the torch CrossEncoder both segfault if their
    first-time native init races across threads."""
    get_retriever()
    if settings.use_reranker:
        from src.retrieval.rerank import warm_reranker

        warm_reranker(settings.reranker_model)


def get_retriever():
    """Hybrid retriever (BM25 + dense), built once per thread.

    When use_reranker is on, the ensemble over-retrieves `rerank_pool` candidates
    and a cross-encoder reranks them to top_k; otherwise it returns the plain
    ensemble at top_k.
    """
    cached = getattr(_local, "retriever", None)
    if cached is not None:
        return cached

    with _build_lock:  # serialise Chroma construction (racy bindings singleton)
        embeddings = _embeddings()
        store = Chroma(
            collection_name="judgments",
            embedding_function=embeddings,
            persist_directory=str(settings.chroma_dir),
        )

    # Over-retrieve a pool when reranking; otherwise retrieve exactly top_k.
    k = settings.rerank_pool if settings.use_reranker else settings.top_k
    dense = store.as_retriever(search_kwargs={"k": k})
    bm25 = BM25Retriever.from_documents(get_chunks())
    bm25.k = k

    # Dense weighted slightly above keyword.
    ensemble = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.4, 0.6])

    if not settings.use_reranker:
        _local.retriever = ensemble
        return ensemble

    from langchain_core.runnables import RunnableLambda

    from src.retrieval.rerank import rerank
    from src.tracing import add_trace_metadata, traceable

    @traceable(run_type="retriever", name="hybrid_retrieve")
    def _retrieve_and_rerank(query: str):
        candidates = ensemble.invoke(query)
        ranked = rerank(query, candidates, top_n=settings.top_k,
                        model_name=settings.reranker_model)
        # Trace the "how it ranked" signal: pool size in, doc_ids out.
        add_trace_metadata(
            n_candidates=len(candidates),
            returned_doc_ids=[d.metadata.get("doc_id") for d in ranked],
        )
        return ranked

    result = RunnableLambda(_retrieve_and_rerank)
    _local.retriever = result
    return result
