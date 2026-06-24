"""Hybrid retriever: BM25 (keyword) + dense (semantic) via an ensemble.

Why hybrid: legal queries hinge on exact terms (statute names, "Section 149",
"MV Act", "pay and recover") where keyword search shines, AND on conceptual
similarity ("policy void because driver unlicensed") where dense embeddings
shine. Blending both beats either alone.
"""
from __future__ import annotations

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
def get_retriever():
    """Build the hybrid retriever once per process (cached).

    Streamlit re-runs the script on every interaction, so caching the embedding
    model + BM25 index here is essential — otherwise we'd rebuild them per click.

    When `use_reranker` is on (I1), the ensemble over-retrieves `rerank_pool`
    candidates and a cross-encoder reranks them down to top_k — measured to lift
    recall@8 with no per-query regression (docs/EXPERIMENTS.md). Otherwise it
    returns the plain ensemble at top_k.
    """
    embeddings = HuggingFaceEmbeddings(model_name=settings.embed_model)
    store = Chroma(
        collection_name="judgments",
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )

    # Over-retrieve when reranking (the reranker needs a candidate pool); otherwise
    # retrieve exactly top_k.
    k = settings.rerank_pool if settings.use_reranker else settings.top_k
    dense = store.as_retriever(search_kwargs={"k": k})
    bm25 = BM25Retriever.from_documents(get_chunks())
    bm25.k = k

    # Weight dense slightly higher than keyword; tuned by eyeballing results.
    ensemble = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.4, 0.6])

    if not settings.use_reranker:
        return ensemble

    from langchain_core.runnables import RunnableLambda

    from src.retrieval.rerank import rerank

    def _retrieve_and_rerank(query: str):
        candidates = ensemble.invoke(query)
        return rerank(query, candidates, top_n=settings.top_k,
                      model_name=settings.reranker_model)

    return RunnableLambda(_retrieve_and_rerank)
