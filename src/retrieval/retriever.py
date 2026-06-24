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
def get_retriever() -> EnsembleRetriever:
    """Build the hybrid retriever once per process (cached).

    Streamlit re-runs the script on every interaction, so caching the embedding
    model + BM25 index here is essential — otherwise we'd rebuild them per click.
    """
    embeddings = HuggingFaceEmbeddings(model_name=settings.embed_model)
    store = Chroma(
        collection_name="judgments",
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )
    dense = store.as_retriever(search_kwargs={"k": settings.top_k})

    chunks = get_chunks()
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = settings.top_k

    # Weight dense slightly higher than keyword; tuned by eyeballing results.
    return EnsembleRetriever(retrievers=[bm25, dense], weights=[0.4, 0.6])
