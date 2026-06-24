"""Embed all chunks and persist them to a local Chroma store. Run ONCE.

Idempotent: re-running skips work if the collection is already populated
(unless force=True).
"""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings
from src.ingest.chunk import get_chunks


def build_index(force: bool = False) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=settings.embed_model)
    store = Chroma(
        collection_name="judgments",
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )

    count = store._collection.count()
    if not force and count > 0:
        print(f"Index already has {count} chunks. Use force=True to rebuild.")
        return store

    if force and count > 0:
        # wipe and rebuild cleanly
        store.reset_collection()

    chunks = get_chunks()
    # Chroma batches internally; add in one call.
    store.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks.")
    return store


if __name__ == "__main__":
    build_index()
