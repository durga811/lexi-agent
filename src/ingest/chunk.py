"""Split judgments into overlapping chunks, carrying doc_id metadata.

Why recursive splitting: judgments are long, unstructured PDFs. Splitting at
paragraph -> sentence -> word boundaries keeps each chunk's reasoning intact,
and carrying doc_id in metadata lets the agent map any snippet back to its
source judgment for citation.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.ingest.parse import load_documents


def chunk_documents(docs: list[dict]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Document] = []
    for d in docs:
        for i, piece in enumerate(splitter.split_text(d["text"])):
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        "doc_id": d["doc_id"],
                        "source": d["source"],
                        "chunk": i,
                        # judgment-level metadata lifted from the page furniture
                        "title": d["title"],
                        "date": d["date"],
                        "kanoon_url": d["kanoon_url"],
                        "kanoon_id": d["kanoon_id"],
                    },
                )
            )
    return chunks


@lru_cache(maxsize=1)
def get_chunks() -> list[Document]:
    """Cached chunk list — BM25 needs these in memory each process."""
    return chunk_documents(load_documents())
