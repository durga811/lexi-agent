"""Smoke test for hybrid retrieval. Run: uv run pytest tests/test_retrieval.py -s"""
from src.retrieval.retriever import get_retriever


def test_retrieval_returns_results():
    r = get_retriever()
    hits = r.invoke("driver without valid licence insurance liability")
    assert len(hits) > 0
    seen = set()
    for h in hits[:5]:
        doc_id = h.metadata["doc_id"]
        seen.add(doc_id)
        print(doc_id, "→", h.page_content[:120].replace("\n", " "))
    # hits should carry doc_id metadata for citation
    assert all("doc_id" in h.metadata for h in hits)
