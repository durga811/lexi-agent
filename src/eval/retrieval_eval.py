"""Retriever-level evaluation — isolates retrieval quality from the agent.

The end-to-end eval (`run_eval.py`) scores the agent's *cited* set, so a recall
miss could be the retriever OR the agent. This module removes the agent: it runs
each gold query straight through a ranking function and measures whether the gold
docs actually appear in the retrieved chunks, at several cut-offs.

Why cut-offs matter for the reranker decision (I1): if recall@8 is low but
recall@40 is high, the gold docs ARE reachable — they're just ranked past where
the agent looks. That is exactly the gap an over-retrieve→rerank step closes. If
recall@40 is also low, the candidate pool itself is missing them and a reranker
cannot help.

Fast (seconds, no LLM) — use this for tight A/B loops; reserve run_eval.py for
milestones. The harness takes a `ranking_fn(query) -> [doc_id, ...]` (ranked,
one entry per retrieved chunk, repeats allowed) so any retriever/reranker config
can be scored by the same code.
"""
from __future__ import annotations

from collections import OrderedDict

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:  # pragma: no cover
    try:
        from langchain.retrievers import EnsembleRetriever
    except ImportError:
        from langchain_community.retrievers import EnsembleRetriever

from src.config import settings
from src.eval.gold_set import GOLD
from src.ingest.chunk import get_chunks

# Cut-offs measured in number of retrieved CHUNKS (what the agent would see).
CUTOFFS = (8, 20, 40)
POOL = 50  # over-retrieve depth for the candidate pool


def build_ensemble(k: int) -> EnsembleRetriever:
    """An ensemble retriever returning up to k hits per sub-retriever (uncached)."""
    embeddings = HuggingFaceEmbeddings(model_name=settings.embed_model)
    store = Chroma(
        collection_name="judgments",
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )
    dense = store.as_retriever(search_kwargs={"k": k})
    bm25 = BM25Retriever.from_documents(get_chunks())
    bm25.k = k
    return EnsembleRetriever(retrievers=[bm25, dense], weights=[0.4, 0.6])


def ensemble_ranking_fn(pool: int = POOL):
    """Baseline ranking: fused BM25+dense, returning doc_ids in ranked order."""
    retriever = build_ensemble(pool)

    def rank(query: str) -> list[str]:
        return [d.metadata["doc_id"] for d in retriever.invoke(query)]

    return rank


def _unique_in_order(doc_ids: list[str]) -> list[str]:
    return list(OrderedDict.fromkeys(doc_ids))


def evaluate_ranking(ranking_fn, cutoffs=CUTOFFS) -> dict:
    """Score a ranking_fn against the gold set. Returns per-query + mean metrics.

    recall@k = of the gold docs, how many appear within the first k retrieved
               CHUNKS (deduped to docs). first_hit = rank (1-based, in chunks) of
               the first gold doc, or None.
    Empty-gold queries (the negative test) are skipped — retrieval recall is
    undefined when nothing is relevant.
    """
    rows = []
    for query, g in GOLD.items():
        gold = set(g["supporting"]) | set(g["adverse"])
        if not gold:
            continue
        ranked = ranking_fn(query)  # one doc_id per chunk, ranked
        first_hit = next((i + 1 for i, d in enumerate(ranked) if d in gold), None)
        row = {"query": query[:46], "gold": len(gold), "first_hit": first_hit}
        for k in cutoffs:
            found = set(_unique_in_order(ranked[:k])) & gold
            row[f"r@{k}"] = round(len(found) / len(gold), 3)
        rows.append(row)

    means = {"query": "** MEAN **", "gold": "", "first_hit": ""}
    for k in cutoffs:
        means[f"r@{k}"] = round(sum(r[f"r@{k}"] for r in rows) / len(rows), 3)
    return {"rows": rows, "mean": means}


def print_report(result: dict, label: str) -> None:
    cutoffs = [c for c in result["mean"] if c.startswith("r@")]
    header = f"{'query':48} {'gold':>4} {'1st':>4} " + " ".join(f"{c:>6}" for c in cutoffs)
    print(f"\n=== Retriever-level eval: {label} ===")
    print(header)
    print("-" * len(header))
    for r in result["rows"]:
        print(
            f"{r['query']:48} {r['gold']:>4} {str(r['first_hit']):>4} "
            + " ".join(f"{r[c]:>6}" for c in cutoffs)
        )
    m = result["mean"]
    print("-" * len(header))
    print(f"{m['query']:48} {'':>4} {'':>4} " + " ".join(f"{m[c]:>6}" for c in cutoffs))


if __name__ == "__main__":
    res = evaluate_ranking(ensemble_ranking_fn())
    print_report(res, "baseline ensemble (BM25 0.4 + dense 0.6)")
