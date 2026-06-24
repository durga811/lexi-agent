"""A/B: baseline ensemble vs ensemble+cross-encoder reranker (I1).

Runs the fast retriever-level eval for both and prints recall@k side by side.
Usage: PYTHONPATH=. uv run python scripts/exp_rerank.py [reranker_model]
"""
import sys
import time

from src.eval.retrieval_eval import (
    CUTOFFS,
    ensemble_ranking_fn,
    evaluate_ranking,
    reranked_ranking_fn,
)

model = sys.argv[1] if len(sys.argv) > 1 else None

t0 = time.time()
base = evaluate_ranking(ensemble_ranking_fn())
t1 = time.time()
rer = evaluate_ranking(reranked_ranking_fn(model_name=model))
t2 = time.time()

ks = [f"r@{k}" for k in CUTOFFS]
print(f"\n{'query':46} " + " ".join(f"{k:>14}" for k in ks))
print("-" * (46 + 15 * len(ks)))
for b, r in zip(base["rows"], rer["rows"]):
    cells = []
    for k in ks:
        d = r[k] - b[k]
        sign = "+" if d > 0 else ""
        cells.append(f"{b[k]:.3f}->{r[k]:.3f}({sign}{d:.2f})")
    print(f"{b['query']:46} " + " ".join(f"{c:>14}" for c in cells))
print("-" * (46 + 15 * len(ks)))
bm, rm = base["mean"], rer["mean"]
cells = []
for k in ks:
    d = rm[k] - bm[k]
    sign = "+" if d > 0 else ""
    cells.append(f"{bm[k]:.3f}->{rm[k]:.3f}({sign}{d:.2f})")
print(f"{'** MEAN **':46} " + " ".join(f"{c:>14}" for c in cells))
print(f"\nbaseline: {t1-t0:.1f}s   reranked: {t2-t1:.1f}s   reranker={model or 'BAAI/bge-reranker-base'}")
