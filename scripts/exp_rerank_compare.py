"""Compare baseline vs pure-rerank vs RRF-fused-rerank (I1 attempt B).

Usage: PYTHONPATH=. uv run python scripts/exp_rerank_compare.py [reranker_model]
"""
import sys
import time

from src.eval.retrieval_eval import (
    CUTOFFS,
    ensemble_ranking_fn,
    evaluate_ranking,
    fused_rerank_ranking_fn,
    reranked_ranking_fn,
)

model = sys.argv[1] if len(sys.argv) > 1 else "cross-encoder/ms-marco-MiniLM-L-6-v2"
ks = [f"r@{k}" for k in CUTOFFS]

configs = []
t = time.time()
configs.append(("baseline", evaluate_ranking(ensemble_ranking_fn()), time.time() - t))
t = time.time()
configs.append(("pure-rerank", evaluate_ranking(reranked_ranking_fn(model_name=model)), time.time() - t))
t = time.time()
configs.append(("fused-RRF", evaluate_ranking(fused_rerank_ranking_fn(model_name=model)), time.time() - t))

# per-query r@8 / r@20
print(f"\nreranker = {model}\n")
for k in ("r@8", "r@20"):
    print(f"--- {k} ---")
    print(f"{'query':46} " + " ".join(f"{name:>12}" for name, _, _ in configs))
    for i in range(len(configs[0][1]["rows"])):
        q = configs[0][1]["rows"][i]["query"]
        vals = " ".join(f"{cfg[1]['rows'][i][k]:>12.3f}" for cfg in configs)
        print(f"{q:46} {vals}")
    means = " ".join(f"{cfg[1]['mean'][k]:>12.3f}" for cfg in configs)
    print(f"{'** MEAN **':46} {means}\n")

print("timings:", ", ".join(f"{name}={dt:.0f}s" for name, _, dt in configs))
