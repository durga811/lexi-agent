"""Measure case-brief adverse_recall over N runs (no judges -> fast).

The 5-doc adverse set is high-variance, so a single eval run can't tell a real
prompt effect from noise. This drives the agent directly and reports the backbone
metrics per run. Usage: PYTHONPATH=. uv run python scripts/check_adverse_variance.py [N]
"""
import sys

from src.agent.graph import agent
from src.eval.gold_set import GOLD
from src.eval.metrics import cited_doc_ids
from src.utils import message_text

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
query, gold = next(iter(GOLD.items()))  # Q1 case brief
support = set(gold["supporting"])
adverse = set(gold["adverse"])
all_gold = support | adverse

print(f"adverse gold ({len(adverse)}): {sorted(adverse)}\n")
arec, prec, rec = [], [], []
for i in range(N):
    ans = message_text(agent.invoke({"messages": [("user", query)]})["messages"][-1].content)
    pred = cited_doc_ids(ans)
    a = len(pred & adverse) / len(adverse)
    p = len(pred & all_gold) / len(pred) if pred else 0.0
    r = len(pred & all_gold) / len(all_gold)
    arec.append(a); prec.append(p); rec.append(r)
    print(f"run {i+1}: adverse_recall={a:.2f} (found {sorted(pred & adverse)})  "
          f"precision={p:.2f}  recall={r:.2f}")

mean = lambda xs: sum(xs) / len(xs)
print(f"\nMEAN over {N}: adverse_recall={mean(arec):.2f}  precision={mean(prec):.2f}  recall={mean(rec):.2f}")
