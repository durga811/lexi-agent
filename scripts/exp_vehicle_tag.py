"""I7 honesty check: can an INDEPENDENT heuristic tag commercial vehicles well?

Tags each doc by counting commercial vs private vehicle keywords in the cleaned
text (no use of the gold labels), then scores that flag against the gold Q3
commercial set. If precision/recall of the heuristic flag is high, metadata
filtering is worth wiring; if not, the tag is too noisy to help.
"""
from __future__ import annotations

import re

from src.eval.gold_set import GOLD
from src.ingest.parse import load_documents

COMMERCIAL = [
    r"truck", r"lorr(y|ies)", r"tanker", r"tempo", r"dumper", r"canter",
    r"goods vehicle", r"goods carriage", r"\bbus\b", r"maxi[\s-]?cab",
    r"\btaxi\b", r"auto[\s-]?rickshaw", r"\bauto\b", r"matador", r"omni[\s-]?bus",
]
PRIVATE = [
    r"\bcar\b", r"\bjeep\b", r"maruti", r"\bomni\b", r"scooter", r"motor[\s-]?cycle",
    r"motorcycle", r"motor[\s-]?bike", r"moped", r"two[\s-]?wheeler",
]
TRACTOR = [r"tractor"]


def _count(patterns, text):
    return sum(len(re.findall(p, text, re.I)) for p in patterns)


def tag(text: str) -> dict:
    c, p, t = _count(COMMERCIAL, text), _count(PRIVATE, text), _count(TRACTOR, text)
    # refined: need >=2 commercial mentions, not outweighed by private OR tractor
    is_commercial = c >= 2 and c >= p and c > t
    return {"commercial_hits": c, "private_hits": p, "tractor_hits": t,
            "is_commercial": is_commercial}


# Operative insurer-outcome signal (independent of gold labels).
EXON = [
    r"insurance company is not liable", r"insurer is not liable",
    r"not liable to (pay|indemnify)", r"is exonerated", r"stand(s)? exonerated",
    r"absolved", r"company is exonerated", r"no liability (can|could) be fastened",
]
PAYREC = [r"pay and recover", r"pay and recovery", r"recover the (same|amount)",
          r"right (of|to) recover"]


def outcome_probe() -> None:
    """Does an exoneration regex separate the adverse (exonerated) docs from the
    supporting (liable / pay-and-recover) docs?"""
    g = GOLD["Client's husband was killed by a commercial truck whose driver had no "
             "valid driving licence; the insurer denies the motor accident claim as "
             "void because the driver was unlicensed. Find supporting and adverse "
             "precedents and recommend a strategy."]
    support, adverse = set(g["supporting"]), set(g["adverse"])
    print("\n--- insurer-outcome probe (exoneration vs pay/recover hits) ---")
    for d in load_documents():
        if d["doc_id"] not in support and d["doc_id"] not in adverse:
            continue
        e, pr = _count(EXON, d["text"]), _count(PAYREC, d["text"])
        label = "ADVERSE" if d["doc_id"] in adverse else "support"
        flag = "exon>payrec" if e > pr else ("payrec" if pr else "neutral")
        print(f"  {d['doc_id']} [{label:7}] exon={e:2} payrec={pr:2} -> {flag}")


def main() -> None:
    gold_comm = set(GOLD["Which of these judgments involve commercial vehicles?"]["supporting"])
    docs = load_documents()
    flagged = set()
    rows = []
    for d in docs:
        r = tag(d["text"])
        if r["is_commercial"]:
            flagged.add(d["doc_id"])
        rows.append((d["doc_id"], r, d["doc_id"] in gold_comm))

    tp = len(flagged & gold_comm)
    precision = tp / len(flagged) if flagged else 0.0
    recall = tp / len(gold_comm)
    print(f"heuristic flagged {len(flagged)} commercial; gold has {len(gold_comm)}")
    print(f"  precision={precision:.3f}  recall={recall:.3f}  tp={tp}")
    print("\nDisagreements (heuristic vs gold):")
    for doc_id, r, in_gold in rows:
        flag = r["is_commercial"]
        if flag != in_gold:
            kind = "FALSE-POS (flagged, not in gold)" if flag else "FALSE-NEG (missed)"
            print(f"  {doc_id}: {kind}  c={r['commercial_hits']} p={r['private_hits']} t={r['tractor_hits']}")


if __name__ == "__main__":
    main()
    outcome_probe()
