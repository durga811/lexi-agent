"""Deterministic, transparent set-based metrics on cited doc_ids.

These are the backbone of the eval: they need no LLM and are fully reproducible.
Precision/recall are computed by comparing the doc_ids the agent CITES in its
final answer against a hand-labelled gold set.
"""
from __future__ import annotations

import re


def cited_doc_ids(answer: str) -> set[str]:
    """Extract every DOC_### id the agent cited in its answer."""
    return set(re.findall(r"DOC_\d{3}", answer.upper()))


def precision_recall(predicted: set[str], gold: set[str]) -> dict:
    """Set-based precision/recall.

    precision = of what the agent cited, how much was actually relevant.
    recall    = of what should have been found, how much the agent found.

    Negative/abstention queries (empty gold by design — e.g. a topic absent from
    the corpus) are scored on whether the agent correctly cited NOTHING: an empty
    prediction is a perfect 1.0 (no fabricated precedents), any citation is a
    false positive and scores 0.0. This keeps the "don't hallucinate" failure
    mode measurable instead of reading as a spurious zero.
    """
    if not gold:
        ok = 1.0 if not predicted else 0.0
        return {
            "precision": ok, "recall": ok, "f1": ok,
            "tp": 0, "predicted": len(predicted), "gold": 0,
        }

    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "tp": tp,
        "predicted": len(predicted),
        "gold": len(gold),
    }
