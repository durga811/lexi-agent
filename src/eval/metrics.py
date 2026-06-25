"""Deterministic, transparent set-based metrics on cited doc_ids.

These are the backbone of the eval: they need no LLM and are fully reproducible.
Precision/recall are computed by comparing the doc_ids the agent CITES in its
final answer against a hand-labelled gold set.

This module also owns the metric-GATING policy (which metric applies to which
query kind) and the multi-sample AGGREGATION helper, so `run_eval.py` never
hardcodes either.
"""
from __future__ import annotations

import re
import statistics

# --- metric gating -------------------------------------------------------------
# Which metrics are meaningful for which query kind. Metrics not listed for a kind
# are reported as n/a (None), exactly like adverse_recall already degrades today.
#   advocacy    — full battery: the only kind with an adversary + a strategy section
#   explanatory — deep but non-adversarial: no adverse / no strategy
#   lookup      — simple list; still judged for reasoning + grounding (a list can
#                 cite a wrong/hallucinated doc_id), per the design decision
#   negative    — abstention; faithfulness resolves to n/a when no context retrieved
METRIC_APPLICABILITY: dict[str, set[str]] = {
    "advocacy": {
        "precision", "recall", "adverse_recall",
        "reasoning", "adverse_honesty", "strategy", "faithfulness",
    },
    "explanatory": {"precision", "recall", "reasoning", "faithfulness"},
    "lookup": {"precision", "recall", "reasoning", "faithfulness"},
    "negative": {"precision", "recall", "reasoning", "faithfulness"},
}


def applies(kind: str, metric: str) -> bool:
    """True if `metric` should be scored for a query of this `kind`."""
    return metric in METRIC_APPLICABILITY.get(kind, set())


def aggregate_samples(samples: list[dict], keys: list[str]) -> dict:
    """Aggregate per-sample metric dicts into mean ± std per metric.

    Each sample is a flat dict {metric: float|None, ...}. None means n/a for that
    sample and is ignored. Returns {metric: {"mean", "std", "n"}}; if a metric is
    n/a in every sample, mean and std are None. Uses population std (pstdev) so a
    single sample yields std 0.0 rather than raising.
    """
    out: dict[str, dict] = {}
    for k in keys:
        vals = [s[k] for s in samples if s.get(k) is not None]
        if not vals:
            out[k] = {"mean": None, "std": None, "n": 0}
        else:
            out[k] = {
                "mean": round(statistics.fmean(vals), 3),
                "std": round(statistics.pstdev(vals), 3),
                "n": len(vals),
            }
    return out


def cited_doc_ids(answer: str) -> set[str]:
    """Extract every DOC_### id the agent cited in its answer."""
    return set(re.findall(r"DOC_\d{3}", answer.upper()))


def strategy_coverage(answer: str, groups: list[list[str]]) -> float | None:
    """Fraction of required Strategy concepts the answer covers (advocacy A1).

    Each `group` is a list of acceptable synonyms; the group counts as covered if
    ANY synonym appears (case-insensitive substring). Deterministic, reproducible
    companion to the Strategy G-Eval — checks the actionable answer actually
    contains its required elements (multiplier method, future prospects, the
    third-party theory, that the insurer still pays, an honest risk assessment).
    Returns None when there is nothing to score.
    """
    if not groups:
        return None
    al = answer.lower()
    hit = sum(1 for g in groups if any(k.lower() in al for k in g))
    return round(hit / len(groups), 3)


def mentions_amount_in_range(answer: str, lo_lakh: float, hi_lakh: float,
                             tol: float = 0.25) -> bool | None:
    """Does the answer state a rupee figure within [lo, hi] lakh (±tol band)?

    A soft check that A1's Strategy gives a *quantified* compensation range near
    the corpus-grounded value, not a hand-waved one. Parses figures written as
    'Rs 50 lakh', '50,00,000', '50.9 lakh', 'Rs 51,80,000'. Returns None if the
    answer states no parseable amount.
    """
    al = answer.lower().replace(",", "")
    found: list[float] = []
    # "<n> lakh" / "<n> lac"
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:lakh|lac)", al):
        found.append(float(m.group(1)))
    # bare rupee amounts >= 1,00,000 -> convert to lakh
    for m in re.finditer(r"(?:rs\.?|inr|₹)\s*(\d{6,})", al):
        found.append(float(m.group(1)) / 1e5)
    if not found:
        return None
    lo, hi = lo_lakh * (1 - tol), hi_lakh * (1 + tol)
    return any(lo <= v <= hi for v in found)


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
