"""Run the full evaluation across all four dimensions and write a markdown report.

Dimensions:
  1. Precision    — of cited precedents, % that are in the gold relevant set
  2. Recall       — of gold relevant precedents, % the agent cited
  3. Reasoning    — DeepEval G-Eval (Gemini judge): sound legal mapping + logic
  4. Adverse      — gold adverse recall + DeepEval G-Eval honesty rubric
  5. Faithfulness — custom G-Eval "Grounding" (Gemini judge) over the chunks the
                    agent ACTUALLY retrieved: is every claim supported by the
                    retrieval_context? This is the "confident hallucination"
                    detector. We deliberately do NOT use DeepEval's stock
                    FaithfulnessMetric: it only penalises CONTRADICTIONS (a claim
                    the context refutes), scoring an UNSUPPORTED/fabricated claim
                    as "idk" → faithful. For a legal agent the dangerous failure is
                    fabrication (inventing a holding the text never states), so the
                    G-Eval rubric penalises any claim not traceable to a chunk.
                    (Verified: grounded answer → 1.0, fabricated → 0.0; the stock
                    metric scored both 1.0.)

The set-based backbone (1, 2, 4-recall) always runs and is deterministic. The
LLM-judge layer (3, 4-honesty, 5-faithfulness) runs if `deepeval` is installed;
otherwise the report notes it was skipped so the script never hard-fails.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# The eval runs many agents concurrently, and each agent's cross-encoder reranker
# is CPU-bound (torch). By default torch grabs ALL cores per call, so N concurrent
# reranks oversubscribe an 8-core box ~N*8-fold and the whole run thrashes to a
# stall. Pin torch to 1 thread so each worker's rerank uses 1 core; the dominant
# cost (sequential Gemini calls) is I/O and still parallelises across workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
# Keep the batch eval's traces out of the interactive LangSmith project — route
# them to their own project so a failing gold score is easy to find without
# drowning the live-app traces. (setdefault: an explicit override still wins.)
os.environ.setdefault("LANGSMITH_PROJECT", "lexi-agent-eval")
# HuggingFace tokenizers fork worker processes that leak semaphores under the
# ThreadPoolExecutor (the "leaked semaphore" warning) and accumulate until the OS
# OOM-kills the run (exit 137). Disable tokenizer parallelism — each rerank already
# runs on its own thread, so this costs nothing here.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    import torch

    torch.set_num_threads(1)
except Exception:  # pragma: no cover
    pass

from src.agent.graph import agent
from src.agent.tools import get_retrieval_log, reset_retrieval_log
from src.eval.gold_set import GOLD
from src.eval.metrics import (
    aggregate_samples,
    applies,
    cited_doc_ids,
    mentions_amount_in_range,
    precision_recall,
    strategy_coverage,
)
from src.utils import message_text

# DeepEval is optional — degrade gracefully if missing.
try:
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    from src.eval.judge import GeminiJudge

    _HAVE_DEEPEVAL = True
except Exception as e:  # pragma: no cover
    _HAVE_DEEPEVAL = False
    _DEEPEVAL_ERR = str(e)


def _build_judges(judge=None):
    """Construct the four GEval judges. Pass a shared `judge` model to avoid
    rebuilding the LLM client per sample; GEval instances themselves are cheap and
    must be fresh per measurement (they mutate `.score`), so build these per call.

    Criteria are domain-NEUTRAL: this is a general precedent agent over a mixed
    corpus (motor / criminal / health / fraud / trademark), so no metric assumes a
    motor-accident "client" or a "compensation" figure.
    """
    judge = judge or GeminiJudge()
    reasoning_metric = GEval(
        name="Reasoning Quality",
        criteria=(
            "Does the analysis rely only on retrieved judgments, correctly map "
            "their facts and holdings to the query's legal issue, and reach legally "
            "coherent conclusions?"
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge,
        async_mode=False,  # we parallelise at the sample level (threads), not here
    )
    adverse_metric = GEval(
        name="Adverse Honesty",
        criteria=(
            "Does the answer surface precedents that cut AGAINST the position the "
            "query advocates, and honestly assess their risk and distinguishability, "
            "rather than presenting only favorable cases?"
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge,
        async_mode=False,  # we parallelise at the sample level (threads), not here
    )
    strategy_metric = GEval(
        name="Strategy Quality",
        criteria=(
            "For an advocacy task, does the answer (a) prioritise concrete arguments "
            "grounded in the cited precedents, (b) give a realistic outcome or "
            "quantum range whose basis is traceable to the cited judgments (not "
            "invented), and (c) flag the key risks or adverse authorities the "
            "position must overcome? Penalise generic advice, unsupported numbers, "
            "and missing risk assessment."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge,
        async_mode=False,  # we parallelise at the sample level (threads), not here
    )
    # Custom grounding/faithfulness — penalises any unsupported claim, not just
    # contradictions (see module docstring). Needs RETRIEVAL_CONTEXT on the case.
    faithfulness_metric = GEval(
        name="Grounding",
        criteria=(
            "Is EVERY specific claim in the actual output — each holding, case name, "
            "statute or section number, or numeric figure (award, quantum, "
            "percentage) — directly supported by the retrieval context? Penalise "
            "heavily any claim that is NOT present in the retrieval context (a "
            "fabricated or unsupported claim), even if it is not explicitly "
            "contradicted. A perfect answer makes no claim that cannot be traced to "
            "a retrieved chunk."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=judge,
        async_mode=False,  # we parallelise at the sample level (threads), not here
    )
    return reasoning_metric, adverse_metric, strategy_metric, faithfulness_metric


# Metric keys aggregated/reported (in display order).
_METRIC_KEYS = [
    "precision", "recall", "f1", "adverse_recall",
    "reasoning", "adverse_honesty", "strategy", "faithfulness",
    "strategy_coverage", "strategy_range_ok", "n_context",
]


def _is_rate_limit(e: Exception) -> bool:
    s = str(e).lower()
    return any(t in s for t in ("429", "resourceexhausted", "rate limit", "quota"))


def _measure(metric, tc):
    """Score a GEval; one retry on rate-limit, else degrade to None (never fatal)."""
    for attempt in range(2):
        try:
            metric.measure(tc)
            return round(metric.score, 3)
        except Exception as e:  # noqa: BLE001
            if attempt == 0 and _is_rate_limit(e):
                time.sleep(2)
                continue
            print(f"[warn] {getattr(metric, 'name', 'judge')} failed: {e}")
            return None


def _invoke_agent(query: str, config: dict | None = None) -> str:
    for attempt in range(2):
        try:
            return message_text(
                agent.invoke({"messages": [("user", query)]}, config=config or {})[
                    "messages"
                ][-1].content
            )
        except Exception as e:  # noqa: BLE001
            if attempt == 0 and _is_rate_limit(e):
                time.sleep(3)
                continue
            raise


def _run_one_sample(query: str, gold: dict, kind: str, judge_model, idx: int) -> dict:
    """One agent run + all metrics APPLICABLE to `kind`, for one sample.

    Thread/context-safe: resets and reads the retrieval log inside this call, so
    the ContextVar copy in this worker thread holds only this sample's chunks.
    """
    reset_retrieval_log()
    from src.tracing import run_config

    answer = _invoke_agent(
        query, config=run_config(source="eval", query_kind=kind, sample=idx)
    )
    retrieval_context = get_retrieval_log()
    predicted = cited_doc_ids(answer)

    gold_support = set(gold.get("supporting", []))
    gold_adverse = set(gold.get("adverse", []))
    pr = precision_recall(predicted, gold_support | gold_adverse)

    row = {
        "sample_idx": idx,
        "precision": pr["precision"], "recall": pr["recall"], "f1": pr["f1"],
        "tp": pr["tp"], "predicted": pr["predicted"], "gold": pr["gold"],
        "n_context": len(retrieval_context),
        "answer": answer,
        "adverse_recall": None, "reasoning": None,
        "adverse_honesty": None, "strategy": None, "faithfulness": None,
        "strategy_coverage": None, "strategy_range_ok": None,
    }

    if applies(kind, "adverse_recall") and gold_adverse:
        row["adverse_recall"] = round(len(predicted & gold_adverse) / len(gold_adverse), 3)

    # Deterministic actionability check (advocacy A1): does the Strategy section
    # actually contain its required elements + a grounded compensation range?
    if gold.get("strategy_must_mention"):
        row["strategy_coverage"] = strategy_coverage(answer, gold["strategy_must_mention"])
    if gold.get("strategy_range_lakh"):
        lo, hi = gold["strategy_range_lakh"]
        in_range = mentions_amount_in_range(answer, lo, hi)
        row["strategy_range_ok"] = None if in_range is None else float(in_range)

    if judge_model is not None:
        reasoning_m, adverse_m, strategy_m, faith_m = _build_judges(judge_model)
        tc = LLMTestCase(
            input=query, actual_output=answer,
            retrieval_context=retrieval_context or None,
        )
        if applies(kind, "reasoning"):
            row["reasoning"] = _measure(reasoning_m, tc)
        if applies(kind, "adverse_honesty"):
            row["adverse_honesty"] = _measure(adverse_m, tc)
        if applies(kind, "strategy"):
            row["strategy"] = _measure(strategy_m, tc)
        if applies(kind, "faithfulness") and retrieval_context:
            row["faithfulness"] = _measure(faith_m, tc)
    return row


def run() -> None:
    judge_model = GeminiJudge() if _HAVE_DEEPEVAL else None
    if not _HAVE_DEEPEVAL:
        print(f"[warn] deepeval unavailable ({_DEEPEVAL_ERR}); LLM-judge dims = n/a.")

    only = os.environ.get("EVAL_ONLY", "").strip()
    n_samples = int(os.environ.get("EVAL_SAMPLES", "1"))
    # Gemini scales ~linearly to 16+ concurrent calls (benchmarked: 14x at 16
    # workers — it's latency-bound, not throttled), and agent runs are sequential
    # internally, so high worker counts are the main speed lever. Default 12.
    max_workers = int(os.environ.get("EVAL_MAX_WORKERS", "12"))

    items = list(GOLD.items())
    if only:
        idx = {int(i) - 1 for i in only.split(",")}
        items = [it for n, it in enumerate(items) if n in idx]
        print(f"[subset] {len(items)}/{len(GOLD)} queries: {sorted(int(i) for i in only.split(','))}")
    print(f"[config] samples={n_samples} max_workers={max_workers}")

    # Initialise the embedding model + Chroma bindings on THIS thread before
    # spawning workers, so the first concurrent client builds don't race the
    # process-wide singleton (see retriever.warmup / _build_lock).
    from src.retrieval.retriever import warmup
    warmup()

    # One task per (query, sample). Threads are safe: the agent is stateless, the
    # retriever/reranker are read-only after warmup, and the retrieval log is a
    # per-context ContextVar (see tools.py).
    # Interleave by SAMPLE (sample-0 of every query, then sample-1, ...) so each
    # concurrency wave mixes fast lookups with slow advocacy queries instead of
    # the pool's first wave all grabbing the heaviest query — balances load and
    # gives early completions.
    tasks = [
        (q, g, g["kind"], i)
        for i in range(n_samples)
        for (q, g) in items
    ]
    results: dict[str, list[dict]] = {q: [] for q, _ in items}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_run_one_sample, q, g, kind, judge_model, i): q
            for (q, g, kind, i) in tasks
        }
        done = 0
        for fut in as_completed(futures):
            q = futures[fut]
            results[q].append(fut.result())
            done += 1
            print(f"  [{done}/{len(tasks)}] sample done: {q[:50]}…")

    rows = []
    for q, g in items:
        samples = sorted(results[q], key=lambda s: s["sample_idx"])
        agg = aggregate_samples(samples, _METRIC_KEYS)
        rows.append({"query": q, "kind": g["kind"], "n_samples": len(samples),
                     "agg": agg, "samples": samples})

    suffix = f".n{n_samples}" if n_samples > 1 else ""
    base = "eval_results.subset" if only else "eval_results"
    _write_report(rows, f"{base}{suffix}.md")


def _cell(agg: dict) -> str:
    """Render a metric as 'mean' (n<=1 / no spread) or 'mean±std', or 'n/a'."""
    if agg["mean"] is None:
        return "n/a"
    if agg["n"] <= 1 or not agg["std"]:
        return f"{agg['mean']:.3f}"
    return f"{agg['mean']:.3f}±{agg['std']:.3f}"


def _write_report(rows: list[dict], out_path: str = "eval_results.md") -> None:
    cols = ["precision", "recall", "f1", "adverse_recall",
            "reasoning", "adverse_honesty", "strategy", "faithfulness",
            "strategy_coverage", "strategy_range_ok"]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation Results\n\n")
        f.write(
            "Each metric is scored only for the query KINDS where it applies "
            "(`n/a` otherwise): adverse_recall / adverse_honesty / strategy → "
            "advocacy queries only; precision/recall/reasoning/faithfulness → all. "
            "Backbone (precision/recall/adverse_recall) is deterministic; "
            "reasoning/adverse_honesty/strategy/faithfulness are Gemini LLM-judge "
            "scores (0-1). With n_samples>1, cells show mean±std over independent "
            "agent runs (de-noises agent + judge stochasticity).\n\n"
        )
        f.write("| Query | kind | n | " + " | ".join(cols) + " |\n")
        f.write("|---|---|---|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            cells = " | ".join(_cell(r["agg"][c]) for c in cols)
            f.write(f"| {r['query'][:46]}… | {r['kind']} | {r['n_samples']} | {cells} |\n")
        f.write("\n---\n\n")
        for r in rows:
            f.write(f"## {r['query']}\n\n")
            f.write(f"- **kind**: {r['kind']}\n- **n_samples**: {r['n_samples']}\n")
            for c in cols + ["n_context"]:
                f.write(f"- **{c}**: {_cell(r['agg'][c])}\n")
            if r["n_samples"] > 1:
                f.write("\n**Per-sample** (precision/recall/reasoning/faithfulness):\n\n")
                f.write("| # | prec | rec | reasoning | faithfulness |\n|---|---|---|---|---|\n")
                for s in r["samples"]:
                    f.write(f"| {s['sample_idx']} | {s['precision']} | {s['recall']} | "
                            f"{s.get('reasoning')} | {s.get('faithfulness')} |\n")
            f.write(f"\n<details><summary>Agent answer (sample 0)</summary>\n\n"
                    f"{r['samples'][0]['answer']}\n\n</details>\n\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    run()
