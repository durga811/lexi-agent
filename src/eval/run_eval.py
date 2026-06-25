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

from src.agent.graph import agent
from src.agent.tools import get_retrieval_log, reset_retrieval_log
from src.eval.gold_set import GOLD
from src.eval.metrics import cited_doc_ids, precision_recall
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


def _build_judges():
    judge = GeminiJudge()
    reasoning_metric = GEval(
        name="Reasoning Quality",
        criteria=(
            "Does the analysis cite only retrieved judgments, correctly map their "
            "facts to the client's situation, and reach legally coherent conclusions?"
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge,
    )
    adverse_metric = GEval(
        name="Adverse Honesty",
        criteria=(
            "Does the answer surface precedents AGAINST the client and honestly "
            "assess their risk, rather than only presenting favorable cases?"
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge,
    )
    # Custom grounding/faithfulness — penalises any unsupported claim, not just
    # contradictions (see module docstring). Needs RETRIEVAL_CONTEXT on the case.
    faithfulness_metric = GEval(
        name="Grounding",
        criteria=(
            "Is EVERY specific factual/legal claim in the actual output — each "
            "holding, case name, statute or section number, and compensation figure "
            "— directly supported by the retrieval context? Penalise heavily any "
            "claim that is NOT present in the retrieval context (a fabricated or "
            "unsupported claim), even if it is not explicitly contradicted. A "
            "perfect answer makes no claim that cannot be traced to a retrieved chunk."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=judge,
    )
    return reasoning_metric, adverse_metric, faithfulness_metric


def run() -> None:
    reasoning_metric = adverse_metric = faithfulness_metric = None
    if _HAVE_DEEPEVAL:
        reasoning_metric, adverse_metric, faithfulness_metric = _build_judges()
    else:
        print(f"[warn] deepeval unavailable ({_DEEPEVAL_ERR}); skipping LLM-judge dims.")

    # Optional subset for targeted experiments: EVAL_ONLY="1,2,4" runs the 1st,
    # 2nd, 4th gold queries only (1-based). Writes to eval_results.subset.md so it
    # never clobbers the full baseline report.
    import os

    only = os.environ.get("EVAL_ONLY", "").strip()
    items = list(GOLD.items())
    if only:
        idx = {int(i) - 1 for i in only.split(",")}
        items = [it for n, it in enumerate(items) if n in idx]
        print(f"[subset] running {len(items)} of {len(GOLD)} queries: {sorted(int(i) for i in only.split(','))}")

    rows: list[dict] = []
    for query, gold in items:
        print(f"\n=== Running: {query[:70]}...")
        reset_retrieval_log()  # capture exactly the chunks this run retrieves
        answer = message_text(
            agent.invoke({"messages": [("user", query)]})["messages"][-1].content
        )
        retrieval_context = get_retrieval_log()
        predicted = cited_doc_ids(answer)

        gold_support = set(gold.get("supporting", []))
        gold_adverse = set(gold.get("adverse", []))
        gold_all = gold_support | gold_adverse

        pr = precision_recall(predicted, gold_all)
        adverse_recall = (
            round(len(predicted & gold_adverse) / len(gold_adverse), 3)
            if gold_adverse
            else None
        )

        row = {
            "query": query,
            "deep": gold.get("deep", False),
            **pr,
            "adverse_recall": adverse_recall,
            "answer": answer,
        }

        if reasoning_metric is not None:
            tc = LLMTestCase(
                input=query,
                actual_output=answer,
                retrieval_context=retrieval_context or None,
            )
            # A judge hiccup on one query shouldn't kill the whole run — degrade
            # that dimension to None and keep the deterministic metrics.
            try:
                reasoning_metric.measure(tc)
                row["reasoning"] = round(reasoning_metric.score, 3)
            except Exception as e:
                print(f"[warn] reasoning judge failed: {e}")
                row["reasoning"] = None
            try:
                adverse_metric.measure(tc)
                row["adverse_honesty"] = round(adverse_metric.score, 3)
            except Exception as e:
                print(f"[warn] adverse judge failed: {e}")
                row["adverse_honesty"] = None
            # Faithfulness: only meaningful if the agent actually retrieved context.
            if retrieval_context:
                try:
                    faithfulness_metric.measure(tc)
                    row["faithfulness"] = round(faithfulness_metric.score, 3)
                except Exception as e:
                    print(f"[warn] faithfulness judge failed: {e}")
                    row["faithfulness"] = None
            else:
                row["faithfulness"] = None
            row["n_context"] = len(retrieval_context)

        rows.append(row)

    _write_report(rows, "eval_results.subset.md" if only else "eval_results.md")


def _write_report(rows: list[dict], out_path: str = "eval_results.md") -> None:
    with open(out_path, "w") as f:
        f.write("# Evaluation Results\n\n")
        f.write(
            "Backbone metrics (precision/recall/adverse_recall) are deterministic, "
            "computed from doc_ids cited in the agent's answer vs. the hand-labelled "
            "gold set. Reasoning/adverse_honesty/faithfulness are DeepEval LLM-judge "
            "scores (Gemini), 0-1. Faithfulness checks the answer's claims against the "
            "chunks the agent actually retrieved (n_context = how many).\n\n"
        )
        # summary table
        f.write("| Query | precision | recall | f1 | adverse_recall | reasoning | adverse_honesty | faithfulness |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['query'][:50]}… | {r['precision']} | {r['recall']} | {r['f1']} | "
                f"{r.get('adverse_recall')} | {r.get('reasoning', '—')} | "
                f"{r.get('adverse_honesty', '—')} | {r.get('faithfulness', '—')} |\n"
            )
        f.write("\n---\n\n")
        for r in rows:
            f.write(f"## {r['query']}\n\n")
            for k in (
                "deep", "precision", "recall", "f1", "tp", "predicted", "gold",
                "adverse_recall", "reasoning", "adverse_honesty", "faithfulness",
                "n_context",
            ):
                if k in r:
                    f.write(f"- **{k}**: {r[k]}\n")
            f.write(f"\n<details><summary>Agent answer</summary>\n\n{r['answer']}\n\n</details>\n\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    run()
