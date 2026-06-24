"""Run the full evaluation across all four dimensions and write a markdown report.

Dimensions:
  1. Precision  — of cited precedents, % that are in the gold relevant set
  2. Recall     — of gold relevant precedents, % the agent cited
  3. Reasoning  — DeepEval G-Eval (Gemini judge): faithful to source + sound logic
  4. Adverse    — gold adverse recall + DeepEval G-Eval honesty rubric

The set-based backbone (1, 2, 4-recall) always runs and is deterministic. The
LLM-judge layer (3, 4-honesty) runs if `deepeval` is installed; otherwise the
report notes it was skipped so the script never hard-fails.
"""
from __future__ import annotations

from src.agent.graph import agent
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
    return reasoning_metric, adverse_metric


def run() -> None:
    reasoning_metric = adverse_metric = None
    if _HAVE_DEEPEVAL:
        reasoning_metric, adverse_metric = _build_judges()
    else:
        print(f"[warn] deepeval unavailable ({_DEEPEVAL_ERR}); skipping LLM-judge dims.")

    rows: list[dict] = []
    for query, gold in GOLD.items():
        print(f"\n=== Running: {query[:70]}...")
        answer = message_text(
            agent.invoke({"messages": [("user", query)]})["messages"][-1].content
        )
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
            tc = LLMTestCase(input=query, actual_output=answer)
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

        rows.append(row)

    _write_report(rows)


def _write_report(rows: list[dict]) -> None:
    with open("eval_results.md", "w") as f:
        f.write("# Evaluation Results\n\n")
        f.write(
            "Backbone metrics (precision/recall/adverse_recall) are deterministic, "
            "computed from doc_ids cited in the agent's answer vs. the hand-labelled "
            "gold set. Reasoning/adverse_honesty are DeepEval G-Eval scores (Gemini "
            "judge), 0-1.\n\n"
        )
        # summary table
        f.write("| Query | precision | recall | f1 | adverse_recall | reasoning | adverse_honesty |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['query'][:50]}… | {r['precision']} | {r['recall']} | {r['f1']} | "
                f"{r.get('adverse_recall')} | {r.get('reasoning', '—')} | "
                f"{r.get('adverse_honesty', '—')} |\n"
            )
        f.write("\n---\n\n")
        for r in rows:
            f.write(f"## {r['query']}\n\n")
            for k in (
                "deep", "precision", "recall", "f1", "tp", "predicted", "gold",
                "adverse_recall", "reasoning", "adverse_honesty",
            ):
                if k in r:
                    f.write(f"- **{k}**: {r[k]}\n")
            f.write(f"\n<details><summary>Agent answer</summary>\n\n{r['answer']}\n\n</details>\n\n")
    print("\nWrote eval_results.md")


if __name__ == "__main__":
    run()
