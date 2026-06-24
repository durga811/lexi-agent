# Lexi — Legal Precedent Research Agent

An agent that researches a corpus of Indian court judgments (Motor Accident
Claims Tribunal / High Court appeals) and produces strategic case analysis:
**supporting precedents, adverse precedents, and a strategy recommendation**.

It handles both general questions ("Which judgments involve commercial
vehicles?") and deep precedent research ("Find precedents supporting our
argument on contributory negligence") — **deciding its own workflow** rather than
following hard-coded steps.

> See [`ADR.md`](ADR.md) for the architecture decision record and
> [`eval_results.md`](eval_results.md) (generated) for evaluation results.

---

## Architecture at a glance

```
PDFs ─▶ parse (PyMuPDF) ─▶ chunk (recursive) ─▶ embed ─▶ Chroma (dense)
                                              └────────▶ BM25 (keyword)
                                                              │
                          search_corpus / get_document  ◀── hybrid ensemble
                                      │
                          Gemini 3.5 Flash agent (ReAct, LangGraph)
                                      │
                          Streamlit UI (streams the reasoning trace)
```

- **Retrieval:** hybrid — BM25 (keyword) + dense embeddings (`bge-small-en-v1.5`)
  blended with `EnsembleRetriever`. Legal queries need both exact-term and
  semantic matching.
- **Agent:** `langchain.agents.create_agent` (LangGraph ReAct loop) with **two
  tools** — `search_corpus` (breadth) and `get_document` (depth). The agent
  composing these in different orders/counts *is* the dynamic workflow; there is
  no `if query_type == ...` branching.
- **UI:** Streamlit, a thin layer over `src/agent/graph.py`. Streams tool calls
  and retrieved snippets so you can see *how* it reasoned, not just the answer.
- **Eval:** deterministic set-based precision/recall on cited doc_ids + DeepEval
  G-Eval (Gemini-as-judge) for reasoning quality and adverse-honesty.

```
src/
├── config.py              # paths, model + chunk/retrieval params
├── ingest/{parse,chunk,build_index}.py
├── retrieval/retriever.py # hybrid BM25 + dense
├── agent/{tools,prompts,schemas,graph}.py   # graph.py is the importable core
└── eval/{gold_set,metrics,judge,run_eval}.py
app.py                     # Streamlit UI
tests/test_retrieval.py
```

---

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# 1. install deps (already pinned in pyproject / uv.lock)
uv sync

# 2. add your Gemini API key
cp .env.example .env        # then edit .env and set GEMINI_API_KEY

# 3. build the vector index (run once; idempotent)
uv run python -m src.ingest.build_index

# 4. (optional) sanity-check retrieval
uv run pytest tests/test_retrieval.py -s
```

## Run the app

```bash
uv run streamlit run app.py
```

Then open the URL Streamlit prints. Paste a case brief (or any prompt) and watch
the tool calls + retrieved snippets stream in before the final analysis.

## Run the evaluation

```bash
uv run python -m src.eval.run_eval     # writes eval_results.md
```

## Deployment

`app.py` is the only deployment surface. Deploy to Streamlit Community Cloud
(point it at `app.py`, set `GEMINI_API_KEY` as a secret). The index is built from
`data/raw/` on first run. To swap to FastAPI later, replace `app.py` with
endpoints that call `agent.astream(...)` — `src/` is untouched.

## Hosted URL

> _TODO: add the deployed Streamlit URL here before submission._
