# Architecture Decision Record — Lexi Precedent Research Agent

*A 2-page explanation of what we built and why. Plain language, no jargon for its
own sake.*

## 1. What this is

An agent that reads a corpus of ~56 Indian court judgments and answers a lawyer's
questions about them — from a quick lookup ("which cases involve commercial
vehicles?") to a full case analysis (the precedents that **help**, the precedents
that **hurt**, and a **strategy**). The worked example is Mrs. Lakshmi Devi: her
husband was killed by a commercial truck whose driver had no licence, and the
insurer says the policy is void.

## 2. The big picture

```
  56 PDFs ─▶ split into chunks ─▶ keyword index (BM25)  ┐
                                └▶ meaning index (embeddings) ┘
                                          │  blend both (hybrid search)
                                          ▼
                            re-rank the top ~40 → keep best 8
                                          │
                     two tools:  search_corpus   ·   get_document
                                          │
                     a reasoning agent (LangGraph, Gemini 3.5 Flash)
                                          │
                          Streamlit UI that shows its work
```

The one idea to hold onto: **there is no hard-coded "if it's a lookup do X, if it's
research do Y."** The agent has two tools and decides for itself how to use them.
That's §4.

## 3. The main choices, and why

**Agent framework — LangGraph (a "ReAct" agent).** The agent thinks → calls a tool →
reads the result → thinks again. It's a *loop*, not a fixed script, and LangGraph is
built for exactly that. A bonus: the graph doubles as the "show your work" trace the
assessment asks for. We kept it to **one agent with two tools** — a multi-agent setup
(a researcher + a "devil's advocate") would be over-engineering for 56 documents.
(Ruled out CrewAI / drag-and-drop builders — not allowed, and we wanted to be able to
explain every line.)

**Retrieval — hybrid search + a re-ranker.** Legal questions need two kinds of
matching at once: **exact terms** ("Section 149", "pay and recover") and **meaning**
("policy void because the driver was unlicensed"). So we blend keyword search (BM25)
with semantic search (embeddings) — the strengths cover each other's gaps. Then a
second step: a **cross-encoder re-ranker** re-reads the top ~40 candidates against the
question and keeps the best 8. This was our **biggest retrieval win — measured +0.19
precision and +0.09 recall.** Embeddings are a small open-source model (`bge-small`);
the store is Chroma; it all runs locally with no extra API cost.

**Chunking — simple on purpose.** We split each judgment into ~1,200-token chunks with
overlap. We deliberately did **not** build a clever "split by legal section" parser:
these judgments are continuous prose with no reliable "Held:" heading, so a section
parser would be brittle across 56 different documents. Reliable beat clever.

**Observability — three layers, kept separate.** (1) *In-app transparency* — the
Streamlit UI streams the agent's steps (each search, the documents it pulled, the
final answer), so you see *how* it concluded, not just the output. (2) *Developer
tracing* — **LangSmith** traces every run (the agent loop, LLM calls, tool calls)
with near-zero code because we're on LangChain/LangGraph; we added one span around
the retrieve-and-rerank step so the **reranker's per-document scores** are visible —
the "how it ranked" signal generic tracing misses. (3) *Offline quality* — the
gold-set eval harness is the regression gate. Eval traces go to a separate project so
they don't drown the live ones, and the whole thing degrades to a no-op if no
LangSmith key is set, so observability never changes behaviour.

## 4. How it decides "quick answer" vs "deep research" (the no-if-else part)

This is the question the assessment specifically asks. We do **not** branch in code.
The system prompt tells the agent to match its effort *and* its format to the question:

- **Lookup** → one or two searches, then a short cited list.
- **Explanation** → research it, then explain it plainly from the judgments.
- **Research / advocacy** → search from several angles *and* search for the
  counter-arguments, read the key cases in full, then answer in three sections:
  **Supporting · Adverse · Strategy.**

The agent reads the question and picks. You can *see* it in the trace: a lookup makes
1–2 tool calls; the full case brief makes ~15–20. **Same code, no switch statement** —
the flexibility lives in the prompt and the two composable tools, which is what makes
it handle prompts we never hard-coded for.

## 5. Tradeoffs we made

- **Prose answers, not a rigid JSON schema.** More natural and easy to cite inline;
  the cost is a required field could occasionally be thin. (An enforced schema is on
  the "another week" list.)
- **Shared models, per-thread clients in the eval** — the vector store and the
  re-ranker aren't thread-safe, so the multi-run eval gives each thread its own client
  (a real bug we hit and fixed).
- **The re-ranker adds a few seconds per search** — worth it for a research tool.
- **The answer key is one careful annotator's reading, scored pass/fail** — honest and
  documented, not a higher authority.

## 6. If the corpus were 5,000 docs, not 50

- Move off in-memory keyword search to a **managed hybrid store** (Qdrant / Weaviate).
- Add a **parent-document / summary index** so "list every case about X" stays complete.
- **Batch-precompute** embeddings and ship the prebuilt index.
- **Metadata filtering** (court, vehicle, outcome) stops being optional — it's how you
  keep recall up at scale.

## 7. If we had another week

1. **The #1 item: an ingest-time "outcome classifier"** to fix adverse-precedent
   retrieval — our weakest number, with a clear fix (explained in the eval doc, §6).
2. A **verification pass** that checks every cited claim against the retrieved text.
3. An **enforced output schema**.
4. A **domain lawyer's sign-off** on the few borderline labels.
