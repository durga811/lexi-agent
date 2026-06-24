# Architecture Decision Record — Lexi Precedent Research Agent

## Context

Build an agent that researches a corpus of ~56 Indian court judgments and
produces strategic case analysis (supporting precedents, adverse precedents,
strategy). It must handle both shallow lookups and deep research, deciding its
own workflow rather than running a hard-coded pipeline.

---

## 1. Why this architecture

### Agent framework: LangChain `create_agent` (LangGraph ReAct loop)
A precedent-research task has an *unknown number of steps*: a lookup needs one
search; "find supporting and adverse precedents and recommend a strategy" needs
several searches plus full-text reads. A ReAct agent (reason → act → observe →
repeat) models exactly this — the LLM decides, at each step, whether it has
enough to answer or needs another tool call. LangGraph gives us a durable,
streamable graph for free (used by the UI to show the trace), and
`create_agent` is the thin, supported wrapper over it. We avoided CrewAI-style
drag-and-drop builders (explicitly disallowed) and avoided writing our own
control flow because the *whole point* is that the model — not our `if`
statements — chooses the workflow.

### The dynamic workflow is two tools, not branching code
- `search_corpus(query)` — breadth: hybrid retrieval over all judgments.
- `get_document(doc_id)` — depth: full text of one judgment to verify a snippet.

The agent composing these in different orders and counts **is** the dynamic
workflow. A general query ("which judgments involve commercial vehicles?")
resolves in one or two `search_corpus` calls; the case brief triggers several
searches from different angles plus `get_document` reads before synthesis. The
behavioural difference comes entirely from the system prompt + the model's
planning — there is zero `if query_type == ...` branching. That is the
architecture answer to the flexibility requirement.

### Retrieval strategy: hybrid (BM25 + dense)
Legal queries are bimodal. Some hinge on **exact terms** — "Section 149", "MV
Act", "pay and recover", a specific statute — where keyword (BM25) wins. Others
are **conceptual** — "policy void because the driver was unlicensed" — where
dense embeddings win. An `EnsembleRetriever` (weights 0.4 BM25 / 0.6 dense)
blends both with reciprocal-rank fusion, so we don't have to pick. Dense uses
the open-source `BAAI/bge-small-en-v1.5` (no API cost, runs locally, strong for
its size). Chroma is the vector store: embedded, persists to disk, zero infra —
right for 56 docs.

### Chunking: recursive character splitting (1200 / 200 overlap)
Judgments are long, unstructured PDFs. Recursive splitting at paragraph →
sentence → word boundaries keeps each chunk's reasoning intact, and 200-token
overlap avoids cutting a holding in half at a boundary. Every chunk carries its
`doc_id` in metadata, which is what lets the agent cite a source for any snippet.

---

## 2. Tradeoffs made (and why)

| Decision | Tradeoff accepted | Why it's right *here* |
|---|---|---|
| Two generic tools, not many specialized ones | Less hand-holding for the model | Forces genuine dynamic planning; fewer tools = fewer failure modes |
| Prose three-section output, not enforced Pydantic | Output isn't machine-validated | More natural for the LLM, easier to render with inline citations; schema exists (`schemas.py`) as the upgrade path |
| BM25 rebuilt in-memory each process | Slow cold start, doesn't scale | Trivial at 56 docs; correct call to defer infra (see §3) |
| `bge-small` over `bge-base`/API embeddings | A few points of retrieval quality | Free, fast, local; quality is enough — verified by the retrieval smoke test |
| Small hand-labelled gold set | Not statistically large | Honest, defensible recall denominator beats a guessed one; methodology matters more than N at this stage |
| Default Gemini sampling params | No fine control | Gemini 3.x reasoning is tuned for its defaults; overriding `temperature`/`top_p` hurts tool planning |

---

## 3. How the agent decides simple vs. deep workflow

It is **not** decided by us. The system prompt tells the agent to do the minimum
for general questions and to run multi-angle research + full-text reads for
precedent tasks; the model then plans accordingly via the ReAct loop. We verify
this empirically rather than asserting it: the eval tags each query `deep:
true/false` and the Streamlit trace shows the tool-call count. A simple query
makes ~1 tool call and answers concisely; the case brief makes several searches,
reads documents, and returns the three-section structure. **That contrast is the
proof of dynamic workflow.** If we wanted a hard guarantee we could add a
planner node, but that would re-introduce exactly the rigidity the assessment
warns against.

---

## 4. What I'd change for 5,000 documents instead of 50

- **Drop in-memory BM25.** Rebuilding the BM25 index from all chunks on every
  process boot is fine for 56 docs, fatal for 5,000. Move to a server-backed
  store with **native hybrid search** (Qdrant or Weaviate) so keyword + dense
  live in one queryable index.
- **Add a reranker.** A cross-encoder (e.g. `bge-reranker`) over the top ~50
  ensemble hits to sharpen precision before the agent sees them — over-retrieval
  hurts more as the corpus grows.
- **Two-stage / parent-document retrieval.** A per-judgment summary index for
  doc-level questions ("which judgments involve commercial vehicles?"), then
  drill into chunks. Pure chunk retrieval answers doc-level questions poorly.
- **Precompute embeddings as a batch job** and persist to a volume / managed
  store; never embed at request time.
- **Metadata filtering.** Extract court, year, vehicle type, statute at ingest so
  the agent can pre-filter ("commercial vehicle" judgments) instead of relying on
  semantic recall alone.

## 5. What I'd change with another week

- **Structure-aware chunking** (split on Facts / Held / Ratio sections) so
  retrieval returns the *holding*, not boilerplate cause-title.
- **Enforced structured output** — a final Pydantic pass (`ResearchReport` in
  `schemas.py`) so the three sections are machine-validated, not prose-parsed.
- **RAGAS retrieval metrics** (`ContextPrecision`/`ContextRecall`) to measure
  *retrieval* quality separately from the agent's final citations — today a
  recall miss could be the retriever or the agent; this disentangles them.
- **A larger gold set built from traces** — label real agent runs to grow ground
  truth honestly, and add adverse-precedent labels per query.
- **Adverse-precedent retrieval as a first-class step** — an explicit "now search
  for what cuts against the client" tool/prompt phase to push adverse recall up.

---

## 6. Evaluation approach (summary; numbers in `eval_results.md`)

Four dimensions, each with at least one automated metric. Strategy: a
**deterministic, transparent backbone** plus a **best-tool LLM-judge layer**.

| Dimension | Method | Why |
|---|---|---|
| 1. Precision | set overlap of cited doc_ids vs. gold relevant | reproducible, no LLM variance |
| 2. Recall | cited ∩ gold / gold | needs a defensible denominator → hand-labelled gold set |
| 3. Reasoning quality | DeepEval **G-Eval**, Gemini judge | qualitative; rubric = faithful-to-source + sound legal logic |
| 4. Adverse identification | gold adverse recall + G-Eval honesty rubric | catches the dangerous failure mode: only finding favorable cases |

**Gold-set methodology:** every one of the 56 judgments was read and reduced to
objective facts (court, vehicle, licence defect, insurer-liability outcome,
compensation method, contributory negligence), then labels were *derived* by an
explicit rubric and **independently verified** (a second adversarial pass re-read
each labelled doc; 33/34 confirmed, 1 corrected). Full record in
`docs/GOLDEN_TEST_SET.md`; the mechanism in `docs/EVALUATION.md`. The set is small
but verified — the recall denominator is defined and audited, not guessed.

### Results (baseline run 2026-06-24, `gemini-3.5-flash`, 7-query verified gold set)

| Query | precision | recall | f1 | adverse_recall | reasoning | adverse_honesty |
|---|---|---|---|---|---|---|
| Q1 Case brief — unlicensed driver, insurer denies (deep) | 0.90 | 0.50 | 0.64 | **0.20** | 0.90 | 1.00 |
| Q2 Compensation calc — death of earner (deep) | 0.43 | 0.33 | 0.38 | n/a | 0.30 | 1.00 |
| Q3 Which judgments involve commercial vehicles? (simple) | 0.75 | 0.43 | 0.55 | n/a | 0.90 | 0.00 |
| Q4 Contributory negligence (deep) | 0.67 | 0.80 | 0.73 | 1.00 | 0.80 | 1.00 |
| Q5 "Pay and recover" — lexical/BM25 (simple) | 0.63 | 0.50 | 0.56 | n/a | 0.50 | 1.00 |
| Q6 Hit-and-run — negative/abstention (simple) | n/a* | n/a* | n/a* | n/a | 0.70 | 1.00 |
| Q7 Passenger-in-vehicle theme (simple) | 0.43 | 0.60 | 0.50 | n/a | 0.50 | 1.00 |

\* Q6 has an **empty gold by design** (the corpus contains no hit-and-run/untraced
precedent). The agent correctly reported that no such precedent exists and only
cited DOC_017 as an explicitly "closest-but-not-on-point" reference, so the strict
abstention rule scores it 0.0 — a metric artifact, **not** a fabrication. The
qualitative reasoning judge (0.70) reflects the correct behaviour.

> **Why these differ from the 2026-06-18 run:** that earlier 3-query baseline used
> a partly-incorrect gold set (it mislabelled the pro-claimant Supreme Court
> landmarks Iyyapan/Dhut as *adverse*). The corrected labels raise case-brief
> precision (0.64→0.90) and, importantly, drop case-brief adverse-recall to an
> **honest 0.20** — the old 0.50 was inflated by counting pro-claimant docs as
> adverse. Side-by-side comparison in `eval_results.md`.

Dynamic-workflow proof (from traces): simple queries resolve in ~1–2 tool calls
and answer concisely; the case brief makes many multi-angle searches +
`get_document` reads before the three-section report. Same code, no branching.

### Measured improvements (one variable at a time — full log in `docs/EXPERIMENTS.md`)

| Change | Method | precision | recall | adverse_recall (Q1) |
|---|---|---|---|---|
| Baseline (hybrid, no rerank) | run_eval | 0.64 | 0.53 | ~0.20† |
| **+ I1 cross-encoder reranker** (MiniLM, over-retrieve 40 → top-8) | run_eval | **0.83 (+0.19)** | **0.62 (+0.09)** | ~0.20† (unchanged) |
| I2 counter-query prompt | A/B ×3 | (within noise) | (within noise) | ~0.20† (no effect — reverted) |

*How I1 was chosen:* a fast retriever-level metric (I8) showed gold docs sit in the
candidate pool but past the agent's top-8 (recall 0.32@8 → 0.71@40). I A/B'd three
rerankers — `bge-reranker-base` gave +0.02 r@8 but wrecked the lexical query and
ran 10× slower; **MiniLM gave +0.07 r@8 with no per-query regression**; RRF-fusion
was middle. Shipped MiniLM; end-to-end confirmed +0.19 precision / +0.09 recall
(both anchored by the deterministic retriever metric, so not agent noise).

† **adverse_recall (Q1) is high-variance** (denominator 5; ranges 0.00–0.40 per
run). Measured over 3 runs it is ~0.20 regardless of reranker or counter-query
prompt — neither moved it. The adverse docs rank 5–16 (DOC_002 unreachable), so
they don't reach the agent's top-8; the fix is structural (I7 metadata
tagging/filtering), not retrieval-ranking or prompting. *(G-Eval reasoning scores
are likewise too noisy to A/B on single runs — the deterministic backbone is the
signal.)*

### Failure analysis (where it fails, what I'd fix first)

1. **Adverse recall on the case brief is the headline weakness (0.20).** Now that
   the adverse set is honest (5 distinguishable gratuitous-passenger/use-breach
   cases), the agent surfaces only 1 of 5. A system that misses what cuts against
   the client is the dangerous failure mode (Dimension 4). **First fix:** a
   first-class **counter-query step** — an explicit "now search for what exonerates
   the insurer" phase — to push adverse recall up independently of overall recall.
2. **Recall is the consistent weak axis (0.33–0.60).** The agent finds the most
   on-point judgments but misses relevant ones past `top_k=8`; doc-level queries
   (Q3 commercial 9/21, Q7 passenger) suffer most. **Fix:** raise `top_k` + add a
   **reranker** (widen retrieval, refilter to a clean top-k), and a per-document
   summary index for the doc-level "which judgments involve X" questions.
3. **Q2 methodology is the weakest deep query (prec 0.43, reasoning 0.30).** The
   agent over-cites methodology-adjacent cases and the judge found the multiplier
   reasoning thin — a content-quality gap, not just retrieval. Worth a targeted
   look (structure-aware chunking to surface the *Held/quantum* section).
4. **`adverse_honesty=0.0` on Q3 is expected, not a bug:** a "list commercial
   vehicles" query has no adverse dimension. Lesson: gate the adverse judge on
   adverse-bearing queries rather than scoring every query.
5. **Negative-query metric is too strict** (see Q6): it penalizes a doc the agent
   itself flags as off-point. A refinement is to score abstention on the agent's
   *asserted* precedents, not every doc_id mention.
6. **Residual faithfulness risk:** landmark precedents (*Swaran Singh*, *Pranay
   Sethi*, *Sarla Verma*) the agent surfaces genuinely appear *inside* the
   retrieved judgments — legitimate. The remaining hardening is an enforced "every
   cited proposition maps to a retrieved span" verification pass.
