# Design Discussion — How Each Decision Should Actually Be Made

*A working document, not a spec. For each of the four phases — ingestion/chunking,
retrieval, agent, evaluation — it answers three questions: (1) what does the task
actually demand here, (2) what would I choose and why, and (3) **how do you decide
which method is right** (i.e. how do you evaluate the choice itself). It also says
plainly where I disagree with the obvious answer or with the current implementation.*

---

## The one idea that governs everything

Before the four phases, one framing that changes every decision below:

**The reviewer grades the unit "document" (DOC_xxx), not "chunk", and grades your
*judgment about choices*, not your absolute scores.**

Two consequences:

1. **The answer unit is the judgment, not the passage.** They ask "*which judgments*
   involve commercial vehicles" and they cite precedents by `DOC_xxx`. They score
   precision/recall over *documents*. So chunking, embedding, and retrieval are all
   just machinery whose only job is: *make sure at least one chunk of every relevant
   judgment ranks high enough to surface that judgment.* Keep your eye on the
   document, not the perfect passage.

2. **Every choice must come with "why this and not the alternative."** The learning
   guide is a buffet of every technique that exists (HyDE, semantic chunking,
   multi-agent, Self-RAG, six vector DBs…). A strong submission *subtracts*: picks
   3–4 techniques that fit 50 legal PDFs and explicitly rejects the rest with a
   reason. Adding everything is as weak a signal as adding nothing.

So throughout, the high-value sentence is never "I used X." It is "I used X instead
of Y because the corpus has property Z, and here is the eval number that shows it
helped."

---

## Phase 1 — Ingestion & Chunking

### What the task actually demands
Turn 50 messy judgment PDFs into something retrievable, such that any relevant
*judgment* can be surfaced and *cited*. That's it. Perfect passage boundaries are a
nice-to-have; reliable doc-level recall and a clean citation handle are the job.

### My opinion (and where I'd push back)

**Chunk size is a minor tunable. Stop optimizing it.** Recursive character splitting
at ~1000–1200 tokens with ~200 overlap is a fine default and the current impl already
does this. The marginal return on tuning 800 vs 1200 is small. Don't spend your
budget here.

**Structure-aware chunking (split on Facts / Held / Ratio) is the "textbook best"
answer — and I'd reject it for this build.** Reason: Indian MACT / High Court
judgments are *not* cleanly structured. They're continuous prose with inconsistent
paragraph numbering and no reliable "Held:" heading. Building a parser that segments
50 heterogeneous PDFs by legal section is high-effort, brittle, and unverifiable. It
belongs in the ADR's "with another week" section as the right *aspiration*, not in
the build. Naming it and *consciously declining it* is the senior move.

**The two ingestion upgrades that actually pay off here are not about chunk size:**

1. **Per-document metadata extraction (highest leverage in the whole phase).** One
   cheap LLM pass per judgment (50 calls, one-time) to extract: case name, court,
   year, **vehicle type** (commercial/private), **claim type** (death/injury), key
   statutes (S.149, S.166 MV Act), and — the important one — **outcome /
   disposition** (did the insurer escape liability? was it "pay and recover"?).
   - Why it's the highest leverage: it turns "*which judgments involve commercial
     vehicles*" from a fragile semantic guess into a **metadata filter**. The current
     eval gets 12/22 on exactly that query *because it answers a doc-level question
     with chunk retrieval*. Metadata fixes the query type the reviewer explicitly
     named.
   - It also turns **adverse identification** (Dimension 4) from luck into a filter:
     `outcome = insurer-not-liable` *is* the adverse set for this client.
   - Honest caveat: extracting these with an LLM is itself a faithfulness risk —
     you must spot-check a sample. But for 50 docs that's an afternoon.

2. **Contextual prepending ("small-to-big" lite).** Before embedding each chunk,
   prepend one generated sentence situating it ("This is from a High Court judgment
   on insurer liability where the driver was unlicensed; this passage discusses…").
   Cheap at 50 docs, and it fixes the classic failure where "the appellant" or "the
   said policy" is meaningless out of context. Strong, defensible, low-cost.

### How you decide which chunking/ingestion choice is *right* (the user's real question)

This is the part most people get wrong, so be explicit:

> **You never evaluate chunking in isolation. Chunking only matters through its effect
> on retrieval, so you measure it with a *retrieval* metric, holding everything else
> fixed.**

Concrete procedure (fast, deterministic, no agent, no LLM judge):

1. Build a tiny offline harness: for each gold query, run retrieval and record the
   **set of `doc_id`s that appear in the top-k retrieved chunks**.
2. Metric = **document-level recall@k** (did the gold docs show up at all?) and
   **precision@k** (how much junk came with them).
3. Sweep one variable at a time: `chunk_size ∈ {512, 800, 1200}`,
   `overlap ∈ {100, 200}`, `strategy ∈ {recursive, +contextual-prepend}`.
4. Pick the config on the recall/precision curve — the smallest, simplest config that
   maxes recall@k without tanking precision.

The deliverable insight: **the metric for chunking is retrieval recall, not answer
quality.** Decoupling them is itself a senior signal, because it lets you say later
"this miss was the retriever, not the agent" — which most submissions cannot.

---

## Phase 2 — Retrieval

### What the task actually demands
Given the agent's search query, return the chunks that surface the relevant
*judgments* — for both supporting *and* adverse positions — ranked well enough that
top-k contains them.

### My opinion (and where I'd push back)

**Hybrid (BM25 + dense) is correct; keep it.** Legal text is full of exact tokens
("Section 149", party names, "pay and recover") where dense embeddings are weak, and
conceptual queries where keyword is weak. Ensemble with RRF gets both. The current
0.4/0.6 BM25/dense split is reasonable. No disagreement.

**The single highest-leverage thing missing is a reranker — add it first.** A
cross-encoder (`bge-reranker-base`, open-source, free, local) over the top ~30 hybrid
hits, keeping the top ~8. Why it's the #1 fix:
- It improves **precision** (pushes topically-adjacent junk down) *and* effective
  **recall** (promotes buried-but-relevant docs into the top-k window) from one
  change. The current eval shows recall (~0.47–0.55) is the weak axis — this is the
  lever for it.
- At 50 docs the latency cost is nil. There is no reason not to.

**The retrieval insight specific to THIS task — and it's the one most candidates
miss: adverse retrieval is a *query-construction* problem, not a ranking problem.**
If you embed the client's framing ("insurer must pay despite unlicensed driver"),
semantic search returns cases *like that* — i.e. supporting cases. Adverse cases are
semantically *about the opposite proposition* ("policy void, insurer not liable") and
will systematically rank low. **No reranker and no better embedding fixes this,
because the query points the wrong way.** The fix is architectural: someone must
issue retrieval queries from the *opponent's* perspective. Two ways:
- a dedicated tool (`search_adverse(position)` that reframes), or
- the system prompt instructs the ReAct agent to always also search the counter-
  position.

I lean toward the prompt approach — keep two generic tools, push the behavior into
reasoning — because it preserves the clean architecture. (The current prompt *does*
ask for the "counter-argument" angle, which is right; the question is whether it's
reliable enough or needs to be a named tool. That's measurable — see below.)

**On "efficiency": be honest that it doesn't matter at 50 docs.** Everything is
sub-second; in-memory BM25 rebuild on boot is fine. Pretending to optimize latency
now is a *negative* signal (premature optimization). Efficiency is strictly a
**5,000-doc** conversation for the ADR: native hybrid index (Qdrant/Weaviate),
batch/precomputed embeddings, metadata pre-filtering, rerank-only-the-candidates.
Put it there, not in the build.

### How you decide which retrieval method is *right*

Same discipline as chunking — don't argue it, layer it and measure the delta on the
gold set:

| Layer | What you measure | What it should show |
|---|---|---|
| dense-only top-k | recall@k, precision@k, MRR | baseline |
| + BM25 hybrid (RRF) | same | recall ↑ on keyword-heavy queries |
| + reranker | same, esp. precision & NDCG | precision ↑, buried docs promoted |
| + adverse counter-query | **adverse-recall** specifically | adverse docs now surface |

The ADR sentence writes itself: "hybrid added +X recall, reranking added +Y
precision, the counter-query step took adverse-recall from A to B." That measured
chain *is* the "reason about alternatives" signal they grade.

### Tools — how to define them (you asked this here, and it belongs here)

The agent's behavior is ~90% determined by **(a) tool descriptions and (b) the system
prompt** — not by your Python. So tool design is prompt engineering, and it's
underrated.

Principles:
- **Few, generic, composable tools beat many specialized ones.** Two or three is
  right: `search_corpus(query)`, `get_document(doc_id)`, and — if you do metadata —
  `filter_documents(field, value)`. More tools = more ways for the model to choose
  wrong.
- **The description is a prompt.** "search_corpus: hybrid semantic+keyword search over
  judgments; call it multiple times with different phrasings, *including the opposing
  side's framing*, to surface both supporting and adverse cases" — that one
  description is what drives adverse behavior. Vague descriptions = bad tool choices.
- **Every result must carry `doc_id`** — that's the citation handle *and* the thing
  your precision/recall is computed from. Return snippets, not full documents (context
  budget); `get_document` is the depth escape hatch.
- **Real pushback on the current two-tool design:** the doc-level query "which
  judgments involve commercial vehicles" is the wrong job for chunk search — it
  returns chunks, not an enumerated, complete set of documents, and recall suffers
  (12/22). A `filter_documents` tool over extracted metadata answers that query type
  far better. So metadata (Phase 1) isn't polish — it unlocks a *tool* the current
  architecture lacks. Phases 1 and 2 are the same decision.

---

## Phase 3 — The Agent

### What the task actually demands
One agent that **decides its own workflow** (no if-else): minimal work for simple
queries, multi-step research for precedent tasks, with **visible** intermediate
reasoning, producing the three-section output for research.

### My opinion: yes, the system prompt is primary — but four things beyond it matter

You're right that for a single-agent ReAct design the **system prompt is the main
control surface.** But "is there anything beyond the prompt?" — yes, four things, and
underweighting them is the common mistake:

1. **Tool descriptions (co-equal with the prompt).** Covered above. The model picks
   tools from these. Treat them as part of the prompt, not as code comments.

2. **Termination / step control.** A ReAct loop can over-retrieve, loop, or stop
   early. You need a **max-step guard** and prompt language for "when you have enough,
   stop." Note the current deep query made **22 tool calls** — that *might* be
   thoroughness, or it might be flailing. You can't tell without inspecting, which is
   why you bound it *and* surface the trace.

3. **The output contract — schema vs. prose (a real trade-off).** The current impl
   asks for the three sections as *prose*. Prose is natural and easy to render with
   inline citations — but it is **not machine-checkable**, and required sub-fields
   (`fact_alignment`, `distinguishing_argument`, `risk`) can silently go missing. For
   a product where "honest adverse risk assessment" is explicitly graded, I lean
   toward a **soft schema**: the prompt names the required fields per precedent (and
   optionally a final structured-output pass). Trade: guaranteed completeness &
   evaluability vs. naturalness. At minimum, *know* you made this trade.

4. **Grounding mechanics (beyond the prompt).** This is about how you *feed* context,
   not just what you instruct:
   - give the agent the **client facts explicitly** as structured input so fact-
     alignment is grounded, not recalled;
   - require a `doc_id` on **every** claim;
   - a "**say not-found** if it's not in the corpus" escape hatch.
   The current impl had a genuine hallucination bug (external reporter citations,
   mis-attributed case names) that was fixed *only by tightening the prompt*. That
   fragility is the lesson: **prompt-only grounding is brittle.** The durable fix is a
   **verification pass** — "every cited proposition must map to a retrieved span" —
   which is a known gap worth naming honestly.

### Where I'd reconsider the architecture

- **Single agent is the right call.** Multi-agent (Researcher + Devil's-advocate) is
  elegant for the supporting/adverse split but over-engineering for 50 docs. Mention
  it as a future direction; don't build it.
- **Pure ReAct is fine, but consider a "plan-first" prompt for the observability
  score.** The brief grades *visible reasoning*. A 22-step ReAct log is a worse
  reasoning artifact than an explicit up-front plan ("1. find supporting 2. find
  adverse 3. assess quantum 4. synthesize"). You can get a visible plan **without any
  new framework** — just instruct the single agent to emit a short plan before acting.
  That's a cheap lift on a graded dimension. Worth doing.

### How you decide if the agent design is *right*
- **Dynamic-workflow proof:** the contrast in tool-call counts (simple ≈ 1–2, deep ≈
  many) *is* the evidence that it's not a hard-coded pipeline. The current eval already
  shows 2 vs 22 — that's the right thing to measure and display.
- **Test with off-brief prompts** (contributory negligence; commercial vehicles), not
  just the Lakshmi Devi brief — that's exactly how the reviewer will test, and it's
  the only way to catch accidental over-fitting to the case.

---

## Phase 4 — Evaluation (you called this the most important; I agree)

### First, a foundational distinction: fixed questions vs. live questions

Before any metric, get this clear, because the interview will probe it: **the
deterministic metrics (precision, recall, NDCG) can ONLY run against a fixed set of
pre-labeled questions. They cannot run on an arbitrary live question.**

Why: precision and recall are *comparisons*. Precision = |predicted ∩ gold| /
|predicted|; recall = |predicted ∩ gold| / |gold|. Both formulas contain **`gold`** —
the set of truly-relevant docs. If someone types a brand-new question, *there is no
`gold`* — nobody has labeled which of the 50 docs are relevant to it — so the
denominator literally does not exist. You cannot compute the number. This isn't a
limitation of the implementation; it's inherent to what precision/recall *are*.

So metrics split into two families:

| | Needs a labeled "right answer"? | Fixed gold question | Arbitrary live question |
|---|---|---|---|
| **precision / recall / NDCG** | yes (ground-truth) | ✅ works | ❌ impossible — no gold to compare to |
| **faithfulness / groundedness / "did it cite", "did it surface adverse"** | no (reference-free — judges the answer against the *retrieved text*) | ✅ works | ✅ works |

Consequences for the design:

- **The eval harness runs a fixed, hand-labeled question set** (the gold set). That's
  not laziness — producing the `gold` list means a human reading the corpus once per
  question, and you can only do that for a fixed set.
- **A live query in the Streamlit app gets no precision/recall** — only the
  reference-free checks (faithfulness against retrieved chunks, citation presence,
  adverse-surfacing). Useful, but not the graded P/R numbers.
- **What the reviewer is doing:** "*we'll evaluate recall against **our** internal
  benchmark*" means *they* have their own fixed gold set (their labeled questions +
  right docs) and run this same fixed-question eval against your live agent. **Your**
  gold set is therefore your *hypothesis* of theirs.
- **So your fixed question set must cover query *categories*, not memorize one brief:**
  at least one deep precedent-research query, one doc-level filter query ("which
  involve commercial vehicles"), and one off-brief query ("contributory negligence").
  Covering categories is robustness; over-fitting to the Lakshmi Devi brief is the
  hard-coded-pipeline trap.

### Second distinction: offline (batch) eval vs. online (inline) eval

A natural misread of "automated eval" is *"the agent attaches a score to every
response as it answers."* That's a real pattern, but it is **not** what the brief
requires, and for the ground-truth metrics it's impossible. Keep the two patterns
separate:

| | **Offline / batch eval** ← what the brief wants | **Online / inline eval** ← the thing people picture |
|---|---|---|
| When it runs | separately, on demand (`run_eval.py`) | attached to every live response, in real time |
| Runs on | a **fixed, labeled** question set (gold set) | whatever arbitrary question the user typed |
| Output | a report (`eval_results.md`) read once | a score shown next to each answer |
| Also called | "the eval harness" | "guardrails" / "online evals" |

Three things to be precise about (all separate axes):

- **"Automated" ≠ "deterministic" ≠ "inline."** *Automated* means a script grades it,
  not a human (the brief's only requirement). *Deterministic* means exact math (set
  overlap) vs. an LLM-judge — **an LLM-judge is still automated.** *Inline* means
  scored live per response — which is a deployment choice, not the requirement.
- **The ground-truth metrics can't run inline.** Precision/recall/adverse-recall need
  `gold`; a live arbitrary question has none, so they only run offline over the fixed
  gold set. This is the same denominator argument as the fixed-vs-live distinction
  above — it's *why* the harness is a separate batch job.
- **Only reference-free checks (faithfulness) *could* run inline**, as an optional
  guardrail, because they judge against the retrieved text, not a gold label. Nice
  polish; not required, and it can't cover the four graded dimensions.

So: **the brief's "≥1 automated eval per dimension" is a floor, met by an offline
batch harness** (the current build already ticks it). Everything else in this section
is about making those four automated metrics *good* — which is the part actually
graded ("we evaluate your judgment on **how** to measure").

### Precision & Recall — the exact mechanics you asked about

"How do you know the actual number vs. the predicted number?" Precisely:

- **Predicted set** = the set of `doc_id`s the agent **cites** as relevant in its
  answer. (Extract with regex `DOC_\d{3}`, dedup → a set.)
- **Gold (actual) set** = the `doc_id`s *you* labeled as truly relevant for that
  query.
- **Precision** = |predicted ∩ gold| / |predicted| — *of what the agent claimed, how
  much was right* (penalizes over-retrieval).
- **Recall** = |predicted ∩ gold| / |gold| — *of what's truly relevant, how much it
  found* (penalizes misses).

It is just set overlap — deterministic, no LLM. **Your instinct (deterministic golden
set) is correct.** But there are two subtleties that separate a good eval from a naive
one, and I'd push on both:

#### Pushback A — measure precision/recall at TWO levels, not one
The current impl measures only the **agent's final citations**. That conflates two
different failures:
- the retriever never surfaced the doc (a **retrieval** failure), vs.
- the retriever surfaced it but the agent didn't cite it (an **agent** failure).

You can't fix what you can't separate. So report **both**:

| Level | Predicted set | Measures | Used to tune |
|---|---|---|---|
| **Retrieval-level** | doc_ids in top-k retrieved | chunking + retrieval | Phases 1–2 |
| **Answer-level (end-to-end)** | doc_ids the agent cites | the whole system | final score |

The reviewer's "internal benchmark" most resembles the answer-level number, but the
retrieval-level number is what lets you *diagnose*. Reporting both is the senior move
and directly answers their failure-analysis question.

#### Pushback B — the gold set is the entire ballgame; treat it that way
They said "*we'll evaluate recall against **our** internal benchmark.*" So your gold
set is a *hypothesis* of theirs. Two honesty requirements:
- **Use exhaustive labeling and say so.** At 50 docs you have the rare luxury of
  reading *every* document and labeling its relevance to each test query — that gives
  *true* recall, not an estimate. Lean into the small-corpus advantage; state the
  methodology explicitly (the current gold set was built this way — make that
  visible).
- **Consider graded relevance, not binary.** "Relevant" isn't yes/no for precedents —
  some are directly on point, some adjacent. Binary labeling punishes precision for
  "topically adjacent" docs a human might count as half-relevant (this is *exactly*
  why the case-brief precision is 0.64). Graded labels + **NDCG** reward ranking the
  on-point cases higher and give a fairer, more defensible number.

### Reasoning quality (LLM-judge) — you're right that the rubric is the key; here's the catch

LLM-as-judge is the correct tool, and yes, **the rubric/prompt is what determines the
score's worth.** Two things I'd push hard on:

1. **The judge must SEE THE SOURCE, not just the answer.** This is a concrete flaw in
   the current eval: the G-Eval reasoning metric is given only `INPUT` and
   `ACTUAL_OUTPUT` — *not the retrieved chunks.* A faithfulness judge that can't see
   the source can only check whether the answer is *plausible*, which is the opposite
   of what you want (plausible-but-ungrounded is the exact failure mode in law). Pass
   the retrieved context to the judge so it can check **groundedness** — does every
   claim trace to a retrieved span. This is the single most important fix in the eval.

2. **Decompose "reasoning quality" instead of one scalar.** Split into: (a)
   **faithfulness** (grounded in retrieved text — semi-automatable, RAGAS-style claim
   checking), (b) **fact-alignment correctness** (does the claimed overlap with the
   client's facts actually hold), (c) **distinguishing-argument coherence** for
   adverse cases. A single "0.9" hides which one failed.

**And name the judge's weakness honestly:** the current setup is *Gemini judging
Gemini* with no calibration — that's self-preference bias (a model rates its own
style generously). Mitigations to state: use a *different* model as judge if you can,
give it the source context, randomize, and **calibrate against a handful of
human-scored examples** so you can say *how much* to trust the number.

### Adverse identification — two parts, and one test-design bug to fix

- **Part 1, deterministic:** adverse-recall = |predicted ∩ gold_adverse| /
  |gold_adverse| — did it surface the *known* adverse docs.
- **Part 2, LLM-judge:** honesty — did it actually *present* them as risks with
  distinguishing arguments, not bury them. The current impl does both. Good.
- **Bug to fix:** the current run scores `adverse_honesty = 0.0` on the "commercial
  vehicles" simple query and calls it "expected." It's not expected — it's a
  **test-design error**: you shouldn't run the adverse judge on a query that has no
  adverse dimension; a spurious 0.0 pollutes any aggregate. **Gate each metric to the
  queries where it's meaningful.**
- **The deeper point:** the eval *measures* adverse coverage but can't *create* it.
  Low adverse-recall traces back to single-perspective querying (Phase 2). So the
  failure analysis should connect the *measured* number to the *upstream* fix —
  exactly the loop they're hiring for.

### What "good evaluation" means here (the meta-point)
They grade *judgment about measurement*, not the scores. So the value of your eval is
in: (1) separating **retrieval-level from answer-level** metrics, (2) an **honest,
exhaustive gold set** with stated methodology, (3) a faithfulness judge that
**actually sees the source**, (4) **calibrating** the judge, (5) **gating** each
metric to the queries where it applies, and (6) a **failure analysis that ties a
measured number to a specific first fix.** That last one, they said, is the answer
they care about most.

---

## Summary — what I'd actually change vs. the current build

| Phase | Keep | Change / add | Priority |
|---|---|---|---|
| Ingestion | recursive 1200/200 | **per-doc metadata (esp. outcome)**, contextual prepend | high |
| Retrieval | hybrid BM25+dense | **reranker**, metadata-filter tool, reliable counter-query | high |
| Agent | single ReAct, two tools | plan-first prompt; soft output schema; grounding verification pass | medium |
| Eval | deterministic P/R + gold set | **retrieval-level metrics**, **judge sees source**, gate metrics, graded relevance + NDCG, judge calibration | high |

**The through-line:** every one of these is justified by a property of *this* corpus
(50 docs, doc-level grading, legal keyword density, adverse-coverage requirement) and
is *measurable*. That — not the technique list — is what's being graded.
