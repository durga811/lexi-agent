# Building a Legal Precedent Research Agent — A Learning Guide

*A study companion for the Lexi Backend Engineer take-home. This is not a solution. It is a map of everything you need to understand so that you can design the solution yourself, defend it in an interview, and know where it will break.*

---

## How to use this document

Read Part 0 first — it reframes the assignment so the rest makes sense. Then read Parts 1–6 in order; they build a mental model from the ground up. Parts 7–11 go deep on the things this specific assignment cares about most (legal precedent, observability, evaluation). Part 12 lays out concrete architecture options you can choose between. Part 13 is a learning sequence with small experiments. Part 14 is interview ammunition.

You already build production systems (Next.js, Supabase, Cloudflare Workers, CI/CD, Stripe). So I am **not** going to explain APIs, deployment, environment variables, or databases-in-general. I *am* going to go slow on the ML/LLM-specific machinery — embeddings, vector search, chunking, agentic loops, and evaluation — because that is the genuinely new surface area for you. Where useful I'll connect new ideas to things you already run.

---

## Part 0 — What is actually being tested

The brief is a legal task, but read the rubric and the "Note on Legal Domain Knowledge" carefully. They tell you outright: *"A strong submission from an engineer with zero legal background will outperform a mediocre submission from someone who happens to know the law."* That is a giant hint. They are testing four engineering competencies:

1. **Architecture judgment.** Did you pick a retrieval strategy, chunking approach, and agent design that fit the problem, and can you articulate *why* over the alternatives? The ADR is where this is graded, and they say it's "one of the most important deliverables."
2. **Faithful information use.** Does your system extract and use what's actually in the documents, rather than hallucinating plausible-sounding legal reasoning? Grounding and citation are the heart of this.
3. **Measurement.** Can you build an evaluation framework that honestly measures quality — including the uncomfortable parts (recall: what did you *miss*?), and adverse precedents (what works *against* the client)?
4. **Reasoning about failure.** They explicitly ask "where does your agent fail and what would you fix first?" Knowing your system's weaknesses is treated as a senior signal.

There are also three traps planted in the brief, each of which fails a submission outright:

- **The hard-coded pipeline trap.** "The agent must not be a hard-coded pipeline... not through if-else branching." They will test with *different* prompts ("Which judgments involve commercial vehicles?" vs. "Find precedents on contributory negligence"). If your code is a fixed `parse → embed → retrieve → summarize` script keyed to the Lakshmi Devi brief, you fail. They want the *agent* to decide its own workflow.
- **The only-favorable trap.** "A system that only finds favorable cases is dangerous in legal practice." If your retrieval only surfaces documents similar to the client's position, you will score zero on Dimension 4. Surfacing *adverse* precedents is an architectural decision, not an afterthought — see Part 7.
- **The black-box trap.** "Intermediate reasoning steps must be visible... Do not show only the final output." The UI must expose what was retrieved, how it was ranked, and how conclusions were reached.

**Reframed, the task is:** *Build an agent that, given an arbitrary natural-language task over a 50-PDF corpus, dynamically decides whether to answer directly or run a multi-step retrieval-and-analysis workflow, returns grounded results with visible reasoning, and is accompanied by an evaluation harness that measures retrieval precision/recall, reasoning quality, and adverse-case coverage.*

Everything below is in service of being able to design that and defend every choice.

---

## Part 1 — The core mental model: LLMs, RAG, and Agents

### 1.1 What an LLM can and cannot do here

A large language model is a next-token predictor with a fixed **context window** (the working memory it can "see" in one call). Three properties drive every design decision:

- **It has no built-in knowledge of your 50 PDFs.** It was trained on the public internet up to some cutoff. Your corpus is private and specific. So the documents must be *put into the context window* at answer time.
- **The context window is finite** (today's models range from ~128K to ~1M tokens). 50 court judgments easily exceed even large windows, and stuffing everything in every time is slow, expensive, and degrades quality ("lost in the middle" — models attend poorly to information buried in very long contexts). So you must *select* what to show.
- **It hallucinates.** Asked about something it half-remembers, it will produce fluent, confident, wrong text. In a legal product this is catastrophic. The antidote is **grounding**: forcing the model to answer only from retrieved source text, and citing it.

### 1.2 RAG = Retrieval-Augmented Generation

RAG is the pattern that solves "the model doesn't know my docs and can't fit them all." It has two phases:

- **Indexing (offline, once):** Parse documents → split into **chunks** → convert each chunk to a numeric **embedding** vector → store vectors (and the text + metadata) in a searchable index.
- **Querying (online, per request):** Convert the user's question to an embedding → find the most similar chunks → put those chunks into the prompt as context → the LLM generates an answer grounded in them.

The whole game of RAG quality is: *did you retrieve the right chunks, and did the model faithfully use them?* That single sentence is the lens for Parts 2–5.

### 1.3 "Naive RAG" vs. reality

The 30-line tutorial version (one embedding model, fixed chunk size, top-5 cosine similarity, stuff into prompt) works in demos and falls over in production. Classic failure modes, all relevant here:

- The answer spans multiple chunks, but you only retrieved one.
- The right chunk exists but ranked 8th, and you only took top-5 (a *recall* failure).
- You retrieved 5 chunks that are superficially similar but irrelevant (a *precision* failure).
- The chunk got cut mid-sentence and lost the meaning.
- The query and the document use different words for the same concept (lexical gap), so similarity search misses it.

Parts 2–4 are essentially the catalogue of techniques people layer on to fix these.

### 1.4 What "agent" means

An **agent** is an LLM in a loop that can **decide to take actions** (call **tools**) based on what it observes, rather than running a fixed script. The canonical loop is **ReAct** (Reason + Act):

```
Thought:  I need precedents about unlicensed drivers and insurer liability.
Action:   retrieve("insurer liability unlicensed driver motor accident")
Observation: [5 chunks returned]
Thought:  These support the client. Now I should look for cases the insurer could use.
Action:   retrieve("policy void breach of condition driver license valid defence")
Observation: [4 chunks]
Thought:  I have enough. Compose the supporting/adverse/strategy analysis.
Final Answer: ...
```

The key difference from naive RAG: **the agent chooses what to retrieve, how many times, and when to stop.** That is exactly the "dynamic workflow, not if-else" property the brief demands. The agent's tools are things like `search_corpus`, `filter_by_metadata`, `read_full_document(id)`, and maybe `compare_cases`. The model's reasoning decides the sequence.

### 1.5 The two axes of this assignment

Hold these two independent axes in your head; most design choices live on one of them:

- **Retrieval quality axis** (Parts 2–5): chunking, embeddings, hybrid search, reranking, metadata. Determines whether the right evidence ever reaches the model. This drives precision and recall.
- **Agentic flexibility axis** (Part 6): how the system decides its workflow per query. Determines whether you pass the "not a hard-coded pipeline" requirement and handle arbitrary prompts.

A strong submission is competent on both. A common mistake is to over-invest in a fancy multi-agent framework (flexibility axis) while shipping naive top-5 retrieval (quality axis), then scoring poorly on precision/recall.

---

## Part 2 — Document processing and chunking

This is unglamorous and decides more of your final score than anything else. Garbage chunks → garbage retrieval → garbage answers, no matter how clever the agent.

### 2.1 PDF parsing: the messy reality

Court judgments are some of the nastiest PDFs: multi-page, dense, sometimes two-column, sometimes scanned images, with headers/footers, paragraph numbering, citations, and footnotes. Your first job is text extraction that preserves structure (paragraph breaks, headings, the natural reading order).

Tools, roughly from "fast and simple" to "smart and heavy":

| Tool | What it is | When to use |
|---|---|---|
| **PyMuPDF (`fitz`)** | Very fast, accurate text + layout extraction from native (text-based) PDFs | Default first try; great for clean digital judgments |
| **pdfplumber** | Strong at tables and precise character positioning | When layout/tables matter |
| **`unstructured`** | Library that segments docs into titles, narrative text, lists, tables | When you want structure-aware elements out of the box |
| **Docling** (IBM, open source) | Modern document-understanding pipeline; good layout + tables, exports to structured formats | When parsing quality is a differentiator and you want metadata-rich output |
| **LlamaParse** (hosted) | LlamaIndex's managed parser, very good on complex layouts | When you'll accept a hosted dependency for quality |
| **OCR (Tesseract, or vision models)** | Turns scanned images into text | Only if some DOCs are scanned images, not digital text |

**Action item before you design anything:** open 5–6 of the 50 PDFs and *look* at them. Are they digital text or scans? One column or two? Numbered paragraphs? Consistent header with case name/court/date? The answers change your whole pipeline. A two-column scanned judgment needs OCR + careful reading-order logic; a clean digital judgment needs almost nothing.

### 2.2 Chunking: why and how

You split documents because (a) embeddings have a token limit, (b) retrieval should return the *relevant passage*, not a whole 30-page judgment, and (c) precision improves when chunks are coherent units of meaning.

The strategies, with trade-offs:

- **Fixed-size (e.g. 512 tokens, 50-token overlap).** Simple, predictable. Blind to structure — happily cuts a sentence or a legal holding in half. The overlap exists so a concept split across a boundary still appears whole in one chunk. Fine as a baseline.
- **Recursive character splitting.** Splits on a hierarchy of separators (paragraphs → sentences → words) trying to keep semantic units intact up to a size cap. The pragmatic default; what most people ship.
- **Structure-aware / document-based.** Split on the document's own structure — headings, numbered paragraphs (judgments are often numbered ¶1, ¶2...), sections (Facts / Issues / Held). For legal text this is *much* better because a "holding" is a natural unit. Worth the extra parsing effort here.
- **Semantic chunking.** Use embeddings to detect topic shifts and split there. Produces coherent chunks but is slower and can be unpredictable.
- **Parent-document / auto-merging / hierarchical.** Embed and search over *small* chunks (precise matching) but return the *larger parent* passage to the LLM (full context). LlamaIndex's auto-merging retriever and the "small-to-big" pattern. Excellent when you need precise hits but rich context — very applicable to legal reasoning where the surrounding paragraphs matter.
- **Contextual retrieval (the "prepend a summary" trick).** Before embedding each chunk, prepend a short LLM-generated sentence situating it in the document ("This passage is from a 2018 Supreme Court judgment on insurer liability, discussing..."). Dramatically improves retrieval because isolated chunks lose context ("the appellant" means nothing without knowing the case). For 50 docs this is cheap to do and a strong, defensible choice.

**Chunk size trade-off:** small chunks = precise matches but fragmented context; large chunks = rich context but noisy matches and you retrieve irrelevant text alongside the relevant bit. The parent-document pattern is the standard way to get both.

**The legal-specific insight:** a precedent's value lives in its *ratio* (the legal principle) and the *facts* it turned on. If your chunking shreds those apart, the model can't reason about "which facts align." Structure-aware chunking + parent-document retrieval directly addresses this.

### 2.3 Metadata — quietly the highest-leverage thing you'll do

Each chunk should carry metadata: source doc id, case name, court, year/date, judges, the section it came from (Facts/Held), and ideally extracted entities (vehicle type, claim type, statutes cited). You can extract a lot of this with one cheap LLM pass per document at index time.

Why it matters so much:
- It powers **filtered retrieval** ("which judgments involve commercial vehicles?" → filter on a `vehicle_type` field rather than hoping similarity search nails it).
- It powers **citations** in the output (the brief implicitly needs "Judgment X establishes principle Y").
- It is your **agent's tool surface** — `filter_by(court="Supreme Court", year>2010)` is a tool the agent can call.

For a 50-document corpus, spending real effort on a clean per-document metadata extraction step is one of the best ROI moves available, and it's a great thing to talk about in the ADR.

---

## Part 3 — Embeddings and semantic search

### 3.1 What an embedding is

An embedding model maps a piece of text to a fixed-length vector of floats (e.g. 768 or 1024 numbers). The geometry encodes meaning: texts about similar topics land near each other. "Similarity" is usually **cosine similarity** (the angle between vectors). So semantic search = embed the query, find the nearest chunk vectors.

The power: it matches *meaning*, not keywords. "Insurer refuses to pay" can match "the insurance company denied liability" even with no shared words. The weakness: it can miss exact terms, rare names, and statute numbers (lexical precision), which is why you'll later add keyword search back in (Part 4).

### 3.2 Dense vs. sparse

- **Dense** (what I described): neural embeddings, semantic. Great for paraphrase and concept matching.
- **Sparse** (BM25, TF-IDF): classic keyword search, scores by term frequency. Great for exact terms, names, citations, statute numbers — precisely where dense models are weakest.
- **Hybrid** uses both and fuses the results. For legal text (full of specific terms like "Section 149", "National Insurance Co.", "contributory negligence") hybrid is almost always better than dense alone. More in Part 4.

### 3.3 The model landscape (as of early–mid 2026)

The open-source embedding ecosystem has caught up to and in places passed the commercial APIs. Current notable options:

- **BGE-M3** (BAAI, open): the workhorse open-source default. ~100+ languages, and unusually it produces dense, sparse, *and* multi-vector representations from one model — convenient for hybrid search. Excellent quality-to-cost ratio. A safe, defensible self-hosted pick.
- **Qwen3-Embedding** (open): tops the MTEB v2 leaderboard among open-weight models; instruction-aware (you can prepend a task instruction to nudge it), flexible output dimensions. Strong choice if you want top benchmark quality.
- **Nomic Embed v2 / GTE / E5** family (open): solid, long-context options.
- **OpenAI `text-embedding-3-small`/`-large`** (API): the common, frictionless default; small is cheap and good, large is higher quality. Supports Matryoshka (truncate dimensions to trade quality for storage).
- **Voyage AI (`voyage-3` family)** and **Cohere `embed-v4`** (API): retrieval-optimized commercial models that consistently rank well; Voyage in particular is favored for RAG retrieval, and both pair naturally with rerankers.
- **Google Gemini Embedding** (API): multimodal, strong multilingual.

**How to actually choose** (and what to say in the ADR):
1. Benchmarks (MTEB) are a starting filter, not a verdict — vendors self-report and generic scores don't always transfer to *your* domain. Benchmark 2–3 candidates on *your* corpus with a handful of real queries.
2. For a 50-doc take-home, an API model (OpenAI small, or Voyage) gives you zero-ops simplicity; an open model (BGE-M3) gives you a "no external dependency, runs anywhere" story and avoids per-call cost. Either is defensible; *state the trade-off*.
3. **Query/document asymmetry:** some models (Voyage, instruction-tuned ones) want you to flag whether you're embedding a query or a document, or to prepend an instruction. Read the model card; using it wrong silently degrades retrieval.
4. **Dimensions** affect storage and speed. 768–1024 is the usual sweet spot. At 50 docs it's irrelevant; mention it for the "5,000 docs" question.
5. **Domain fit / fine-tuning:** generic models can underperform on legalese. Fine-tuning an embedding model on legal pairs yields meaningful gains (+10–30% in specialized domains is commonly cited) but is overkill for a take-home. Knowing this exists, and saying "if I had another week I'd evaluate a legal-domain or fine-tuned embedder," is a strong ADR point.

---

## Part 4 — Retrieval strategies (where precision and recall are won)

Retrieval is the single biggest lever on your eval scores. Layer these as needed:

### 4.1 Baseline: top-k dense similarity
Embed query, return k nearest chunks. Set k thoughtfully — too small hurts recall, too large hurts precision and floods the context. k is a tunable you should sweep in your eval.

### 4.2 Hybrid search (dense + sparse)
Run dense and BM25 in parallel, fuse the ranked lists. The standard fusion method is **Reciprocal Rank Fusion (RRF)** — simple, robust, no score-normalization headaches. Hybrid is the biggest single quality upgrade for keyword-heavy domains like law. Many vector stores (Weaviate, Qdrant, Milvus, pgvector with extensions) support this natively; BGE-M3 even gives you both vectors from one model.

### 4.3 Reranking (do not skip this)
First-stage retrieval (dense/hybrid) optimizes for speed over a large set; it's approximate and order is rough. A **reranker** is a heavier **cross-encoder** model that takes (query, chunk) *pairs* and scores true relevance precisely. Pattern: retrieve top ~25–50 candidates cheaply, rerank, keep the top ~5–8 for the LLM.

Why it matters: cross-encoders see the query and candidate *together*, so they catch relevance a bi-encoder embedding misses. This is often the difference between "the right doc was retrieved but ranked 9th" and "it's now ranked 1st." Options: **Cohere Rerank**, **bge-reranker** (open), **Qwen3-Reranker** (open), Voyage rerank. For this assignment reranking directly boosts **precision** (Dimension 1) and helps **recall@k** by promoting buried-but-relevant hits.

### 4.4 Metadata filtering
Combine semantic search with hard filters on your metadata fields. "Which judgments involve commercial vehicles?" should be a filter, not a vibe. This is also where pre-filtering vs. post-filtering matters (filter before ANN search for correctness/speed, where the store supports it).

### 4.5 Query transformations (the techniques the *agent* will wield)
These convert a single user prompt into better retrieval calls — and they're naturally expressed as agent actions, which is how you satisfy "dynamic workflow":

- **Query rewriting / expansion:** turn a verbose brief into a focused search query, or expand with synonyms ("unlicensed driver" → "without valid driving license", "breach of policy condition").
- **Multi-query:** generate several phrasings of the question and union the results — boosts recall.
- **Query decomposition:** split a complex task into sub-questions, retrieve for each, then synthesize. ("Find precedents supporting the claim" decomposes into: precedents on insurer liability despite license defect; precedents on quantum for a 42-year-old earner; precedents on dependents' compensation.)
- **HyDE (Hypothetical Document Embeddings):** ask the LLM to *write a fake ideal answer*, embed *that*, and search with it. Often beats embedding the bare question because the hypothetical answer is lexically closer to the target documents. Cheap, clever, worth knowing.
- **Step-back prompting:** generate a more general question to retrieve foundational principles before the specific one.

### 4.6 The recall-for-adverse-cases insight (read twice)
Standard retrieval finds documents *similar to the query*. If your query encodes the client's position, you will retrieve documents supporting the client and systematically miss the ones that hurt them — because adverse precedents are semantically *about the opposing argument*. To find adverse cases you must **deliberately issue retrieval queries from the opponent's perspective** ("cases where insurer was *not* liable due to license breach", "policy held void for breach of condition"). This is an *architectural* decision: your agent needs a step/tool that searches for counter-arguments, not just supporting ones. Nailing this is how you score on Dimension 4 — and it's a great ADR/interview point because most candidates won't see it.

---

## Part 5 — Vector storage and indexing

### 5.1 What a vector database does
It stores vectors and does fast **approximate nearest-neighbor (ANN)** search. Exact nearest-neighbor over millions of vectors is too slow, so ANN indexes (the dominant one is **HNSW**, a navigable graph) trade a tiny bit of accuracy for huge speed. They also handle metadata filtering, persistence, and (some) hybrid search.

### 5.2 The landscape and when each fits

| Option | Nature | Sweet spot |
|---|---|---|
| **In-memory / FAISS / NumPy** | A library, not a DB | Tiny corpora, prototypes. **For 50 docs this is genuinely enough.** |
| **Chroma** | Embedded, dead-simple Python | Prototyping and small apps; run it embedded or single-server |
| **pgvector** (Postgres extension) | Vectors inside Postgres | You already run Postgres/Supabase; under ~5–50M vectors; want one system, ACID, SQL filters. *Your existing Supabase familiarity makes this a natural, defensible choice.* |
| **Qdrant** | Open-source, Rust, strong filtering | Pure-vector workloads where filter performance and speed matter |
| **Weaviate** | Open-source, native hybrid search | When you want hybrid retrieval and rich schema out of the box |
| **Milvus** | Distributed | Billions of vectors, you have an ops team |
| **Pinecone** | Managed, proprietary | Want zero ops and will pay for it |
| **LanceDB** | Embedded, multimodal | Local-first, multimodal |

### 5.3 The honest take for *this* assignment
With 50 documents you have a few thousand chunks at most. You do **not** need a heavyweight vector DB; an in-memory index or Chroma is fine and keeps the "no local infra required to evaluate" constraint easy (the index can build at app startup or load from a committed artifact). Using pgvector/Supabase is also reasonable and lets you reuse skills you already have. The mistake would be reaching for Milvus/Pinecone here — it signals poor scale judgment.

### 5.4 The "what if 5,000 documents?" question (they ask this explicitly)
This is an ADR question testing whether you understand how things break at scale. Things that change:
- In-memory stops being viable; you need a real ANN index (HNSW) and a persistent store (Qdrant/Weaviate/pgvector/Milvus).
- **Indexing cost and pipeline:** embedding 5,000 long judgments is a batch job with retries, idempotency, and incremental updates (you know this world from your Cloudflare queue work — same instincts apply).
- **Metadata filtering and pre-filtering** become essential to keep latency and precision sane.
- **Recall evaluation gets harder** (you can't read 5,000 docs to label) — you lean on pooling and sampling (Part 9).
- **Cost control:** caching, smaller/cheaper embedding dims, reranking only the top candidates.
- Possibly **hierarchical or graph retrieval** (route to a sub-corpus first). The agent's routing matters more.

---

## Part 6 — Agent architecture (the heart of the assignment)

This is where you pass or fail "must not be a hard-coded pipeline."

### 6.1 Decoding the requirement
They want the *system's control flow to be decided at runtime by reasoning*, not by `if "precedent" in query: ...`. Concretely: an LLM (the agent/planner) looks at the incoming task and chooses which tools to call, in what order, how many times, and when it's done. A simple factual query ("which docs mention commercial vehicles?") should resolve in one filtered retrieval; a deep research task should trigger multiple retrievals (supporting + adverse + quantum), maybe reading full documents, then synthesis.

### 6.2 The pattern catalogue (these five cover ~95% of agentic RAG)

1. **Router.** A first LLM step classifies the query and routes it: direct answer vs. simple retrieval vs. deep research workflow vs. metadata lookup. This is the cleanest way to implement "simple query → straightforward answer, complex query → deeper workflow" *without if-else* — the *LLM* makes the routing decision. Lowest latency, easy to reason about. Often the right backbone here.
2. **ReAct (tool-use loop).** The agent interleaves reasoning and tool calls until it decides to stop (Part 1.4). Flexible and general; the natural fit for "dynamically determine its own workflow." Risk: it can loop, over-retrieve, or stop early — you need step limits and good tool descriptions.
3. **Plan-and-execute.** The agent first writes an explicit plan (a list of sub-tasks), then executes each (often retrieval per sub-task), then synthesizes. Great for the precedent task because the plan *is* the research workflow: [find supporting] → [find adverse] → [assess quantum] → [synthesize strategy]. Very explainable (the plan is visible — perfect for the "show reasoning" requirement).
4. **Self-RAG / Corrective RAG (CRAG).** The agent grades its own retrieved chunks for relevance; if they're weak, it rewrites the query and retrieves again, or abstains. Adds a self-correction loop that directly improves precision and reduces hallucination. More moving parts.
5. **Multi-agent.** Separate agents (e.g., a Researcher and a Critic/Devil's-advocate that hunts adverse precedents). Conceptually elegant for the supporting-vs-adverse split, but for 50 docs it's usually over-engineering. Worth *mentioning* as a "what I'd do with more time / more scale" option.

A strong, realistic design for this take-home is often **Router → (for research) Plan-and-execute or ReAct over a small set of retrieval tools, with reranking and an adverse-search step.** Plan-and-execute and router patterns also give you the most legible "intermediate reasoning" to display.

### 6.3 Tool design
Your tools are the agent's API to the corpus. Design them like you'd design any clean interface:
- `search(query, k, filters)` — hybrid retrieval + rerank under the hood.
- `filter_documents(metadata)` — pure metadata queries.
- `get_document(doc_id)` / `get_full_context(chunk_id)` — read more around a hit.
- Maybe `search_counterarguments(position)` — an explicit adverse-search tool, which bakes the Part 4.6 insight into the architecture.

Tool *descriptions* are prompt engineering: the model decides what to call based on them. Vague descriptions = bad tool choices. This is a real and underrated skill.

### 6.4 State, loops, and termination
Agents need: a place to accumulate findings (state/memory), a loop with a **max-steps** guard, and a **stopping condition** (the agent declares it has enough, or an evaluator says so). Uncontrolled loops burn tokens and latency. Plan for it.

### 6.5 Frameworks — what to build on
You're allowed LangChain, LlamaIndex, "any LLM/agent SDK." Forbidden: CrewAI and drag-and-drop builders. The realistic choices:

- **LangGraph** (in the LangChain ecosystem): models the agent as a **stateful graph** with nodes and edges, supporting cycles (loop back to retrieve), conditional routing, and human-in-the-loop. It's the strongest fit when control flow is non-trivial — exactly the router/plan/loop patterns above. More code than LlamaIndex for equivalent RAG, but you get explicit, inspectable control flow (which doubles as your "show the reasoning" story). This is the common 2026 choice for *agent orchestration*.
- **LlamaIndex**: data-/retrieval-first. Less code to get high-quality RAG, with batteries-included retrievers (auto-merging, sub-question/decomposition, hybrid). Its agent layer is capable though historically less rich than LangGraph's. Excellent if retrieval quality is your priority and you want to ship fast.
- **The 2026 "use both" pattern:** LlamaIndex as the retrieval layer (wrap its query engine as a tool), LangGraph as the orchestration layer. A LlamaIndex retriever can be exposed as a LangChain/LangGraph tool. This is a legitimately strong, defensible architecture and a good ADR talking point.
- **Raw SDK (Anthropic / OpenAI tool-calling), no framework:** modern model APIs have native tool-use; you can build a clean ReAct loop in a few hundred lines with zero framework lock-in and *total* control/explainability. For a 50-doc system this is very viable and arguably the most defensible in an interview because you'll be able to explain *every* line (which is exactly what they say they'll probe). The cost is you reimplement retries, state, and tracing yourself.

**How to choose for the ADR:** weigh *control & explainability* (raw SDK or LangGraph) against *speed-to-quality-RAG* (LlamaIndex). Whatever you pick, the rubric line that matters is: *can you reason about alternatives and justify the choice?* "I used LangGraph because the workflow needs cyclic, conditional control flow that's awkward to express as a linear chain, and the graph is also what I render as the visible reasoning trace" is a winning sentence.

---

## Part 7 — The legal precedent problem, specifically

The three required outputs (Supporting / Adverse / Strategy) each impose design requirements.

### 7.1 Supporting precedents — "which facts align, what principle each establishes"
This needs **structured extraction**, not a vibe summary. For each candidate precedent the system should pull: the material facts, the legal issue, the holding/ratio, and *the specific overlap with the client's facts*. That argues for: structure-aware chunking (Part 2.2) so holdings stay intact, and a generation step that fills a **structured schema** (Part 10.3) rather than free-texting. The "fact alignment" is essentially the model comparing client facts (truck, unlicensed driver, insurer denial, 42-year-old earner, dependents) against each precedent's facts — give it both, explicitly.

### 7.2 Adverse precedents — the differentiator
As established in 4.6, this is a *retrieval* problem before it's a generation problem. The system must actively search for cases supporting the *insurer's* position. Then for each, generate an honest **risk assessment** and a **distinguishing argument** (why the client's facts differ, e.g. "in that case the owner was complicit; here the owner had verified the license"). A system that only returns favorable cases is explicitly called "dangerous" — so build the adverse search as a first-class step and test it (Dimension 4).

### 7.3 Strategy — synthesis grounded in the above
Compensation range, argument prioritization, risks. This is the model reasoning *over its own retrieved evidence*. Keep it grounded: the strategy should reference the precedents found, not invent figures. (You're told legal accuracy isn't graded, but *faithfulness to the corpus* is.)

### 7.4 Distinguishing precedents — a concept worth knowing
In common-law reasoning, you "distinguish" an adverse precedent by showing the facts differ materially. Your adverse-analysis prompt should explicitly ask the model to attempt distinguishing each adverse case using the client's specific facts. This single instruction noticeably raises perceived reasoning quality (Dimension 3).

---

## Part 8 — Showing the reasoning (observability is a graded requirement)

"Intermediate reasoning steps must be visible." Two layers:

### 8.1 In the app UI (required)
Streamlit or Gradio (they say either is fine; they're not grading frontend). You must surface: the query the agent issued, *which documents/chunks were retrieved*, their *scores/ranking*, the agent's plan or thought steps, and the final grounded answer with citations. Streamlit's `st.status`/`st.expander` and streaming, or Gradio's `ChatInterface` with intermediate steps, are the usual ways. Render the plan-and-execute plan or the ReAct trace directly — this is *why* those patterns help: their internal state is legible.

### 8.2 Tracing (for you, and a bonus to mention)
For debugging and for the ADR, instrument the agent with tracing so you can see every LLM call, tool call, latency, and token cost:
- **LangSmith** (LangChain ecosystem), **Langfuse** (open source), **Arize Phoenix** (open source, OpenTelemetry-based). Any of these gives you a timeline of the agent's steps — invaluable when retrieval misbehaves, and great evidence of engineering maturity in the writeup. You already use Sentry; this is the LLM-shaped cousin of that instinct.

---

## Part 9 — Evaluation (treated as a top deliverable — invest here)

They want **one automated eval per dimension minimum**, and a written failure analysis. This is where many strong coders under-invest; doing it well is a real differentiator.

### 9.1 The foundational problem: you need a labeled set
Almost all of this rests on a **golden dataset**: a set of queries with known-correct answers / known-relevant documents. For a corpus this small you can bootstrap one:
- Read a subset of the 50 judgments (you don't need to be a lawyer — you're labeling topical relevance).
- For a handful of representative queries, label which documents are relevant (and ideally graded: highly/somewhat/not).
- Use an LLM to *propose* labels, then spot-check them by hand. This "LLM-assisted labeling with human verification" is standard and honest — describe it transparently in the ADR.

### 9.2 Dimension 1 — Precision
Of the precedents the agent returned, what fraction are actually relevant? Compute **Precision@k** against your labels. Also report **NDCG** if you have graded relevance, because it rewards correct *ordering* (a relevant doc ranked 1st is worth more than ranked 8th) and correlates better with end-to-end quality than binary precision. Reranking (4.3) is your main lever here.

### 9.3 Dimension 2 — Recall (the hard, honest one)
Of the precedents that *should* have been found, how many were? The trap: you can't measure recall without knowing the full set of relevant docs — the thing you don't fully know. Methodologies:
- **Exhaustive labeling (feasible at 50 docs):** since the corpus is tiny, you can plausibly label every doc's relevance to your test queries, giving true recall. Lean into the small-corpus advantage.
- **Pooling:** run *several* retrieval strategies, pool all their hits, label that pool, and treat it as the relevant universe (standard IR technique for when exhaustive labeling is infeasible — relevant for the 5,000-doc story).
- Report **Recall@k** and **MRR** (how high the first relevant doc ranks). They say they'll test recall against their own internal benchmark — so a system that over-retrieves to inflate recall will get caught by their precision check. Tune the precision/recall balance, don't game one.

### 9.4 Dimension 3 — Reasoning quality (qualitative → automatable)
"Does the reasoning hold up?" The standard tool is **LLM-as-judge**: a separate model scores each answer against a **rubric** (Is each claim grounded in a cited chunk? Does the fact-alignment actually match? Are distinguishing arguments coherent?). Know the pitfalls so you can defend it:
- **Bias:** judges favor longer answers, their own style, and the first option in pairwise comparisons (position bias) — mitigate with rubrics, randomized order, and pairwise + reference answers.
- **Faithfulness/groundedness** is the most important sub-metric here: does every claim trace to retrieved text? This is exactly what catches hallucination. RAGAS and DeepEval ship faithfulness metrics.
- Calibrate the judge against a few human-scored examples so you can state how much to trust it.

### 9.5 Dimension 4 — Adverse identification
Build a test where you *know* adverse precedents exist in the corpus (you labeled them), and measure whether the agent surfaces them. Metric: recall over the adverse subset, plus an LLM-judge check that the risk assessment is honest (not hand-waved). This directly tests the Part 7.2 architecture.

### 9.6 Frameworks
- **RAGAS** — pioneered the four-metric pattern (context precision, context recall, faithfulness, answer relevance); notebook/data-pipeline ergonomics. Natural fit for the retrieval dimensions.
- **DeepEval** — pytest-style; cleanest CI integration; broad metric library. Good if you want eval as test cases.
- **Promptfoo** — prompt-level A/B and red-teaming; good for comparing prompt/model variants.
- **Phoenix/Arize** — tracing + eval together.

You don't need all of them. A defensible move: RAGAS (or your own simple metrics) for retrieval precision/recall + faithfulness, plus a custom LLM-judge rubric for reasoning and adverse quality, all runnable from one script. *Including the eval code and results is required.*

### 9.7 The required failure analysis
After running evals, write "where it fails and what I'd fix first." Strong answers connect a *measured* failure to a *specific* fix: "Recall@5 on adverse cases is 0.4 because my single-query retrieval encodes the client's framing; first fix is an explicit counter-argument retrieval step + multi-query." That sentence demonstrates exactly the loop they're hiring for.

---

## Part 10 — LLM choice, prompting, and grounding

### 10.1 Model choice
Options: Claude, GPT, Gemini (APIs); open models (Llama, Qwen, Mistral) self-hosted or via providers. Trade-offs: capability vs. cost vs. latency vs. context length. For an agent doing multi-step reasoning over legal text, a strong frontier model for the *reasoning/synthesis* steps and a cheaper/faster model for *routing and grading* is a common cost-aware split — worth mentioning.

### 10.2 Grounding and citations
The core anti-hallucination technique: instruct the model to answer *only* from provided context, to say "not found in corpus" when it isn't there, and to **cite the chunk/doc id** for each claim. Citations also feed your faithfulness eval. This is non-negotiable in a legal product.

### 10.3 Structured output
The three-part deliverable (supporting/adverse/strategy, each with sub-fields) should come back as **structured JSON** via the model's structured-output / tool-calling / JSON-schema mode, not free text you regex. Benefits: reliable parsing, easier rendering of intermediate steps, and easier evaluation. Define a schema (e.g. `{supporting: [{case, principle, fact_alignment, citation}], adverse: [{case, risk, distinguishing_argument, citation}], strategy: {...}}`).

### 10.4 Prompt engineering that matters here
- Give the model the **client facts explicitly** as structured input so fact-alignment is grounded.
- Ask for **distinguishing arguments** explicitly (Part 7.4).
- Separate **retrieval-query generation** prompts from **synthesis** prompts — different jobs.
- Few-shot examples of a good supporting/adverse analysis raise quality and consistency.

---

## Part 11 — Deployment and the "hosted URL must work" constraint

You know deployment; here are only the AI-specific wrinkles:
- **Streamlit Community Cloud, Hugging Face Spaces, Render, Railway** all host a Streamlit/Gradio app easily. (You've used Railway already.)
- **The index has to exist at runtime.** Either build it on app startup (slow cold start, but self-contained) or **pre-build and commit/ship the index** (faster, more reliable) — for 50 docs the prebuilt index is small and the better choice.
- **Secrets:** embedding/LLM API keys go in the host's secret store, never in the repo.
- **The eval must not require local infra** — make sure your hosted URL and a clear README are the only things they need.
- **Cost/latency awareness:** cache embeddings; don't re-embed the corpus per request; rerank only the candidate set. Mention these even if the corpus is tiny — it shows scale instinct.

---

## Part 12 — Architecture options to evaluate (pick and defend one)

Here are four coherent designs along a complexity gradient. The point is for *you* to choose; each is defensible in a different context.

**Option A — Lean agentic RAG (router + hybrid retrieve + rerank).**
Router classifies the query; research queries trigger a single strong retrieval (hybrid + rerank + metadata) then structured synthesis. *Fits:* time-boxed build, prioritizing retrieval quality and clarity. *Risk:* may under-serve multi-faceted research tasks and adverse coverage unless you add an explicit adverse step. Cheapest to reason about.

**Option B — Plan-and-execute / ReAct with multi-strategy retrieval (recommended sweet spot).**
Router → planner decomposes research into sub-tasks (supporting, adverse, quantum) → each runs hybrid retrieval + rerank → synthesis into the structured deliverable. The plan is your visible reasoning trace. *Fits:* this assignment's exact shape; directly addresses adverse coverage and "dynamic workflow." *Risk:* more moving parts; needs step limits. This is what I'd point a strong submission toward.

**Option C — Corrective / Self-RAG (Option B + self-grading loops).**
Add relevance-grading of retrieved chunks and query-rewrite-and-retry when retrieval is weak. *Fits:* maximizing precision and faithfulness, willing to spend complexity/latency. *Risk:* over-engineering for 50 docs; great as a "with another week" item.

**Option D — Multi-agent (Researcher + Devil's-advocate Critic).**
Separate agents for finding support vs. attacking the case. *Fits:* conceptually clean for supporting/adverse split; better at larger scale. *Risk:* usually overkill here; harder to make legible and to debug. Best mentioned as a future direction.

**A decision matrix to fill in for the ADR:**

| Concern | A | B | C | D |
|---|---|---|---|---|
| Handles arbitrary prompts (not hard-coded) | ✓ (router) | ✓✓ | ✓✓ | ✓✓ |
| Adverse coverage | depends | ✓✓ | ✓✓ | ✓✓ |
| Precision/faithfulness | ✓ | ✓ | ✓✓ | ✓ |
| Explainable / visible steps | ✓✓ | ✓✓ | ✓ | ✓ |
| Build time | low | medium | high | high |
| Defensibility in interview | high | high | medium | medium |
| Right at 5,000 docs | partial | ✓ | ✓ | ✓ |

The honest meta-point: a *clean* Option B with excellent retrieval and a real eval harness beats a *sloppy* Option D. They said as much.

---

## Part 13 — A learning path (with small experiments)

Work bottom-up; build intuition with tiny experiments before committing to an architecture.

1. **Embeddings & similarity (½ day).** Embed 20 sentences with `sentence-transformers` (e.g. BGE-M3) and an API model; compute cosine similarity; see what clusters. Goal: *feel* semantic search.
2. **Parsing & chunking (½ day).** Parse 3 of the real judgments with PyMuPDF and with `unstructured`/Docling; try fixed vs. recursive vs. structure-aware chunking; eyeball chunk quality. Goal: see why chunking dominates.
3. **Minimal RAG (½ day).** 50 docs → chunks → Chroma/in-memory → top-k retrieval → one LLM call with grounding + citations. Ask the Lakshmi Devi question. Goal: a working baseline + first failures.
4. **Upgrade retrieval (1 day).** Add BM25 + RRF hybrid, then a reranker. Add metadata extraction + filtering. Re-ask. Goal: watch precision/recall improve and learn the levers.
5. **Make it an agent (1 day).** Wrap retrieval as tools; build a ReAct or plan-and-execute loop (raw SDK first to understand it, then optionally LangGraph). Add the explicit adverse-search step. Goal: dynamic workflow + adverse coverage.
6. **Evaluation (1 day).** Hand-label a small golden set; implement Precision@k, Recall@k, MRR; add an LLM-judge rubric for reasoning + adverse honesty; try RAGAS for faithfulness. Run it, read the failures. Goal: the graded eval deliverable + your failure analysis.
7. **UI + deploy + ADR (½–1 day).** Streamlit with visible intermediate steps; deploy; write the ADR around the decisions you actually made and the alternatives you rejected.

**Keywords to search as you go:** *RAG, dense vs sparse retrieval, BM25, reciprocal rank fusion, cross-encoder reranker, HNSW, ANN, parent-document retriever, auto-merging retrieval, contextual retrieval, HyDE, query decomposition, ReAct, plan-and-execute, router agent, corrective RAG, self-RAG, LLM-as-judge, faithfulness, context precision/recall, RAGAS, DeepEval, LangGraph, LlamaIndex query engine, structured outputs / tool calling.*

---

## Part 14 — Pitfalls and interview ammunition

**Likely failure modes to anticipate (and have fixes for):**
- Retrieval misses adverse cases (single-query-from-client's-framing) → explicit counter-argument retrieval + multi-query.
- Chunk boundaries split holdings → structure-aware chunking + parent-document retrieval.
- Model hallucinates legal reasoning → strict grounding, citations, faithfulness eval, "not in corpus" escape hatch.
- Agent loops or over-retrieves → max-steps, relevance-grading, stopping conditions.
- Precision inflated by dumping many chunks → reranking + tuned k, report precision *and* recall together.
- Eval that only checks the final answer → measure retrieval *and* generation separately (the three-layer view: retrieval / generation / end-to-end).

**Questions they will probably ask (from the brief itself), and what a good answer sounds like:**
- *"Why this architecture?"* → name the pattern, the alternative you rejected, and the property of the problem that decided it (cyclic/conditional control flow; adverse coverage; explainability).
- *"How does the agent decide simple vs. deep?"* → the router/planner makes an LLM-based decision; show the routing logic is reasoning, not if-else.
- *"What at 5,000 docs?"* → persistent HNSW store, batch indexing pipeline, pre-filtering, pooled recall eval, cost controls, maybe hierarchical routing.
- *"With another week?"* → fine-tuned/legal embedding model, corrective-RAG loop, larger labeled eval set, multi-agent critic, caching/latency work.
- *"Where does it fail?"* → cite a *measured* number and the first fix. This is the answer they care about most.

**The one-paragraph thesis to keep in your head:** *This is a retrieval-quality problem and an agent-flexibility problem joined by an evaluation problem. Win retrieval with hybrid search + reranking + good chunking + metadata. Win flexibility with a router/plan-execute agent whose steps are visible. Win adverse coverage by searching the opponent's position on purpose. Win the grade by measuring all of it honestly and saying clearly where it breaks.*

---

*Built from the assignment brief plus current (early–mid 2026) tooling research. Tool rankings and model leaders move fast — re-verify the specific embedding model / vector DB picks against the latest MTEB and your own corpus before committing.*
