# Lexi Agent — Complete Build Process & Engineering Journal

> A step-by-step record of how this prototype was planned, built, debugged, and
> validated — including the reasoning behind each decision, the parallelization
> strategy, the exact commands run, every bug hit, and the test results.
>
> Audience: anyone reviewing *how* the system was engineered (not just the final
> code). Pairs with [`../ADR.md`](../ADR.md) (the *why* of the architecture) and
> [`../README.md`](../README.md) (how to run it).

---

## 0. Starting state

When work began the repo was an empty `uv` scaffold:

```
lexi-agent/
├── main.py                # "Hello from lexi-agent!" template
├── pyproject.toml         # no dependencies
├── .env                   # GEMINI_API_KEY=...
├── docs/
│   ├── Lexi BE EG 2026 Research Assessment.pdf   # the brief
│   └── LEXI_AGENT_BUILD_GUIDE.md                  # a phase-by-phase build guide
└── data/raw/doc_001.pdf … doc_056.pdf            # the corpus (56 PDFs)
```

Two inputs drove everything: the **assessment PDF** (the requirements) and the
**build guide** (a recommended implementation path). The job: turn this into a
working, locally-tested legal precedent research agent.

---

## 1. Understand the task (before writing any code)

**Logic:** never start coding from a guide blindly — read the actual requirements
first, because the guide is one person's interpretation and the *grader* uses the
PDF.

### Commands / actions
- Listed the project tree and counted PDFs:
  ```bash
  find . -type f -not -path '*/.git/*' … | head -100
  ls data/raw/*.pdf | wc -l        # -> 56
  cat .python-version               # -> 3.11
  ```
- Read `main.py`, `pyproject.toml`, `.env` (confirmed key name is `GEMINI_API_KEY`).
- Read the full build guide (`LEXI_AGENT_BUILD_GUIDE.md`).
- Read all 6 pages of the assessment PDF with the `Read` tool's `pages` param.
- Sampled `data/raw/doc_001.pdf` (pages 1–2) to see what the corpus actually is.

### What I learned (the requirements that shaped the build)
From the **PDF**:
- Build a legal precedent research agent over ~50 Indian court judgments.
- Core capability = **deep precedent research** producing three sections:
  **Supporting precedents / Adverse precedents / Strategy**.
- **Critical: flexibility.** Must handle both general queries *and* deep research
  by **dynamically deciding its own workflow** — explicitly *"not through if-else
  branching."*
- Deliverables: hosted web app (intermediate reasoning must be visible), GitHub
  repo, **ADR**, and an **evaluation framework on 4 dimensions** (precision,
  recall, reasoning quality, adverse identification) + a failure analysis.
- **Not allowed:** CrewAI/drag-drop builders, hard-coded pipelines.

From the **corpus sample**: real Motor Accident Claims Tribunal (MACT) / High
Court judgments (e.g. *United India Insurance v. Neelam Devi*) — exactly the
client's domain (motor accident, insurer liability, compensation).

**Decision:** follow the build guide's architecture (it's sound and uses current
tooling) but treat the PDF as the source of truth for what must exist.

---

## 2. The plan

**Target architecture** (matches the guide, validated against the PDF):

```
PDFs ─▶ parse (PyMuPDF) ─▶ chunk (recursive) ─▶ embed ─▶ Chroma (dense)
                                              └────────▶ BM25 (keyword)
                                                              │
                          search_corpus / get_document  ◀── hybrid ensemble
                                      │
                          Gemini 3.5 Flash agent (ReAct / LangGraph)
                                      │
                          Streamlit UI (streams the reasoning trace)
                          + Eval framework (4 dimensions)
```

**Build order** (dependency-driven):
1. Config + ingestion (parse → chunk → build index)
2. Hybrid retrieval + retrieval smoke test
3. Agent core (tools, prompt, graph)
4. Streamlit UI
5. Eval framework + gold set
6. ADR + README + process doc

**Key design decisions made up front (the "logic"):**
| Decision | Reasoning |
|---|---|
| Two generic tools (`search_corpus`, `get_document`), no query-type branching | The agent composing them in different orders/counts *is* the dynamic workflow — this is the single most-graded requirement |
| Hybrid retrieval (BM25 + dense) | Legal queries need exact-term matching ("Section 149") *and* semantic matching ("policy void") |
| Open-source embeddings (`bge-small-en-v1.5`) | No API cost, runs locally, allowed by the rules |
| `app.py` is a thin layer over `src/agent/graph.py` | Lets us swap Streamlit→FastAPI later without touching the agent |
| Deterministic set-based metrics + LLM-judge layer | Reproducible backbone + qualitative depth for the 4 eval dimensions |

### How I decided to parallelize the work
Three things were independent and slow, so I ran them concurrently instead of
serially:
1. **Dependency install** (`uv add …`) — slow network/compile → **background**.
2. **Corpus analysis for the gold set** — required reading many PDFs → **delegated
   to a subagent** (see §3).
3. **Writing source files** — pure authoring, no waiting → done on the main thread
   while 1 and 2 ran.

This is the core orchestration logic: *put slow I/O and read-heavy fan-out work
in the background; keep authoring on the main thread.*

---

## 3. Agent orchestration (how I used subagents)

**Why a subagent for the gold set:** building an honest evaluation gold set means
reading ~56 judgments and classifying each (which support the client vs. which are
adverse). That's a large, read-heavy, parallelizable task whose *conclusion* (a
doc_id → label mapping) is small. Perfect for delegation — the subagent burns its
own context reading PDFs and returns just the labels.

**What I did NOT delegate:** the actual code. The implementation is interdependent
and I had the full build guide, so writing it directly on the main thread was more
reliable than fanning out (which risks inconsistency between files).

### The subagent I launched

Tool: `Agent` (subagent_type `general-purpose`, `run_in_background: true`).

Prompt (paraphrased): *"You are building a gold-standard eval set for a legal
precedent agent. The corpus is ~56 MACT/HC judgment PDFs at `data/raw/`. The
client case = husband killed by a commercial truck whose driver had no valid
licence; insurer denies the claim as void. Read the first 2–3 pages of each PDF
and classify by theme: (1) unlicensed-driver insurer-liability — split into
SUPPORTS_CLAIMANT vs ADVERSE, (2) commercial vehicle, (3) compensation
calculation for death, (4) contributory negligence. Return doc_id → one-line
summary plus theme groupings I can paste into a Python dict. Prioritize accuracy
on theme #1."*

**What it returned** (high-value findings that directly shaped the gold set):
- The corpus has **only ~40 motor-accident cases**; docs 036–038, 040, 044,
  046–056 are off-topic (health/IP/excise/criminal). → Don't pollute the gold set
  with these.
- **doc_024 duplicates doc_022.**
- docs 015/016/017 are **criminal** convictions for unlicensed driving (no insurer
  party) → exclude from the insurer-liability theme to protect precision.
- Theme #1 split:
  - SUPPORTS_CLAIMANT: 001, 003, 022, 027, 033, 034, 035, 041, 042
  - ADVERSE: 005, 025, 026, 028, 031, 032 (031 & 032 are Supreme Court — the most
    dangerous)
- Commercial-vehicle and death-compensation doc lists.

This mapping became `src/eval/gold_set.py` more or less verbatim (normalized to
`DOC_###`).

---

## 4. Implementation — step by step

All source files were written while the dependency install and the subagent ran
in the background.

### 4.1 Config (`src/config.py`)
**Logic:** one place for all knobs; never hard-code paths/hyperparameters.
- Used `pydantic-settings` `BaseSettings` reading from `.env`.
- **Deviation from the guide:** the guide assumed `GOOGLE_API_KEY`, but this
  project's `.env` uses `GEMINI_API_KEY`. So I accept **both** (`api_key`
  property) and `os.environ.setdefault("GOOGLE_API_KEY", …)` so the
  langchain-google-genai SDK finds it regardless.
- Defaults: `chunk_size=1200`, `chunk_overlap=200`, `top_k=8`, embeddings
  `BAAI/bge-small-en-v1.5`. Model left configurable via `LLM_MODEL`.

### 4.2 Ingestion (`src/ingest/`)
- `parse.py` — PyMuPDF: `"\n".join(page.get_text() …)` per PDF → `{doc_id,
  source, text}`. `@lru_cache` because both BM25 and `get_document` call it (no
  point parsing 56 PDFs twice). Normalized `doc_id` to uppercase `DOC_001`.
- `chunk.py` — `RecursiveCharacterTextSplitter` at paragraph→sentence→word
  boundaries, carrying `doc_id` in metadata (so any snippet maps back to its
  source for citation). `get_chunks()` cached.
- `build_index.py` — embeds chunks → Chroma, persisted to `data/chroma/`.
  **Idempotent**: skips if the collection already has chunks unless `force=True`.

### 4.3 Retrieval (`src/retrieval/retriever.py`)
- `EnsembleRetriever([bm25, dense], weights=[0.4, 0.6])` — dense weighted higher.
- `@lru_cache(maxsize=1)` because Streamlit re-runs the script on every
  interaction; without caching we'd rebuild the embedding model + BM25 index per
  click.

### 4.4 Agent core (`src/agent/`)
- `tools.py` — the two `@tool`s. `search_corpus` returns ranked snippets each
  prefixed `[DOC_###]`; `get_document` returns full text capped at 20k chars.
- `prompts.py` — the system prompt: "decide your own workflow," grounding rules
  (cite every claim with a doc_id, never rely on unretrieved judgments), and the
  mandatory three-section output for deep research.
- `schemas.py` — Pydantic `ResearchReport` documenting the target structure (hook
  for an enforced-structured-output upgrade).
- `graph.py` — `create_agent(model, tools, system_prompt)` = LangGraph ReAct loop.
  Singleton via `@lru_cache`. **The importable core** — UI and eval import only
  this.

### 4.5 UI (`app.py`)
- Thin Streamlit layer. Streams with `agent.stream(…, stream_mode="updates")` and
  renders each step: `AIMessage.tool_calls` → "🔧 tool(args)", `ToolMessage` →
  expandable retrieved snippets, final `AIMessage` → the analysis. Satisfies the
  "show intermediate reasoning, not just the final output" requirement.

### 4.6 Eval (`src/eval/`)
- `metrics.py` — `cited_doc_ids()` (regex `DOC_\d{3}` over the answer) +
  `precision_recall()` (set overlap, plus f1).
- `gold_set.py` — the hand-labelled ground truth from the subagent's analysis.
- `judge.py` — `GeminiJudge(DeepEvalBaseLLM)` so DeepEval's G-Eval uses Gemini,
  not OpenAI (one provider, one key).
- `run_eval.py` — for each query: invoke agent → extract cited doc_ids →
  precision/recall/adverse_recall (deterministic) + two G-Eval scores (reasoning
  quality, adverse honesty) → writes `eval_results.md`. DeepEval is optional
  (import guarded) so the deterministic metrics always run.

---

## 5. The exact command sequence

```bash
# --- dependency install (run in BACKGROUND while writing code) ---
uv add langchain langgraph langchain-google-genai langchain-core \
       langchain-text-splitters chromadb langchain-chroma \
       langchain-huggingface sentence-transformers langchain-community \
       rank-bm25 pymupdf streamlit pydantic-settings python-dotenv
uv add deepeval

# --- verify which Gemini model the key can actually use ---
curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=$KEY" \
  | python3 -c "...filter models with generateContent..."
# -> confirmed gemini-3.5-flash IS available (the guide was right)

# --- build the index (BACKGROUND; downloads the embedding model) ---
uv run python -m src.ingest.build_index
# -> "Indexed 3200 chunks."
uv run python -m src.ingest.build_index      # re-run -> idempotency check
# -> "Index already has 3200 chunks. Use force=True to rebuild."

# --- Phase 2 test: retrieval ---
uv run pytest tests/test_retrieval.py -s

# --- Phase 3 test: agent (simple vs deep) ---
uv run python -c "from src.agent.graph import agent; ..."

# --- Phase 4 test: Streamlit boots cleanly (headless + health poll) ---
uv run streamlit run app.py --server.headless true --server.port 8599 &
curl -s …/_stcore/health         # -> 200

# --- Phase 5: full evaluation ---
uv run python -m src.eval.run_eval           # -> eval_results.md
```

---

## 6. Bugs hit and how I fixed them

This is the real engineering — the guide's code did **not** run as-is against the
current library versions and the actual model.

### Bug 1 — `pytest` can't import `src`
```
ModuleNotFoundError: No module named 'src'
```
**Cause:** pytest doesn't add the project root to `sys.path`.
**Fix:** added to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
pythonpath = ["."]
```

### Bug 2 — `EnsembleRetriever` import fails
```
ImportError: cannot import name 'EnsembleRetriever' from 'langchain_community.retrievers'
```
**Cause:** in LangChain 1.x it moved to `langchain_classic.retrievers`.
**Diagnosis command:**
```bash
grep -rl "class EnsembleRetriever" .venv/lib/python3.11/site-packages/
# -> langchain_classic/retrievers/ensemble.py
```
**Fix:** import-fallback chain `langchain_classic → langchain → langchain_community`.

### Bug 3 (the important one) — Gemini 3.x returns `.content` as a *list*
The agent worked, but `msgs[-1].content` was
`[{'type': 'text', 'text': '...', 'extras': {'signature': ...}}]`, not a string.
This silently broke two things:
- the eval's `re.findall(DOC_\d{3}, answer)` (regex needs a string),
- the Streamlit `st.markdown(final)` rendering,
- and it **crashed DeepEval**:
  ```
  AttributeError: 'list' object has no attribute 'find'
  ```
  (DeepEval does `result.find("{")` to parse the judge's JSON.)

**Fix:** added `src/utils.py::message_text()` that flattens content blocks to
plain text, and wired it through `app.py`, `run_eval.py`, and the `GeminiJudge`
(`generate`/`a_generate`). This is the kind of bug the build guide couldn't
anticipate because it depends on the *actual* model's response shape.

### Bug 4 — one judge failure killed the whole eval run
**Fix:** wrapped each `metric.measure()` in try/except so a judge hiccup degrades
that one dimension to `None` instead of aborting; the deterministic metrics always
survive.

### Faithfulness correction (not a crash, a quality fix)
The first deep-research run appended **external reporter citations** ("Swaran
Singh (2004) 3 SCC 297") and **mis-attributed case names to doc_ids** (claimed
DOC_001 = "Pappu v. Vinod Kumar Lamba" when DOC_001 is actually *Neelam Devi*).
**Fix:** tightened the system prompt's grounding rule — refer to judgments by
doc_id, describe holdings in your own words from the retrieved text, and don't add
external case names/citations unless they appear verbatim in retrieved text. After
the fix, the landmark cases it cites (*Pranay Sethi*, *Sarla Verma*) are ones that
genuinely appear *inside* the retrieved judgments — legitimate grounding.

---

## 7. Testing & validation results

### Phase 1 — Ingestion ✅
- `Indexed 3200 chunks.` from 56 PDFs.
- Re-run is idempotent (`already has 3200 chunks`).

### Phase 2 — Retrieval ✅
`uv run pytest tests/test_retrieval.py` → **1 passed**. For the query
*"driver without valid licence insurance liability"* the top hits were
**DOC_027, DOC_003, DOC_031** — all in the unlicensed-driver gold set. Strong
signal that retrieval quality is good (this caps everything downstream).

### Phase 3 — Agent / dynamic workflow ✅ (the headline result)
| Query type | Tool calls | Behaviour |
|---|---|---|
| "Which judgments involve commercial vehicles?" | **2** | concise list, no over-research |
| Full case brief (unlicensed driver) | **22** | multi-angle `search_corpus` + several `get_document` reads, then synthesis |

Same code, no `if query_type` — the contrast *is* the proof of dynamic workflow.
The deep answer had all three sections and cited 11 doc_ids.

### Phase 4 — Streamlit ✅
Headless boot returned health `200` with no import/runtime errors; the reasoning
trace logic renders tool calls + retrieved snippets before the final answer.

### Phase 5 — Evaluation ✅ (all 4 dimensions produced)
| Query | precision | recall | f1 | adverse_recall | reasoning | adverse_honesty |
|---|---|---|---|---|---|---|
| Case brief (deep) | 0.64 | 0.47 | 0.54 | 0.50 | 0.90 | 1.00 |
| Compensation (deep) | 0.75 | 0.50 | 0.60 | n/a | 0.80 | 1.00 |
| Commercial vehicles (simple) | 0.86 | 0.55 | 0.67 | n/a | 1.00 | 0.00 |

Interpretation (full version in the ADR):
- **Recall is the weak axis** — relevant docs buried past `top_k=8`. First fix: a
  reranker + higher `top_k` (improves precision *and* recall from one change).
- **`adverse_honesty=0.0` on the simple query is correct, not a bug** — that query
  has no adverse dimension to surface.

### Final regression check ✅
```bash
uv run python -m src.ingest.build_index   # idempotent
uv run python -c "import <every module>; import app"   # ALL IMPORTS OK
uv run pytest tests/test_retrieval.py     # 1 passed
```

---

## 8. Final repository layout

```
lexi-agent/
├── ADR.md                      # architecture decisions + eval results + failure analysis
├── README.md                   # setup + run instructions
├── pyproject.toml / uv.lock    # pinned deps + pytest config
├── .env / .env.example         # GEMINI_API_KEY (gitignored)
├── app.py                      # Streamlit UI (thin)
├── data/
│   ├── raw/doc_001..056.pdf    # corpus
│   └── chroma/                 # persisted index (gitignored)
├── src/
│   ├── config.py  utils.py
│   ├── ingest/{parse,chunk,build_index}.py
│   ├── retrieval/retriever.py
│   ├── agent/{tools,prompts,schemas,graph}.py
│   └── eval/{gold_set,metrics,judge,run_eval}.py
├── tests/test_retrieval.py
└── docs/
    ├── Lexi BE EG 2026 Research Assessment.pdf
    ├── LEXI_AGENT_BUILD_GUIDE.md
    └── BUILD_PROCESS.md         # this file
```

---

## 9. What's intentionally left for the submission step
- **Deploy** to Streamlit Community Cloud; put the live URL in `README.md`
  (placeholder present) and set `GEMINI_API_KEY` as a host secret.
- **Git commit + push** to a private repo and send the submission email
  (`[LEXI-BE-2026] <Name> - Research Assessment Submission`). Not done
  automatically — these are outward-facing actions that need explicit sign-off.

---

## 10. Summary of the engineering logic in one paragraph
Read the requirements before the guide; parallelize the slow/independent work
(deps install in background, corpus labelling delegated to a subagent, code
authored on the main thread); build dependency-first (ingest → retrieve → agent →
UI → eval); test each phase before moving on (idempotent index, on-topic
retrieval, simple-vs-deep tool-call contrast, clean UI boot, four eval
dimensions); and fix the integration bugs the guide couldn't predict — most
importantly the Gemini-3.x list-shaped `.content`, which required a single
normalization helper threaded through the UI, eval, and judge.
