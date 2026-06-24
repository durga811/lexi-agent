# Lexi Precedent Research Agent — Build & Test Guide

A step-by-step, phase-by-phase guide to build a **working, locally-testable** legal precedent research agent for the Lexi take-home — using **Gemini 3.5 Flash + LangGraph 1.x**, with current tool versions and best practices.

**Scope of this guide:** get the core feature + evaluation working *locally* with the best tools and clean architecture. No deployment yet. The structure is deliberately FastAPI-ready so you can rebuild later without touching the agent core.

---

## 0. Tool versions (verified June 2026)

| Layer | Package | Version | Notes |
|---|---|---|---|
| Runtime | Python | 3.12 | 3.10–3.13 all fine; 3.12 is the safe default |
| LLM | `langchain-google-genai` | 4.2.5 | uses the consolidated `google-genai` SDK |
| Model | — | `gemini-3.5-flash` | GA, agentic/coding-optimized, 1M context |
| Agent | `langchain` | 1.3.x | `langchain.agents.create_agent` (built on LangGraph) |
| Orchestration | `langgraph` | 1.1.x | pulled in transitively; `langgraph.prebuilt` is **deprecated** |
| Vector store | `chromadb` + `langchain-chroma` | 1.5.x / latest | embedded, persists to disk, zero infra |
| Embeddings | `sentence-transformers` + `langchain-huggingface` | latest | open-source `BAAI/bge-small-en-v1.5` |
| Keyword search | `rank-bm25` | 0.2.2 | for hybrid retrieval |
| Hybrid + community retrievers | `langchain-community` | 4.0.x | `BM25Retriever`, `EnsembleRetriever` |
| PDF parsing | `pymupdf` | latest | fast, reliable text extraction |
| UI | `streamlit` | latest | shows intermediate reasoning steps |
| Eval (metrics) | `ragas` | 0.4.3 | v0.4 returns metric *objects*, not floats |
| Eval (test runner / G-Eval) | `deepeval` | 3.9.9 | pytest-style, LLM-as-judge with custom Gemini judge |
| Config | `pydantic-settings`, `python-dotenv` | latest | env + settings |

> **Gemini 3.x gotcha:** don't override `temperature` / `top_p` / `top_k`. The 3.x models are tuned for their defaults — changing them hurts reasoning. Leave them unset.

---

## 1. Environment setup

You already created the `uv` project `lexi-agent`. From inside it:

```bash
# Core agent + LLM
uv add langchain langgraph langchain-google-genai langchain-core

# Retrieval
uv add chromadb langchain-chroma langchain-huggingface sentence-transformers
uv add langchain-community rank-bm25

# Parsing + UI + config
uv add pymupdf streamlit pydantic-settings python-dotenv

# Evaluation
uv add ragas deepeval

# Dev
uv add --dev pytest
```

Create a `.env` (and `.env.example` with the same keys, blank values):

```bash
# .env
GOOGLE_API_KEY=your_google_ai_studio_key_here
```

> `langchain-google-genai` checks `GOOGLE_API_KEY` first, then `GEMINI_API_KEY`. Get a key from Google AI Studio.

Add to `.gitignore`:

```
.env
data/chroma/
__pycache__/
.venv/
*.pyc
```

Drop the corpus into `data/raw/` (`DOC_001.pdf` … `DOC_050.pdf`).

---

## 2. Project structure

```
lexi-agent/
├── pyproject.toml
├── .env / .env.example / .gitignore
├── README.md
├── ADR.md                       # architecture decision record (key deliverable)
│
├── data/
│   ├── raw/                     # DOC_001.pdf ... DOC_050.pdf
│   └── chroma/                  # persisted index (gitignored)
│
├── src/
│   ├── __init__.py
│   ├── config.py                # settings, paths, model + chunk params
│   ├── ingest/
│   │   ├── parse.py             # PyMuPDF -> text + metadata
│   │   ├── chunk.py             # chunking with doc_id metadata
│   │   └── build_index.py       # embed + persist to Chroma (run ONCE)
│   ├── retrieval/
│   │   └── retriever.py         # hybrid: BM25 + dense (EnsembleRetriever)
│   ├── agent/
│   │   ├── tools.py             # search_corpus(), get_document()
│   │   ├── prompts.py           # system prompt (the brain)
│   │   ├── schemas.py           # Pydantic models for structured research output
│   │   └── graph.py             # the agent (create_agent) — IMPORTABLE CORE
│   └── eval/
│       ├── gold_set.py          # {query: relevant doc_ids} hand-labeled
│       ├── judge.py             # Gemini-as-judge wrapper for DeepEval
│       ├── metrics.py           # set-based precision/recall + adverse recall
│       └── run_eval.py          # runs all 4 dimensions -> results.md
│
├── app.py                       # Streamlit UI — THIN layer over src.agent.graph
└── tests/
    └── test_retrieval.py
```

**The one rule that makes the rebuild painless:** `app.py` only imports `src.agent.graph` and renders. All intelligence lives in `src/`. When you swap Streamlit for FastAPI, `app.py` is the only file you replace.

---

## 3. Config (`src/config.py`)

```python
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str

    # paths
    raw_dir: Path = ROOT / "data" / "raw"
    chroma_dir: Path = ROOT / "data" / "chroma"

    # models
    llm_model: str = "gemini-3.5-flash"
    embed_model: str = "BAAI/bge-small-en-v1.5"   # open-source, fast; bge-base for higher quality

    # chunking
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # retrieval
    top_k: int = 8

settings = Settings()
```

---

## 4. PHASE 1 — Ingestion

**Goal:** turn 50 PDFs into a searchable index. No agent yet.

### 4.1 Parse (`src/ingest/parse.py`)

```python
import pymupdf
from src.config import settings

def load_documents() -> list[dict]:
    """Return [{doc_id, source, text}] for every PDF in raw_dir."""
    docs = []
    for pdf in sorted(settings.raw_dir.glob("*.pdf")):
        with pymupdf.open(pdf) as f:
            text = "\n".join(page.get_text() for page in f)
        docs.append({"doc_id": pdf.stem, "source": pdf.name, "text": text.strip()})
    return docs
```

### 4.2 Chunk (`src/ingest/chunk.py`)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config import settings

def chunk_documents(docs: list[dict]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Document] = []
    for d in docs:
        for i, piece in enumerate(splitter.split_text(d["text"])):
            chunks.append(Document(
                page_content=piece,
                metadata={"doc_id": d["doc_id"], "source": d["source"], "chunk": i},
            ))
    return chunks
```

> **Why this chunking:** judgments are long and unstructured PDFs. Recursive splitting at paragraph boundaries keeps reasoning intact, and carrying `doc_id` in metadata lets the agent map any chunk back to its source judgment. **ADR upgrade note:** a structure-aware splitter (Facts / Held / Ratio) or a per-document summary index improves doc-level queries like *"which judgments involve commercial vehicles?"* — mention this as a "what I'd do with another week" item.

### 4.3 Build index (`src/ingest/build_index.py`)

```python
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import settings
from src.ingest.parse import load_documents
from src.ingest.chunk import chunk_documents

def build_index(force: bool = False):
    embeddings = HuggingFaceEmbeddings(model_name=settings.embed_model)
    store = Chroma(
        collection_name="judgments",
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )
    # idempotent: skip if already populated
    if not force and store._collection.count() > 0:
        print(f"Index already has {store._collection.count()} chunks. Use force=True to rebuild.")
        return store

    docs = load_documents()
    chunks = chunk_documents(docs)
    store.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks from {len(docs)} documents.")
    return store

if __name__ == "__main__":
    build_index()
```

### ✅ Test Phase 1

```bash
uv run python -m src.ingest.build_index
```

Expect: `Indexed N chunks from 50 documents.` Re-run → it should say "already has N chunks" (idempotency works).

---

## 5. PHASE 2 — Hybrid retrieval

**Goal:** a single `retriever` object that blends keyword (BM25) + semantic (dense) search. Keyword matters here because legal queries hinge on exact terms (statute names, "Section 149", "MV Act").

### `src/retrieval/retriever.py`

```python
from functools import lru_cache
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever  # fallback: langchain_community.retrievers
from src.config import settings
from src.ingest.parse import load_documents
from src.ingest.chunk import chunk_documents

@lru_cache(maxsize=1)
def get_retriever():
    embeddings = HuggingFaceEmbeddings(model_name=settings.embed_model)
    store = Chroma(
        collection_name="judgments",
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )
    dense = store.as_retriever(search_kwargs={"k": settings.top_k})

    # BM25 needs the raw chunks in memory
    chunks = chunk_documents(load_documents())
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = settings.top_k

    return EnsembleRetriever(retrievers=[bm25, dense], weights=[0.4, 0.6])
```

> `lru_cache` keeps the embedding model + BM25 index loaded once per process (important — Streamlit re-runs the script on every interaction).

### ✅ Test Phase 2 (`tests/test_retrieval.py`)

```python
from src.retrieval.retriever import get_retriever

def test_retrieval_returns_results():
    r = get_retriever()
    hits = r.invoke("driver without valid licence insurance liability")
    assert len(hits) > 0
    for h in hits[:3]:
        print(h.metadata["doc_id"], "→", h.page_content[:120])
```

```bash
uv run pytest tests/test_retrieval.py -s
```

**Eyeball the output.** If the top hits aren't on-topic, fix retrieval *now* — a weak retriever caps everything downstream. Try bumping `top_k`, switching `embed_model` to `bge-base-en-v1.5`, or adjusting ensemble weights.

---

## 6. PHASE 3 — The agent (the core)

**Goal:** an agent that **decides its own workflow** — one tool call for a lookup, many for deep research — with no `if query_type == ...` branching. This is the single most-graded requirement.

### 6.1 Tools (`src/agent/tools.py`)

```python
from langchain_core.tools import tool
from src.retrieval.retriever import get_retriever
from src.ingest.parse import load_documents

@tool
def search_corpus(query: str) -> str:
    """Search the corpus of court judgments for passages relevant to `query`.
    Returns ranked snippets with their source doc_id. Call multiple times with
    different queries to research a topic from several angles."""
    hits = get_retriever().invoke(query)
    if not hits:
        return "No relevant passages found."
    return "\n\n".join(
        f"[{h.metadata['doc_id']}] {h.page_content}" for h in hits[:8]
    )

@tool
def get_document(doc_id: str) -> str:
    """Retrieve the full text of a single judgment by its doc_id (e.g. 'DOC_017').
    Use after search_corpus to read a promising judgment in full before relying on it."""
    for d in load_documents():
        if d["doc_id"].lower() == doc_id.lower():
            return d["text"][:20000]  # cap to stay within context
    return f"No document found with id {doc_id}."
```

> Two tools is enough. `search_corpus` for breadth, `get_document` for depth. The agent composing these in different orders/counts **is** the dynamic workflow — that's the architecture answer, not branching code.

### 6.2 Schemas (`src/agent/schemas.py`) — optional structured pass

```python
from pydantic import BaseModel, Field

class Precedent(BaseModel):
    doc_id: str
    principle: str = Field(description="legal principle the judgment establishes")
    relevance: str = Field(description="how its facts align with the client's case")

class ResearchReport(BaseModel):
    supporting: list[Precedent]
    adverse: list[Precedent] = Field(description="precedents the opposing side could use")
    strategy: str = Field(description="prioritized arguments, realistic compensation range, risks")
```

### 6.3 System prompt (`src/agent/prompts.py`)

```python
SYSTEM_PROMPT = """You are a legal precedent research agent working over a corpus of
Indian court judgments. You have two tools: search_corpus and get_document.

Decide your own workflow based on the request:
- Simple/general questions (e.g. "which judgments involve commercial vehicles?"):
  do the minimum — one or two searches, then answer concisely.
- Deep precedent research (e.g. "find precedents supporting our argument on X"):
  run several searches from different angles, read the most relevant judgments in
  full with get_document, then synthesize.

NEVER rely on a judgment you have not actually retrieved. Cite every claim with its
doc_id. Do not invent case names, sections, or holdings.

For deep precedent-research tasks, structure your final answer in three sections:
1. SUPPORTING PRECEDENTS — judgments that help the client, with the specific facts
   that align and the legal principle each establishes (cite doc_id).
2. ADVERSE PRECEDENTS — judgments the opposing side could use, an honest assessment
   of the risk each poses, and how it might be distinguished or countered.
3. STRATEGY — which arguments to prioritize, a realistic compensation range, and the
   key risks the client should know.

Surfacing unfavorable precedents honestly is mandatory — a system that only finds
favorable cases is dangerous in legal practice."""
```

### 6.4 The graph (`src/agent/graph.py`)

```python
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import settings
from src.agent.tools import search_corpus, get_document
from src.agent.prompts import SYSTEM_PROMPT

def build_agent():
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        # leave temperature unset for Gemini 3.x
    )
    return create_agent(
        model=llm,
        tools=[search_corpus, get_document],
        system_prompt=SYSTEM_PROMPT,
    )

agent = build_agent()
```

> If your installed version doesn't expose `create_agent` yet, the drop-in fallback is `from langgraph.prebuilt import create_react_agent` with the same `model` / `tools` / `prompt` args — it's the same ReAct loop under the hood.

### ✅ Test Phase 3 (quick CLI smoke test)

```python
# scratch.py
from src.agent.graph import agent

# simple query — should make ~1 tool call
out = agent.invoke({"messages": [("user", "Which judgments involve commercial vehicles?")]})
print(out["messages"][-1].content)

# deep query — should make several
out = agent.invoke({"messages": [("user",
    "Client's truck driver had no valid licence; insurer denies the motor accident "
    "claim as void. Find supporting and adverse precedents and recommend a strategy.")]})
print(out["messages"][-1].content)
```

```bash
uv run python scratch.py
```

Confirm: the simple query answers fast with few/one tool call; the deep query searches multiple times, reads docs, and returns the three-section structure with `doc_id` citations. **This contrast is your proof of dynamic workflow** — screenshot it for the ADR.

---

## 7. PHASE 4 — Streamlit UI with visible reasoning

**Goal:** input any prompt, and **show the intermediate steps** — which docs were retrieved and how the agent reasoned. The assessment explicitly says: *do not show only the final output.*

### `app.py`

```python
import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage
from src.agent.graph import agent

st.set_page_config(page_title="Lexi Precedent Agent", layout="wide")
st.title("⚖️ Legal Precedent Research Agent")

query = st.text_area("Enter any prompt or case brief:", height=140)

if st.button("Run") and query.strip():
    final = None
    st.subheader("Reasoning trace")
    # stream_mode="updates" yields one dict per node execution
    for update in agent.stream({"messages": [("user", query)]}, stream_mode="updates"):
        for node, payload in update.items():
            for msg in payload.get("messages", []):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        with st.status(f"🔧 {tc['name']}({tc['args']})", state="complete"):
                            st.write("Agent decided to call this tool.")
                elif isinstance(msg, ToolMessage):
                    with st.expander(f"📄 Tool result — {msg.name}"):
                        st.text(msg.content[:3000])
                elif isinstance(msg, AIMessage) and msg.content:
                    final = msg.content

    st.subheader("Final analysis")
    st.markdown(final or "_No answer produced._")
```

### ✅ Test Phase 4

```bash
uv run streamlit run app.py
```

Paste the case brief → watch tool calls and retrieved snippets stream in, then the structured analysis. Paste a one-liner → confirm it short-circuits. **This is a graded deliverable**, so make the trace genuinely readable.

---

## 8. PHASE 5 — Evaluation framework (the differentiator)

**Goal:** one automated eval per dimension. The grading rewards your *judgment* on how to measure agent quality. Strategy: a **transparent, deterministic backbone** (set-based metrics on doc IDs) + a **best-tool LLM-judge layer** (DeepEval G-Eval with a Gemini judge).

| Dimension | Method | Tool |
|---|---|---|
| 1. Precision | of cited precedents, % that are in the gold relevant set | custom set-based (`metrics.py`) |
| 2. Recall | of gold relevant precedents, % the agent cited | custom set-based (`metrics.py`) |
| 3. Reasoning quality | rubric: faithful to source + sound legal logic | DeepEval **G-Eval** (Gemini judge) |
| 4. Adverse identification | recall on the gold *adverse* set + honesty rubric | custom + DeepEval G-Eval |

### 8.1 Gold set (`src/eval/gold_set.py`)

Hand-label by skimming a handful of judgments. Keep it small but honest — this is your ground truth.

```python
# Map each eval query to the doc_ids that SHOULD appear.
GOLD = {
    "unlicensed driver insurer denies motor accident claim": {
        "supporting": ["DOC_003", "DOC_011", "DOC_027"],   # <-- replace with real labels
        "adverse":    ["DOC_019", "DOC_042"],
    },
    "compensation calculation for death of 42-year-old earning member": {
        "supporting": ["DOC_007", "DOC_015"],
        "adverse":    [],
    },
}
```

> **How to label without legal expertise:** run `search_corpus` for each query, read the top ~10 judgments' holdings, and mark which genuinely support vs. cut against the client. Document this process in the ADR — defining your own recall methodology is explicitly asked for.

### 8.2 Set-based metrics (`src/eval/metrics.py`)

```python
import re

def cited_doc_ids(answer: str) -> set[str]:
    return set(re.findall(r"DOC_\d{3}", answer.upper()))

def precision_recall(predicted: set[str], gold: set[str]) -> dict:
    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "tp": 0}
    tp = len(predicted & gold)
    precision = tp / len(predicted)
    recall = tp / len(gold) if gold else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "tp": tp}
```

### 8.3 Gemini judge for DeepEval (`src/eval/judge.py`)

DeepEval defaults to an OpenAI judge; wire it to Gemini so the whole stack is one provider.

```python
from deepeval.models import DeepEvalBaseLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import settings

class GeminiJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            model=settings.llm_model, google_api_key=settings.google_api_key
        )
    def load_model(self):
        return self.model
    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content
    async def a_generate(self, prompt: str) -> str:
        return (await self.model.ainvoke(prompt)).content
    def get_model_name(self):
        return settings.llm_model
```

### 8.4 Runner (`src/eval/run_eval.py`)

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from src.agent.graph import agent
from src.eval.gold_set import GOLD
from src.eval.metrics import cited_doc_ids, precision_recall
from src.eval.judge import GeminiJudge

judge = GeminiJudge()

reasoning_metric = GEval(
    name="Reasoning Quality",
    criteria="Does the analysis cite only retrieved judgments, correctly map their "
             "facts to the client's situation, and reach legally coherent conclusions?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=judge,
)
adverse_metric = GEval(
    name="Adverse Honesty",
    criteria="Does the answer surface precedents AGAINST the client and honestly "
             "assess their risk, rather than only presenting favorable cases?",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=judge,
)

def run():
    rows = []
    for query, gold in GOLD.items():
        answer = agent.invoke({"messages": [("user", query)]})["messages"][-1].content
        predicted = cited_doc_ids(answer)

        gold_support = set(gold["supporting"])
        gold_adverse = set(gold["adverse"])

        pr = precision_recall(predicted, gold_support | gold_adverse)
        adverse_recall = (
            len(predicted & gold_adverse) / len(gold_adverse) if gold_adverse else None
        )

        tc = LLMTestCase(input=query, actual_output=answer)
        reasoning_metric.measure(tc)
        adverse_metric.measure(tc)

        rows.append({
            "query": query,
            **pr,
            "adverse_recall": adverse_recall,
            "reasoning": round(reasoning_metric.score, 3),
            "adverse_honesty": round(adverse_metric.score, 3),
        })

    # write a markdown report for the ADR
    with open("eval_results.md", "w") as f:
        f.write("# Evaluation Results\n\n")
        for r in rows:
            f.write(f"## {r['query']}\n")
            for k, v in r.items():
                if k != "query":
                    f.write(f"- **{k}**: {v}\n")
            f.write("\n")
    print("Wrote eval_results.md")

if __name__ == "__main__":
    run()
```

### ✅ Test Phase 5

```bash
uv run python -m src.eval.run_eval
```

Produces `eval_results.md` with precision/recall, adverse recall, and two LLM-judged scores per query. **Then write the required failure analysis** — where does it miss? Common culprits: over-retrieval (low precision), missing a relevant doc buried past `top_k` (low recall), or skipping adverse precedents (low adverse recall). Name the top fix.

> **Best-tool upgrade (optional):** add RAGAS 0.4.3 `ContextPrecision` / `ContextRecall` on the retrieved chunks to measure *retrieval* quality separately from the agent's final citations. Note the v0.4 change: metrics return objects (read `.value`), and you pass a judge via `llm_factory()`.

---

## 9. Acceptance checklist (before you call it "working")

- [ ] `build_index` runs once, is idempotent on re-run
- [ ] `pytest` retrieval test returns on-topic hits for the case brief
- [ ] Simple query → few tool calls; deep query → multiple searches + `get_document`
- [ ] Deep-research output always has the three sections, every claim cites a `doc_id`
- [ ] Agent surfaces at least one adverse precedent on the case brief
- [ ] Streamlit shows tool calls + retrieved snippets, not just the final answer
- [ ] `run_eval` produces numbers for all four dimensions
- [ ] No hallucinated doc_ids (spot-check 2–3 cited docs against `get_document`)
- [ ] You can explain every architectural choice out loud (the interview tests this)

---

## 10. What changes later (don't build now, just note in ADR)

- **FastAPI rebuild:** replace `app.py` with FastAPI endpoints that call `agent.astream(...)`; stream steps over SSE/WebSocket. `src/` is untouched.
- **Deployment:** containerize, push to Railway/Render; persist Chroma to a volume or move to Chroma Cloud / Qdrant.
- **5,000 docs instead of 50:** drop BM25-in-memory (rebuilds every boot) for a server-backed hybrid store (Qdrant/Weaviate native hybrid); add a reranker (cross-encoder) and a summary/parent-doc index; precompute embeddings in a batch job. *This is a literal ADR question — answer it.*
- **With another week:** structure-aware chunking (Facts/Held/Ratio), enforced Pydantic structured output, RAGAS retrieval metrics, a larger gold set built from traces, and a reranking layer.

---

### Suggested build order (≈6–8h)
1. Phase 1 + 2 (ingest + retrieval) and *verify retrieval quality* — ~2h
2. Phase 3 (agent) until the three-section output looks right — ~2h
3. Phase 4 (Streamlit trace) — ~1h
4. Phase 5 (eval + gold labeling + failure analysis) — ~2h
5. Write the **ADR** last, while everything is fresh — ~1h

The ADR and eval honesty carry disproportionate weight. Budget for them.
