"""The agent's two tools.

Two primitives only:
  - search_corpus : breadth — find relevant passages across all judgments
  - get_document  : depth   — read one judgment in full before relying on it

The agent *composing* these in different orders and counts IS the dynamic
workflow — a simple lookup is one search, deep research is several searches plus
get_document reads. No if/else branching on query type.
"""
from __future__ import annotations

from langchain_core.tools import tool

from src.ingest.parse import load_documents
from src.retrieval.retriever import get_retriever


# Judgment metadata (title/date/url) is attached to each hit and surfaced in the
# tool output for citation — deliberately NOT prepended into the embedded chunk
# text, where a constant title prefix would add little dense signal (titles are
# homogeneous) and pollute BM25 with the same party/year tokens on every chunk.


def _hit_header(meta: dict) -> str:
    """e.g. '[#1 · relevance 8.42 · DOC_031 · National Insurance ... 2007]'.

    The rank + reranker score (added by rerank()) make "how it ranked" visible in
    the tool output, and so in the Streamlit reasoning trace.
    """
    doc_id = meta["doc_id"]
    title = (meta.get("title") or "").strip()
    rank, score = meta.get("rank"), meta.get("rerank_score")
    prefix = f"#{rank} · " if rank else ""
    prefix += f"relevance {score:.2f} · " if score is not None else ""
    body = f"{prefix}{doc_id}"
    return f"[{body} · {title}]" if title else f"[{body}]"


# --- retrieval recorder (for faithfulness eval) ---------------------------
# The faithfulness metric needs the chunks the agent actually saw across its many
# tool calls, so we accumulate them here; the eval resets before a run and reads
# the union after. A ContextVar (not a module global) so the multi-sample harness
# can run agent invocations concurrently without their logs interleaving — each
# worker gets its own copied context. Default None = no eval active → zero
# overhead in production.
import contextvars

_retrieval_log: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "retrieval_log", default=None
)


def reset_retrieval_log() -> None:
    _retrieval_log.set([])


def get_retrieval_log() -> list[str]:
    """Deduped chunk texts retrieved since the last reset (preserves order)."""
    return list(dict.fromkeys(_retrieval_log.get() or []))


@tool
def search_corpus(query: str) -> str:
    """Search the corpus of Indian court judgments for passages relevant to `query`.

    Returns ranked snippets, each prefixed with its source doc_id and case name,
    e.g. [DOC_017 · United India Insurance ... vs Neelam Devi on 6 November, 2023].
    Call this multiple times with different phrasings to research a topic from
    several angles (e.g. once for the legal principle, once for the fact pattern).
    """
    hits = get_retriever().invoke(query)
    if not hits:
        return "No relevant passages found."
    used = hits[:8]
    log = _retrieval_log.get()
    if log is not None:  # record what the agent saw (only during eval)
        log.extend(h.page_content for h in used)
    return "\n\n".join(
        f"{_hit_header(h.metadata)} {h.page_content}" for h in used
    )


@tool
def get_document(doc_id: str) -> str:
    """Retrieve the full text of a single judgment by its doc_id (e.g. 'DOC_017').

    Use after search_corpus to read a promising judgment in full before relying
    on it for a citation. Verifies the judgment actually says what a snippet hinted.
    """
    target = doc_id.strip().upper()
    for d in load_documents():
        if d["doc_id"] == target:
            # Lead with the authoritative case name + verifiable source link so the
            # agent can cite accurately without re-deriving them from the body.
            header = (
                f"{d['doc_id']} — {d['title']}\n"
                f"Source: {d['kanoon_url']}\n"
                f"{'-' * 60}\n"
            )
            body = d["text"][:20000]  # cap to stay within context budget
            # Record the read for the faithfulness judge (only during eval) — the
            # agent grounds claims in these full reads, not just search snippets.
            # The logged copy is capped smaller than the body the agent gets, to
            # keep eval memory bounded under concurrency.
            log = _retrieval_log.get()
            if log is not None:
                log.append(f"{_hit_header(d)} {body[:12000]}")
            return header + body
    return f"No document found with id {doc_id}."
