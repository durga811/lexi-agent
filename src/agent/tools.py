"""The agent's two tools.

Deliberately just two primitives:
  - search_corpus : breadth (find relevant passages across all judgments)
  - get_document  : depth  (read one judgment in full before relying on it)

The agent *composing* these in different orders and counts IS the dynamic
workflow. A simple lookup might be one search_corpus call; deep precedent
research is several searches from different angles plus get_document reads.
There is no if/else branching on query type — that's the architecture answer.
"""
from __future__ import annotations

from langchain_core.tools import tool

from src.ingest.parse import load_documents
from src.retrieval.retriever import get_retriever


# NOTE: the judgment metadata (title/date/kanoon_url) is surfaced here in the
# TOOL OUTPUT only — it is deliberately NOT prepended into the embedded chunk
# text at index time. A constant title+date prefix on every chunk is the
# degenerate (non-chunk-specific) form of contextual retrieval: it adds little
# dense signal because our titles are homogeneous ("<X> Insurance vs <Y> Devi"),
# and it actively pollutes BM25 by injecting the same party/year tokens into
# every chunk of a document. We keep the index clean; the agent gets the case
# name + source for citations from the metadata attached to each hit instead.


def _hit_header(meta: dict) -> str:
    """'[DOC_031 · National Insurance Co. Ltd vs Laxmi Narain Dhut on 2 March, 2007]'."""
    doc_id = meta["doc_id"]
    title = (meta.get("title") or "").strip()
    return f"[{doc_id} · {title}]" if title else f"[{doc_id}]"


# --- retrieval recorder (for faithfulness eval, I9) -----------------------
# The faithfulness metric needs the chunks the agent ACTUALLY saw — but the agent
# issues many search_corpus calls per run, so we accumulate the retrieved chunk
# texts here. The eval resets this before a run and reads the union after, to
# check every claim in the answer against what was genuinely retrieved. No effect
# on normal operation; it's just an append-only log.
_RETRIEVAL_LOG: list[str] = []


def reset_retrieval_log() -> None:
    _RETRIEVAL_LOG.clear()


def get_retrieval_log() -> list[str]:
    """Deduped chunk texts retrieved since the last reset (preserves order)."""
    return list(dict.fromkeys(_RETRIEVAL_LOG))


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
    _RETRIEVAL_LOG.extend(h.page_content for h in used)  # record what the agent saw
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
            return header + d["text"][:20000]  # cap to stay within context budget
    return f"No document found with id {doc_id}."
