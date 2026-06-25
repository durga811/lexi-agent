"""Streamlit UI — a thin layer over src.agent.graph.

Streams the agent's intermediate reasoning (tool calls + retrieved/ranked
snippets) as it works, not just the final answer — a graded deliverable: the
reviewer must see which documents were retrieved, how they ranked, and how the
agent concluded. Swapping this for FastAPI later touches nothing in src/.
"""
from __future__ import annotations

import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage

st.set_page_config(page_title="Lexi Precedent Agent", layout="wide")


@st.cache_resource(show_spinner="Preparing the index and models (first run only)…")
def _bootstrap():
    """Build the vector index from data/raw if it's missing, then preload models.

    The Chroma index isn't committed (it's rebuildable), so on a fresh host like
    Hugging Face Spaces this builds it once on first launch. Cached so it runs a
    single time per container.
    """
    from src.ingest.build_index import build_index
    from src.retrieval.retriever import warmup

    build_index()  # idempotent — returns early if already populated
    warmup()
    return True


_bootstrap()

from src.agent.graph import agent  # noqa: E402  (import after the index exists)
from src.tracing import run_config  # noqa: E402
from src.utils import message_text  # noqa: E402

st.title("⚖️ Legal Precedent Research Agent")
st.caption(
    "Ask a general question about the judgment corpus, or give a full case brief "
    "for deep precedent research. The agent decides its own workflow."
)

with st.sidebar:
    st.header("Try these")
    st.markdown(
        "- *Which judgments involve commercial vehicles?* (simple)\n"
        "- *Find precedents on contributory negligence.* (focused)\n"
        "- *Client's husband (42, ₹35,000/mo, wife + 2 kids) killed by a commercial "
        "truck whose driver had no valid licence; insurer denies the claim as void. "
        "Find supporting and adverse precedents and recommend a strategy.* (deep)"
    )

query = st.text_area("Enter any prompt or case brief:", height=160)

if st.button("Run", type="primary") and query.strip():
    final = None
    st.subheader("🧠 Reasoning trace")
    trace = st.container()

    with st.spinner("Agent researching…"):
        for update in agent.stream(
            {"messages": [("user", query)]},
            stream_mode="updates",
            config=run_config(source="streamlit"),
        ):
            for _node, payload in update.items():
                for msg in payload.get("messages", []):
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            args = tc.get("args", {})
                            label = args.get("query") or args.get("doc_id") or args
                            trace.info(f"🔧 **{tc['name']}** → `{label}`")
                    elif isinstance(msg, ToolMessage):
                        with trace.expander(
                            f"📄 Retrieved & ranked — {msg.name} "
                            "(ordered by reranker relevance; each headed "
                            "`#rank · relevance score · DOC_id`)"
                        ):
                            st.text(message_text(msg.content)[:6000])
                    elif isinstance(msg, AIMessage):
                        text = message_text(msg.content)
                        if text:
                            final = text

    st.subheader("📋 Final analysis")
    st.markdown(final or "_No answer produced._")
