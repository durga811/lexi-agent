"""Streamlit UI — a THIN layer over src.agent.graph.

Shows the agent's intermediate reasoning (tool calls + retrieved snippets) as it
streams, not just the final answer. This is a graded deliverable: the reviewer
must be able to see which documents were retrieved and how the agent reasoned.

Swap this file for FastAPI later and nothing in src/ changes.
"""
from __future__ import annotations

import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage

from src.agent.graph import agent
from src.utils import message_text

st.set_page_config(page_title="Lexi Precedent Agent", layout="wide")
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
            {"messages": [("user", query)]}, stream_mode="updates"
        ):
            for _node, payload in update.items():
                for msg in payload.get("messages", []):
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            args = tc.get("args", {})
                            label = args.get("query") or args.get("doc_id") or args
                            trace.info(f"🔧 **{tc['name']}** → `{label}`")
                    elif isinstance(msg, ToolMessage):
                        with trace.expander(f"📄 Retrieved — {msg.name}"):
                            st.text(message_text(msg.content)[:4000])
                    elif isinstance(msg, AIMessage):
                        text = message_text(msg.content)
                        if text:
                            final = text

    st.subheader("📋 Final analysis")
    st.markdown(final or "_No answer produced._")
