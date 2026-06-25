"""The agent — the importable core.

Built on LangGraph's ReAct loop via langchain's `create_agent`. The agent is a
single object that plans and executes its own tool-use sequence; the UI and the
eval both import THIS and nothing else from the agent layer. Swapping Streamlit
for FastAPI later touches only the UI, never this file.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from src.agent.prompts import SYSTEM_PROMPT
from src.agent.tools import get_document, search_corpus
from src.config import settings

# `create_agent` moved around across langchain 1.x; support both, and fall back
# to the LangGraph prebuilt ReAct agent (same loop under the hood).
try:
    from langchain.agents import create_agent  # langchain 1.x
except ImportError:  # pragma: no cover
    from langgraph.prebuilt import create_react_agent as create_agent


@lru_cache(maxsize=1)
def build_agent():
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        # Leave temperature/top_p/top_k unset — Gemini reasoning is tuned for
        # its defaults; overriding them hurts tool-planning quality.
    )
    tools = [search_corpus, get_document]
    try:
        return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    except TypeError:
        # prebuilt create_react_agent uses `prompt=` instead of `system_prompt=`
        return create_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT)


# Module-level singleton for convenient `from src.agent.graph import agent`.
agent = build_agent()
