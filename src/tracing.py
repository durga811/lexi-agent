"""LangSmith observability helpers — safe no-ops when langsmith is unavailable.

LangChain/LangGraph auto-trace the agent loop, LLM calls, and tool calls once
`LANGSMITH_TRACING=true` is set (config.py mirrors the .env vars into os.environ).
This module adds the parts auto-tracing can't see: a `hybrid_retrieve` span with
the reranker's doc_ids/scores and pool size, plus run-level tags so traces are
filterable. Everything degrades to a no-op if langsmith is absent or tracing off.
"""
from __future__ import annotations

from src.config import settings

try:
    from langsmith import traceable

    _HAVE_LANGSMITH = True
except Exception:  # pragma: no cover - langsmith optional
    _HAVE_LANGSMITH = False

    def traceable(*d_args, **d_kwargs):  # type: ignore[misc]
        """No-op stand-in supporting both `@traceable` and `@traceable(...)`."""
        if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
            return d_args[0]
        return lambda fn: fn


def add_trace_metadata(**fields) -> None:
    """Attach metadata (e.g. reranker scores, doc_ids) to the active trace span.
    No-op if langsmith is unavailable or no span is active. Must NEVER raise."""
    if not _HAVE_LANGSMITH:
        return
    try:
        from langsmith.run_helpers import get_current_run_tree

        rt = get_current_run_tree()
        if rt is not None:
            rt.metadata.update(fields)
    except Exception:  # pragma: no cover - tracing must never break the app
        pass


# Stamp every traced run with the config that produced it, so two runs can be
# compared in LangSmith by model / retriever settings rather than guessing.
_RUN_TAGS = ["lexi-agent"]
_RUN_METADATA = {
    "llm_model": settings.llm_model,
    "embed_model": settings.embed_model,
    "use_reranker": settings.use_reranker,
    "rerank_pool": settings.rerank_pool,
    "top_k": settings.top_k,
}


def run_config(**extra_metadata) -> dict:
    """A RunnableConfig fragment to pass to `agent.invoke(...)` / `agent.stream(...)`
    so the trace carries our config tags + metadata (plus any caller extras, e.g.
    `query_kind` from the eval)."""
    return {"tags": _RUN_TAGS, "metadata": {**_RUN_METADATA, **extra_metadata}}
