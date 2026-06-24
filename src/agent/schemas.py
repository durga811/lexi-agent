"""Pydantic models describing the shape of a structured research report.

Currently the agent returns prose in the three-section format (more natural for
the LLM and easier to render with citations). These schemas document the target
structure and are the hook for an optional enforced structured-output pass — see
the ADR "what I'd do with another week" note.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Precedent(BaseModel):
    doc_id: str
    principle: str = Field(description="legal principle the judgment establishes")
    relevance: str = Field(description="how its facts align with the client's case")


class ResearchReport(BaseModel):
    supporting: list[Precedent]
    adverse: list[Precedent] = Field(
        description="precedents the opposing side could use against the client"
    )
    strategy: str = Field(
        description="prioritized arguments, realistic compensation range, key risks"
    )
