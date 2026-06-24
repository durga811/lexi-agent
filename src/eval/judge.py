"""Gemini-as-judge adapter so DeepEval's G-Eval uses Gemini, not OpenAI.

DeepEval defaults to an OpenAI judge. Wiring it to Gemini keeps the whole stack
on one provider (one API key, no OpenAI dependency).
"""
from __future__ import annotations

from deepeval.models import DeepEvalBaseLLM
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings
from src.utils import message_text


class GeminiJudge(DeepEvalBaseLLM):
    def __init__(self) -> None:
        self.model = ChatGoogleGenerativeAI(
            model=settings.llm_model, google_api_key=settings.google_api_key
        )

    def load_model(self):
        return self.model

    # NB: Gemini 3.x returns `.content` as a list of blocks; DeepEval needs a
    # plain string (it does `str.find("{")` to parse JSON), so flatten it.
    def generate(self, prompt: str, *args, **kwargs) -> str:
        return message_text(self.model.invoke(prompt).content)

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        return message_text((await self.model.ainvoke(prompt)).content)

    def get_model_name(self) -> str:
        return settings.llm_model
