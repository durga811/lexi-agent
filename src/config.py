"""Central configuration: paths, model names, chunking + retrieval params.

Every tunable knob lives here, read from the environment / `.env`.
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API key — langchain-google-genai reads GOOGLE_API_KEY; accept GEMINI_API_KEY
    # (the AI Studio convention) as an alias so either name in .env works.
    google_api_key: str = Field(
        "", validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY")
    )

    # Paths
    raw_dir: Path = ROOT / "data" / "raw"          # source PDFs (local only)
    corpus_file: Path = ROOT / "data" / "corpus.jsonl"  # extracted text (committed)
    chroma_dir: Path = ROOT / "data" / "chroma"

    # Models — LLM_MODEL in .env overrides the chat model; embeddings run locally.
    llm_model: str = "gemini-3.5-flash"
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # Chunking
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # Retrieval — when use_reranker is on, over-retrieve `rerank_pool` ensemble
    # candidates and a cross-encoder reranks them down to top_k (measured +0.07
    # recall@8). Set use_reranker=False for the plain ensemble.
    top_k: int = 8
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_pool: int = 40

    # Observability — mirrored into os.environ below because LangChain auto-tracing
    # reads LANGSMITH_* from the environment, not from this object.
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "lexi-agent"
    langsmith_endpoint: str = "https://api.smith.langchain.com"


settings = Settings()

# Mirror the key into the environment so langchain-google-genai picks it up.
if settings.google_api_key:
    os.environ.setdefault("GOOGLE_API_KEY", settings.google_api_key)

# Activate tracing only when explicitly enabled and a key is present. setdefault
# so a real exported env var always wins over the .env value.
if settings.langsmith_tracing and settings.langsmith_api_key:
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)
