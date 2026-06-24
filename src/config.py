"""Central configuration: paths, model names, chunking + retrieval params.

All tunable knobs live here so the rest of the codebase never hard-codes a path
or a hyperparameter. Settings are read from the environment / `.env`.
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- API key -----------------------------------------------------------
    # langchain-google-genai reads GOOGLE_API_KEY from the environment.
    google_api_key: str = ""

    # --- paths -------------------------------------------------------------
    raw_dir: Path = ROOT / "data" / "raw"
    chroma_dir: Path = ROOT / "data" / "chroma"

    # --- models ------------------------------------------------------------
    # Gemini chat model. Overridable via LLM_MODEL in .env so we can swap if a
    # given model id isn't available on this key.
    llm_model: str = "gemini-3.5-flash"
    # Open-source embeddings — no API cost, runs locally.
    embed_model: str = "BAAI/bge-small-en-v1.5"

    # --- chunking ----------------------------------------------------------
    chunk_size: int = 1200
    chunk_overlap: int = 200

    # --- retrieval ---------------------------------------------------------
    top_k: int = 8


settings = Settings()

# Mirror the key into the environment so langchain-google-genai picks it up.
if settings.google_api_key:
    os.environ.setdefault("GOOGLE_API_KEY", settings.google_api_key)
