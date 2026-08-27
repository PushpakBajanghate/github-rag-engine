"""Centralized environment configs & API keys."""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"


class AppSettings(BaseModel):
    """Central application settings."""
    # API Keys
    github_token: str = Field(
        default_factory=lambda: os.getenv("GITHUB_TOKEN", ""),
        description="GitHub Personal Access Token"
    )
    openai_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        description="OpenAI API key"
    )

    # Vector store paths
    chroma_persist_dir: str = Field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIRECTORY", str(CHROMA_DIR)),
        description="Local ChromaDB directory"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model"
    )

    # LLM Settings
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Primary LLM"
    )
    temperature: float = Field(
        default=0.1,
        description="LLM temperature"
    )

    # Retrieval & Reranker
    flashrank_model: str = Field(
        default_factory=lambda: os.getenv("FLASH_RANK_MODEL", "ms-marco-TinyBERT-L-2-v2"),
        description="FlashRank model"
    )
    top_k_vector: int = Field(
        default=15,
        description="Stage 1 vector retrieval count"
    )
    top_k_rerank: int = Field(
        default=5,
        description="Stage 2 FlashRank rerank count"
    )

    # Chunking
    chunk_size: int = Field(default=1000, description="Chunk character size")
    chunk_overlap: int = Field(default=150, description="Chunk overlap")


settings = AppSettings()
