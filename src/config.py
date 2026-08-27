"""Centralized environment configurations and multi-provider API keys."""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file with override
load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"


class AppSettings(BaseModel):
    """Application configuration parameters supporting Gemini and OpenAI."""

    # API Keys
    google_api_key: str = Field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", ""),
        description="Google Gemini API Key"
    )
    openai_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        description="OpenAI API Key"
    )
    github_token: str = Field(
        default_factory=lambda: os.getenv("GITHUB_TOKEN", ""),
        description="GitHub Personal Access Token"
    )

    # Provider Selection ('gemini' or 'openai')
    @property
    def active_provider(self) -> str:
        if self.google_api_key.strip():
            return "gemini"
        if self.openai_api_key.strip():
            return "openai"
        return "none"

    # Vector store paths
    chroma_persist_dir: str = Field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR") or os.getenv("CHROMA_PERSIST_DIRECTORY", str(CHROMA_DIR)),
        description="Local directory for ChromaDB vector storage"
    )

    # Gemini Models
    gemini_model: str = Field(
        default="gemini-3.6-flash",
        description="Gemini LLM model name"
    )
    gemini_embedding_model: str = Field(
        default="models/gemini-embedding-001",
        description="Gemini Embedding model name"
    )

    # OpenAI Models
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI LLM model name"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI Embedding model name"
    )

    # FlashRank Reranking
    flashrank_model: str = Field(
        default_factory=lambda: os.getenv("FLASH_RANK_MODEL", "ms-marco-TinyBERT-L-2-v2"),
        description="FlashRank neural reranker model identifier"
    )
    top_k_vector: int = Field(
        default=15,
        description="Initial candidates retrieved via vector search"
    )
    top_k_rerank: int = Field(
        default=5,
        description="Top candidates retained after reranking"
    )

    # Chunking
    chunk_size: int = Field(default=1000, description="Target character size per chunk")
    chunk_overlap: int = Field(default=150, description="Character overlap between consecutive chunks")
    temperature: float = Field(default=0.2, description="Generation temperature for LLM")


settings = AppSettings()
