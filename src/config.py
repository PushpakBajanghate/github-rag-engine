import os
from pydantic_settings import BaseSettings, SettingsConfigDict

def get_secret(key: str, default: str = "") -> str:
    """Safely resolves configuration from environment variables or Streamlit Cloud secrets."""
    # 1. Check OS Environment
    val = os.getenv(key)
    if val:
        return val
    # 2. Check Streamlit Cloud Secrets (if running in Streamlit)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default

class Settings(BaseSettings):
    GOOGLE_API_KEY: str = get_secret("GOOGLE_API_KEY", "")
    GITHUB_TOKEN: str = get_secret("GITHUB_TOKEN", "")
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    
    # Embedding and LLM specifications
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"

    # Primary LLM model (experimentally verified working models ordered by quota headroom)
    LLM_MODEL: str = "gemini-3.5-flash"

    # Ordered fallback chain — tried in sequence when primary hits 429 / 404 / 503
    LLM_MODEL_FALLBACKS: list = [
        "gemini-3.5-flash-lite",      # ~1500 rpm, very generous free tier
        "gemini-flash-lite-latest",   # alias for latest lite flash
        "gemini-3.6-flash",           # 20 req/day limit — last resort only
    ]
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

config = Settings()