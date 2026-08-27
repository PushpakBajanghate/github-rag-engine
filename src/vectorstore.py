"""ChromaDB embedding & persistence logic supporting Gemini and OpenAI."""

import os
import re
import time
import logging
from typing import List, Optional, Callable
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.config import settings

logger = logging.getLogger(__name__)

# Gemini free-tier embedding rate limits
# Free tier: 100 embed requests/minute total
GEMINI_BATCH_SIZE = 5           # Docs per API call (keep small)
GEMINI_DELAY_SECONDS = 12       # Wait 12s between batches (5 batches/min = 60 req/min with buffer)
GEMINI_MAX_RETRIES = 5          # Max 429-retry attempts
GEMINI_BACKOFF_BASE = 60        # Initial backoff seconds on 429

OPENAI_BATCH_SIZE = 100         # OpenAI has very generous limits


def _is_rate_limit_error(e: Exception) -> bool:
    """Detects 429/RESOURCE_EXHAUSTED/quota errors from any provider."""
    err = str(e).upper()
    return any(k in err for k in ("429", "RESOURCE_EXHAUSTED", "QUOTA", "RATE_LIMIT", "RATELIMIT"))


def sanitize_collection_name(name: str) -> str:
    """Sanitizes repository identifier to a valid ChromaDB collection name."""
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    s = re.sub(r"^[^a-zA-Z0-9]+", "", s)
    s = re.sub(r"[^a-zA-Z0-9]+$", "", s)
    return f"repo_{s}"[:63] if len(s) < 3 else s[:63]


def _embed_batch_with_retry(
    vectorstore: Chroma,
    batch: List[Document],
    batch_idx: int,
    total_batches: int,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """Embeds one batch with exponential backoff on rate-limit errors."""
    for attempt in range(GEMINI_MAX_RETRIES + 1):
        try:
            vectorstore.add_documents(batch)
            return
        except Exception as e:
            if _is_rate_limit_error(e):
                if attempt == GEMINI_MAX_RETRIES:
                    raise RuntimeError(
                        f"Embedding quota exceeded after {GEMINI_MAX_RETRIES} retries. "
                        "The Gemini free tier allows ~100 embed requests/min. "
                        "Wait 1-2 minutes and try again, or reduce 'Max Files to Ingest'."
                    ) from e
                wait = GEMINI_BACKOFF_BASE * (2 ** attempt)
                msg = (
                    f"Rate limit hit on batch {batch_idx}/{total_batches}. "
                    f"Retrying in {wait}s (attempt {attempt + 1}/{GEMINI_MAX_RETRIES})..."
                )
                logger.warning(msg)
                if progress_callback:
                    progress_callback(f"⏳ {msg}")
                time.sleep(wait)
            else:
                raise


class VectorStoreManager:
    """
    Manages ChromaDB vector collections with automatic rate-limit handling.

    Gemini free tier: batches 5 docs per call, waits 12s between batches,
    and retries automatically on 429 RESOURCE_EXHAUSTED with exponential backoff.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        os.makedirs(self.persist_directory, exist_ok=True)

        selected_provider = provider or settings.active_provider
        self.provider = selected_provider

        if selected_provider == "gemini":
            key = api_key or settings.google_api_key
            if not key:
                raise ValueError(
                    "GOOGLE_API_KEY / GEMINI_API_KEY required. Add it to your .env file."
                )
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=settings.gemini_embedding_model,
                google_api_key=key,
            )
            self.batch_size = GEMINI_BATCH_SIZE
            self.inter_batch_delay = GEMINI_DELAY_SECONDS

        elif selected_provider == "openai":
            key = api_key or settings.openai_api_key
            if not key:
                raise ValueError(
                    "OPENAI_API_KEY required. Add it to your .env file."
                )
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                openai_api_key=key,
            )
            self.batch_size = OPENAI_BATCH_SIZE
            self.inter_batch_delay = 0

        else:
            raise ValueError(
                "No API key found! Configure GOOGLE_API_KEY (Gemini) or "
                "OPENAI_API_KEY in your .env file."
            )

    def create_or_update_vectorstore(
        self,
        repo_name: str,
        documents: List[Document],
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Chroma:
        """
        Creates or updates a ChromaDB collection, respecting Gemini rate limits.

        Strategy for Gemini free tier:
          - Batch size: 5 docs per API call
          - Delay: 12 seconds between each batch
          - Retry: exponential backoff (60s, 120s, 240s...) on 429 errors
        """
        collection_name = sanitize_collection_name(repo_name)
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

        if not documents:
            return vectorstore

        batches = [
            documents[i : i + self.batch_size]
            for i in range(0, len(documents), self.batch_size)
        ]
        total_batches = len(batches)
        total_docs = len(documents)

        for batch_idx, batch in enumerate(batches, start=1):
            docs_done = min(batch_idx * self.batch_size, total_docs)
            msg = (
                f"Embedding batch {batch_idx}/{total_batches} "
                f"({docs_done}/{total_docs} docs embedded)..."
            )
            if progress_callback:
                progress_callback(f"🧠 {msg}")
            else:
                logger.info(msg)

            _embed_batch_with_retry(
                vectorstore=vectorstore,
                batch=batch,
                batch_idx=batch_idx,
                total_batches=total_batches,
                progress_callback=progress_callback,
            )

            # Rate-limit pause between batches (skip after final batch)
            if self.inter_batch_delay > 0 and batch_idx < total_batches:
                pause_msg = (
                    f"Pausing {self.inter_batch_delay}s to respect Gemini API rate limits "
                    f"(batch {batch_idx}/{total_batches} complete)..."
                )
                if progress_callback:
                    progress_callback(f"⏳ {pause_msg}")
                time.sleep(self.inter_batch_delay)

        return vectorstore

    def get_vectorstore(self, repo_name: str) -> Chroma:
        """Loads an existing Chroma collection for a repository."""
        collection_name = sanitize_collection_name(repo_name)
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def clear_collection(self, repo_name: str) -> None:
        """Deletes all vectors for a given repository collection."""
        collection_name = sanitize_collection_name(repo_name)
        try:
            vs = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            vs.delete_collection()
        except Exception:
            pass
