import time
import random
import logging
from typing import List, Optional, Callable
from functools import lru_cache
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import config

logger = logging.getLogger(__name__)

GEMINI_MICRO_BATCH_SIZE = 10
GEMINI_DELAY_SECONDS = 1.0

class ResilientGoogleEmbeddings(Embeddings):
    """
    Wrapper around GoogleGenerativeAIEmbeddings that gracefully handles
    429 RESOURCE_EXHAUSTED rate limits with exponential backoff and safe micro-batching.
    """
    def __init__(
        self,
        model: str,
        google_api_key: str,
        max_retries: int = 8,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        micro_batch_size: int = GEMINI_MICRO_BATCH_SIZE,
        delay_between_batches: float = GEMINI_DELAY_SECONDS,
    ):
        self.underlying = GoogleGenerativeAIEmbeddings(
            model=model,
            google_api_key=google_api_key
        )
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.micro_batch_size = micro_batch_size
        self.delay_between_batches = delay_between_batches

    def _embed_with_retry(self, fn: Callable, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = (
                    "429" in err_str
                    or "resource_exhausted" in err_str
                    or "quota" in err_str
                    or "rate limit" in err_str
                )
                if attempt == self.max_retries - 1 or not is_rate_limit:
                    raise
                # Exponential backoff with jitter
                sleep_time = min(
                    self.max_delay,
                    self.base_delay * (2 ** attempt) + random.uniform(0.5, 2.0)
                )
                print(f"[Rate Limit 429] RESOURCE_EXHAUSTED. Retrying in {sleep_time:.1f}s (Attempt {attempt+1}/{self.max_retries})...")
                time.sleep(sleep_time)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.micro_batch_size):
            batch = texts[i : i + self.micro_batch_size]
            embeddings = self._embed_with_retry(self.underlying.embed_documents, batch)
            all_embeddings.extend(embeddings)
            if i + self.micro_batch_size < len(texts):
                time.sleep(self.delay_between_batches)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._embed_with_retry(self.underlying.embed_query, text)

_CACHED_VECTORSTORE: Optional[Chroma] = None

@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    return ResilientGoogleEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY
    )

def index_documents(
    documents: List[Document],
    collection_name: str = "github_repo",
    batch_size: int = 15,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Chroma:
    """Creates/replaces vector index in local ChromaDB in safe rate-limited batches."""
    global _CACHED_VECTORSTORE
    embeddings = get_embedding_model()
    
    # Initialize Chroma instance
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR
    )
    
    # Reset existing collection if present
    try:
        vector_store.delete_collection()
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=config.CHROMA_PERSIST_DIR
        )
    except Exception:
        pass
    
    total = len(documents)
    # Ingest in small batches with rate limit safety
    for idx, i in enumerate(range(0, total, batch_size)):
        batch = documents[i : i + batch_size]
        if progress_callback:
            progress_callback(f"Embedding chunks {i + 1}-{min(i + batch_size, total)} of {total}...")
        vector_store.add_documents(batch)
        if i + batch_size < total:
            time.sleep(0.5)
        
    _CACHED_VECTORSTORE = vector_store
    return vector_store

def load_vectorstore(collection_name: str = "github_repo") -> Chroma:
    """Loads or returns existing cached ChromaDB vector index."""
    global _CACHED_VECTORSTORE
    if _CACHED_VECTORSTORE is not None:
        return _CACHED_VECTORSTORE
        
    embeddings = get_embedding_model()
    _CACHED_VECTORSTORE = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR
    )
    return _CACHED_VECTORSTORE