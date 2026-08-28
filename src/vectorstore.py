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

try:
    from fastembed import TextEmbedding
    HAS_FASTEMBED = True
except ImportError:
    HAS_FASTEMBED = False

class FastEmbedLocalEmbeddings(Embeddings):
    """Local, ultra-fast ONNX embedding engine (zero API cost, zero rate limits)."""
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = list(self.model.embed(texts))
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embeddings = list(self.model.embed([text]))
        return embeddings[0].tolist()

class HybridResilientEmbeddings(Embeddings):
    """
    Intelligent embedding engine:
    Uses fast, local BAAI/bge-small ONNX embeddings by default for sub-second, zero-quota processing,
    with graceful fallback across providers.
    """
    def __init__(
        self,
        model: str,
        google_api_key: str,
        prefer_local: bool = True
    ):
        self.prefer_local = prefer_local
        self.local_model = FastEmbedLocalEmbeddings() if HAS_FASTEMBED else None
        self.google_model = None
        self.google_api_key = google_api_key
        self.model = model

    def _get_google_model(self):
        if self.google_model is None and self.google_api_key:
            try:
                self.google_model = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=self.google_api_key
                )
            except Exception:
                pass
        return self.google_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.prefer_local and self.local_model:
            try:
                return self.local_model.embed_documents(texts)
            except Exception as e:
                logger.warning(f"Local embedding failed: {e}. Falling back to Google Embeddings.")

        # Remote Google Embeddings
        gm = self._get_google_model()
        if gm:
            try:
                return gm.embed_documents(texts)
            except Exception as e:
                if self.local_model:
                    logger.warning(f"Google embeddings quota/error ({e}). Falling back to local FastEmbed.")
                    return self.local_model.embed_documents(texts)
                raise e
        elif self.local_model:
            return self.local_model.embed_documents(texts)
        raise RuntimeError("No embedding provider available.")

    def embed_query(self, text: str) -> List[float]:
        if self.prefer_local and self.local_model:
            try:
                return self.local_model.embed_query(text)
            except Exception:
                pass

        gm = self._get_google_model()
        if gm:
            try:
                return gm.embed_query(text)
            except Exception:
                if self.local_model:
                    return self.local_model.embed_query(text)
        elif self.local_model:
            return self.local_model.embed_query(text)
        raise RuntimeError("No embedding provider available.")

_CACHED_VECTORSTORE: Optional[Chroma] = None

@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    return HybridResilientEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        prefer_local=True
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