from typing import List, Optional
from functools import lru_cache
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import config

_CACHED_VECTORSTORE: Optional[Chroma] = None

@lru_cache(maxsize=1)
def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY
    )

def index_documents(
    documents: List[Document],
    collection_name: str = "github_repo",
    batch_size: int = 64
) -> Chroma:
    """Creates/replaces vector index in local ChromaDB in safe batches."""
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
    
    # Ingest in batches to prevent hitting API payload limits
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        vector_store.add_documents(batch)
        
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