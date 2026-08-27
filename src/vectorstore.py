from typing import List
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import config

def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY
    )

def index_documents(documents: List[Document], collection_name: str = "github_repo") -> Chroma:
    """Creates/replaces vector index in local ChromaDB."""
    embeddings = get_embedding_model()
    
    # Initialize Chroma vector store with persistent storage
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=config.CHROMA_PERSIST_DIR
    )
    return vector_store

def load_vectorstore(collection_name: str = "github_repo") -> Chroma:
    """Loads existing ChromaDB vector index."""
    embeddings = get_embedding_model()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIR
    )