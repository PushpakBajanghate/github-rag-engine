"""ChromaDB embedding & persistence logic supporting Gemini and OpenAI."""

import os
import re
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.config import settings


def sanitize_collection_name(name: str) -> str:
    """Sanitizes repository name for ChromaDB collection naming."""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    sanitized = re.sub(r"^[^a-zA-Z0-9]+", "", sanitized)
    sanitized = re.sub(r"[^a-zA-Z0-9]+$", "", sanitized)
    if len(sanitized) < 3:
        sanitized = f"repo_{sanitized}"
    return sanitized[:63]


class VectorStoreManager:
    """Handles vector store persistence, embedding generation, and Chroma collections."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        os.makedirs(self.persist_directory, exist_ok=True)

        selected_provider = provider or settings.active_provider
        self.provider = selected_provider

        if selected_provider == "gemini":
            key = api_key or settings.google_api_key
            if not key:
                raise ValueError("GOOGLE_API_KEY / GEMINI_API_KEY is required for Gemini embeddings.")
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=settings.gemini_embedding_model,
                google_api_key=key
            )
        elif selected_provider == "openai":
            key = api_key or settings.openai_api_key
            if not key:
                raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings.")
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                openai_api_key=key
            )
        else:
            raise ValueError(
                "No API Key detected! Please configure GOOGLE_API_KEY (Gemini) or OPENAI_API_KEY in your .env file."
            )

    def create_or_update_vectorstore(
        self,
        repo_name: str,
        documents: List[Document]
    ) -> Chroma:
        """Creates or updates a Chroma vector collection for a repository."""
        collection_name = sanitize_collection_name(repo_name)
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

        if documents:
            batch_size = 80
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                vectorstore.add_documents(batch)

        return vectorstore

    def get_vectorstore(self, repo_name: str) -> Chroma:
        """Loads an existing vector store collection for a repo."""
        collection_name = sanitize_collection_name(repo_name)
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def clear_collection(self, repo_name: str) -> None:
        """Deletes an indexed repository collection."""
        collection_name = sanitize_collection_name(repo_name)
        vs = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        try:
            vs.delete_collection()
        except Exception:
            pass
