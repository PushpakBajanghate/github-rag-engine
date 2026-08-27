"""ChromaDB embedding & persistence logic."""

import os
import re
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from src.config import settings


def sanitize_collection_name(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    sanitized = re.sub(r"^[^a-zA-Z0-9]+", "", sanitized)
    sanitized = re.sub(r"[^a-zA-Z0-9]+$", "", sanitized)
    if len(sanitized) < 3:
        sanitized = f"repo_{sanitized}"
    return sanitized[:63]


class VectorStoreManager:
    """Manages local ChromaDB vector store embeddings and collections."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        os.makedirs(self.persist_directory, exist_ok=True)

        api_key = openai_api_key or settings.openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for embedding generation.")

        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=api_key
        )

    def create_or_update_vectorstore(
        self,
        repo_name: str,
        documents: List[Document]
    ) -> Chroma:
        collection_name = sanitize_collection_name(repo_name)
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

        if documents:
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                vectorstore.add_documents(batch)

        return vectorstore

    def get_vectorstore(self, repo_name: str) -> Chroma:
        collection_name = sanitize_collection_name(repo_name)
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def clear_collection(self, repo_name: str) -> None:
        collection_name = sanitize_collection_name(repo_name)
        vs = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        vs.delete_collection()
