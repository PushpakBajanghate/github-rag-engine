"""Two-stage retrieval (Vector Search + FlashRank)."""

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma
from flashrank import Ranker, RerankRequest
from src.config import settings


class TwoStageRetriever:
    """Two-stage retriever combining ChromaDB similarity search with FlashRank neural reranker."""

    def __init__(
        self,
        vectorstore: Chroma,
        top_k_vector: int = settings.top_k_vector,
        top_k_rerank: int = settings.top_k_rerank,
        model_name: str = settings.flashrank_model
    ):
        self.vectorstore = vectorstore
        self.top_k_vector = top_k_vector
        self.top_k_rerank = top_k_rerank
        try:
            self.ranker = Ranker(model_name=model_name)
        except Exception:
            self.ranker = Ranker()

    def get_relevant_documents(self, query: str) -> List[Document]:
        # Stage 1: Vector similarity search
        initial_docs: List[Document] = self.vectorstore.similarity_search(
            query=query,
            k=self.top_k_vector
        )

        if not initial_docs:
            return []

        if len(initial_docs) <= self.top_k_rerank:
            return initial_docs

        # Stage 2: FlashRank Reranking
        passages = [
            {"id": idx, "text": doc.page_content, "meta": doc.metadata}
            for idx, doc in enumerate(initial_docs)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = self.ranker.rerank(rerank_request)

        final_docs: List[Document] = []
        for result in reranked_results[: self.top_k_rerank]:
            doc_idx = result["id"]
            score = result.get("score", 0.0)
            original_doc = initial_docs[doc_idx]
            
            enriched_meta = {**original_doc.metadata, "rerank_score": score}
            final_docs.append(
                Document(page_content=original_doc.page_content, metadata=enriched_meta)
            )

        return final_docs
