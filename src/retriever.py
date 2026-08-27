from typing import List
from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document
from src.vectorstore import load_vectorstore

ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

def retrieve_and_rerank(query: str, top_k: int = 15, final_k: int = 4) -> List[Document]:
    """
    1. Vector retrieval: extracts top_k candidates via Cosine distance.
    2. Cross-encoder re-ranking: scores and trims to final_k most relevant chunks.
    """
    vector_store = load_vectorstore()
    initial_docs = vector_store.similarity_search(query, k=top_k)
    
    if not initial_docs:
        return []
        
    # Format candidates for FlashRank
    passages = [
        {"id": idx, "text": doc.page_content, "meta": doc.metadata}
        for idx, doc in enumerate(initial_docs)
    ]
    
    rerank_request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(rerank_request)
    
    # Sort and pick top results
    top_results = results[:final_k]
    
    reranked_docs = [
        Document(
            page_content=res["text"],
            metadata=res["meta"]
        )
        for res in top_results
    ]
    
    return reranked_docs