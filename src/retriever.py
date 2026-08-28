import re
import hashlib
from typing import List, Optional, Set
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest
from langchain_core.documents import Document
from src.vectorstore import load_vectorstore
from src.models import SubQuery

_RANKER: Optional[Ranker] = None
_CACHED_BM25: Optional[BM25Okapi] = None
_CACHED_BM25_DOCS: List[Document] = []

def get_ranker() -> Ranker:
    global _RANKER
    if _RANKER is None:
        _RANKER = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    return _RANKER

def tokenize(text: str) -> List[str]:
    """Tokenizes text for BM25 keyword matching."""
    return re.findall(r"\w+", text.lower())

def get_bm25_index(force_refresh: bool = False) -> tuple[Optional[BM25Okapi], List[Document]]:
    """Loads or builds BM25 index from stored Chroma documents."""
    global _CACHED_BM25, _CACHED_BM25_DOCS
    if _CACHED_BM25 is not None and not force_refresh:
        return _CACHED_BM25, _CACHED_BM25_DOCS

    try:
        vector_store = load_vectorstore()
        all_data = vector_store.get()
        documents = all_data.get("documents", [])
        metadatas = all_data.get("metadatas", [])

        if not documents:
            return None, []

        docs = [
            Document(page_content=doc_text, metadata=meta or {})
            for doc_text, meta in zip(documents, metadatas)
        ]

        tokenized_corpus = [tokenize(doc.page_content) for doc in docs]
        _CACHED_BM25 = BM25Okapi(tokenized_corpus)
        _CACHED_BM25_DOCS = docs
        return _CACHED_BM25, _CACHED_BM25_DOCS
    except Exception:
        return None, []

def bm25_search(query: str, top_k: int = 10) -> List[Document]:
    """Executes keyword-based BM25 search over the corpus."""
    bm25, docs = get_bm25_index()
    if not bm25 or not docs:
        return []

    tokenized_query = tokenize(query)
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    top_indices = [idx for idx in ranked_indices[:top_k] if scores[idx] > 0]
    return [docs[idx] for idx in top_indices]

def compute_doc_hash(doc: Document) -> str:
    """Generates unique deterministic hash for document deduplication."""
    source = doc.metadata.get("source", "")
    content = doc.page_content.strip()
    return hashlib.sha256(f"{source}::{content}".encode("utf-8")).hexdigest()

def hybrid_retrieve_and_deduplicate(
    sub_queries: List[SubQuery],
    context_resolved_query: str,
    top_k_per_query: int = 10,
    final_k: int = 5
) -> List[Document]:
    """
    1. Multi-query Hybrid Retrieval: Executes Vector (Dense) + BM25 (Sparse) for every sub-query.
    2. Context Merging & Deduplication: Combines all candidates and removes duplicates by content hash.
    3. Cross-Encoder Re-ranking: Re-ranks merged candidate pool using FlashRank.
    """
    vector_store = load_vectorstore()
    candidate_pool: List[Document] = []
    seen_hashes: Set[str] = set()

    # Collect all queries to search (sub-queries + resolved full query)
    all_query_strings = [sub.query for sub in sub_queries]
    if context_resolved_query and context_resolved_query not in all_query_strings:
        all_query_strings.append(context_resolved_query)

    for query_str in all_query_strings:
        # 1. Dense Vector Search
        try:
            vector_docs = vector_store.similarity_search(query_str, k=top_k_per_query)
            for doc in vector_docs:
                d_hash = compute_doc_hash(doc)
                if d_hash not in seen_hashes:
                    seen_hashes.add(d_hash)
                    candidate_pool.append(doc)
        except Exception:
            pass

        # 2. Sparse BM25 Keyword Search
        try:
            bm25_docs = bm25_search(query_str, top_k=top_k_per_query)
            for doc in bm25_docs:
                d_hash = compute_doc_hash(doc)
                if d_hash not in seen_hashes:
                    seen_hashes.add(d_hash)
                    candidate_pool.append(doc)
        except Exception:
            pass

    if not candidate_pool:
        return []

    # If few candidates, return directly
    if len(candidate_pool) <= final_k:
        return candidate_pool

    # 3. Cross-Encoder Re-ranking via FlashRank
    try:
        ranker = get_ranker()
        passages = [
            {"id": idx, "text": doc.page_content, "meta": doc.metadata}
            for idx, doc in enumerate(candidate_pool)
        ]
        rerank_query = context_resolved_query or (sub_queries[0].query if sub_queries else "")
        rerank_request = RerankRequest(query=rerank_query, passages=passages)
        results = ranker.rerank(rerank_request)

        top_results = results[:final_k]
        return [
            Document(page_content=res["text"], metadata=res["meta"])
            for res in top_results
        ]
    except Exception:
        # Fallback to candidate pool slice if reranker encounters issue
        return candidate_pool[:final_k]

def retrieve_and_rerank(query: str, top_k: int = 15, final_k: int = 5) -> List[Document]:
    """Single-query fallback interface."""
    sub_q = SubQuery(query=query, target_focus="implementation_logic")
    return hybrid_retrieve_and_deduplicate([sub_q], context_resolved_query=query, final_k=final_k)