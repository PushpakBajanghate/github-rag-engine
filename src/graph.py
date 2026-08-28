import logging
from typing import Dict, Any, Generator, Tuple, List
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.models import RAGGraphState, NormalizedQueryOutput
from src.query_normalizer import normalize_and_decompose_query
from src.retriever import hybrid_retrieve_and_deduplicate
from src.chain import get_llm, SYSTEM_PROMPT, format_docs

logger = logging.getLogger(__name__)

def normalize_query_node(state: RAGGraphState) -> Dict[str, Any]:
    """Node 1: Analyzes and normalizes Hinglish/colloquial text into structured sub-queries."""
    question = state["question"]
    normalized = normalize_and_decompose_query(question)
    return {
        "normalized_query": normalized
    }

def hybrid_retrieve_node(state: RAGGraphState) -> Dict[str, Any]:
    """Node 2: Executes multi-query BM25 + Vector hybrid retrieval with deduplication & reranking."""
    normalized: NormalizedQueryOutput = state.get("normalized_query")
    
    if normalized and normalized.retrieval_queries:
        sub_queries = normalized.retrieval_queries
        resolved_query = normalized.context_resolved_query
    else:
        from src.models import SubQuery
        sub_queries = [SubQuery(query=state["question"], target_focus="implementation_logic")]
        resolved_query = state["question"]
        
    retrieved_docs = hybrid_retrieve_and_deduplicate(
        sub_queries=sub_queries,
        context_resolved_query=resolved_query,
        top_k_per_query=12,
        final_k=7
    )
    
    context_str = format_docs(retrieved_docs)
    
    sources = [
        {
            "source": doc.metadata.get("source", "Unknown"),
            "url": doc.metadata.get("html_url", ""),
            "type": doc.metadata.get("type", "code")
        }
        for doc in retrieved_docs
    ]
    
    return {
        "retrieved_docs": retrieved_docs,
        "context_str": context_str,
        "sources": sources
    }

def generate_answer_node(state: RAGGraphState) -> Dict[str, Any]:
    """Node 3: Generates grounded technical response using context."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    answer = chain.invoke({
        "context": state.get("context_str", ""),
        "question": state["question"]
    })
    
    return {
        "answer": answer
    }

def build_rag_graph():
    """Builds and compiles the LangGraph state graph."""
    workflow = StateGraph(RAGGraphState)
    
    workflow.add_node("normalize_query", normalize_query_node)
    workflow.add_node("hybrid_retrieve", hybrid_retrieve_node)
    workflow.add_node("generate_answer", generate_answer_node)
    
    workflow.add_edge(START, "normalize_query")
    workflow.add_edge("normalize_query", "hybrid_retrieve")
    workflow.add_edge("hybrid_retrieve", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    return workflow.compile()

# Global compiled graph instance
rag_graph_app = build_rag_graph()

def run_rag_graph(question: str) -> Dict[str, Any]:
    """Runs the full LangGraph pipeline synchronously."""
    initial_state: RAGGraphState = {
        "question": question
    }
    return rag_graph_app.invoke(initial_state)

def stream_rag_graph(
    question: str,
    top_k_per_query: int = 10,
    final_k: int = 5
) -> Tuple[Generator[str, None, None], List[Document], NormalizedQueryOutput]:
    """
    Executes normalization and hybrid retrieval through LangGraph,
    then streams the response tokens for UI responsiveness.
    """
    # 1. Normalize query
    normalized = normalize_and_decompose_query(question)
    
    # 2. Hybrid Retrieve & Deduplicate
    retrieved_docs = hybrid_retrieve_and_deduplicate(
        sub_queries=normalized.retrieval_queries,
        context_resolved_query=normalized.context_resolved_query,
        top_k_per_query=top_k_per_query,
        final_k=final_k
    )
    
    # 3. Format Context
    context_str = format_docs(retrieved_docs)
    
    # 4. Stream LLM Generation
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    
    stream_gen = chain.stream({
        "context": context_str,
        "question": question
    })
    
    return stream_gen, retrieved_docs, normalized

