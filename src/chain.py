from functools import lru_cache
from typing import Tuple, List, Generator, Optional, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from src.config import config
from src.retriever import retrieve_and_rerank

SYSTEM_PROMPT = """You are a Principal Software Engineer and Codebase Architect reviewing a GitHub repository.
Use the following retrieved code snippets, configuration files, notebooks, and GitHub issues to answer the user question accurately.

Strict Grounding & Anti-Hallucination Guidelines:
1. Grounding: Answer ONLY using facts, code implementations, endpoints, models, and architectures directly verified in the provided Context.
2. Negative Assertion: If the user asks about a library, service, framework, endpoint, model, or configuration (e.g. Redis caching, Stripe billing, Kubernetes Helm charts) that is NOT present in the retrieved context, you MUST explicitly state that it is NOT implemented in this codebase. NEVER invent or hallucinate non-existent features, endpoints, or files.
3. Code Accuracy: When explaining functions, classes, or workflows, quote or reference the exact function names, variables, and file paths.
4. Multilingual / Hinglish: If the question is in Hinglish or informal language, respond in the requested language tone while maintaining technical precision and clear code references.
5. Citations: Explicitly cite relevant file paths (e.g., `backend/app/models/user.py`) for every major claim or explanation.

Context:
{context}
"""

@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.1
    )

def format_docs(docs: List[Document]) -> str:
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        url = doc.metadata.get("html_url", "")
        doc_type = doc.metadata.get("type", "content")
        formatted.append(f"--- SOURCE [{doc_type}]: {source} ({url}) ---\n{doc.page_content}\n")
    return "\n".join(formatted)

def get_rag_chain():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    return prompt | llm | StrOutputParser()

def ask_repo(query: str) -> Tuple[str, List[Document], Optional[Any]]:
    """Runs query through the LangGraph normalization and hybrid retrieval pipeline."""
    from src.graph import run_rag_graph
    result = run_rag_graph(query)
    return result.get("answer", ""), result.get("retrieved_docs", []), result.get("normalized_query")

def ask_repo_stream(query: str) -> Tuple[Generator[str, None, None], List[Document], Optional[Any]]:
    """Streaming query response utilizing LangGraph query normalization & hybrid retrieval."""
    from src.graph import stream_rag_graph
    return stream_rag_graph(query)