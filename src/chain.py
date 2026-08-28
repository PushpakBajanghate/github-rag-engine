from functools import lru_cache
from typing import Tuple, List, Generator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from src.config import config
from src.retriever import retrieve_and_rerank

SYSTEM_PROMPT = """You are an expert software engineer reviewing a codebase.
Use the following retrieved code snippets, notebooks, and GitHub issues to answer the user question accurately.

Rules:
1. ONLY answer using facts grounded directly in the provided Context.
2. If the context does not contain enough info, clearly state what is missing.
3. For every code file, notebook, or issue referenced, mention its file path or issue URL as cited sources.
4. When explaining code from Jupyter notebooks or scripts, refer to the exact logic and functions retrieved.

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

def ask_repo(query: str) -> Tuple[str, List[Document]]:
    """Non-streaming query response."""
    retrieved_docs = retrieve_and_rerank(query=query)
    context_str = format_docs(retrieved_docs)
    chain = get_rag_chain()
    
    response = chain.invoke({
        "context": context_str,
        "question": query
    })
    return response, retrieved_docs

def ask_repo_stream(query: str) -> Tuple[Generator[str, None, None], List[Document]]:
    """Streaming query response for fast real-time UI rendering."""
    retrieved_docs = retrieve_and_rerank(query=query)
    context_str = format_docs(retrieved_docs)
    chain = get_rag_chain()
    
    stream_generator = chain.stream({
        "context": context_str,
        "question": query
    })
    return stream_generator, retrieved_docs