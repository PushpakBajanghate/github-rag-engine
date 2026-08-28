from functools import lru_cache
import logging
from typing import Tuple, List, Generator, Optional, Any, Iterator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.language_models.chat_models import BaseChatModel
from src.config import config

logger = logging.getLogger(__name__)

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

# -----------------------------------------------------------------------
# Resilient LLM — waterfalls across models on 429 / 404 / 503
# -----------------------------------------------------------------------
_MODEL_CHAIN = [config.LLM_MODEL] + list(config.LLM_MODEL_FALLBACKS)
_QUOTA_ERRORS = ("429", "resource_exhausted", "quota", "rate limit", "503", "unavailable")
_NOT_FOUND_ERRORS = ("404", "not_found", "not found", "no longer available")

def _is_retriable_error(e: Exception) -> bool:
    err = str(e).lower()
    return any(k in err for k in _QUOTA_ERRORS + _NOT_FOUND_ERRORS)

def _build_llm(model: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.1,
        max_retries=0,  # we handle retries ourselves
    )

def get_llm(model: Optional[str] = None) -> ChatGoogleGenerativeAI:
    """Returns the primary configured LLM (no fallback logic here — use get_resilient_chain for production)."""
    return _build_llm(model or config.LLM_MODEL)

def _invoke_with_fallback(prompt_value) -> str:
    """Try each model in the waterfall chain. Returns first successful response."""
    last_err = None
    for model in _MODEL_CHAIN:
        try:
            llm = _build_llm(model)
            result = llm.invoke(prompt_value)
            content = result.content if hasattr(result, "content") else str(result)
            # Strip LangChain structured output artifacts if any
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            logger.info(f"LLM answered via model: {model}")
            return content
        except Exception as e:
            last_err = e
            if _is_retriable_error(e):
                logger.warning(f"Model '{model}' failed ({type(e).__name__}). Trying next fallback...")
                continue
            raise  # non-retriable error — propagate immediately
    raise RuntimeError(
        f"All LLM models exhausted. Last error: {last_err}"
    ) from last_err

def _stream_with_fallback(prompt_value) -> Iterator[str]:
    """Try streaming from each model in the waterfall chain."""
    last_err = None
    for model in _MODEL_CHAIN:
        try:
            llm = _build_llm(model)
            for chunk in llm.stream(prompt_value):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                yield content
            logger.info(f"LLM streamed via model: {model}")
            return  # success — stop iteration
        except Exception as e:
            last_err = e
            if _is_retriable_error(e):
                logger.warning(f"Streaming model '{model}' failed ({type(e).__name__}). Trying next...")
                continue
            raise
    raise RuntimeError(
        f"All LLM streaming models exhausted. Last error: {last_err}"
    ) from last_err

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