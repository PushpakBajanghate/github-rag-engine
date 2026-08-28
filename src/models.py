from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.documents import Document

class SubQuery(BaseModel):
    """Decomposed technical search sub-query with specific code/issue focus."""
    query: str = Field(
        description="Formal English technical search query targeting codebase symbols, functions, or concepts."
    )
    target_focus: Literal["schema_definition", "implementation_logic", "usage_example"] = Field(
        description="Target focus of the search query in the codebase: schema_definition, implementation_logic, or usage_example."
    )

class NormalizedQueryOutput(BaseModel):
    """Structured query analysis and normalization output from LLM."""
    detected_language: str = Field(
        description="Detected input language or mix (e.g., 'Hinglish', 'Hindi-Latin', 'English', 'Colloquial English')."
    )
    response_tone: str = Field(
        description="Recommended response tone (e.g., 'technical', 'explanatory', 'bilingual_friendly')."
    )
    is_multi_intent: bool = Field(
        description="True if the user prompt asks multiple distinct technical questions or sub-tasks."
    )
    retrieval_queries: List[SubQuery] = Field(
        description="List of decomposed, formal technical sub-queries optimized for vector and keyword search."
    )
    context_resolved_query: str = Field(
        description="Fully expanded, canonical English query resolving conversational context and colloquial expressions."
    )

class RAGGraphState(TypedDict, total=False):
    """State definition for LangGraph RAG execution pipeline."""
    question: str
    normalized_query: Optional[NormalizedQueryOutput]
    retrieved_docs: List[Document]
    context_str: str
    answer: str
    sources: List[Dict[str, Any]]
