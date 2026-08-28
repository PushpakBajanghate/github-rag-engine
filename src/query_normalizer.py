import logging
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import config
from src.models import NormalizedQueryOutput, SubQuery

logger = logging.getLogger(__name__)

NORMALIZER_SYSTEM_PROMPT = """You are a Principal AI Architect & Code Retrieval Specialist.
Your mission is to eliminate semantic mismatch between informal, colloquial, or Hinglish/multilingual user queries and formal English source codebases, issue trackers, and technical documentation.

Instructions:
1. Detect the input language (e.g., 'Hinglish', 'Hindi-Latin', 'English', 'Colloquial English', etc.).
2. Translate informal or Hinglish colloquialisms into formal, canonical English technical terminology.
3. If the query asks about multiple distinct code components, sub-tasks, or architectural layers, set `is_multi_intent = True` and decompose it into focused `SubQuery` items.
4. For each `SubQuery`, set `target_focus` to one of:
   - 'schema_definition': Models, schemas, structs, data types, interfaces, database tables.
   - 'implementation_logic': Core algorithms, functions, classes, calculations, business rules.
   - 'usage_example': Endpoints, workflow scripts, call sites, CLI commands, tests, integrations.
5. Create a `context_resolved_query` which is a clean, comprehensive canonical English query representing the entire user requirement.

Few-Shot Examples:
- User: "bhai email me pdf attach kaise kare aur background me send kaise kare"
  Output:
    detected_language: "Hinglish"
    response_tone: "explanatory"
    is_multi_intent: true
    retrieval_queries:
      - query: "fastapi-mail MessageSchema attachments UploadFile PDF attachment" (target_focus: "implementation_logic")
      - query: "fastapi-mail FastMail send_message BackgroundTasks async background task" (target_focus: "usage_example")
    context_resolved_query: "How to attach PDF to email and send it asynchronously in the background using fastapi-mail BackgroundTasks"

- User: "cosine similarity calculate kaise ho rahi h movie recommender me"
  Output:
    detected_language: "Hinglish"
    response_tone: "technical"
    is_multi_intent: false
    retrieval_queries:
      - query: "cosine_similarity sklearn metrics pairwise movie recommender notebook" (target_focus: "implementation_logic")
    context_resolved_query: "How is cosine similarity calculated for movie recommendations in the recommender notebook or script"

- User: "where is the Chroma vectorstore initialized and how are embeddings cached?"
  Output:
    detected_language: "English"
    response_tone: "technical"
    is_multi_intent: true
    retrieval_queries:
      - query: "Chroma load_vectorstore index_documents persist_directory collection_name" (target_focus: "implementation_logic")
      - query: "get_embedding_model ResilientGoogleEmbeddings lru_cache caching" (target_focus: "implementation_logic")
    context_resolved_query: "Chroma vectorstore initialization and embedding caching mechanism"
"""

def normalize_and_decompose_query(user_query: str) -> NormalizedQueryOutput:
    """
    Analyzes, normalizes Hinglish/colloquial text, and decomposes the query into
    structured sub-queries with targeted codebase focus.
    """
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.0
    )
    
    structured_llm = llm.with_structured_output(NormalizedQueryOutput)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", NORMALIZER_SYSTEM_PROMPT),
        ("human", "Analyze and normalize this user query:\n\n{query}")
    ])
    
    chain = prompt | structured_llm
    
    try:
        result: NormalizedQueryOutput = chain.invoke({"query": user_query})
        if not result.retrieval_queries:
            result.retrieval_queries = [
                SubQuery(query=user_query, target_focus="implementation_logic")
            ]
        return result
    except Exception as e:
        logger.warning(f"Error during query normalization: {e}. Falling back to default query.")
        return NormalizedQueryOutput(
            detected_language="Unknown",
            response_tone="technical",
            is_multi_intent=False,
            retrieval_queries=[
                SubQuery(query=user_query, target_focus="implementation_logic")
            ],
            context_resolved_query=user_query
        )
