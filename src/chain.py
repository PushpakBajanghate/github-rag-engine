from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import config
from src.retriever import retrieve_and_rerank

SYSTEM_PROMPT = """You are an expert software engineer reviewing a codebase.
Use the following retrieved code snippets and GitHub issues to answer the user question accurately.

Rules:
1. ONLY answer using facts grounded directly in the provided Context.
2. If the context does not contain enough info, state what is missing.
3. For every code file or issue referenced, mention its file path or issue URL as cited sources.

Context:
{context}
"""

def format_docs(docs) -> str:
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        url = doc.metadata.get("html_url", "")
        doc_type = doc.metadata.get("type", "content")
        formatted.append(f"--- SOURCE [{doc_type}]: {source} ({url}) ---\n{doc.page_content}\n")
    return "\n".join(formatted)

def ask_repo(query: str):
    llm = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.1
    )
    
    retrieved_docs = retrieve_and_rerank(query=query)
    context_str = format_docs(retrieved_docs)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "context": context_str,
        "question": query
    })
    
    return response, retrieved_docs