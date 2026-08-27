"""LangChain RAG pipeline & citation formatter supporting Gemini and OpenAI."""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import settings
from src.retriever import TwoStageRetriever

RAG_SYSTEM_PROMPT = """You are an expert GitHub Codebase Architect and Engineering Assistant.
Answer the user's question about the repository using ONLY the provided code context and issue records.

Guidelines:
1. Provide technical, accurate, and insightful explanations based on the context.
2. Refer explicitly to relevant files, classes, and function names.
3. Structure your response with clear markdown headings, bullet points, and syntax-highlighted code blocks.
4. If the provided context is insufficient to answer completely, acknowledge what is known and specify what details are missing.
5. Emphasize architectural relationships, dependencies, and design patterns when asked.

Context:
{context}
"""


class RAGChain:
    """RAG pipeline execution with citation formatting."""

    def __init__(
        self,
        retriever: TwoStageRetriever,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = settings.temperature
    ):
        self.retriever = retriever
        selected_provider = provider or settings.active_provider

        if selected_provider == "gemini":
            key = api_key or settings.google_api_key
            model = model_name or settings.gemini_model
            from langchain_google_genai import ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=key
            )
        elif selected_provider == "openai":
            key = api_key or settings.openai_api_key
            model = model_name or settings.openai_model
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                openai_api_key=key
            )
        else:
            raise ValueError("No API Key detected! Please configure GOOGLE_API_KEY or OPENAI_API_KEY.")

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{question}")
        ])
        self.output_parser = StrOutputParser()

    @staticmethod
    def format_context(documents: List[Document]) -> str:
        """Formats retrieved documents with source headers and line numbers."""
        formatted_chunks = []
        for idx, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "Unknown")
            html_url = doc.metadata.get("html_url", "")
            doc_type = doc.metadata.get("type", "code")
            
            chunk_header = f"=== [Document {idx}] Source: {source} ({doc_type}) ==="
            if html_url:
                chunk_header += f"\nGitHub URL: {html_url}"
            
            formatted_chunks.append(f"{chunk_header}\n{doc.page_content}\n")
        return "\n".join(formatted_chunks)

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Executes full RAG workflow with citations."""
        relevant_docs = self.retriever.get_relevant_documents(question)
        context_str = self.format_context(relevant_docs)

        chain = self.prompt | self.llm | self.output_parser
        raw_response = chain.invoke({"context": context_str, "question": question})

        # Ensure string response format
        if isinstance(raw_response, list):
            response_text = "".join(
                item.get("text", str(item)) if isinstance(item, dict) else str(item)
                for item in raw_response
            )
        else:
            response_text = str(raw_response)

        citations = []
        for doc in relevant_docs:
            citations.append({
                "source": doc.metadata.get("source", "Unknown"),
                "file_path": doc.metadata.get("file_path", ""),
                "html_url": doc.metadata.get("html_url", ""),
                "score": doc.metadata.get("rerank_score", None),
                "type": doc.metadata.get("type", "code"),
                "preview": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else "")
            })

        return {
            "answer": response_text,
            "citations": citations,
            "raw_documents": relevant_docs
        }
