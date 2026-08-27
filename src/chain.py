"""LangChain RAG pipeline & citation formatter."""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from src.config import settings
from src.retriever import TwoStageRetriever

RAG_SYSTEM_PROMPT = """You are an expert GitHub Codebase and Architecture AI assistant.
Your job is to answer questions about the repository using only the provided context chunks.

Instructions:
1. Base your answer strictly on the provided context. If the context does not contain enough information, state clearly what is missing.
2. When explaining code, refer to specific files, functions, and architecture patterns shown in the context.
3. Always include precise citations using markdown links with file paths and line references.
4. Format code blocks with the appropriate programming language identifier.

Context:
{context}
"""


class RAGChain:
    """RAG pipeline execution with citation formatting."""

    def __init__(
        self,
        retriever: TwoStageRetriever,
        openai_api_key: str = settings.openai_api_key,
        model_name: str = settings.llm_model,
        temperature: float = settings.temperature
    ):
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=openai_api_key
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{question}")
        ])
        self.output_parser = StrOutputParser()

    @staticmethod
    def format_context(documents: List[Document]) -> str:
        formatted_chunks = []
        for idx, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "Unknown")
            html_url = doc.metadata.get("html_url", "")
            doc_type = doc.metadata.get("type", "code")
            
            chunk_header = f"--- Document [{idx}] | Source: {source} | Type: {doc_type} ---"
            if html_url:
                chunk_header += f"\nLink: {html_url}"
            
            formatted_chunks.append(f"{chunk_header}\n{doc.page_content}\n")
        return "\n".join(formatted_chunks)

    def answer_question(self, question: str) -> Dict[str, Any]:
        relevant_docs = self.retriever.get_relevant_documents(question)
        context_str = self.format_context(relevant_docs)

        chain = self.prompt | self.llm | self.output_parser
        response_text = chain.invoke({"context": context_str, "question": question})

        citations = []
        for doc in relevant_docs:
            citations.append({
                "source": doc.metadata.get("source", "Unknown"),
                "file_path": doc.metadata.get("file_path", ""),
                "html_url": doc.metadata.get("html_url", ""),
                "score": doc.metadata.get("rerank_score", None),
                "preview": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else "")
            })

        return {
            "answer": response_text,
            "citations": citations,
            "raw_documents": relevant_docs
        }
