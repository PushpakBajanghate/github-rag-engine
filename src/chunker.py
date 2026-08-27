"""Language-aware & Markdown AST text splitters."""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
    MarkdownHeaderTextSplitter
)
from src.config import settings

EXT_TO_LANGUAGE: Dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".cpp": Language.CPP,
    ".c": Language.CPP,
    ".h": Language.CPP,
    ".hpp": Language.CPP,
    ".html": Language.HTML,
    ".md": Language.MARKDOWN,
    ".sol": Language.SOL,
    ".php": Language.PHP,
    ".rb": Language.RUBY,
    ".swift": Language.SWIFT,
    ".kt": Language.KOTLIN,
    ".scala": Language.SCALA,
}

MARKDOWN_HEADERS = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]


class CodeAwareChunker:
    """Splits documents using syntax and AST awareness."""

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ";", " ", ""]
        )
        self.markdown_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=MARKDOWN_HEADERS
        )

    def split_document(self, doc: Document) -> List[Document]:
        ext = doc.metadata.get("extension", "").lower()

        if ext == ".md":
            return self._split_markdown(doc)

        if ext in EXT_TO_LANGUAGE:
            lang = EXT_TO_LANGUAGE[ext]
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            chunks = splitter.split_documents([doc])
            return self._enrich_metadata(chunks)

        chunks = self.default_splitter.split_documents([doc])
        return self._enrich_metadata(chunks)

    def _split_markdown(self, doc: Document) -> List[Document]:
        try:
            header_splits = self.markdown_header_splitter.split_text(doc.page_content)
            result_docs = []
            for h_split in header_splits:
                merged_meta = {**doc.metadata, **h_split.metadata}
                temp_doc = Document(page_content=h_split.page_content, metadata=merged_meta)
                sub_chunks = self.default_splitter.split_documents([temp_doc])
                result_docs.extend(sub_chunks)
            return self._enrich_metadata(result_docs)
        except Exception:
            chunks = self.default_splitter.split_documents([doc])
            return self._enrich_metadata(chunks)

    def _enrich_metadata(self, chunks: List[Document]) -> List[Document]:
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            lines = chunk.page_content.count("\n") + 1
            chunk.metadata["chunk_line_count"] = lines
        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        all_chunks: List[Document] = []
        for doc in documents:
            chunks = self.split_document(doc)
            all_chunks.extend(chunks)
        return all_chunks
