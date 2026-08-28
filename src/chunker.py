from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language
)

EXTENSION_TO_LANGUAGE = {
    ".py": Language.PYTHON,
    ".ipynb": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".cpp": Language.CPP,
    ".c": Language.CPP,
    ".h": Language.CPP,
    ".hpp": Language.CPP,
    ".cs": getattr(Language, "CSHARP", None),
    ".go": Language.GO,
    ".rs": getattr(Language, "RUST", None),
    ".rb": getattr(Language, "RUBY", None),
    ".php": getattr(Language, "PHP", None),
    ".scala": getattr(Language, "SCALA", None),
    ".kt": getattr(Language, "KOTLIN", None),
    ".html": getattr(Language, "HTML", None),
    ".rst": Language.MARKDOWN,
    ".md": Language.MARKDOWN,
}

def chunk_code_and_docs(
    documents: List[Document],
    chunk_size: int = 2000,
    chunk_overlap: int = 250
) -> List[Document]:
    """
    Chunks code, Jupyter notebooks, and documentation based on language syntax rules.
    Default chunk size is increased to 2000 chars (250 overlap) to preserve full function
    definitions and mathematical calculations across files and notebooks.
    """
    chunked_docs = []
    
    for doc in documents:
        file_path = doc.metadata.get("source", "")
        ext = "." + file_path.split(".")[-1].lower() if "." in file_path else ""
        
        language = EXTENSION_TO_LANGUAGE.get(ext)
        
        if language:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
        splits = splitter.split_documents([doc])
        chunked_docs.extend(splits)
        
    return chunked_docs