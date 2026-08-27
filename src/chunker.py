from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language
)

EXTENSION_TO_LANGUAGE = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".java": Language.JAVA,
    ".cpp": Language.CPP,
    ".go": Language.GO,
    ".rst": Language.MARKDOWN,
    ".md": Language.MARKDOWN,
}

def chunk_code_and_docs(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
    """
    Chunks code and markdown documents based on language syntax rules.
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