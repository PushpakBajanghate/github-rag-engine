# GitHub RAG Engine ⚡

A Retrieval-Augmented Generation (RAG) engine designed to index, query, and understand any GitHub codebase using language-aware chunking and two-stage retrieval.

---

## 🏛️ Project Architecture

```
├── data/                       # Local vector database storage (ChromaDB)
├── src/
│   ├── __init__.py
│   ├── config.py               # Centralized environment configs & API keys
│   ├── ingestion.py            # GitHub API client & file/issue extractor
│   ├── chunker.py              # Language-aware & Markdown AST text splitters
│   ├── vectorstore.py          # ChromaDB embedding & persistence logic
│   ├── retriever.py            # Two-stage retrieval (Vector Search + FlashRank)
│   └── chain.py                # LangChain RAG pipeline & citation formatter
├── .env.example                # Template for API keys
├── .gitignore                  # Prevent committing DB files and API keys
├── requirements.txt            # Pinned production dependencies
└── app.py                      # Streamlit frontend entrypoint
```

---

## 🚀 Key Components

1. **Ingestion (`src/ingestion.py`)**: Fetches repository structure, code files, and issues via GitHub API.
2. **Chunker (`src/chunker.py`)**: Language-aware syntax chunker preserving class/function boundaries across Python, JS/TS, Java, Go, Rust, C++, and Markdown AST.
3. **Vector Store (`src/vectorstore.py`)**: Manages local ChromaDB vector collections, embeddings, and persistence.
4. **Two-Stage Retriever (`src/retriever.py`)**: Hybrid retrieval combining initial dense vector search with neural reranking via **FlashRank**.
5. **Chain (`src/chain.py`)**: LangChain RAG pipeline providing answers with direct source citations and line links.
6. **Frontend (`app.py`)**: Streamlit interactive web interface with real-time indexing status and Q&A chat.

---

## 🛠️ Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/PushpakBajanghate/github-rag-engine.git
cd github-rag-engine
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

### 3. Run Application
```bash
streamlit run app.py
```
