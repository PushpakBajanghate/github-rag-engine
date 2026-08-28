# GitHub RAG Engine ⚡

A production-grade, state-of-the-art **Retrieval-Augmented Generation (RAG)** engine designed to index, analyze, and query complex GitHub codebases and issue trackers using **LangGraph, Pydantic, Hybrid Search (BM25 + Vector), Cross-Encoder Re-ranking, and Multilingual/Hinglish Query Normalization**.

---

## 🏛️ System Architecture Flow

```
[ GitHub Repository / Issues / .ipynb Notebooks ]
                         │
                         ▼ (Git Tree API + ThreadPool Concurrent Ingestion)
          [ Document Ingestion & Clean Extraction ]
                         │
                         ▼ (Language-Aware AST Splitter - 2000 Chars)
          [ Syntax-Preserved Document Chunks ]
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
[ BM25 Sparse Index ]      [ Resilient Embedding Engine ]
                                       │ (Safe Micro-Batching & Backoff)
                                       ▼
                             [ ChromaDB Vector Store ]
```

### Query Pre-Processing & LangGraph RAG Workflow

```
User Query (English / Hinglish / Colloquial)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ 🧠 Node 1: Query Normalizer & Decomposer (LangGraph)        │
│ • Detects Language & Tone (e.g. Hinglish -> Technical)      │
│ • Pydantic Structured Output: Context-Resolved Query        │
│ • Decomposes into Sub-Queries with Target Focus:             │
│   ['schema_definition', 'implementation_logic', 'usage_example']
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 🔍 Node 2: Hybrid Retrieval & Context Fusion                │
│ • Multi-Query Dense Vector Search (ChromaDB)                │
│ • Multi-Query Sparse Keyword Search (BM25)                  │
│ • SHA-256 Document Hash Deduplication                      │
│ • Cross-Encoder Re-Ranking (FlashRank ms-marco-MiniLM)      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ ⚡ Node 3: Grounded Answer Generation & Token Streaming     │
│ • Grounded LLM Response with Direct Source Citations        │
│ • Real-Time Token Streaming in UI (Streamlit)               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🌟 Key Features & Innovations

### 1. 🌐 Multilingual & Hinglish Query Normalization (Phase 1)
- **Eliminates Semantic Mismatch**: Translates colloquial Hinglish (e.g., *"bhai email me pdf attach kaise kare aur background me send kaise kare"*) into precise technical search queries.
- **Pydantic Structured Output**: Decomposes multi-intent questions into structured sub-queries with explicit focus targets (`schema_definition`, `implementation_logic`, `usage_example`).
- **LangGraph State Pipeline**: Orchestrates query normalization, hybrid retrieval, context deduplication, and generation via compiled state graphs.

### 2. 📓 Jupyter Notebook (`.ipynb`) Ingestion & Parsing
- Cleanly parses notebook JSON structures, extracting executable code and markdown explanations while stripping bulky execution outputs, base64 images, and metadata.
- Preserves complete data science, machine learning, and math routines (e.g. cosine similarity calculations, model training scripts).

### 3. 🚀 High-Speed Ingestion for Large Repositories
- **Git Tree API Crawling**: Fetches the entire repository hierarchy in a single API call instead of recursive directory crawling.
- **Multithreaded Fetching**: Ingests source code, documentation, and issues concurrently using `ThreadPoolExecutor`.
- Supports 30+ programming languages (`.py`, `.ipynb`, `.js`, `.ts`, `.tsx`, `.java`, `.cpp`, `.rs`, `.go`, `.sql`, etc.).

### 4. 🛡️ Resilient Embedding Engine (Anti-429 Quota Guard)
- **Exponential Backoff & Jitter**: Automatically catches `429 RESOURCE_EXHAUSTED` errors and retries gracefully without pipeline failure.
- **Adaptive Micro-Batching**: Processes embeddings in rate-limited batches to comfortably remain within Gemini API quotas.

### 5. 🔬 Hybrid Retrieval & Cross-Encoder Re-ranking
- **Dense + Sparse Search**: Combines semantic embeddings (ChromaDB) with lexical keyword matching (BM25).
- **Deterministic Deduplication**: Eliminates redundant context chunks across sub-queries using SHA-256 chunk hashing.
- **FlashRank Re-ranking**: Re-ranks the deduplicated candidate pool with a cross-encoder model to return only the most relevant snippets.

---

## 📂 Project Structure

```
├── data/                       # Local vector database storage (ChromaDB)
├── src/
│   ├── __init__.py
│   ├── config.py               # Centralized environment configs & API keys
│   ├── models.py               # Pydantic state & sub-query output schemas
│   ├── query_normalizer.py     # Hinglish normalization & query decomposition
│   ├── ingestion.py            # Git Tree API, notebook parser & file extractor
│   ├── chunker.py              # Language-aware & Markdown AST text splitters
│   ├── vectorstore.py          # Resilient ChromaDB embedding & persistence logic
│   ├── retriever.py            # Hybrid search (BM25 + Vector + FlashRank)
│   ├── graph.py                # LangGraph state graph compilation & execution
│   └── chain.py                # RAG pipeline interface & response generator
├── .env.example                # Template for API keys
├── .gitignore                  # Prevents committing local DB and secrets
├── requirements.txt            # Production dependencies
└── app.py                      # Streamlit interactive frontend
```

---

## 🛠️ Quick Start

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/PushpakBajanghate/github-rag-engine.git
cd github-rag-engine
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file from `.env.example`:
```bash
GOOGLE_API_KEY="your_gemini_api_key"
GITHUB_TOKEN="your_optional_github_token"
```

### 3. Launch the Application
```bash
streamlit run app.py
```

