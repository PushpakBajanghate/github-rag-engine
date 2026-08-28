# 📜 Architectural Decisions & Error Resolution Log (`decision.md`)

This living document tracks all major engineering challenges, production errors, architectural decisions, and permanent fixes implemented across the lifecycle of the **GitHub RAG Engine** project.

---

## 📑 Table of Contents
1. [DEC-001: Jupyter Notebook (.ipynb) Extraction & AST Chunking](#dec-001-jupyter-notebook-ipynb-extraction--ast-chunking)
2. [DEC-002: Scalable Ingestion via Git Tree API & ThreadPool](#dec-002-scalable-ingestion-via-git-tree-api--threadpool)
3. [DEC-003: Multi-Intent Hinglish & Colloquial Query Normalization](#dec-003-multi-intent-hinglish--colloquial-query-normalization)
4. [DEC-004: Hybrid Retrieval with BM25, SHA-256 Deduplication & FlashRank](#dec-004-hybrid-retrieval-with-bm25-sha-256-deduplication--flashrank)
5. [DEC-005: 404 Ingestion Error on Non-Main Default Branches](#dec-005-404-ingestion-error-on-non-main-default-branches)
6. [DEC-006: Ingestion Noise & Token Burn on Lockfiles (package-lock.json)](#dec-006-ingestion-noise--token-burn-on-lockfiles-package-lockjson)
7. [DEC-007: Anti-Hallucination & Strict Negative Assertion Guardrails](#dec-007-anti-hallucination--strict-negative-assertion-guardrails)
8. [DEC-008: Embedding Model 404 NOT_FOUND & Zero-Quota FastEmbed Fallback](#dec-008-embedding-model-404-not_found--zero-quota-fastembed-fallback)
9. [DEC-009: Streamlit Developer Console & Dynamic UI State Isolation](#dec-009-streamlit-developer-console--dynamic-ui-state-isolation)

---

### DEC-001: Jupyter Notebook (`.ipynb`) Extraction & AST Chunking
* **Error / Challenge:**
  Jupyter notebooks contain complex JSON structures with heavy execution outputs, base64 images, and noisy metadata. Raw text ingestion resulted in severe token pollution and broken code representations, causing the RAG pipeline to omit critical machine learning and data analysis logic.
* **Fix / Decision Taken:**
  Built a dedicated `parse_jupyter_notebook()` function in `src/ingestion.py` and mapped `.ipynb` files to Python AST splitters in `src/chunker.py`.
* **Brief Description:**
  The extractor iterates over notebook cells, separates `# [Notebook Cell X - Code]` from `<!-- [Notebook Cell X - Markdown] -->`, strips all execution payloads/base64 strings, and feeds pure readable Python code into AST recursive splitters (2,000 characters, 250 overlap).

---

### DEC-002: Scalable Ingestion via Git Tree API & ThreadPool
* **Error / Challenge:**
  Recursive directory traversal using standard GitHub Contents API caused rate-limiting bottlenecks and high latency on large repositories with deep folder hierarchies.
* **Fix / Decision Taken:**
  Switched to GitHub's Git Tree API (`repo.get_git_tree(recursive=True)`) combined with a `ThreadPoolExecutor(max_workers=10)` parallel fetch pipeline.
* **Brief Description:**
  The full repository tree is retrieved in a single API call, filtered against supported extensions, and source files are downloaded concurrently, improving ingestion throughput by over **8x**.

---

### DEC-003: Multi-Intent Hinglish & Colloquial Query Normalization
* **Error / Challenge:**
  Semantic mismatch between informal user queries (e.g. *"bhai email me pdf attach kaise kare aur background me send kaise kare"*) and formal English codebases caused poor retrieval recall and missing context.
* **Fix / Decision Taken:**
  Implemented a LangGraph pre-processing node (`normalize_query_node`) powered by Gemini structured output using Pydantic schemas (`NormalizedQueryOutput`, `SubQuery`).
* **Brief Description:**
  Translates Hinglish queries into canonical English, detects language tone, and decomposes multi-intent prompts into targeted sub-queries (`schema_definition`, `implementation_logic`, `usage_example`).

---

### DEC-004: Hybrid Retrieval with BM25, SHA-256 Deduplication & FlashRank
* **Error / Challenge:**
  Dense vector search alone struggled with exact symbol lookups (e.g. function names, config keys, error codes), while multi-query expansion produced duplicate context chunks.
* **Fix / Decision Taken:**
  Architected a Hybrid Retrieval pipeline merging ChromaDB (Dense Vector) with `rank_bm25` (Sparse BM25), deterministic SHA-256 content deduplication, and FlashRank cross-encoder re-ranking (`ms-marco-MiniLM-L-12-v2`).
* **Brief Description:**
  For each sub-query, candidate documents from vector and lexical searches are pooled, duplicate snippets are purged via SHA-256 hashes, and the merged pool is re-ranked by FlashRank to select the top $K$ most relevant passages.

---

### DEC-005: 404 Ingestion Error on Non-Main Default Branches
* **Error / Challenge:**
  `404 NOT_FOUND {"message": "No commit found for the ref main"}` occurred when attempting to ingest repositories whose primary branch is `master`, `trunk`, or `develop` instead of `main`.
* **Fix / Decision Taken:**
  Implemented `resolve_repo_branch()` in `src/ingestion.py` to dynamically inspect and resolve the repository's true default branch from GitHub API metadata.
* **Brief Description:**
  The ingestion engine queries `repo.default_branch` and resolves the root commit SHA, falling back through candidate branches (`main`, `master`, `develop`) gracefully without crashing.

---

### DEC-006: Ingestion Noise & Token Burn on Lockfiles (`package-lock.json`)
* **Error / Challenge:**
  Indexing large JavaScript/TypeScript repositories crawled `package-lock.json` (150KB+), generating 80+ chunks of dependency trees. This consumed API quotas rapidly and diluted semantic search results.
* **Fix / Decision Taken:**
  Implemented strict file blacklists (`IGNORED_FILENAMES` and `IGNORED_FILE_SUFFIXES`) in `src/ingestion.py`.
* **Brief Description:**
  The system automatically ignores `package-lock.json`, `yarn.lock`, `pnpm-lock.yaml`, `poetry.lock`, `*.min.js`, `*.map`, `*.svg`, and test mocks, ensuring only valuable source code and docs are chunked.

---

### DEC-007: Anti-Hallucination & Strict Negative Assertion Guardrails
* **Error / Challenge:**
  When asked out-of-scope questions about non-existent technologies (e.g. *"What Redis cache clustering and Stripe billing webhooks are configured?"*), the LLM tended to extrapolate generic architectures.
* **Fix / Decision Taken:**
  Enforced strict **Negative Assertion Guidelines** within `SYSTEM_PROMPT` in `src/chain.py`.
* **Brief Description:**
  The system prompt strictly mandates: *"If a requested library, service, framework, endpoint, model, or configuration is NOT present in the retrieved context, you MUST explicitly state that it is NOT implemented in this codebase. NEVER invent or hallucinate non-existent features."*

---

### DEC-008: Embedding Model 404 NOT_FOUND & Zero-Quota FastEmbed Fallback
* **Error / Challenge:**
  - `404 NOT_FOUND: models/text-embedding-004 is not found for API version v1beta`.
  - Daily free-tier limit of 1,000 requests on Google Gemini Embeddings caused `429 RESOURCE_EXHAUSTED` crashes during repository indexing.
* **Fix / Decision Taken:**
  Created `HybridResilientEmbeddings` in `src/vectorstore.py` with local **FastEmbed** (`BAAI/bge-small-en-v1.5`) as the primary zero-cost engine and seamless fallback.
* **Brief Description:**
  FastEmbed runs lightweight quantized ONNX embeddings locally on device (~1,000 chunks/sec) with **zero API quota limits, zero network latency, and 100% uptime**, completely eliminating 404 and 429 errors.

---

### DEC-009: Streamlit Developer Console & Dynamic UI State Isolation
* **Error / Challenge:**
  Initial UI had hardcoded repository presets, cluttered hero banners, redundant input fields, and lacked execution trace observability.
* **Fix / Decision Taken:**
  Redesigned `app.py` into a modern Obsidian Dark Slate developer console with glassmorphic cards, clean sidebar controls, and multi-agent trace tabs.
* **Brief Description:**
  Added dynamic starter prompt cards, multi-agent observability inspector (`Query Normalization`, `Syntax Specialist`, `Architecture Arbiter`), and syntax-highlighted code chunk inspectors tagged by language.
