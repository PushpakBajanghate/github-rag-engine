"""Ultra-modern Developer Dashboard for GitHub RAG Engine."""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

from src.config import settings
from src.ingestion import GitHubIngestion
from src.chunker import CodeAwareChunker
from src.vectorstore import VectorStoreManager
from src.retriever import TwoStageRetriever
from src.chain import RAGChain

# Page Configuration
st.set_page_config(
    page_title="GitHub RAG Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling (Dark Modern Glassmorphism)
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    code, pre {
        font-family: 'Fira Code', monospace !important;
    }
    
    /* Header Gradient & Badge */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 14px 18px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.3);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Provider Status Pills */
    .pill-gemini {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .pill-openai {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #60a5fa;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Quick Query Pill Button */
    .stButton button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    
    /* Primary action button */
    div[data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
        border: none !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.39) !important;
    }
    div[data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #4338ca 100%) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5) !important;
        transform: translateY(-1px);
    }
    
    /* Citation Box */
    .citation-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Detect active provider
active_prov = settings.active_provider
has_gemini = bool(settings.google_api_key.strip())
has_openai = bool(settings.openai_api_key.strip())

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_repo" not in st.session_state:
    st.session_state.indexed_repo = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "repo_stats" not in st.session_state:
    st.session_state.repo_stats = None

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Engine Control")

    # Status Pill
    if has_gemini:
        st.markdown('<div class="pill-gemini">🟢 Google Gemini Active</div>', unsafe_allow_html=True)
    elif has_openai:
        st.markdown('<div class="pill-openai">🟢 OpenAI Active</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ No API Key in .env (Add GOOGLE_API_KEY)")

    st.write("")

    # Provider Selection
    provider_options = []
    if has_gemini:
        provider_options.append("Google Gemini")
    if has_openai:
        provider_options.append("OpenAI")
    if not provider_options:
        provider_options = ["Google Gemini", "OpenAI"]

    selected_provider_label = st.selectbox("LLM Provider", options=provider_options, index=0)
    provider_key = "gemini" if "Gemini" in selected_provider_label else "openai"

    st.divider()
    st.markdown("#### 📦 Repository Source")

    repo_input = st.text_input(
        "GitHub Repository",
        value=st.session_state.indexed_repo or "PushpakBajanghate/github-rag-engine",
        placeholder="e.g. owner/repo or full GitHub URL",
        help="Paste a public or private GitHub repository URL or owner/repo format."
    )

    # Quick Samples
    st.caption("Quick Load:")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("github-rag-engine", use_container_width=True):
            repo_input = "PushpakBajanghate/github-rag-engine"
    with col_s2:
        if st.button("pallets/flask", use_container_width=True):
            repo_input = "pallets/flask"

    st.write("")
    
    github_token = st.text_input(
        "GitHub Token (PAT)",
        value=settings.github_token,
        type="password",
        help="Prefilled from .env. Required for private repos and high API rate limits."
    )

    st.divider()
    st.markdown("#### ⚡ Ingestion Settings")
    max_files = st.slider("Max Files to Ingest", min_value=10, max_value=300, value=100, step=10)
    include_issues = st.checkbox("Include GitHub Issues & PRs", value=True)

    st.write("")
    ingest_btn = st.button("🚀 Ingest & Index Repository", type="primary", use_container_width=True)

    if st.session_state.indexed_repo:
        st.write("")
        if st.button("🧹 Clear Current Index", use_container_width=True):
            try:
                vsm = VectorStoreManager(provider=provider_key)
                vsm.clear_collection(st.session_state.indexed_repo)
                st.session_state.indexed_repo = None
                st.session_state.rag_chain = None
                st.session_state.messages = []
                st.session_state.repo_stats = None
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error clearing collection: {e}")

# Main Header / Hero Section
st.markdown("""
<div class="hero-container">
    <div class="hero-title">⚡ GitHub RAG Engine</div>
    <div class="hero-subtitle">Language-aware code chunking & two-stage neural retrieval for GitHub codebases</div>
</div>
""", unsafe_allow_html=True)

# Metrics Strip if indexed
if st.session_state.repo_stats:
    stats = st.session_state.repo_stats
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{stats["files"]}</div><div class="metric-label">Files Indexed</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{stats["chunks"]}</div><div class="metric-label">Code Chunks</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{stats["issues"]}</div><div class="metric-label">Issues Loaded</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">FlashRank</div><div class="metric-label">Reranker Engine</div></div>', unsafe_allow_html=True)
    st.write("")

# Ingestion Flow
if ingest_btn:
    if not repo_input.strip():
        st.sidebar.error("⚠️ Please specify a GitHub repository name or URL.")
    else:
        with st.status("🚀 Processing repository pipeline...", expanded=True) as status:
            try:
                st.write("📡 Step 1/4: Connecting to GitHub API & extracting files...")
                ingestion = GitHubIngestion(token=github_token)
                docs = ingestion.fetch_repo_files(repo_input, max_files=max_files)
                
                issue_count = 0
                if include_issues:
                    try:
                        st.write("📋 Step 1b: Fetching repository issues & discussions...")
                        issue_docs = ingestion.fetch_repo_issues(repo_input)
                        issue_count = len(issue_docs)
                        docs.extend(issue_docs)
                    except Exception as ie:
                        st.write(f"ℹ️ Note: Issues skipped ({ie})")

                st.write(f"✂️ Step 2/4: Applying AST language-aware chunking on {len(docs)} documents...")
                chunker = CodeAwareChunker()
                chunks = chunker.split_documents(docs)

                total_chunks = len(chunks)
                st.write(f"🧠 Step 3/4: Generating vector embeddings for {total_chunks} chunks via {selected_provider_label}...")
                st.caption("⚠️ Gemini free tier: Batching 5 docs at a time with 12s pauses between batches. Large repos may take a few minutes.")
                vsm = VectorStoreManager(provider=provider_key)
                vectorstore = vsm.create_or_update_vectorstore(
                    repo_input,
                    chunks,
                    progress_callback=lambda msg: st.write(msg)
                )

                st.write("🎯 Step 4/4: Initializing Two-Stage Retriever (ChromaDB + FlashRank)...")
                retriever = TwoStageRetriever(vectorstore=vectorstore)
                chain = RAGChain(retriever=retriever, provider=provider_key)

                # Update session state
                st.session_state.rag_chain = chain
                st.session_state.indexed_repo = repo_input
                st.session_state.messages = []
                st.session_state.repo_stats = {
                    "files": len(docs) - issue_count,
                    "chunks": len(chunks),
                    "issues": issue_count
                }

                status.update(label=f"✅ Successfully indexed {repo_input}!", state="complete", expanded=False)
                st.rerun()
            except Exception as e:
                status.update(label="❌ Ingestion failed", state="error", expanded=True)
                st.error(f"Error during ingestion: {str(e)}")

# Q&A Interface Section
st.subheader("💬 Codebase Intelligence & Q&A")

if not st.session_state.indexed_repo or not st.session_state.rag_chain:
    st.info("👈 Enter a repository in the sidebar and click **'🚀 Ingest & Index Repository'** to start querying.")
    
    # Showcase Cards
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        st.markdown("### 🌲 AST Chunking")
        st.caption("Splits code respecting function, class, and grammar boundaries across 20+ programming languages.")
    with col_f2:
        st.markdown("### ⚡ Two-Stage Retrieval")
        st.caption("Combines fast dense vector search with ultra-accurate FlashRank neural cross-encoder reranking.")
    with col_f3:
        st.markdown("### 🔗 Line Citations")
        st.caption("Every answer includes exact file paths, line numbers, relevance scores, and direct GitHub links.")

else:
    st.caption(f"Currently querying indexed repo: **`{st.session_state.indexed_repo}`**")

    # Sample Quick Prompts
    st.markdown("**Quick Prompts:**")
    qp1, qp2, qp3, qp4 = st.columns(4)
    quick_query = None
    with qp1:
        if st.button("🏗️ Explain Architecture", use_container_width=True):
            quick_query = "Explain the overall architecture and main components of this codebase."
    with qp2:
        if st.button("🔍 Ingestion Workflow", use_container_width=True):
            quick_query = "How does the ingestion and data processing pipeline work in this project?"
    with qp3:
        if st.button("🎯 Retriever Logic", use_container_width=True):
            quick_query = "Explain how the two-stage retrieval and FlashRank reranking are implemented."
    with qp4:
        if st.button("📦 Key Dependencies", use_container_width=True):
            quick_query = "What are the core dependencies and configuration parameters of this repository?"

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                with st.expander("📑 Sources & Code References", expanded=False):
                    for c in msg["citations"]:
                        score_text = f" • Match Score: `{c['score']:.3f}`" if c.get("score") is not None else ""
                        url = c.get("html_url")
                        source_label = f"[{c['source']}]({url})" if url else f"**{c['source']}**"
                        st.markdown(f"🔹 {source_label} ({c.get('type', 'code')}){score_text}")
                        st.code(c.get("preview", ""), language="text")

    # User Query Input
    chat_prompt = st.chat_input("Ask any question about functions, classes, bugs, architecture...")
    active_prompt = quick_query or chat_prompt

    if active_prompt:
        st.session_state.messages.append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing codebase & synthesizing response..."):
                try:
                    result = st.session_state.rag_chain.answer_question(active_prompt)
                    answer = result["answer"]
                    citations = result["citations"]

                    st.markdown(answer)
                    if citations:
                        with st.expander("📑 Sources & Code References", expanded=False):
                            for c in citations:
                                score_text = f" • Match Score: `{c['score']:.3f}`" if c.get("score") is not None else ""
                                url = c.get("html_url")
                                source_label = f"[{c['source']}]({url})" if url else f"**{c['source']}**"
                                st.markdown(f"🔹 {source_label} ({c.get('type', 'code')}){score_text}")
                                st.code(c.get("preview", ""), language="text")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations
                    })
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
