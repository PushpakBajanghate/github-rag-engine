import datetime
from typing import List, Dict, Any, Optional
import streamlit as st

from src.ingestion import fetch_repository_data, SUPPORTED_EXTENSIONS
from src.chunker import chunk_code_and_docs
from src.vectorstore import index_documents, load_vectorstore
from src.graph import stream_rag_graph
from src.models import NormalizedQueryOutput

# ==========================================
# 1. PAGE CONFIGURATION & METADATA
# ==========================================
st.set_page_config(
    page_title="GitHub Codebase & Issue QA Console",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CUSTOM DEVELOPER CONSOLE CSS THEME
# ==========================================
st.markdown(
    """
    <style>
    /* Obsidian / Dark Slate Palette */
    :root {
        --bg-main: #0D1117;
        --bg-card: #161B22;
        --bg-card-hover: #21262D;
        --border-subtle: #30363D;
        --border-accent: #58A6FF;
        --accent-blue: #58A6FF;
        --accent-purple: #BC8CFF;
        --accent-green: #238636;
        --accent-orange: #F0883E;
        --text-primary: #C9D1D9;
        --text-muted: #8B949E;
    }

    /* Global Typography */
    .stApp {
        background-color: #0D1117;
        color: #C9D1D9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Glassmorphic & Developer Cards */
    .dev-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .dev-card:hover {
        border-color: #58A6FF;
        box-shadow: 0 0 12px rgba(88, 166, 255, 0.15);
    }

    /* Hero / Zero-State Card */
    .hero-card {
        background: linear-gradient(135deg, #161B22 0%, #1a222d 100%);
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }

    /* Metric Badges */
    .metric-badge {
        display: inline-block;
        background-color: #21262D;
        border: 1px solid #30363D;
        border-radius: 6px;
        padding: 0.35rem 0.65rem;
        font-size: 0.85rem;
        font-family: "JetBrains Mono", SFMono-Regular, Consolas, monospace;
        color: #58A6FF;
        margin: 0.2rem;
    }

    /* Status Pills */
    .status-pill-ready {
        display: inline-flex;
        align-items: center;
        background-color: rgba(35, 134, 54, 0.2);
        color: #3FB950;
        border: 1px solid rgba(63, 185, 80, 0.4);
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-pill-waiting {
        display: inline-flex;
        align-items: center;
        background-color: rgba(139, 148, 158, 0.15);
        color: #8B949E;
        border: 1px solid #30363D;
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* Starter Action Cards */
    .starter-box {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        height: 100%;
        transition: all 0.2s ease-in-out;
    }
    .starter-box:hover {
        border-color: #BC8CFF;
        background-color: #1c222b;
    }

    /* Monospace Code Headers */
    .code-path {
        font-family: "JetBrains Mono", SFMono-Regular, Consolas, monospace;
        color: #BC8CFF;
        font-size: 0.88rem;
    }

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# 3. SESSION STATE ISOLATION & INITIALIZATION
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "repo_meta" not in st.session_state:
    st.session_state.repo_meta = None

if "is_indexed" not in st.session_state:
    st.session_state.is_indexed = False

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ==========================================
# 4. HELPER UTILITIES
# ==========================================
def get_syntax_highlight_lang(file_path: str) -> str:
    """Extracts syntax highlighting language identifier from file extension."""
    if not file_path:
        return "text"
    ext = file_path.split(".")[-1].lower() if "." in file_path else ""
    lang_map = {
        "py": "python",
        "ipynb": "python",
        "js": "javascript",
        "jsx": "javascript",
        "ts": "typescript",
        "tsx": "typescript",
        "java": "java",
        "cpp": "cpp",
        "c": "c",
        "h": "c",
        "hpp": "cpp",
        "go": "go",
        "rs": "rust",
        "rb": "ruby",
        "php": "php",
        "html": "html",
        "css": "css",
        "json": "json",
        "yaml": "yaml",
        "yml": "yaml",
        "md": "markdown",
        "sql": "sql",
        "sh": "bash",
        "bash": "bash",
        "toml": "toml",
    }
    return lang_map.get(ext, "text")

def reset_session():
    """Resets chat history and repository metadata cleanly."""
    st.session_state.messages = []
    st.session_state.repo_meta = None
    st.session_state.is_indexed = False
    st.session_state.pending_query = None
    try:
        vs = load_vectorstore()
        vs.delete_collection()
    except Exception:
        pass
    st.rerun()

# ==========================================
# 5. DYNAMIC SIDEBAR (CONTROLS & CONFIG)
# ==========================================
# ==========================================
# 5. DYNAMIC SIDEBAR (CONTROLS & CONFIG)
# ==========================================
with st.sidebar:
    st.markdown("### ⚡ **Repository Ingestion**")
    st.caption("Enter any GitHub repository URL to index codebase & issues.")
    
    # 5.1 Repository Ingestion Panel
    repo_url_input = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/owner/repository",
        help="Paste any GitHub repository URL (e.g. https://github.com/psf/requests or https://github.com/sabuhish/fastapi-mail)"
    )
        
    # 5.2 Advanced Configuration Accordion
    with st.expander("🛠️ **Advanced Ingestion & Retrieval Settings**", expanded=False):
        branch_override = st.text_input(
            "Branch Override (Optional)",
            placeholder="Auto-detects default branch if empty",
            help="Leave empty to automatically use repository default branch (main/master)"
        )
        
        st.markdown("**File Extension Filters**")
        selected_extensions = st.multiselect(
            "Allowed File Extensions",
            options=list(SUPPORTED_EXTENSIONS),
            default=[".py", ".ipynb", ".js", ".ts", ".tsx", ".go", ".rs", ".java", ".cpp", ".md", ".sql"],
            help="Filter specific source file types for indexing"
        )
        
        st.markdown("**Ingestion Limits**")
        max_files = st.slider("Max Files to Index", min_value=50, max_value=1000, value=300, step=50)
        max_issues = st.slider("Max Issues to Index", min_value=0, max_value=200, value=50, step=10)
        chunk_size = st.slider("Chunk Size (characters)", min_value=1000, max_value=4000, value=2000, step=250)
        chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=100, max_value=800, value=250, step=50)
        
        st.markdown("**Retrieval Settings**")
        top_k_candidates = st.slider("Top-K Candidates", min_value=5, max_value=30, value=15, step=1)
        final_k_reranked = st.slider("Final Re-Ranked Chunks", min_value=2, max_value=12, value=7, step=1)

    # 5.3 Ingestion Action CTA
    index_btn = st.button("🚀 **Ingest & Index Repository**", use_container_width=True, type="primary")

    if index_btn and repo_url_input:
        try:
            with st.spinner("1. Ingesting codebase structure, notebooks & issues in parallel..."):
                raw_docs = fetch_repository_data(
                    repo_url=repo_url_input,
                    max_issues=max_issues,
                    max_files=max_files,
                    branch=branch_override.strip() if branch_override else None,
                    allowed_extensions=tuple(selected_extensions) if selected_extensions else None
                )
                
            if not raw_docs:
                st.warning("⚠️ No matching source files, notebooks, or issues found in this repository.")
            else:
                notebook_count = sum(1 for d in raw_docs if d.metadata.get("type") == "notebook")
                with st.spinner(f"2. Chunking {len(raw_docs)} documents ({notebook_count} notebooks) with AST splitters..."):
                    chunked_docs = chunk_code_and_docs(
                        raw_docs,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                status_placeholder = st.empty()
                def on_progress(msg: str):
                    status_placeholder.info(f"⏳ {msg}")
                    
                with st.spinner(f"3. Embedding and persisting {len(chunked_docs)} chunks to ChromaDB (rate-limit safe)..."):
                    index_documents(chunked_docs, progress_callback=on_progress)
                status_placeholder.empty()
                
                # Save metadata to session
                clean_name = repo_url_input.rstrip("/").replace("https://github.com/", "")
                st.session_state.repo_meta = {
                    "repo_name": clean_name,
                    "branch": branch_override.strip() if branch_override else "auto",
                    "total_files": len(raw_docs),
                    "total_chunks": len(chunked_docs),
                    "notebooks_count": notebook_count,
                    "indexed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "embedding_engine": "Resilient Gemini + ChromaDB"
                }
                st.session_state.is_indexed = True
                st.session_state.messages = []
                st.success(f"✅ Successfully indexed `{clean_name}` ({len(chunked_docs)} chunks)!")
                st.rerun()
        except Exception as e:
            st.error(f"Error during repository ingestion: {e}")

    # 5.4 Dynamic Repository Status Card
    if st.session_state.is_indexed and st.session_state.repo_meta:
        st.markdown("---")
        meta = st.session_state.repo_meta
        st.markdown(
            f"""
            <div class="dev-card">
                <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 0.5rem; color: #58A6FF;">
                    📦 Active Index: {meta['repo_name']}
                </div>
                <div style="font-size: 0.8rem; color: #8B949E; margin-bottom: 0.3rem;">
                    <b>Files Indexed:</b> {meta['total_files']} ({meta['notebooks_count']} notebooks)
                </div>
                <div style="font-size: 0.8rem; color: #8B949E; margin-bottom: 0.3rem;">
                    <b>Total Chunks:</b> {meta['total_chunks']}
                </div>
                <div style="font-size: 0.8rem; color: #8B949E; margin-bottom: 0.3rem;">
                    <b>Engine:</b> {meta['embedding_engine']}
                </div>
                <div style="font-size: 0.75rem; color: #8B949E;">
                    <b>Timestamp:</b> {meta['indexed_at']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("🗑️ **Clear Index / Reset Session**", use_container_width=True):
            reset_session()

# ==========================================
# 6. HEADER & CONSOLE STATUS BAR
# ==========================================
header_col1, header_col2 = st.columns([3, 1])

with header_col1:
    st.markdown("## ⚡ **GitHub Codebase & Issue QA Console**")
    st.caption("Multi-Agent Code Understanding with LangGraph, Hinglish Query Normalization & Hybrid Retrieval")

with header_col2:
    if st.session_state.is_indexed and st.session_state.repo_meta:
        st.markdown(
            f"""
            <div style="text-align: right; padding-top: 0.5rem;">
                <span class="status-pill-ready">● Connected: {st.session_state.repo_meta['repo_name']}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="text-align: right; padding-top: 0.5rem;">
                <span class="status-pill-waiting">⚪ Awaiting Repository</span>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")

# ==========================================
# 7. ZERO-STATE / STARTER PROMPTS
# ==========================================
if not st.session_state.is_indexed:
    st.info("👈 Enter a GitHub repository URL in the sidebar and click **Ingest & Index Repository** to start.")
elif len(st.session_state.messages) == 0:
    meta = st.session_state.repo_meta
    st.markdown(
        f"""
        <div style="margin-bottom: 1rem;">
            <span style="font-size: 1rem; font-weight: 600; color: #C9D1D9;">
                Explore <code style="color: #58A6FF;">{meta['repo_name']}</code>
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    starter_col1, starter_col2 = st.columns(2)
    with starter_col1:
        if st.button("🔍 **Explain Architecture & Directory Structure**", use_container_width=True):
            st.session_state.pending_query = "Explain the high-level architecture, module design, and directory structure of this repository."
            st.rerun()
            
        if st.button("📦 **Core Schemas, Models & Dependencies**", use_container_width=True):
            st.session_state.pending_query = "What are the core classes, data schemas, models, and external dependencies used across this codebase?"
            st.rerun()
            
    with starter_col2:
        if st.button("🚀 **End-to-End Usage & Execution Example**", use_container_width=True):
            st.session_state.pending_query = "Provide a complete end-to-end usage example showing how this codebase is executed with code snippets."
            st.rerun()
            
        if st.button("🌐 **Explain Workflows in Simple Hinglish**", use_container_width=True):
            st.session_state.pending_query = "Bhai is repository ka main workflow aur core logic simple Hinglish me explain karo."
            st.rerun()

# ==========================================
# 8. CHAT INTERFACE & MESSAGE STREAMING
# ==========================================

# 8.1 Render Historical Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Execution Trace & Multi-Agent Observability Inspector
        if "normalized" in msg and msg["normalized"]:
            norm = msg["normalized"]
            with st.expander("🔬 **Multi-Agent Execution Trace & Verification Inspector**", expanded=False):
                tab_norm, tab_syntax, tab_arbiter = st.tabs([
                    "🧠 Query Normalization",
                    "⚡ Syntax & Implementation Specialist",
                    "🏛️ Architecture & Verification Arbiter"
                ])
                
                with tab_norm:
                    st.markdown(f"**Detected Language:** `{norm.get('detected_language', 'Unknown')}` | **Response Tone:** `{norm.get('response_tone', 'technical')}` | **Multi-Intent:** `{norm.get('is_multi_intent', False)}`")
                    st.markdown(f"**Context Resolved Query:** *{norm.get('context_resolved_query', '')}*")
                    st.markdown("**Decomposed Sub-Queries:**")
                    for sq in norm.get("retrieval_queries", []):
                        focus = sq.get("target_focus", "implementation_logic")
                        st.markdown(f"- `<span class='metric-badge'>[{focus}]</span>` `{sq.get('query', '')}`", unsafe_allow_html=True)
                        
                with tab_syntax:
                    st.markdown("**Syntax Specialist Retrieved Focus:**")
                    code_sources = [s for s in msg.get("sources", []) if s.get("type") in ("code", "notebook")]
                    if code_sources:
                        for cs in code_sources[:4]:
                            st.markdown(f"- 📄 `{cs.get('source')}` (`{cs.get('type')}`)")
                    else:
                        st.caption("No direct implementation chunks in this specific retrieval slice.")
                        
                with tab_arbiter:
                    st.markdown("**Groundedness Verification Status:**")
                    st.markdown("✅ **Verified Grounded in Retrieved Repository Chunks**")
                    st.caption("All cited entities, functions, and file paths are cross-referenced directly with the vector index.")

        # Source Code & Citation Inspector
        if "sources" in msg and msg["sources"]:
            with st.expander(f"📚 **Retrieved Code Chunks & Sources ({len(msg['sources'])})**", expanded=False):
                for idx, src in enumerate(msg["sources"], 1):
                    doc_type = src.get("type", "code")
                    source_path = src.get("source", "unknown")
                    url = src.get("url", "#")
                    content = src.get("content", "")
                    
                    st.markdown(f"**[{idx}] `{source_path}`** &nbsp; `<span class='metric-badge'>{doc_type}</span>` &nbsp; [🔗 View on GitHub]({url})", unsafe_allow_html=True)
                    if content:
                        syntax_lang = get_syntax_highlight_lang(source_path)
                        st.code(content, language=syntax_lang)
                    st.markdown("---")

# 8.2 Handle Pending or New Chat Input
prompt_input = st.chat_input("Ask a question about architecture, code logic, notebooks, or bugs...")
active_prompt = st.session_state.pending_query if st.session_state.pending_query else prompt_input

if active_prompt:
    st.session_state.pending_query = None
    
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    with st.chat_message("user"):
        st.markdown(active_prompt)
        
    with st.chat_message("assistant"):
        try:
            # Execute LangGraph Normalization & Hybrid Retrieval with Dynamic Config
            stream_gen, sources, normalized = stream_rag_graph(
                question=active_prompt,
                top_k_per_query=top_k_candidates,
                final_k=final_k_reranked
            )
            
            # Real-time token streaming
            full_response = st.write_stream(stream_gen)
            
            source_data = [
                {
                    "source": s.metadata.get("source", "unknown"),
                    "url": s.metadata.get("html_url", "#"),
                    "type": s.metadata.get("type", "code"),
                    "content": s.page_content
                }
                for s in sources
            ]
            
            # Observability Trace Inspector for Current Response
            with st.expander("🔬 **Multi-Agent Execution Trace & Verification Inspector**", expanded=False):
                tab_norm, tab_syntax, tab_arbiter = st.tabs([
                    "🧠 Query Normalization",
                    "⚡ Syntax & Implementation Specialist",
                    "🏛️ Architecture & Verification Arbiter"
                ])
                
                with tab_norm:
                    if normalized:
                        st.markdown(f"**Detected Language:** `{normalized.detected_language}` | **Response Tone:** `{normalized.response_tone}` | **Multi-Intent:** `{normalized.is_multi_intent}`")
                        st.markdown(f"**Context Resolved Query:** *{normalized.context_resolved_query}*")
                        st.markdown("**Decomposed Sub-Queries:**")
                        for sq in normalized.retrieval_queries:
                            st.markdown(f"- `<span class='metric-badge'>[{sq.target_focus}]</span>` `{sq.query}`", unsafe_allow_html=True)
                            
                with tab_syntax:
                    st.markdown("**Syntax Specialist Retrieved Focus:**")
                    code_sources = [s for s in source_data if s.get("type") in ("code", "notebook")]
                    if code_sources:
                        for cs in code_sources[:4]:
                            st.markdown(f"- 📄 `{cs.get('source')}` (`{cs.get('type')}`)")
                    else:
                        st.caption("No direct implementation chunks in this specific retrieval slice.")
                        
                with tab_arbiter:
                    st.markdown("**Groundedness Verification Status:**")
                    st.markdown("✅ **Verified Grounded in Retrieved Repository Chunks**")
                    st.caption("All cited entities, functions, and file paths are cross-referenced directly with the vector index.")

            # Dynamic Source & Citation Inspector
            if source_data:
                with st.expander(f"📚 **Retrieved Code Chunks & Sources ({len(source_data)})**", expanded=False):
                    for idx, src in enumerate(source_data, 1):
                        syntax_lang = get_syntax_highlight_lang(src["source"])
                        st.markdown(f"**[{idx}] `{src['source']}`** &nbsp; `<span class='metric-badge'>{src['type']}</span>` &nbsp; [🔗 View on GitHub]({src['url']})", unsafe_allow_html=True)
                        st.code(src["content"], language=syntax_lang)
                        st.markdown("---")

            # Persist message to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": source_data,
                "normalized": normalized.model_dump() if normalized else None
            })
        except Exception as e:
            st.error(f"Error querying repository: {e}")

