import streamlit as st
from src.ingestion import fetch_repository_data
from src.chunker import chunk_code_and_docs
from src.vectorstore import index_documents
from src.chain import ask_repo_stream

st.set_page_config(page_title="GitHub Code & Issue RAG Engine", page_icon="🔍", layout="wide")
st.title("🔍 GitHub Codebase & Issue QA Engine")

with st.sidebar:
    st.header("⚙️ Repository Configuration")
    repo_url = st.text_input("GitHub Repo URL", placeholder="https://github.com/psf/requests")
    
    with st.expander("🛠️ Advanced Ingestion Settings", expanded=False):
        max_files = st.slider("Max Files to Index", min_value=50, max_value=1000, value=300, step=50)
        max_issues = st.slider("Max Issues to Index", min_value=0, max_value=200, value=50, step=10)
        chunk_size = st.slider("Chunk Size (characters)", min_value=1000, max_value=4000, value=2000, step=250)
        chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=100, max_value=800, value=250, step=50)
        
    index_btn = st.button("🚀 Ingest & Index Repository", use_container_width=True)
    
    if index_btn and repo_url:
        try:
            with st.spinner("1. Fetching repository files, notebooks & issues in parallel..."):
                raw_docs = fetch_repository_data(repo_url, max_issues=max_issues, max_files=max_files)
                
            if not raw_docs:
                st.warning("No supported code files, notebooks, or issues found in this repository.")
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
                st.success(f"✅ Indexed {len(chunked_docs)} chunks from {len(raw_docs)} files (including {notebook_count} notebooks)!")
        except Exception as e:
            st.error(f"Error during ingestion/indexing: {e}")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 View Retrieved Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- **[{src.get('type', 'doc')}]** [{src.get('source', 'source')}]({src.get('url', '#')})")

if prompt := st.chat_input("Ask a question about code logic, algorithms, notebooks, or past bugs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        try:
            stream_gen, sources = ask_repo_stream(prompt)
            full_response = st.write_stream(stream_gen)
            
            source_data = [
                {
                    "source": s.metadata.get("source"),
                    "url": s.metadata.get("html_url"),
                    "type": s.metadata.get("type")
                }
                for s in sources
            ]
            
            if source_data:
                with st.expander("📚 View Retrieved Sources"):
                    for s in source_data:
                        st.markdown(f"- **[{s['type']}]** [{s['source']}]({s['url']})")
                        
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": source_data
            })
        except Exception as e:
            st.error(f"Error querying repository: {e}")