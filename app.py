import streamlit as st
from src.ingestion import fetch_repository_data
from src.chunker import chunk_code_and_docs
from src.vectorstore import index_documents
from src.chain import ask_repo

st.set_page_config(page_title="GitHub Code & Issue RAG Engine", page_icon="🔍", layout="wide")
st.title("🔍 GitHub Codebase & Issue QA Engine")

with st.sidebar:
    st.header("⚙️ Repository Configuration")
    repo_url = st.text_input("GitHub Repo URL", placeholder="https://github.com/psf/requests")
    index_btn = st.button("🚀 Ingest & Index Repository")
    
    if index_btn and repo_url:
        with st.spinner("1. Fetching repository files and issues..."):
            raw_docs = fetch_repository_data(repo_url)
        with st.spinner(f"2. Chunking {len(raw_docs)} documents with AST splitters..."):
            chunked_docs = chunk_code_and_docs(raw_docs)
        with st.spinner(f"3. Embedding and persisting {len(chunked_docs)} chunks to ChromaDB..."):
            index_documents(chunked_docs)
        st.success(f"Successfully indexed {len(chunked_docs)} chunks from `{repo_url}`!")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📚 View Retrieved Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- **[{src['type']}]** [{src['source']}]({src['url']})")

if prompt := st.chat_input("Ask a question about the repository or its past bugs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Searching codebase & generating grounded answer..."):
            answer, sources = ask_repo(prompt)
            st.markdown(answer)
            
            source_data = [
                {"source": s.metadata.get("source"), "url": s.metadata.get("html_url"), "type": s.metadata.get("type")}
                for s in sources
            ]
            
            with st.expander("📚 View Retrieved Sources"):
                for s in source_data:
                    st.markdown(f"- **[{s['type']}]** [{s['source']}]({s['url']})")
                    
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": source_data
            })