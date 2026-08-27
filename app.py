"""Streamlit frontend entrypoint for GitHub RAG Engine."""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.config import settings
from src.ingestion import GitHubIngestion
from src.chunker import CodeAwareChunker
from src.vectorstore import VectorStoreManager
from src.retriever import TwoStageRetriever
from src.chain import RAGChain

st.set_page_config(
    page_title="GitHub RAG Engine",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ GitHub RAG Engine")
st.caption("Two-stage retrieval and code intelligence for any GitHub repository")

with st.sidebar:
    st.header("⚙️ Settings")
    
    repo_input = st.text_input(
        "GitHub Repository",
        placeholder="e.g. owner/repo or full GitHub URL",
        value=""
    )
    
    github_token = st.text_input(
        "GitHub Token (Optional)",
        type="password",
        value=os.getenv("GITHUB_TOKEN", ""),
        help="Recommended for rate limits or private repos."
    )
    
    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Required for embeddings and LLM generation."
    )
    
    st.divider()
    
    max_files = st.slider("Max Files to Ingest", min_value=10, max_value=300, value=100)
    include_issues = st.checkbox("Include Open Issues", value=True)
    
    ingest_btn = st.button("📥 Ingest & Index Repository", type="primary", use_container_width=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_repo" not in st.session_state:
    st.session_state.indexed_repo = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if ingest_btn:
    if not repo_input.strip():
        st.sidebar.error("Please provide a GitHub repository URL or owner/repo.")
    elif not openai_key.strip():
        st.sidebar.error("Please provide an OpenAI API Key.")
    else:
        with st.status("Indexing repository...", expanded=True) as status:
            try:
                st.write("🔍 Connecting to GitHub API...")
                ingestion = GitHubIngestion(token=github_token)
                
                st.write("📂 Extracting code files and documentation...")
                docs = ingestion.fetch_repo_files(repo_input, max_files=max_files)
                
                if include_issues:
                    st.write("📋 Fetching repository issues...")
                    issue_docs = ingestion.fetch_repo_issues(repo_input)
                    docs.extend(issue_docs)
                
                st.write(f"✂️ Splitting {len(docs)} files using Language-aware & Markdown chunkers...")
                chunker = CodeAwareChunker()
                chunks = chunker.split_documents(docs)
                
                st.write(f"🧠 Generating embeddings and storing in ChromaDB ({len(chunks)} chunks)...")
                vsm = VectorStoreManager(openai_api_key=openai_key)
                vectorstore = vsm.create_or_update_vectorstore(repo_input, chunks)
                
                st.write("🎯 Initializing Two-Stage Retriever (ChromaDB + FlashRank)...")
                retriever = TwoStageRetriever(vectorstore=vectorstore)
                chain = RAGChain(retriever=retriever, openai_api_key=openai_key)
                
                st.session_state.rag_chain = chain
                st.session_state.indexed_repo = repo_input
                st.session_state.messages = []
                
                status.update(label=f"✅ Successfully indexed {repo_input}!", state="complete", expanded=False)
                st.success(f"Repository **{repo_input}** ready for Q&A! ({len(docs)} files, {len(chunks)} chunks indexed)")
            except Exception as e:
                status.update(label="❌ Ingestion failed", state="error", expanded=True)
                st.error(f"Error: {str(e)}")

st.subheader("💬 Repository Q&A")

if not st.session_state.indexed_repo:
    st.info("👈 Please enter a repository and click **'Ingest & Index Repository'** in the sidebar to start asking questions.")
else:
    st.caption(f"Currently querying: **{st.session_state.indexed_repo}**")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                with st.expander("📑 Sources & Citations"):
                    for c in msg["citations"]:
                        score_str = f" (Score: {c['score']:.3f})" if c.get('score') is not None else ""
                        url = c.get('html_url')
                        if url:
                            st.markdown(f"- [{c['source']}]({url}){score_str}")
                        else:
                            st.markdown(f"- **{c['source']}**{score_str}")
                        st.code(c.get('preview', ''), language="text")

    if prompt := st.chat_input("Ask a question about code architecture, functions, bugs, or issues..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking & Retrieving relevant code..."):
                try:
                    result = st.session_state.rag_chain.answer_question(prompt)
                    answer = result["answer"]
                    citations = result["citations"]
                    
                    st.markdown(answer)
                    if citations:
                        with st.expander("📑 Sources & Citations"):
                            for c in citations:
                                score_str = f" (Score: {c['score']:.3f})" if c.get('score') is not None else ""
                                url = c.get('html_url')
                                if url:
                                    st.markdown(f"- [{c['source']}]({url}){score_str}")
                                else:
                                    st.markdown(f"- **{c['source']}**{score_str}")
                                st.code(c.get('preview', ''), language="text")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations
                    })
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
