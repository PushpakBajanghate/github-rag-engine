
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import os
from dotenv import load_dotenv
load_dotenv(override=True)

from src.config import settings
from src.ingestion import GitHubIngestion
from src.chunker import CodeAwareChunker
from src.vectorstore import VectorStoreManager, GEMINI_BATCH_SIZE, GEMINI_DELAY_SECONDS

print(f"Provider: {settings.active_provider}")
print(f"Batch size: {GEMINI_BATCH_SIZE} docs | Delay: {GEMINI_DELAY_SECONDS}s between batches")

print("\nFetching 5 files from own repo...")
ingestion = GitHubIngestion()
docs = ingestion.fetch_repo_files("PushpakBajanghate/github-rag-engine", max_files=5)
print(f"Fetched {len(docs)} files")

chunker = CodeAwareChunker()
chunks = chunker.split_documents(docs)
print(f"Created {len(chunks)} chunks")

test_chunks = chunks[:10]
print(f"\nRunning embedding pipeline on {len(test_chunks)} chunks (2 batches of 5)...")

def callback(msg):
    print(f"  PROGRESS: {msg}")

vsm = VectorStoreManager(provider="gemini")
vs = vsm.create_or_update_vectorstore("test_e2e_final", test_chunks, progress_callback=callback)
print("\nVectorstore creation: SUCCESS")

from src.retriever import TwoStageRetriever
from src.chain import RAGChain

retriever = TwoStageRetriever(vectorstore=vs)
retrieved = retriever.get_relevant_documents("What is this project about?")
print(f"Retrieval: {len(retrieved)} docs retrieved")

chain = RAGChain(retriever=retriever, provider="gemini")
result = chain.answer_question("What does src/vectorstore.py do?")
answer = result["answer"]
print(f"\nRAG Answer (first 200 chars): {answer[:200]}")
print("\n=== FULL PIPELINE PASSED ===")
vsm.clear_collection("test_e2e_final")
print("Test collection cleaned.")
