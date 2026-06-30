"""
13_advanced_rag.py — Advanced RAG (The Production Stack)
==========================================================

WHAT IS ADVANCED RAG?
    Advanced RAG isn't a single new trick — it's the strategic combination
    of multiple techniques (query rewriting, reranking, hybrid search) to
    produce a highly robust, production-grade system.

    It layers techniques to cover the weaknesses of Simple RAG.

PIPELINE IMPLEMENTED HERE:
    1. Query Expansion (Rewrite the query into multiple variants)
    2. Parallel Retrieval (Run vector search for all variants)
    3. Results Merging & Deduplication
    4. Reranking (Simulated: evaluate relevance of merged chunks)
    5. Generation

BEST FOR:
    High-stakes systems, research tools, and enterprise apps where accuracy
    is critical and cost/latency constraints allow for more compute.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List
from langchain_core.documents import Document

from shared.utils import (
    print_header, print_step, print_result, print_documents,
    load_documents_from_directory, chunk_documents, timer
)
from shared.vector_store import create_vector_store, similarity_search, delete_collection
from shared.llm import ask_llm, ask_llm_with_context

COLLECTION_NAME = "advanced_rag"

def expand_query(question: str) -> List[str]:
    """Generates varied search queries to improve recall."""
    print("  [Technique: Query Expansion]")
    prompt = f"""Generate 2 related search queries for the following question to maximize retrieval recall.
Question: {question}
Return ONLY the queries separated by a newline."""
    response = ask_llm(
        prompt=prompt,
        system_prompt="You are a search query generator."
    )
    queries = [q.strip() for q in response.split('\n') if q.strip()]
    return [question] + queries[:2]

def rerank_documents(question: str, documents: List[Document], top_k: int = 3) -> List[Document]:
    """
    Simulates a Cross-Encoder Reranker using the LLM.
    (In real life, use sentence-transformers/all-MiniLM-L6-v2 cross encoder)
    """
    print(f"  [Technique: Reranking] Scoring {len(documents)} docs...")

    scored_docs = []
    for doc in documents:
        # Ask LLM to score relevance 1-10
        prompt = f"Score relevance from 1-10.\nQ: {question}\nText: {doc.page_content}\nScore (number only):"
        try:
            score = int(ask_llm(prompt, "You output only numbers 1-10").strip())
        except ValueError:
            score = 5
        scored_docs.append((score, doc))

    # Sort by score descending and take top_k
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    best_docs = [doc for score, doc in scored_docs[:top_k]]
    return best_docs

def advanced_rag_query(question: str) -> str:
    # 1. Query Expansion
    print_step(1, "Expanding query...")
    queries = expand_query(question)
    print(f"  Queries: {queries}")

    # 2. Multi-Query Retrieval
    print_step(2, "Parallel retrieval...")
    all_docs = []
    for q in queries:
        docs = similarity_search(q, collection_name=COLLECTION_NAME, k=3)
        all_docs.extend(docs)

    # 3. Deduplication
    unique_contents = set()
    deduped_docs = []
    for d in all_docs:
        if d.page_content not in unique_contents:
            unique_contents.add(d.page_content)
            deduped_docs.append(d)
    print(f"  Retrieved {len(deduped_docs)} unique chunks across all queries.")

    # 4. Reranking
    print_step(3, "Reranking candidates...")
    final_docs = rerank_documents(question, deduped_docs, top_k=3)
    print("  Top Reranked Docs:")
    print_documents(final_docs, max_chars=100)

    # 5. Generation
    print_step(4, "Generating final answer...")
    context = "\n\n".join(d.page_content for d in final_docs)
    return ask_llm_with_context(question, context)

def setup_knowledge_base():
    docs = load_documents_from_directory()
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    delete_collection(COLLECTION_NAME)
    create_vector_store(chunks, collection_name=COLLECTION_NAME)

def run():
    print_header("🚀 ADVANCED RAG (Production Pipeline)")
    print("Combines Query Expansion, Multi-Retrieval, Deduplication, and Reranking.")
    setup_knowledge_base()

    q = "What is the role of positional encoding in transformers?"
    print(f"\nQuestion: {q}")
    ans = advanced_rag_query(q)
    print(f"\nFinal Answer: {ans}")

if __name__ == "__main__":
    run()
