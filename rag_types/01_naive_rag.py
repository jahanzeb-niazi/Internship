"""
01_naive_rag.py — Naive RAG (The Simplest Baseline)
=====================================================

WHAT IS NAIVE RAG?
    The most basic form of RAG. It converts your question into simple
    search terms, pulls the top matching documents, and passes them
    straight to the LLM with NO filtering, NO reranking, NO query
    optimization.

    Think of it like copying the first Google results into ChatGPT
    without even reading them first.

WHY LEARN THIS?
    It's the baseline. By seeing its weaknesses, you'll appreciate
    why every other RAG type exists.

WEAKNESSES:
    - Often retrieves irrelevant documents that hurt answer quality
    - No filtering = noise gets passed to the LLM
    - No query optimization = bad questions get bad results
    - No reranking = the most relevant doc might not be first

PIPELINE:
    User Query → Embed → Vector Search (top-K) → Stuff into Prompt → LLM → Answer
    (No preprocessing, no postprocessing, no quality checks)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.utils import (
    print_header, print_step, print_result, print_documents,
    print_warning, load_documents_from_directory, chunk_documents, timer
)
from shared.vector_store import create_vector_store, similarity_search, delete_collection
from shared.llm import ask_llm_with_context


COLLECTION_NAME = "naive_rag"


@timer
def index_documents():
    """
    STEP 1: Load, chunk, and index documents.

    In Naive RAG, we use basic fixed-size chunking with no special
    preprocessing — just split and embed.
    """
    print_step(1, "Loading documents from data/ directory...")
    docs = load_documents_from_directory()
    print_result("Documents loaded", str(len(docs)))

    # ── Chunk with basic settings ──
    # Naive RAG doesn't optimize chunk size — we just use a reasonable default
    print_step(2, "Chunking documents (basic fixed-size)...")
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    print_result("Chunks created", str(len(chunks)))

    # ── Store in vector database ──
    print_step(3, "Creating embeddings and storing in ChromaDB...")
    delete_collection(COLLECTION_NAME)  # Clean slate
    store = create_vector_store(chunks, collection_name=COLLECTION_NAME)
    print_result("Vector store created", f"Collection '{COLLECTION_NAME}'")

    return store


@timer
def naive_retrieve_and_generate(question: str, k: int = 4):
    """
    STEP 2: Retrieve and Generate (the naive way).

    NO query optimization — we use the raw question as-is.
    NO reranking — we trust the vector DB's ordering.
    NO filtering — we pass ALL retrieved docs to the LLM.
    """
    print_step(4, f"Retrieving top-{k} documents for: '{question}'")

    # ── Retrieve: Just do a simple similarity search ──
    # This is the ONLY retrieval step in Naive RAG
    docs = similarity_search(question, collection_name=COLLECTION_NAME, k=k)

    print("\n  📄 Retrieved Documents:")
    print_documents(docs, max_chars=150)

    # ── Generate: Stuff ALL documents into the prompt ──
    # No filtering, no selection — everything goes in
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])

    print_step(5, "Generating answer with LLM (no filtering applied)...")
    answer = ask_llm_with_context(question=question, context=context)

    return answer, docs


def run():
    """
    Main demo: Shows Naive RAG in action and highlights its limitations.
    """
    print_header("🔍 NAIVE RAG — The Simplest Baseline")

    print("""
    Naive RAG is the most basic retrieval-augmented generation pipeline.
    It does NO optimization — just search and generate.

    Watch for:
    ❌ Irrelevant documents in the retrieval results
    ❌ The LLM might get confused by noisy context
    ❌ No quality checking on the output
    """)

    # ── Index our knowledge base ──
    index_documents()

    # ── Test with different types of questions ──
    questions = [
        # Good question: clear and specific → should work OK
        "What is the self-attention mechanism in transformers?",

        # Vague question: naive RAG struggles with these
        "How do you make AI better?",

        # Multi-part question: naive RAG can't handle complexity
        "Compare the differences between CNNs and RNNs and explain when to use each.",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'─'*50}")
        print(f"  Question {i}: {question}")
        print(f"{'─'*50}")

        answer, docs = naive_retrieve_and_generate(question)

        print(f"\n  💬 Answer:")
        print(f"  {answer}")

    # ── Show the limitations ──
    print_header("📊 NAIVE RAG — Limitations Summary")
    print("""
    What you likely noticed:
    1. GOOD QUESTIONS work OK — clear queries retrieve relevant docs.
    2. VAGUE QUESTIONS suffer — "make AI better" matches too many things.
    3. COMPLEX QUESTIONS get partial answers — only 4 docs retrieved,
       and they might not cover all parts of the question.

    These limitations motivate ALL the other RAG types:
    → Simple RAG adds proper embedding search (slightly better retrieval)
    → Advanced RAG adds reranking and query rewriting
    → Self-RAG adds quality checking
    → Corrective RAG re-searches when answers are poor
    → etc.
    """)


if __name__ == "__main__":
    run()
