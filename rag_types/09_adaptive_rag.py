"""
09_adaptive_rag.py — Adaptive RAG (Dynamic Strategy Selection)
================================================================

WHAT IS ADAPTIVE RAG?
    Recognizes the TYPE of question (simple, complex, broad, narrow)
    and adjusts its retrieval and generation style accordingly.

    Instead of a "one size fits all" pipeline, it routes queries:
    - Simple greetings → LLM direct answer (skip retrieval)
    - Fact lookups → Small K retrieval, concise answer
    - Complex analysis → Large K retrieval, detailed synthesis, maybe branching

HOW IT WORKS:
    1. Query Classifier: An LLM analyzes the incoming question.
    2. Routing: Based on the classification, it selects a specific RAG pipeline.
    3. Execution: Runs the chosen pipeline.

BEST FOR:
    General-purpose assistants that receive a wide mix of queries,
    from "hello" to "summarize this 50-page document."
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List

from shared.utils import (
    print_header, print_step, print_result, print_documents,
    load_documents_from_directory, chunk_documents, timer
)
from shared.vector_store import create_vector_store, similarity_search, delete_collection
from shared.llm import ask_llm, ask_llm_with_context


COLLECTION_NAME = "adaptive_rag"


# ════════════════════════════════════════════════════
# ROUTING MODULE
# ════════════════════════════════════════════════════

def classify_query(question: str) -> str:
    """
    Determine the complexity and intent of the query.
    Returns one of: 'conversational', 'factual', 'analytical'
    """
    prompt = f"""Classify the following user query into exactly ONE of these categories:

1. conversational: Greetings, pleasantries, or generic questions that don't require external facts.
2. factual: Specific questions looking for a distinct fact, definition, or short answer.
3. analytical: Complex questions requiring synthesis, comparison, or deep explanation across multiple concepts.

Query: "{question}"

Respond with ONLY the category name (conversational, factual, or analytical)."""

    category = ask_llm(
        prompt=prompt,
        system_prompt="You are a query router. Respond with a single word."
    ).strip().lower()

    if category not in ["conversational", "factual", "analytical"]:
        return "factual"  # Safe fallback

    return category


# ════════════════════════════════════════════════════
# DIFFERENT EXECUTION PATHS
# ════════════════════════════════════════════════════

def route_conversational(question: str) -> str:
    """Path 1: No retrieval, just direct LLM response."""
    print("  ↳ 🛤️  Path: Conversational (Skipping retrieval)")
    return ask_llm(
        prompt=question,
        system_prompt="You are a friendly AI assistant. Answer conversationally."
    )

def route_factual(question: str) -> str:
    """Path 2: Precision retrieval (low K), concise answer."""
    print("  ↳ 🛤️  Path: Factual (High precision retrieval)")

    # Retrieve only the top 2 documents to reduce noise
    docs = similarity_search(question, collection_name=COLLECTION_NAME, k=2)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    return ask_llm_with_context(
        question=question,
        context=context,
        system_prompt="Answer the factual question concisely based ONLY on the context. Be brief and direct."
    )

def route_analytical(question: str) -> str:
    """Path 3: Broad retrieval (high K), detailed synthesis."""
    print("  ↳ 🛤️  Path: Analytical (Broad retrieval & synthesis)")

    # Retrieve more documents (top 6) for broader context
    docs = similarity_search(question, collection_name=COLLECTION_NAME, k=6)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    return ask_llm_with_context(
        question=question,
        context=context,
        system_prompt="You are an expert analyst. Provide a detailed, structured, and comprehensive answer based on the context. Use headings and bullet points where appropriate."
    )


# ════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════

def adaptive_rag_query(question: str) -> str:
    """Full Adaptive RAG pipeline: classify → route → execute."""
    print_step(1, f"Analyzing query intent...")
    category = classify_query(question)
    print_result("Classification", category.upper())

    print_step(2, f"Executing specialized pipeline...")
    if category == "conversational":
        return route_conversational(question)
    elif category == "factual":
        return route_factual(question)
    elif category == "analytical":
        return route_analytical(question)
    else:
        return route_factual(question) # Fallback


@timer
def setup_knowledge_base():
    """Index documents."""
    docs = load_documents_from_directory()
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    delete_collection(COLLECTION_NAME)
    create_vector_store(chunks, collection_name=COLLECTION_NAME)
    print_result("Knowledge base ready", f"{len(chunks)} chunks indexed")


def run():
    """Demo: Show Adaptive RAG routing different questions."""
    print_header("🔀 ADAPTIVE RAG — Dynamic Strategy Selection")

    print("""
    Adaptive RAG changes its behavior based on the question.
    Watch how it handles three different types of queries:
    1. Conversational → Skips retrieval entirely (saves time/money)
    2. Factual → Retrieves top 2 docs, gives a short answer
    3. Analytical → Retrieves top 6 docs, gives a detailed breakdown
    """)

    setup_knowledge_base()

    questions = [
        "Hey there, how are you doing today?",           # Should be conversational
        "Who introduced the Transformer architecture?",  # Should be factual
        "Compare the traditional sequential text processing models with the Transformer architecture. What are the key innovations and trade-offs?", # Should be analytical
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'═'*60}")
        print(f"  Question {i}: {question}")
        print(f"{'═'*60}")

        answer = adaptive_rag_query(question)

        print(f"\n  🤖 Answer:\n  {answer}")

    print_header("📊 ADAPTIVE RAG — Key Takeaways")
    print("""
    Trade-offs:
    + Highly efficient (doesn't waste resources searching for "hello")
    + Tailors answer formatting to user intent
    - The classification step adds a small latency overhead to every query
    - If classification is wrong, the wrong pipeline is used (e.g., skips search for a fact)
    """)

if __name__ == "__main__":
    run()
