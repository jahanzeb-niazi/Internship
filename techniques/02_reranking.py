"""
02_reranking.py — Reranking Retrieved Documents
=================================================

WHAT IT IS:
    "Sort the good answers to the top."
    Vector search (bi-encoder) is fast but sometimes inaccurate.
    Reranking (cross-encoder) is slow but highly accurate.
    We combine them: Fast search grabs top 50, slow reranker sorts them to find the top 5.

HOW IT WORKS:
    Vector Search embeds Query and Document separately (A -> vec, B -> vec).
    A Cross-Encoder passes the Query and Document TOGETHER through the
    transformer layers, allowing the model to see how the words interact.

DEMO APPROACH:
    We'll simulate a cross-encoder using the LLM for simplicity, though in
    production you'd use a dedicated model like `cross-encoder/ms-marco-MiniLM-L-6-v2`.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document

from shared.utils import print_header, print_step, print_documents
from shared.llm import ask_llm

# Simulated retrieved documents from a fast vector search (some good, some bad)
MOCK_RETRIEVED_DOCS = [
    Document(page_content="Neural networks are algorithms inspired by the human brain, designed to recognize patterns. They interpret sensory data through a kind of machine perception.", metadata={"id": 1}),
    Document(page_content="A fishing net is a tool used for catching fish. They are made of mesh usually formed by knotting a relatively thin thread. Modern nets are usually made of artificial polyamides like nylon.", metadata={"id": 2}),
    Document(page_content="The nervous system is a highly complex part of an animal that coordinates its actions by transmitting signals to and from different parts of its body. It consists of the central and peripheral nervous systems.", metadata={"id": 3}),
    Document(page_content="Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks, and convolutional neural networks have been applied to fields including computer vision and speech recognition.", metadata={"id": 4}),
]

def llm_reranker(query: str, docs: list[Document]) -> list[tuple[int, Document]]:
    """Simulates a Cross-Encoder by asking the LLM to score (query, doc) pairs."""
    scored = []
    for doc in docs:
        prompt = f"""Rate how relevant this document is to the query on a scale of 0 to 10.
Query: "{query}"
Document: "{doc.page_content}"
Respond with ONLY the integer score."""

        try:
            score = int(ask_llm(prompt, "You score relevance.").strip())
        except ValueError:
            score = 0
        scored.append((score, doc))

    # Sort descending by score
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def run():
    print_header("⚖️ RERANKING")
    query = "What are neural networks in the context of computer science?"
    print(f"Query: {query}\n")

    print_step(1, "Initial Retrieval (Simulated Vector Search)")
    print("Vector search might pull in docs about 'nets' (fishing) or 'neural' (biology) because of word overlap.")
    for doc in MOCK_RETRIEVED_DOCS:
        print(f"  - Doc {doc.metadata['id']}: {doc.page_content[:60]}...")

    print("\n" + "-"*50 + "\n")

    print_step(2, "Applying Reranker (Cross-Encoder / LLM Scorer)")
    ranked_docs = llm_reranker(query, MOCK_RETRIEVED_DOCS)

    print("\n  🏆 Reranked Results (Highest to Lowest):")
    for score, doc in ranked_docs:
        print(f"  [Score: {score:2d}] Doc {doc.metadata['id']}: {doc.page_content[:60]}...")

    print("""
    Notice how:
    - Doc 1 and 4 (AI concepts) get high scores.
    - Doc 3 (Biology) gets a lower score despite matching "neural".
    - Doc 2 (Fishing net) gets the lowest score despite matching "net".
    """)

if __name__ == "__main__":
    run()
