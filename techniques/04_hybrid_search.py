"""
04_hybrid_search.py — Hybrid Search (Vector + BM25)
======================================================

WHAT IT IS:
    "Use both a dictionary AND your brain."
    Combines Vector Search (Dense, Semantic) with Keyword Search (Sparse, Lexical).

WHY IT MATTERS:
    Vector search is great for meaning ("dog" matches "canine").
    But it sucks at exact matches (searching for ID "SKU-9942", or a specific acronym).
    BM25 (classic keyword search) is great at exact matches but bad at meaning.
    Hybrid search combines both using Reciprocal Rank Fusion (RRF).

REQUIREMENT:
    pip install rank-bm25
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import numpy as np

from shared.utils import print_header, print_step, print_documents
from shared.vector_store import create_vector_store, get_vector_store, delete_collection

COLLECTION_NAME = "hybrid_demo"

# Documents designed to highlight differences between Semantic vs Lexical search
MOCK_DOCS = [
    Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={"id": 1}),
    Document(page_content="A fast auburn canine leaps above a tired hound.", metadata={"id": 2}), # Semantic match to Doc 1, but completely different words
    Document(page_content="Error code ERR-X992 indicates a memory leak in the database subsystem.", metadata={"id": 3}), # Exact keyword target
    Document(page_content="We found an issue where the storage system consumes all available RAM leading to a crash.", metadata={"id": 4}), # Semantic match to Doc 3
]

def reciprocal_rank_fusion(vector_ranks, bm25_ranks, k=60):
    """
    Combines two ranked lists. Documents at the top of either list get a high score.
    Formula: Score = 1 / (k + rank)
    """
    scores = {}
    for rank, doc in enumerate(vector_ranks):
        doc_id = doc.metadata["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        # Store the actual doc object for later
        scores[f"obj_{doc_id}"] = doc

    for rank, doc in enumerate(bm25_ranks):
        doc_id = doc.metadata["id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        scores[f"obj_{doc_id}"] = doc

    # Sort by combined score
    sorted_ids = sorted([k for k in scores.keys() if isinstance(k, int)], key=lambda x: scores[x], reverse=True)
    return [scores[f"obj_{doc_id}"] for doc_id in sorted_ids]


def run():
    print_header("🔀 HYBRID SEARCH (Vector + Keyword)")

    delete_collection(COLLECTION_NAME)
    vector_store = create_vector_store(MOCK_DOCS, collection_name=COLLECTION_NAME)

    # Prepare BM25 keyword index
    tokenized_corpus = [doc.page_content.lower().split(" ") for doc in MOCK_DOCS]
    bm25 = BM25Okapi(tokenized_corpus)

    def hybrid_search(query: str):
        # 1. Vector Search
        vector_results = vector_store.similarity_search(query, k=4)

        # 2. BM25 Keyword Search
        tokenized_query = query.lower().split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_ranked_indices = np.argsort(bm25_scores)[::-1]
        bm25_results = [MOCK_DOCS[i] for i in bm25_ranked_indices if bm25_scores[i] > 0] # Filter 0 score

        # 3. Fuse
        fused_results = reciprocal_rank_fusion(vector_results, bm25_results)

        print("\n  [Vector Results (Meaning)]")
        for i, d in enumerate(vector_results[:2]): print(f"    {i+1}. {d.page_content[:60]}")

        print("\n  [BM25 Results (Exact Word Match)]")
        for i, d in enumerate(bm25_results[:2]): print(f"    {i+1}. {d.page_content[:60]}")

        print("\n  [RRF Fused Results (Best of Both)]")
        for i, d in enumerate(fused_results[:2]): print(f"    {i+1}. {d.page_content[:60]}")


    print("="*60)
    print_step(1, "Semantic Test: Query uses different words but same meaning.")
    q1 = "speedy red dog leaping"
    print(f"Query: '{q1}'")
    hybrid_search(q1)
    print("Notice Vector finds Doc 2, while BM25 fails completely (no shared words).")

    print("\n" + "="*60)
    print_step(2, "Lexical Test: Searching for an exact identifier.")
    q2 = "ERR-X992"
    print(f"Query: '{q2}'")
    hybrid_search(q2)
    print("Notice BM25 easily finds Doc 3. Vector search might struggle if 'ERR-X992' doesn't have a strong embedding representation.")

if __name__ == "__main__":
    run()
