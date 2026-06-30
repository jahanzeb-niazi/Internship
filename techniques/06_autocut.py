"""
06_autocut.py — Autocut (Dynamic Thresholding)
================================================

WHAT IT IS:
    "Don't dump everything into the AI's brain at once."
    Instead of always passing exactly `top_K` documents to the LLM (e.g., exactly 5 docs),
    Autocut looks at the similarity scores. If there's a huge drop-off in score after the 2nd
    document, it "cuts" the rest, passing only the first 2.

WHY IT MATTERS:
    Passing irrelevant documents (noise) confuses the LLM and wastes tokens. Autocut ensures
    only truly relevant documents make it to the prompt.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from shared.utils import print_header, print_step
from shared.vector_store import create_vector_store, similarity_search_with_scores, delete_collection

COLLECTION_NAME = "autocut_demo"

MOCK_DOCS = [
    Document(page_content="Python is a dynamically typed programming language.", metadata={"id": 1}),
    Document(page_content="Python was created by Guido van Rossum.", metadata={"id": 2}),
    Document(page_content="Java uses static typing and runs on the JVM.", metadata={"id": 3}),
    Document(page_content="Snakes of the family Pythonidae are non-venomous.", metadata={"id": 4}),
]

def apply_autocut(scored_results: list[tuple[Document, float]], drop_threshold: float = 0.2) -> list[Document]:
    """
    Looks for a significant gap between adjacent scores.
    ChromaDB returns distance scores (lower is better, 0 = exact match).
    """
    if not scored_results: return []

    accepted_docs = [scored_results[0][0]]
    prev_score = scored_results[0][1]

    for doc, score in scored_results[1:]:
        # If the gap between this score and the previous is larger than threshold, CUT!
        # Note: Depending on embedding model and distance metric (L2, Cosine), this logic varies.
        # Here we just look for a jump in distance.
        if (score - prev_score) > drop_threshold:
            print(f"    ✂️ AUTOCUT TRIGGERED! Distance jumped from {prev_score:.3f} to {score:.3f}")
            break
        accepted_docs.append(doc)
        prev_score = score

    return accepted_docs

def run():
    print_header("✂️ AUTOCUT (Dynamic Filtering)")
    delete_collection(COLLECTION_NAME)
    create_vector_store(MOCK_DOCS, collection_name=COLLECTION_NAME)

    query = "Who invented the Python programming language?"
    print(f"Query: '{query}'\n")

    print_step(1, "Retrieve Top 4 Docs with Scores")
    # Chroma returns distance metrics. (Lower = more similar)
    results = similarity_search_with_scores(query, collection_name=COLLECTION_NAME, k=4)

    for i, (doc, score) in enumerate(results):
        print(f"  [{i}] Distance: {score:.3f} | {doc.page_content}")

    print("\n" + "="*50)
    print_step(2, "Applying Autocut")
    print("  Normally, RAG blindly passes all 4 docs to the LLM.")
    print("  Autocut stops reading when it detects a big drop in relevance.\n")

    final_docs = apply_autocut(results, drop_threshold=0.15) # Threshold tuned for this demo

    print(f"\n  Final docs passed to LLM: {len(final_docs)} (Saved {len(results) - len(final_docs)} irrelevant docs)")
    for d in final_docs:
        print(f"  ✓ {d.page_content}")

if __name__ == "__main__":
    run()
