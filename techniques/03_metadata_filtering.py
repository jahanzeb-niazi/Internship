"""
03_metadata_filtering.py — Metadata Filtering (Pre-filtering)
===============================================================

WHAT IT IS:
    "Use labels to find things faster."
    Instead of relying purely on vector similarity, you attach metadata tags
    (date, author, category, document_id) to chunks. Before running the vector
    search, you filter out irrelevant chunks using standard SQL-like WHERE clauses.

WHY IT MATTERS:
    Vectors don't understand logic. If a user asks, "Show me reports from 2024,"
    a vector search might return a highly relevant report from 2022 because
    the content matches. Metadata filtering enforces hard logical rules.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from shared.utils import print_header, print_step, print_documents
from shared.vector_store import create_vector_store, get_vector_store, delete_collection

COLLECTION_NAME = "metadata_demo"

# Mock documents with rich metadata
MOCK_DOCS = [
    Document(page_content="The company revenue increased by 20% due to AI adoption.", metadata={"year": 2022, "type": "financial_report", "department": "sales"}),
    Document(page_content="We launched the new deep learning platform internally.", metadata={"year": 2022, "type": "press_release", "department": "engineering"}),
    Document(page_content="Q3 saw a dip in sales, but Q4 recovered strongly.", metadata={"year": 2023, "type": "financial_report", "department": "sales"}),
    Document(page_content="The new transformer model improved our chat latency by 50%.", metadata={"year": 2023, "type": "tech_blog", "department": "engineering"}),
    Document(page_content="Record breaking profits this year, driven by enterprise software.", metadata={"year": 2024, "type": "financial_report", "department": "sales"}),
    Document(page_content="We are deprecating the old API in favor of GraphQL.", metadata={"year": 2024, "type": "tech_blog", "department": "engineering"}),
]

def run():
    print_header("🏷️ METADATA FILTERING")

    print_step(1, "Indexing documents with metadata tags (year, type, department)...")
    delete_collection(COLLECTION_NAME)
    store = create_vector_store(MOCK_DOCS, collection_name=COLLECTION_NAME)

    query = "Tell me about profits and revenue."

    print("\n" + "="*50)
    print_step(2, f"Search WITHOUT filtering: '{query}'")
    # Will pull financial reports from any year based purely on text similarity
    results_unfiltered = store.similarity_search(query, k=2)
    print_documents(results_unfiltered, max_chars=100)

    print("="*50)
    print_step(3, f"Search WITH filtering (year == 2024): '{query}'")
    # ChromaDB supports a MongoDB-like filtering syntax
    filter_dict = {"year": {"$eq": 2024}}
    results_filtered = store.similarity_search(query, k=2, filter=filter_dict)
    print_documents(results_filtered, max_chars=100)

    print("="*50)
    print_step(4, f"Search WITH complex filtering (year >= 2023 AND dept == engineering)")
    complex_filter = {
        "$and": [
            {"year": {"$gte": 2023}},
            {"department": {"$eq": "engineering"}}
        ]
    }
    # Notice we change the query slightly to match the filtered subset better
    results_complex = store.similarity_search("model updates and APIs", k=2, filter=complex_filter)
    print_documents(results_complex, max_chars=100)

if __name__ == "__main__":
    run()
