"""
14_hyde_rag.py — HyDE (Hypothetical Document Embeddings)
==========================================================

WHAT IS HyDE?
    Instead of embedding the user's QUESTION to search the database,
    HyDE uses an LLM to generate a "fake" or "hypothetical" ANSWER first.
    It then embeds this fake answer to perform the search.

    Why? Because an answer (even a made-up one) looks much more like the
    target documents in vector space than a short question does.
    A question and an answer belong to different semantic distributions.

HOW IT WORKS:
    1. Ask LLM: "Answer this question hypothetically." -> (Fake Answer)
    2. Embed the Fake Answer.
    3. Search vector DB for documents similar to the Fake Answer.
    4. Generate real answer using retrieved real documents.

BEST FOR:
    Academic, legal, medical search where the user's query is short but
    the desired documents use highly specific, dense terminology.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.utils import (
    print_header, print_step, print_result, print_documents,
    load_documents_from_directory, chunk_documents, timer
)
from shared.vector_store import create_vector_store, similarity_search, delete_collection
from shared.llm import ask_llm, ask_llm_with_context

COLLECTION_NAME = "hyde_rag"

def generate_hypothetical_document(question: str) -> str:
    """Generate a fake answer to use for vector similarity."""
    prompt = f"""Please write a paragraph answering the following question.
Write it in the style of an academic or technical document.
Don't worry if you don't know the exact facts; focus on using the correct terminology and structure.

Question: {question}

Hypothetical Document:"""

    hypo_doc = ask_llm(
        prompt=prompt,
        system_prompt="You generate plausible hypothetical text for search queries."
    )
    return hypo_doc.strip()

def hyde_query(question: str) -> str:
    """HyDE Pipeline: Generate Fake Doc -> Search -> Answer"""

    print_step(1, "Generating hypothetical document (HyDE)...")
    hypo_doc = generate_hypothetical_document(question)
    print(f"  📝 Fake Answer:\n  {hypo_doc[:300]}...\n")

    print_step(2, "Searching vector DB using the HYPOTHETICAL document...")
    # NOTE: We are searching using `hypo_doc`, NOT `question`!
    docs = similarity_search(hypo_doc, collection_name=COLLECTION_NAME, k=3)

    print("\n  📄 Retrieved Real Documents:")
    print_documents(docs, max_chars=150)

    print_step(3, "Generating final answer using REAL documents...")
    context = "\n\n".join(d.page_content for d in docs)
    answer = ask_llm_with_context(question, context)

    return answer

def setup_knowledge_base():
    docs = load_documents_from_directory()
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    delete_collection(COLLECTION_NAME)
    create_vector_store(chunks, collection_name=COLLECTION_NAME)

def run():
    print_header("🎭 HyDE — Hypothetical Document Embeddings")
    print("""
    HyDE bridges the semantic gap between a short question and a long document.
    Watch it generate a "fake" answer first, and use THAT to search!
    """)
    setup_knowledge_base()

    # A short, dense query that might fail direct keyword/vector matching
    q = "RLHF mechanism in model training"
    print(f"\nQuestion: {q}")

    ans = hyde_query(q)
    print(f"\nFinal Answer: {ans}")

if __name__ == "__main__":
    run()
