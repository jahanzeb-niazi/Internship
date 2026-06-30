"""
10_speculative_rag.py — Speculative RAG (Predictive Pre-fetching)
===================================================================

WHAT IS SPECULATIVE RAG?
    Analyzes the current query and conversation history to PREDICT likely
    follow-up questions, pre-fetching relevant documents in the background.

    By the time the user asks their next question, the documents are
    already retrieved and waiting in a cache, reducing latency to near-zero.

HOW IT WORKS:
    1. Process user query normally (Retrieve → Generate).
    2. In the background (or post-generation), ask an LLM: "What will the user ask next?"
    3. Generate 3-5 speculative follow-up queries.
    4. Run retrieval for those speculative queries.
    5. Cache the results.
    6. When the next real query arrives, check if it matches a cached speculation.

BEST FOR:
    Interactive chatbots, tutorials, data exploration where users tend
    to follow predictable "paths" of inquiry.
"""

import sys
import os
import threading
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict

from shared.utils import (
    print_header, print_step, print_result,
    load_documents_from_directory, chunk_documents, timer
)
from shared.vector_store import create_vector_store, similarity_search, delete_collection
from shared.llm import ask_llm, ask_llm_with_context
from shared.embeddings import embed_text

COLLECTION_NAME = "speculative_rag"

# Global cache for pre-fetched documents
# Structure: { "predicted_query": [Document1, Document2, ...] }
SPECULATIVE_CACHE: Dict[str, List] = {}


def generate_predictions(current_question: str, current_answer: str) -> List[str]:
    """Predict 2 likely follow-up questions."""
    prompt = f"""Based on the current question and answer, predict 2 highly likely follow-up questions the user might ask next.

Current Question: {current_question}
Current Answer: {current_answer}

Return ONLY the questions, one per line, no numbering or bullets."""

    response = ask_llm(
        prompt=prompt,
        system_prompt="You predict user follow-up questions. Be concise and realistic."
    )

    predictions = [q.strip() for q in response.split('\n') if q.strip()]
    return predictions[:2]


def prefetch_worker(predictions: List[str]):
    """
    Background worker that fetches documents for predicted queries.
    In a real app, this runs asynchronously via Celery or asyncio.
    """
    print(f"\n    [Background Worker] ⏳ Pre-fetching docs for {len(predictions)} predicted queries...")
    start_time = time.time()

    for pred_q in predictions:
        # Retrieve documents for the prediction
        docs = similarity_search(pred_q, collection_name=COLLECTION_NAME, k=3)
        # Store in cache
        SPECULATIVE_CACHE[pred_q] = docs
        print(f"    [Background Worker] ✓ Cached docs for: '{pred_q}'")

    print(f"    [Background Worker] 🏁 Finished pre-fetching in {time.time()-start_time:.2f}s")


def calculate_similarity(query1: str, query2: str) -> float:
    """
    Calculate semantic similarity between two queries using embeddings.
    Used to check if a new query matches a cached prediction.
    """
    # Quick simulation of cosine similarity for the demo to save API calls
    # In reality, you'd embed both strings and calculate cosine similarity:
    # vec1 = embed_text(query1)
    # vec2 = embed_text(query2)
    # return cosine_similarity(vec1, vec2)

    q1_words = set(query1.lower().split())
    q2_words = set(query2.lower().split())
    intersection = q1_words.intersection(q2_words)
    union = q1_words.union(q2_words)
    return len(intersection) / len(union) if union else 0.0


@timer
def process_query(question: str) -> str:
    """Main pipeline that checks cache first, then falls back to normal retrieval."""

    # 1. Check Cache
    matched_prediction = None
    best_sim = 0.0

    print_step(1, "Checking speculative cache...")
    for cached_q in SPECULATIVE_CACHE.keys():
        sim = calculate_similarity(question, cached_q)
        if sim > 0.4:  # Threshold for a "match" (lowered for demo purposes)
            best_sim = sim
            matched_prediction = cached_q

    if matched_prediction:
        print(f"  ⚡ CACHE HIT! User query matches prediction: '{matched_prediction}'")
        docs = SPECULATIVE_CACHE[matched_prediction]
    else:
        print("  🐌 CACHE MISS. Performing standard vector search...")
        docs = similarity_search(question, collection_name=COLLECTION_NAME, k=3)

    # 2. Generate Answer
    print_step(2, "Generating answer...")
    context = "\n\n".join(d.page_content for d in docs)
    answer = ask_llm_with_context(question=question, context=context)

    # 3. Trigger Speculation (Simulated Async)
    # We pass this to a separate thread so it doesn't block the user's response
    print_step(3, "Triggering background speculation for NEXT query...")
    predictions = generate_predictions(question, answer)
    thread = threading.Thread(target=prefetch_worker, args=(predictions,))
    thread.daemon = True
    thread.start()

    return answer


@timer
def setup_knowledge_base():
    docs = load_documents_from_directory()
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    delete_collection(COLLECTION_NAME)
    create_vector_store(chunks, collection_name=COLLECTION_NAME)


def run():
    """Demo: Show speculative caching speedup."""
    print_header("🔮 SPECULATIVE RAG — Predictive Pre-fetching")

    print("""
    Speculative RAG tries to guess what you will ask next and retrieves
    the documents BEFORE you even ask.

    Watch the sequence:
    1. First query: Normal speed (Cache Miss)
    2. Background: System guesses follow-ups and pre-fetches docs
    3. Second query (if predictable): Lightning fast (Cache Hit)
    """)

    setup_knowledge_base()

    # User's sequence of questions
    queries = [
        "What is the core mechanism of Large Language Models?",
        "What is supervised fine-tuning?", # Highly likely follow-up
    ]

    for i, q in enumerate(queries, 1):
        print(f"\n{'═'*60}")
        print(f"  👤 User asks: {q}")
        print(f"{'═'*60}")

        answer = process_query(q)
        print(f"\n  🤖 Answer: {answer[:150]}...")

        if i < len(queries):
            print("\n  [Simulating user reading the answer for 5 seconds...]")
            time.sleep(5) # Give the background thread time to finish

    print_header("📊 SPECULATIVE RAG — Key Takeaways")
    print("""
    Trade-offs:
    + Reduces perceived wait time for the user (zero-latency retrieval)
    + Creates a more natural, flowing conversation
    - Wastes API calls and compute if the user asks something unpredictable
    - Requires async architecture (Celery, background workers)
    """)

if __name__ == "__main__":
    run()
