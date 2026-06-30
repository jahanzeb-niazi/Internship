"""
11_corrective_rag.py — Corrective RAG (CRAG)
==============================================

WHAT IS CORRECTIVE RAG?
    CRAG evaluates the retrieved documents BEFORE generating an answer.
    If the documents are deemed irrelevant or incomplete, it attempts to
    "correct" the retrieval by running a web search or reformulating the query.

    Key difference from Self-RAG:
    - Self-RAG evaluates the GENERATED ANSWER.
    - Corrective RAG evaluates the RETRIEVED DOCUMENTS.

    This is more efficient because it prevents the LLM from generating
    an answer based on garbage context in the first place.

HOW IT WORKS:
    1. Retrieve documents from local knowledge base.
    2. Evaluate documents (Correct, Incorrect, or Ambiguous).
    3. Action based on evaluation:
       - Correct: Proceed to generation.
       - Incorrect/Ambiguous: Discard bad docs, rewrite query, search Web (or try again).
    4. Generate final answer.

BEST FOR:
    Systems where the local knowledge base might have gaps, and falling
    back to web search is acceptable.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple
from langchain_core.documents import Document

from shared.utils import (
    print_header, print_step, print_result, print_warning,
    load_documents_from_directory, chunk_documents, timer
)
from shared.vector_store import create_vector_store, similarity_search, delete_collection
from shared.llm import ask_llm, ask_llm_with_context


COLLECTION_NAME = "corrective_rag"


def evaluate_retrieval(question: str, documents: List[Document]) -> str:
    """
    CRAG Core: Evaluate the relevance of the retrieved documents to the query.
    Returns one of: 'correct', 'incorrect', 'ambiguous'.
    """
    context = "\n\n".join([f"Doc {i+1}: {d.page_content}" for i, d in enumerate(documents)])

    prompt = f"""Evaluate if the provided documents contain the necessary information to answer the question.

Question: {question}

Documents:
{context}

Classify the documents collectively as ONE of the following:
- "correct": The documents clearly contain the answer.
- "incorrect": The documents are completely irrelevant to the question.
- "ambiguous": The documents are related to the topic but don't fully answer the specific question.

Respond with ONLY the classification word."""

    eval_result = ask_llm(
        prompt=prompt,
        system_prompt="You are a retrieval evaluator. Respond with a single word."
    ).strip().lower()

    if eval_result not in ["correct", "incorrect", "ambiguous"]:
        eval_result = "ambiguous" # Safe fallback

    return eval_result


def rewrite_query(question: str) -> str:
    """Rewrite query for fallback search (simulated web search)."""
    prompt = f"""The original query did not return good results.
Rewrite it to be a broader, optimized search engine query to find the missing information.

Original query: {question}

Optimized query:"""

    return ask_llm(
        prompt=prompt,
        system_prompt="You optimize queries for search engines. Return only the query string."
    ).strip()


def simulated_web_search(query: str) -> List[Document]:
    """
    Simulates a fallback web search (e.g., using Tavily or Google Search API).
    In this demo, we just return a mock document indicating web search was used.
    """
    print(f"    🌐 [Simulating Web Search for: '{query}']")
    return [
        Document(
            page_content="[WEB RESULT] Large Language Models (LLMs) can suffer from hallucinations. To evaluate this, metrics like Groundedness, Faithfulness, and Answer Relevance (from frameworks like RAGAS) are commonly used.",
            metadata={"source": "simulated_web_search"}
        )
    ]


def corrective_rag_query(question: str) -> str:
    """Full CRAG pipeline."""
    # 1. Initial Retrieval
    print_step(1, f"Initial retrieval for: '{question}'")
    docs = similarity_search(question, collection_name=COLLECTION_NAME, k=3)

    # 2. Evaluate Documents
    print_step(2, "Evaluating retrieved documents...")
    eval_status = evaluate_retrieval(question, docs)
    print_result("Evaluation Status", eval_status.upper())

    # 3. Corrective Action
    final_docs = docs
    if eval_status == "correct":
        print("  ✓ Proceeding with local documents.")
    elif eval_status in ["incorrect", "ambiguous"]:
        print_step(3, f"Applying corrective action ({eval_status})...")
        new_query = rewrite_query(question)
        print(f"  ✏️ Rewritten query: '{new_query}'")
        # In full CRAG, we might filter out 'incorrect' local docs entirely.
        # Here we simulate falling back to the web to supplement.
        web_docs = simulated_web_search(new_query)

        if eval_status == "incorrect":
            final_docs = web_docs # Discard local, use only web
        else: # ambiguous
            final_docs = docs + web_docs # Combine local and web

    # 4. Generate Answer
    print_step(4, "Generating final answer...")
    context = "\n\n".join(d.page_content for d in final_docs)
    answer = ask_llm_with_context(question=question, context=context)

    return answer


@timer
def setup_knowledge_base():
    # Only load a subset of documents intentionally to create knowledge gaps
    # We omit doc_09 (evaluation) to force a fallback
    docs = load_documents_from_directory()
    docs = [d for d in docs if "doc_09" not in d.metadata.get("source", "")]

    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    delete_collection(COLLECTION_NAME)
    create_vector_store(chunks, collection_name=COLLECTION_NAME)
    print_result("Knowledge base ready", f"{len(chunks)} chunks indexed (Intentionally omitted some docs)")


def run():
    """Demo: Show CRAG falling back when local knowledge fails."""
    print_header("🛠️ CORRECTIVE RAG (CRAG)")

    print("""
    CRAG checks the retrieved documents BEFORE asking the LLM to generate an answer.
    If the documents don't contain the answer, it triggers a fallback (like a web search).

    We intentionally built a knowledge base missing the evaluation document.
    Watch how CRAG handles a query about evaluation!
    """)

    setup_knowledge_base()

    questions = [
        # Should be found in local KB -> Correct -> Generate
        "What are the main types of neural networks?",
        # Omitted from local KB -> Incorrect/Ambiguous -> Web Search -> Generate
        "What metrics are used in the RAGAS framework to evaluate RAG systems?",
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n{'═'*60}")
        print(f"  Question {i}: {q}")
        print(f"{'═'*60}")
        answer = corrective_rag_query(q)
        print(f"\n  🤖 Answer: {answer}")

    print_header("📊 CRAG — Key Takeaways")
    print("""
    Differences from Self-RAG:
    - CRAG prevents bad generation by gating the retrieval.
    - Self-RAG generates first, then critiques the generation.
    - CRAG is often faster because grading documents is a single LLM call, whereas
      Self-RAG might run generation (slow) multiple times.
    """)

if __name__ == "__main__":
    run()
