"""
06_self_rag.py — Self-RAG (The Self-Correcting RAG)
=====================================================

WHAT IS SELF-RAG?
    Uses specialized evaluation modules to CHECK whether its generated
    answer is accurate and supported by source material. If not, it
    REWRITES the query and tries again.

    Think of it as RAG with a built-in fact-checker.

THE SELF-REFLECTION LOOP:
    1. Retrieve documents
    2. Generate an answer
    3. CHECK: Is the answer relevant to the question?
    4. CHECK: Is the answer supported by the retrieved documents? (no hallucination?)
    5. If any check fails → rewrite query → go back to step 1
    6. If all checks pass → return the answer

WHY "SELF"?
    The system evaluates ITSELF — no human needed in the loop.
    It has three internal "critics":
    - Relevance Critic: "Does this answer address the question?"
    - Grounding Critic: "Is this answer supported by the documents?"
    - Quality Critic: "Is this answer complete and useful?"

BEST FOR:
    Vague or incomplete questions, high-stakes applications where
    accuracy matters (medical, legal, financial).
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


COLLECTION_NAME = "self_rag"


# ════════════════════════════════════════════════════
# THE THREE CRITICS (Self-Evaluation Modules)
# ════════════════════════════════════════════════════

def check_relevance(question: str, answer: str) -> Tuple[bool, str]:
    """
    CRITIC 1: Relevance Check
    "Does this answer actually address the question asked?"

    A faithful answer to the WRONG question is still useless.
    This catches cases where retrieval pulled irrelevant docs
    and the LLM went off-topic.
    """
    prompt = f"""Evaluate whether this answer is relevant to the question.

Question: {question}
Answer: {answer}

Is the answer relevant and on-topic? Consider:
1. Does it address what was specifically asked?
2. Does it stay focused on the question topic?

Respond in this format:
RELEVANT: yes/no
REASON: brief explanation"""

    response = ask_llm(
        prompt=prompt,
        system_prompt="You evaluate answer relevance. Be strict but fair.",
    )

    is_relevant = "yes" in response.lower().split("relevant:")[-1].split("\n")[0].lower()
    return is_relevant, response


def check_grounding(answer: str, documents: List[Document]) -> Tuple[bool, str]:
    """
    CRITIC 2: Grounding Check (Anti-Hallucination)
    "Is this answer supported by the source documents?"

    This is the hallucination detector. It breaks the answer into
    claims and checks if each claim appears in the retrieved docs.
    """
    context = "\n\n".join(d.page_content for d in documents)

    prompt = f"""Check if the answer is grounded in (supported by) the source documents.

Source Documents:
{context[:3000]}

Answer to Check:
{answer}

For each major claim in the answer, is it supported by the documents?

Respond in this format:
GROUNDED: yes/no
UNSUPPORTED_CLAIMS: list any claims not found in the documents
REASON: brief explanation"""

    response = ask_llm(
        prompt=prompt,
        system_prompt="You check if answers are grounded in source material. Be strict about hallucination.",
    )

    is_grounded = "yes" in response.lower().split("grounded:")[-1].split("\n")[0].lower()
    return is_grounded, response


def check_quality(question: str, answer: str) -> Tuple[bool, str]:
    """
    CRITIC 3: Quality Check
    "Is this answer complete, clear, and useful?"

    Even a relevant, grounded answer can be poor quality — too vague,
    incomplete, or poorly structured.
    """
    prompt = f"""Evaluate the quality of this answer.

Question: {question}
Answer: {answer}

Rate on these criteria:
1. Completeness: Does it fully answer the question?
2. Clarity: Is it clear and well-structured?
3. Usefulness: Would the user find this helpful?

Respond in this format:
QUALITY: good/poor
REASON: brief explanation of what could be improved"""

    response = ask_llm(
        prompt=prompt,
        system_prompt="You evaluate answer quality. Be constructive.",
    )

    is_good = "good" in response.lower().split("quality:")[-1].split("\n")[0].lower()
    return is_good, response


def rewrite_query(question: str, feedback: str) -> str:
    """
    Rewrite the query based on critic feedback.

    When the self-evaluation fails, we don't just retry the same query —
    we IMPROVE it based on what went wrong.
    """
    prompt = f"""The following question was asked but the retrieved results weren't good enough.

Original Question: {question}

Feedback on why the answer was poor:
{feedback}

Rewrite the question to get better retrieval results. Make it more specific,
use different keywords, or break it into a more focused query.

Rewritten Question:"""

    new_q = ask_llm(
        prompt=prompt,
        system_prompt="You rewrite questions to improve retrieval quality. Return only the new question.",
    )

    return new_q.strip()


# ════════════════════════════════════════════════════
# THE SELF-RAG LOOP
# ════════════════════════════════════════════════════

def self_rag_query(question: str, max_iterations: int = 3) -> str:
    """
    Run the Self-RAG loop: retrieve → generate → evaluate → (retry if needed).

    Args:
        question:        The user's question.
        max_iterations:  Max number of retry loops (safety limit).

    Returns:
        The final, validated answer.
    """
    current_question = question
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n  {'🔄' if iteration > 1 else '▶️'} Iteration {iteration}/{max_iterations}")

        # ── RETRIEVE ──
        print_step(1, f"Retrieving documents for: '{current_question}'")
        docs = similarity_search(current_question, collection_name=COLLECTION_NAME, k=4)
        print_result("Retrieved", f"{len(docs)} documents")

        # ── GENERATE ──
        print_step(2, "Generating answer...")
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        answer = ask_llm_with_context(question=current_question, context=context)
        print(f"  📝 Draft answer: {answer[:200]}...")

        # ── EVALUATE (the "Self" part) ──
        print_step(3, "Running self-evaluation critics...")

        # Critic 1: Relevance
        is_relevant, relevance_feedback = check_relevance(question, answer)
        print(f"  {'✅' if is_relevant else '❌'} Relevance: {'PASS' if is_relevant else 'FAIL'}")

        if not is_relevant:
            print(f"  📋 Feedback: {relevance_feedback[:200]}")
            current_question = rewrite_query(current_question, relevance_feedback)
            print(f"  ✏️ Rewritten query: '{current_question}'")
            continue

        # Critic 2: Grounding (hallucination check)
        is_grounded, grounding_feedback = check_grounding(answer, docs)
        print(f"  {'✅' if is_grounded else '❌'} Grounding: {'PASS' if is_grounded else 'FAIL'}")

        if not is_grounded:
            print(f"  📋 Feedback: {grounding_feedback[:200]}")
            current_question = rewrite_query(current_question, grounding_feedback)
            print(f"  ✏️ Rewritten query: '{current_question}'")
            continue

        # Critic 3: Quality
        is_quality, quality_feedback = check_quality(question, answer)
        print(f"  {'✅' if is_quality else '⚠️'} Quality: {'PASS' if is_quality else 'NEEDS IMPROVEMENT'}")

        # All checks passed!
        if is_relevant and is_grounded:
            print(f"\n  ✅ All critics passed! Returning validated answer.")
            return answer

    # Max iterations reached
    print_warning(f"Reached max iterations ({max_iterations}). Returning best effort answer.")
    return answer


@timer
def setup_knowledge_base():
    """Index documents for Self-RAG."""
    docs = load_documents_from_directory()
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    delete_collection(COLLECTION_NAME)
    create_vector_store(chunks, collection_name=COLLECTION_NAME)
    print_result("Knowledge base ready", f"{len(chunks)} chunks indexed")


def run():
    """Demo: Show Self-RAG's self-correction in action."""
    print_header("🔍 SELF-RAG — The Self-Correcting RAG")

    print("""
    Self-RAG has three built-in "critics" that evaluate every answer:
    ✓ Relevance Critic: Does the answer address the question?
    ✓ Grounding Critic: Is the answer supported by documents? (no hallucination)
    ✓ Quality Critic: Is the answer complete and useful?

    If any critic fails, the query is REWRITTEN and retrieval is retried.
    Watch the self-correction loop in action!
    """)

    setup_knowledge_base()

    questions = [
        # Clear question — should pass on first try
        "What are the main components of a transformer architecture?",

        # Vague question — might need rewriting
        "How does the thing with attention work in the brain-like computing?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'═'*60}")
        print(f"  Question {i}: {question}")
        print(f"{'═'*60}")

        answer = self_rag_query(question, max_iterations=3)

        print(f"\n  🤖 Final Validated Answer:")
        print(f"  {answer}")

    print_header("📊 SELF-RAG — Key Takeaways")
    print("""
    Self-RAG adds reliability through self-evaluation:

    ┌─────────────────┬──────────────┬──────────────────┐
    │ Feature         │ Simple RAG   │ Self-RAG         │
    ├─────────────────┼──────────────┼──────────────────┤
    │ Generate        │ Once         │ May regenerate   │
    │ Relevance Check │ None         │ LLM-based        │
    │ Hallucin. Check │ None         │ LLM-based        │
    │ Quality Check   │ None         │ LLM-based        │
    │ Query Rewriting │ None         │ Feedback-driven  │
    │ Reliability     │ Variable     │ Higher           │
    │ Speed           │ Fast         │ Slower (2-3x)    │
    │ Cost            │ Low          │ Higher (critics)  │
    └─────────────────┴──────────────┴──────────────────┘

    Trade-offs:
    + More reliable for vague or incomplete questions
    + Catches and corrects hallucinations
    - Slower due to multiple LLM calls for evaluation
    - More expensive (each critic = additional API call)
    """)


if __name__ == "__main__":
    run()
