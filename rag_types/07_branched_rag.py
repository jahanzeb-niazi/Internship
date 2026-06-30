"""
07_branched_rag.py — Branched RAG (Parallel Multi-Interpretation RAG)
======================================================================

WHAT IS BRANCHED RAG?
    Generates responses for MULTIPLE interpretations of a question
    simultaneously, then COMPARES them to pick the most relevant.

    Example: "What is Python?"
    Branch 1 interprets as: "What is the Python programming language?"
    Branch 2 interprets as: "What is a python snake?"
    Branch 3 interprets as: "What is the Python IDE/framework?"
    → Each branch retrieves and generates independently
    → A merger picks the best answer

BEST FOR:
    Open-ended questions, ambiguous queries, complex questions
    that could be interpreted multiple ways.

PIPELINE:
    Question → Generate interpretations → [Branch 1] → Answer 1
                                        → [Branch 2] → Answer 2
                                        → [Branch 3] → Answer 3
              → Compare & Merge → Best Answer
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import List, Dict

from shared.utils import (
    print_header, print_step, print_result,
    load_documents_from_directory, chunk_documents, timer
)
from shared.vector_store import create_vector_store, similarity_search, delete_collection
from shared.llm import ask_llm, ask_llm_with_context


COLLECTION_NAME = "branched_rag"


def generate_interpretations(question: str, num_branches: int = 3) -> List[str]:
    """
    Generate multiple interpretations of a question.

    This is the "branching" step — we acknowledge that a question
    can have multiple valid interpretations and explore all of them.
    """
    prompt = f"""Generate {num_branches} different interpretations of this question.
Each interpretation should focus on a different angle or aspect.

Original Question: {question}

Return a JSON list of {num_branches} rewritten questions.
Example: ["interpretation 1", "interpretation 2", "interpretation 3"]

Interpretations:"""

    response = ask_llm(
        prompt=prompt,
        system_prompt="You generate diverse interpretations of questions. Return only a JSON list.",
    )

    try:
        clean = response.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1]
            clean = clean.rsplit("```", 1)[0]
        interpretations = json.loads(clean)
        return interpretations[:num_branches]
    except (json.JSONDecodeError, TypeError):
        return [question]  # Fallback: just use the original


def process_branch(interpretation: str, branch_id: int) -> Dict:
    """
    Process a single branch: retrieve documents and generate an answer.

    Each branch works independently — like parallel Simple RAG pipelines,
    each with a different interpretation of the question.
    """
    print(f"\n  🌿 Branch {branch_id}: '{interpretation}'")

    # Retrieve documents for this interpretation
    docs = similarity_search(interpretation, collection_name=COLLECTION_NAME, k=3)
    context = "\n\n---\n\n".join(d.page_content for d in docs)

    # Generate answer for this branch
    answer = ask_llm_with_context(question=interpretation, context=context)

    # Get source info
    sources = [d.metadata.get("source", "unknown") for d in docs]

    return {
        "branch_id": branch_id,
        "interpretation": interpretation,
        "answer": answer,
        "sources": sources,
        "num_docs": len(docs),
    }


def merge_branches(
    original_question: str,
    branches: List[Dict],
) -> str:
    """
    Compare branch results and produce the best final answer.

    This is the "comparison" step. The LLM acts as a judge, evaluating
    which branch produced the most relevant and comprehensive answer.
    """
    branch_summaries = ""
    for b in branches:
        branch_summaries += f"""
--- Branch {b['branch_id']} ---
Interpretation: {b['interpretation']}
Answer: {b['answer']}
Sources: {', '.join(b['sources'])}
"""

    prompt = f"""You are evaluating multiple answers to the same question.
Each answer was generated from a different interpretation of the question.

Original Question: {original_question}

{branch_summaries}

Your task:
1. Determine which interpretation best matches the user's likely intent
2. Synthesize the BEST answer by combining the most relevant parts from all branches
3. If multiple branches contribute useful information, merge them

Final Answer:"""

    final_answer = ask_llm(
        prompt=prompt,
        system_prompt=(
            "You merge multiple answer branches into one optimal response. "
            "Be comprehensive but avoid redundancy."
        ),
    )

    return final_answer


def branched_rag_query(question: str, num_branches: int = 3) -> str:
    """
    Full Branched RAG pipeline: interpret → branch → merge.
    """
    # Step 1: Generate interpretations
    print_step(1, "Generating multiple interpretations...")
    interpretations = generate_interpretations(question, num_branches)
    for i, interp in enumerate(interpretations, 1):
        print(f"  📌 Interpretation {i}: {interp}")

    # Step 2: Process each branch independently
    print_step(2, "Processing branches in parallel (simulated)...")
    branches = []
    for i, interp in enumerate(interpretations, 1):
        branch_result = process_branch(interp, i)
        branches.append(branch_result)
        print(f"    ✓ Branch {i}: Generated answer ({branch_result['num_docs']} docs)")

    # Step 3: Merge and select the best answer
    print_step(3, "Merging branch results...")
    final_answer = merge_branches(question, branches)

    return final_answer


@timer
def setup_knowledge_base():
    """Index documents for Branched RAG."""
    docs = load_documents_from_directory()
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    delete_collection(COLLECTION_NAME)
    create_vector_store(chunks, collection_name=COLLECTION_NAME)
    print_result("Knowledge base ready", f"{len(chunks)} chunks indexed")


def run():
    """Demo: Show branched interpretation and merging."""
    print_header("🌿 BRANCHED RAG — Parallel Multi-Interpretation")

    print("""
    Branched RAG explores MULTIPLE interpretations simultaneously.
    Watch how:
    ✓ A single question spawns 3 different interpretations
    ✓ Each interpretation retrieves and generates independently
    ✓ Results are compared and merged into the best answer

    This handles ambiguity that single-path RAG misses.
    """)

    setup_knowledge_base()

    questions = [
        # Ambiguous — could be about multiple things
        "How do neural networks learn and what makes them effective?",

        # Multi-aspect — different angles to explore
        "What are the challenges with modern AI systems?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'═'*60}")
        print(f"  Question {i}: {question}")
        print(f"{'═'*60}")

        answer = branched_rag_query(question)

        print(f"\n  🤖 Final Merged Answer:")
        print(f"  {answer}")

    print_header("📊 BRANCHED RAG — Key Takeaways")
    print("""
    ┌─────────────────────┬─────────────┬──────────────────┐
    │ Feature             │ Simple RAG  │ Branched RAG     │
    ├─────────────────────┼─────────────┼──────────────────┤
    │ Interpretations     │ 1           │ Multiple (3+)    │
    │ Retrieval Paths     │ 1           │ Parallel         │
    │ Answer Candidates   │ 1           │ Multiple         │
    │ Handles Ambiguity   │ Poorly      │ Well             │
    │ Speed               │ Fast        │ 3x slower        │
    │ Cost                │ 1x          │ 3-4x             │
    └─────────────────────┴─────────────┴──────────────────┘

    Trade-offs:
    + Handles open-ended and complex questions well
    + Explores angles a single interpretation would miss
    - Can overwhelm users with information if not filtered properly
    - Significantly more expensive (N branches × API calls)
    """)


if __name__ == "__main__":
    run()
