"""
04_agentic_rag.py — Agentic RAG (The Decision-Making RAG)
===========================================================

WHAT IS AGENTIC RAG?
    An AI agent that DECIDES how to answer your question. Instead of
    blindly retrieving and generating, it breaks tasks into steps,
    searches multiple sources, and keeps iterating until it finds
    a satisfactory answer.

    Think of it as giving the RAG system a "brain" that can:
    - Decide IF it needs to search at all
    - Choose WHICH tool to use (vector search, web search, etc.)
    - Evaluate its own answer and try again if needed
    - Combine information from multiple searches

HOW IT DIFFERS FROM SIMPLE RAG:
    Simple RAG: Query → Search → Generate (one shot, no thinking)
    Agentic RAG: Query → Think → Maybe Search → Think → Maybe Search Again → Generate

BEST FOR:
    Legal research, financial analysis, complex multi-step reasoning,
    any task requiring information from multiple sources.

BUILT WITH:
    LangGraph — a library for building stateful, multi-step AI workflows
    as directed graphs (nodes = actions, edges = decisions).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import TypedDict, List, Annotated
import operator

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from shared.utils import (
    print_header, print_step, print_result, print_documents,
    print_warning, load_documents_from_directory, chunk_documents, timer
)
from shared.vector_store import create_vector_store, delete_collection
from shared.llm import get_llm, ask_llm


COLLECTION_NAME = "agentic_rag"


# ════════════════════════════════════════════════════
# STEP 1: Define the Agent's State
# ════════════════════════════════════════════════════
# The state is a dictionary that flows through the graph.
# Each node can read and update this state.

class AgentState(TypedDict):
    """
    The state that flows through our agent graph.

    Think of this as the agent's "working memory" — it holds everything
    the agent needs to make decisions and produce output.
    """
    question: str                          # The user's original question
    documents: List[Document]              # Retrieved documents
    generation: str                        # The generated answer
    search_count: int                      # How many searches we've done
    max_searches: int                      # Safety limit to prevent infinite loops
    decision: str                          # Agent's decision: "search", "generate", "done"


# ════════════════════════════════════════════════════
# STEP 2: Define the Agent's Nodes (Actions)
# ════════════════════════════════════════════════════

def route_question(state: AgentState) -> AgentState:
    """
    NODE 1: The "Brain" — Decide what to do with the question.

    The agent THINKS before acting. It analyzes the question and decides:
    - "search": Need to retrieve documents
    - "generate": Can answer directly (simple greetings, etc.)

    This is what makes it AGENTIC — it doesn't blindly search for everything.
    """
    question = state["question"]

    print(f"\n  🧠 Agent thinking about: '{question}'")

    decision_prompt = f"""Analyze this question and decide the best action:

Question: {question}

Choose ONE action:
- "search" if the question requires factual information from documents
- "generate" if this is a simple greeting, opinion, or doesn't need retrieval

Respond with ONLY the action word, nothing else."""

    decision = ask_llm(
        prompt=decision_prompt,
        system_prompt="You are a decision-making agent. Respond with only 'search' or 'generate'.",
    ).strip().lower()

    # Default to search if uncertain
    if decision not in ["search", "generate"]:
        decision = "search"

    print(f"  🧠 Decision: {decision}")

    return {**state, "decision": decision}


def retrieve_documents(state: AgentState) -> AgentState:
    """
    NODE 2: Retrieve documents from the vector store.

    This is similar to Simple RAG's retrieval, but the agent
    CHOSE to do this (it wasn't automatic).
    """
    question = state["question"]
    search_count = state.get("search_count", 0)

    print(f"  🔍 Search #{search_count + 1}: Retrieving documents...")

    from shared.vector_store import similarity_search
    docs = similarity_search(question, collection_name=COLLECTION_NAME, k=4)

    # Merge with any previously retrieved documents (from earlier searches)
    existing_docs = state.get("documents", [])
    # Deduplicate by content
    existing_contents = {d.page_content for d in existing_docs}
    new_docs = [d for d in docs if d.page_content not in existing_contents]
    all_docs = existing_docs + new_docs

    print(f"  🔍 Found {len(new_docs)} new documents (total: {len(all_docs)})")

    return {
        **state,
        "documents": all_docs,
        "search_count": search_count + 1,
    }


def grade_documents(state: AgentState) -> AgentState:
    """
    NODE 3: Grade whether retrieved documents are relevant.

    The agent EVALUATES its own retrieval results. If documents
    aren't relevant enough, it can decide to search again with
    a different strategy.

    This is a key difference from Simple RAG — quality control.
    """
    question = state["question"]
    documents = state["documents"]

    print(f"  📊 Grading {len(documents)} documents for relevance...")

    relevant_docs = []
    for doc in documents:
        grade_prompt = f"""Is this document relevant to answering the question?

Question: {question}

Document: {doc.page_content[:300]}

Respond with ONLY 'yes' or 'no'."""

        grade = ask_llm(
            prompt=grade_prompt,
            system_prompt="You grade document relevance. Respond only 'yes' or 'no'.",
        ).strip().lower()

        if "yes" in grade:
            relevant_docs.append(doc)

    print(f"  📊 Relevant: {len(relevant_docs)} / {len(documents)}")

    # Decide next step based on grading results
    if len(relevant_docs) == 0 and state["search_count"] < state["max_searches"]:
        decision = "search"  # No relevant docs — try again
        print(f"  📊 No relevant docs found. Will retry search.")
    else:
        decision = "generate"

    return {
        **state,
        "documents": relevant_docs,
        "decision": decision,
    }


def rewrite_query(state: AgentState) -> AgentState:
    """
    NODE 4: Rewrite the query for better retrieval.

    If the first search didn't find good results, the agent
    tries a DIFFERENT search strategy by rephrasing the question.
    """
    question = state["question"]

    print(f"  ✏️ Rewriting query for better results...")

    rewrite_prompt = f"""The following question didn't retrieve good results from our knowledge base.
Rewrite it to be more specific and use different keywords that might match better.

Original question: {question}

Rewritten question:"""

    new_question = ask_llm(
        prompt=rewrite_prompt,
        system_prompt="Rewrite questions for better document retrieval. Return only the rewritten question.",
    ).strip()

    print(f"  ✏️ Rewritten: '{new_question}'")

    return {**state, "question": new_question}


def generate_answer(state: AgentState) -> AgentState:
    """
    NODE 5: Generate the final answer using retrieved documents.
    """
    question = state["question"]
    documents = state.get("documents", [])

    print(f"  💬 Generating answer from {len(documents)} documents...")

    if documents:
        context = "\n\n---\n\n".join(d.page_content for d in documents)
        prompt = f"""Answer the question based on the following context.

Context:
{context}

Question: {question}

Answer:"""
    else:
        prompt = f"Answer this question to the best of your ability: {question}"

    answer = ask_llm(
        prompt=prompt,
        system_prompt=(
            "You are a helpful assistant. Answer based on the provided context. "
            "If no context is available, use your general knowledge but mention "
            "that no specific documents were found."
        ),
    )

    return {**state, "generation": answer, "decision": "done"}


# ════════════════════════════════════════════════════
# STEP 3: Build the Agent Graph
# ════════════════════════════════════════════════════

def run_agent(question: str, retriever_store=None) -> str:
    """
    Run the agentic RAG pipeline.

    Instead of using LangGraph (which requires additional setup),
    we implement the graph logic manually to make the concept clearer.

    The flow is:
        route_question → [search path] → retrieve → grade → [maybe rewrite & retry] → generate
                       → [generate path] → generate directly
    """
    # Initialize state
    state: AgentState = {
        "question": question,
        "documents": [],
        "generation": "",
        "search_count": 0,
        "max_searches": 3,  # Safety limit
        "decision": "",
    }

    # ── Node 1: Route the question ──
    state = route_question(state)

    if state["decision"] == "generate":
        # Simple question — skip retrieval
        state = generate_answer(state)
        return state["generation"]

    # ── Search loop (the "agentic" part) ──
    while state["decision"] == "search" and state["search_count"] < state["max_searches"]:
        # Node 2: Retrieve
        state = retrieve_documents(state)

        # Node 3: Grade
        state = grade_documents(state)

        # If still needs search, rewrite the query first
        if state["decision"] == "search":
            state = rewrite_query(state)

    # ── Node 5: Generate final answer ──
    state = generate_answer(state)

    return state["generation"]


@timer
def setup_knowledge_base():
    """Index documents for the agentic RAG demo."""
    print_step(1, "Loading and indexing documents...")
    docs = load_documents_from_directory()
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    delete_collection(COLLECTION_NAME)
    store = create_vector_store(chunks, collection_name=COLLECTION_NAME)
    print_result("Knowledge base ready", f"{len(chunks)} chunks indexed")
    return store


def run():
    """Demo: Show the agent making decisions about how to answer."""
    print_header("🤖 AGENTIC RAG — The Decision-Making RAG")

    print("""
    Agentic RAG doesn't just search and answer — it THINKS.
    Watch the agent:
    ✓ Decide WHETHER to search
    ✓ Grade retrieved documents for relevance
    ✓ Rewrite queries if initial results are poor
    ✓ Iterate until it finds a good answer

    The key insight: the agent has AGENCY — it chooses its own path.
    """)

    setup_knowledge_base()

    questions = [
        # Simple greeting — agent should skip retrieval
        "Hello, how are you?",

        # Factual question — agent should search
        "What is the difference between CNNs and RNNs?",

        # Complex question — might require multiple searches
        "How does RAG solve the hallucination problem in LLMs, and what metrics can I use to evaluate this?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'═'*60}")
        print(f"  Question {i}: {question}")
        print(f"{'═'*60}")

        answer = run_agent(question)

        print(f"\n  🤖 Final Answer:")
        print(f"  {answer}")

    print_header("📊 AGENTIC RAG — How It Differs")
    print("""
    Comparison:
    ┌────────────────────┬──────────────┬──────────────────┐
    │ Feature            │ Simple RAG   │ Agentic RAG      │
    ├────────────────────┼──────────────┼──────────────────┤
    │ Retrieval          │ Always       │ Only when needed  │
    │ Quality Check      │ None         │ Document grading  │
    │ Query Optimization │ None         │ Auto-rewrite      │
    │ Retry Logic        │ None         │ Up to N attempts  │
    │ Multi-source       │ No           │ Yes               │
    │ Speed              │ Fast         │ Slower            │
    │ Cost               │ Low          │ Higher            │
    └────────────────────┴──────────────┴──────────────────┘
    """)


if __name__ == "__main__":
    run()
