"""
02_simple_rag.py — Simple RAG (The Standard Approach)
======================================================

WHAT IS SIMPLE RAG?
    The "correct" basic implementation of RAG. Converts a user query into
    embeddings, searches a vector DB for semantically similar content,
    and feeds the results to an LLM.

    Unlike Naive RAG, Simple RAG uses proper embedding-based semantic
    search and structures the prompt carefully.

HOW IT DIFFERS FROM NAIVE RAG:
    - Uses LangChain's retrieval chain (structured pipeline)
    - Better prompt engineering (system message + context formatting)
    - Returns source documents with the answer (transparency)
    - Uses a proper retriever interface (not raw similarity search)

BEST FOR:
    Basic Q&A, chatbots, FAQ systems, internal knowledge bases.

PIPELINE:
    User Query → Embed → Semantic Search → Format Context → LLM Chain → Answer + Sources
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from shared.utils import (
    print_header, print_step, print_result, print_documents,
    load_documents_from_directory, chunk_documents, timer
)
from shared.vector_store import create_vector_store, delete_collection
from shared.llm import get_llm
from shared.embeddings import get_embedding_model


COLLECTION_NAME = "simple_rag"


def format_docs(docs):
    """
    Helper: Join retrieved documents into a single string for the prompt.

    Each document is separated by a divider so the LLM can distinguish
    between different source materials.
    """
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


@timer
def build_simple_rag_chain():
    """
    Build the Simple RAG chain using LangChain's LCEL (LangChain Expression Language).

    LCEL lets us compose retrieval + generation into a single, reusable pipeline.
    This is the RECOMMENDED way to build RAG in LangChain.

    The chain works like this:
        1. Take the user's question
        2. Pass it to the retriever (vector search)
        3. Format retrieved docs as context
        4. Inject context + question into the prompt template
        5. Send to LLM
        6. Parse the output as a string
    """
    # ── Step 1: Load and chunk documents ──
    print_step(1, "Loading and chunking documents...")
    docs = load_documents_from_directory()
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    print_result("Chunks created", str(len(chunks)))

    # ── Step 2: Create vector store and retriever ──
    print_step(2, "Creating vector store and retriever...")
    delete_collection(COLLECTION_NAME)
    vector_store = create_vector_store(chunks, collection_name=COLLECTION_NAME)

    # A retriever is an abstraction over vector search
    # search_kwargs={"k": 4} means "return the 4 most similar documents"
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    print_result("Retriever ready", "top-4 semantic search")

    # ── Step 3: Create the prompt template ──
    # This is MUCH better than Naive RAG's simple string concatenation.
    # The template clearly separates the system instruction, context, and question.
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful AI assistant. Answer the user's question based "
            "ONLY on the provided context. If the context doesn't contain the "
            "answer, say 'I don't have enough information to answer this.'\n\n"
            "Always cite which source document(s) your answer is based on."
        )),
        ("human", (
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )),
    ])

    # ── Step 4: Create the LLM ──
    llm = get_llm()

    # ── Step 5: Compose the chain using LCEL ──
    # The | operator pipes data from one component to the next
    # RunnablePassthrough() passes the question unchanged
    chain = (
        {
            "context": retriever | format_docs,  # Retrieve → format
            "question": RunnablePassthrough(),     # Pass question through
        }
        | prompt    # Inject into template
        | llm       # Generate answer
        | StrOutputParser()  # Extract text from LLM response
    )

    print_result("RAG chain built", "retriever → prompt → LLM → output")

    return chain, retriever


def run():
    """
    Main demo: Shows Simple RAG and compares it to Naive RAG.
    """
    print_header("🔗 SIMPLE RAG — The Standard Approach")

    print("""
    Simple RAG is the industry-standard basic RAG pipeline.
    It's a step up from Naive RAG with:
    ✓ Proper LangChain retrieval chain
    ✓ Structured prompt with system message
    ✓ Source document citation
    ✓ Cleaner code using LCEL (LangChain Expression Language)

    Watch for:
    → Better answer quality due to structured prompting
    → Source citations in the answers
    → But still limited to single-step retrieval (no self-correction)
    """)

    # ── Build the chain ──
    chain, retriever = build_simple_rag_chain()

    # ── Test questions ──
    questions = [
        "What is the self-attention mechanism in transformers?",
        "How do embeddings represent meaning?",
        "What are the main challenges of deploying RAG in production?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'─'*50}")
        print(f"  Question {i}: {question}")
        print(f"{'─'*50}")

        # ── Show what documents were retrieved ──
        print_step(3, "Retrieving relevant documents...")
        retrieved_docs = retriever.invoke(question)
        print("\n  📄 Retrieved Documents:")
        print_documents(retrieved_docs, max_chars=150)

        # ── Run the full chain ──
        print_step(4, "Generating answer with RAG chain...")
        answer = chain.invoke(question)

        print(f"\n  💬 Answer:")
        print(f"  {answer}")

    # ── Compare with Naive RAG ──
    print_header("📊 SIMPLE RAG vs NAIVE RAG")
    print("""
    Key Improvements over Naive RAG:
    ┌─────────────────────┬──────────────┬──────────────┐
    │ Feature             │ Naive RAG    │ Simple RAG   │
    ├─────────────────────┼──────────────┼──────────────┤
    │ Prompt Structure    │ Basic concat │ Template     │
    │ System Message      │ ✗            │ ✓            │
    │ Source Citations    │ ✗            │ ✓            │
    │ Reusable Chain      │ ✗            │ ✓ (LCEL)     │
    │ Retriever Interface │ Raw search   │ Abstracted   │
    ├─────────────────────┼──────────────┼──────────────┤
    │ Query Rewriting     │ ✗            │ ✗            │
    │ Reranking           │ ✗            │ ✗            │
    │ Self-Correction     │ ✗            │ ✗            │
    └─────────────────────┴──────────────┴──────────────┘

    Still Missing:
    → No memory of previous questions (see: 03_simple_rag_with_memory)
    → No query optimization (see: 05_query_rewriting technique)
    → No self-correction (see: 06_self_rag, 11_corrective_rag)
    """)


if __name__ == "__main__":
    run()
