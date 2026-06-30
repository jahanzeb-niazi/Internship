"""
12_modular_rag.py — Modular RAG
=================================

WHAT IS MODULAR RAG?
    Breaks the RAG system into separate, swappable components (retriever,
    reranker, generator) that can be customized or upgraded independently.

    Instead of a hardcoded pipeline, it uses a configurable router/orchestrator.

HOW IT WORKS:
    You define an interface for each component.
    At runtime, you pass the specific implementations you want to use.
    For example:
    - Module 1: Dense Retriever vs BM25 Retriever
    - Module 2: Cross-Encoder Reranker vs LLM Reranker
    - Module 3: Basic Generator vs Streaming Generator

BEST FOR:
    Production environments where you need to A/B test different strategies,
    or enterprise platforms offering RAG-as-a-Service to different clients.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_core.documents import Document

from shared.utils import (
    print_header, print_step, print_result,
    load_documents_from_directory, chunk_documents
)
from shared.vector_store import create_vector_store, similarity_search, delete_collection
from shared.llm import ask_llm_with_context

COLLECTION_NAME = "modular_rag"

# ════════════════════════════════════════════════════
# MODULE INTERFACES
# ════════════════════════════════════════════════════

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> List[Document]:
        pass

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, documents: List[Document]) -> str:
        pass

# ════════════════════════════════════════════════════
# IMPLEMENTATIONS
# ════════════════════════════════════════════════════

class VectorRetriever(BaseRetriever):
    """Standard embedding-based similarity search."""
    def retrieve(self, query: str) -> List[Document]:
        print("    [Module: VectorRetriever] Executing semantic search...")
        return similarity_search(query, collection_name=COLLECTION_NAME, k=3)

class MockBM25Retriever(BaseRetriever):
    """Simulated keyword search for the demo."""
    def retrieve(self, query: str) -> List[Document]:
        print("    [Module: BM25Retriever] Executing exact keyword search...")
        # In a real app, use rank_bm25 here. Simulating by returning top 1 vector result.
        return similarity_search(query, collection_name=COLLECTION_NAME, k=1)


class StandardGenerator(BaseGenerator):
    """Standard LLM generation."""
    def generate(self, query: str, documents: List[Document]) -> str:
        print("    [Module: StandardGenerator] Generating standard response...")
        context = "\n\n".join(d.page_content for d in documents)
        return ask_llm_with_context(query, context)

class BulletPointGenerator(BaseGenerator):
    """Specialized generator for concise formatting."""
    def generate(self, query: str, documents: List[Document]) -> str:
        print("    [Module: BulletPointGenerator] Generating bulleted response...")
        context = "\n\n".join(d.page_content for d in documents)
        return ask_llm_with_context(
            query,
            context,
            system_prompt="Answer the question based on the context. You MUST format your entire response as a Markdown bulleted list."
        )

# ════════════════════════════════════════════════════
# ORCHESTRATOR
# ════════════════════════════════════════════════════

class ModularRAGPipeline:
    def __init__(self, retriever: BaseRetriever, generator: BaseGenerator):
        self.retriever = retriever
        self.generator = generator

    def query(self, question: str) -> str:
        docs = self.retriever.retrieve(question)
        answer = self.generator.generate(question, docs)
        return answer


def setup_knowledge_base():
    docs = load_documents_from_directory()
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    delete_collection(COLLECTION_NAME)
    create_vector_store(chunks, collection_name=COLLECTION_NAME)

def run():
    print_header("🧩 MODULAR RAG")

    print("""
    Modular RAG uses swappable components.
    We will run the exact same query through two completely different pipeline configurations.
    """)

    setup_knowledge_base()
    q = "What are the limitations of neural networks?"
    print(f"\n  Query: {q}")

    print_step(1, "Running Config A (Vector + Standard)")
    pipeline_a = ModularRAGPipeline(
        retriever=VectorRetriever(),
        generator=StandardGenerator()
    )
    ans_a = pipeline_a.query(q)
    print(f"  Result: {ans_a[:100]}...\n")

    print_step(2, "Running Config B (BM25 + BulletPoints)")
    pipeline_b = ModularRAGPipeline(
        retriever=MockBM25Retriever(),
        generator=BulletPointGenerator()
    )
    ans_b = pipeline_b.query(q)
    print(f"  Result:\n{ans_b}")

if __name__ == "__main__":
    run()
