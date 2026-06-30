"""
01_text_chunking.py — Text Chunking Strategies
================================================

WHAT IT IS:
    Breaking large documents into smaller pieces (chunks) before embedding.
    "Don't feed the AI a whole textbook at once."

WHY IT MATTERS:
    Embedding models have token limits (context windows). More importantly,
    a 10-page document has a very "diluted" embedding vector because it
    covers too many topics. A 5-sentence chunk has a focused vector that
    matches queries much better.

STRATEGIES SHOWN HERE:
    1. Fixed-Size Chunking (Basic)
    2. Recursive Character Splitting (Standard/LangChain Default)
    3. Semantic Chunking (Advanced, LLM/Embedding-based)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
# Using a simple experimental semantic chunker for demonstration
from langchain_experimental.text_splitter import SemanticChunker

from shared.utils import print_header, print_step
from shared.embeddings import get_embedding_model


SAMPLE_TEXT = """Artificial Intelligence (AI) has rapidly evolved over the last decade. It began with simple rule-based systems but has now transitioned into complex neural networks.

Machine learning, a subset of AI, involves training algorithms on large datasets. These algorithms learn patterns and make predictions. Deep learning goes even further, using multi-layered neural networks.

Natural Language Processing (NLP) allows machines to understand human language. This has led to the development of chatbots and translation services. Computer vision, on the other hand, enables machines to interpret visual data from the world.

The ethical implications of AI are heavily debated. Issues regarding bias, privacy, and job displacement are at the forefront of discussions. As AI continues to integrate into daily life, regulations and guidelines are being drafted globally to ensure its safe deployment."""

def demo_fixed_size():
    """
    Fixed-size splitting chops text blindly every N characters.
    Problem: It often cuts words or sentences in half, destroying meaning.
    """
    print_step(1, "Fixed-Size Chunking (chunk_size=100, overlap=20)")
    splitter = CharacterTextSplitter(
        separator="", # Empty separator forces fixed size cutoff
        chunk_size=100,
        chunk_overlap=20
    )
    chunks = splitter.split_text(SAMPLE_TEXT)

    for i, c in enumerate(chunks):
        print(f"  [{i}] ({len(c)} chars): {repr(c)}")
    print("\n  Notice how words are cut in half (e.g., 'transi-tioned').")

def demo_recursive():
    """
    Recursive splitting tries to keep paragraphs together, then sentences, then words.
    It's the standard best practice for general text.
    """
    print_step(2, "Recursive Character Splitting (chunk_size=150, overlap=20)")
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "], # Tries these in order
        chunk_size=150,
        chunk_overlap=20
    )
    chunks = splitter.split_text(SAMPLE_TEXT)

    for i, c in enumerate(chunks):
        print(f"  [{i}] ({len(c)} chars): {repr(c)}")
    print("\n  Notice how it splits cleanly at sentence boundaries.")

def demo_semantic():
    """
    Semantic chunking uses embeddings to detect shifts in meaning (topic changes).
    It groups sentences into chunks based on how mathematically similar they are.
    """
    print_step(3, "Semantic Chunking (Topic-based splitting)")
    embedding_model = get_embedding_model()

    # Percentile threshold: lower = more splits (smaller chunks)
    splitter = SemanticChunker(embedding_model, breakpoint_threshold_type="percentile")
    docs = splitter.create_documents([SAMPLE_TEXT])

    for i, d in enumerate(docs):
        print(f"  [{i}] Topic Chunk ({len(d.page_content)} chars):\n      {repr(d.page_content)}")
    print("\n  Notice how chunks correspond strictly to topics (Evolution -> Subsets -> Subfields -> Ethics).")


def run():
    print_header("✂️ TEXT CHUNKING STRATEGIES")
    print(f"Sample Text length: {len(SAMPLE_TEXT)} chars\n")

    demo_fixed_size()
    print("-" * 50 + "\n")

    demo_recursive()
    print("-" * 50 + "\n")

    demo_semantic()

if __name__ == "__main__":
    run()
