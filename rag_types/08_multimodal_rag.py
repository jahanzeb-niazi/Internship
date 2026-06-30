"""
08_multimodal_rag.py — Multimodal RAG (Text + Images)
=======================================================

WHAT IS MULTIMODAL RAG?
    Converts different content types (text, images, videos, audio, charts)
    into a searchable format and combines them to answer queries.

    Instead of just searching text documents, it can find relevant images
    and use them to generate an answer.

HOW THIS DEMO WORKS:
    Since true multimodal embeddings (like CLIP) require complex setup,
    we use the "Image-to-Text" approach which is common in production:
    1. Use a Vision LLM to generate a detailed text description of an image
    2. Embed that text description and store it in the vector DB
    3. When searching, retrieve the text description (and its image link)
    4. Answer the query using the extracted information

BEST FOR:
    Medical imaging analysis, technical manuals with diagrams,
    e-commerce product search, slide deck Q&A.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document

from shared.utils import (
    print_header, print_step, print_result,
    load_documents_from_directory, chunk_documents, timer
)
from shared.vector_store import create_vector_store, similarity_search, delete_collection
from shared.llm import ask_llm, ask_llm_with_context


COLLECTION_NAME = "multimodal_rag"

# Simulated image dataset
# In a real app, these would be actual image files that you pass to
# Gemini's Vision API to generate these descriptions.
MOCK_IMAGES = {
    "diagram_transformer.png": "A diagram showing the Transformer architecture. On the left is the Encoder stack with Multi-Head Attention and Feed Forward layers. On the right is the Decoder stack with Masked Multi-Head Attention. Both sides use Positional Encoding at the input.",
    "chart_llm_growth.png": "A line chart showing the growth of LLM parameter sizes from 2018 to 2024. GPT-1 is at the bottom left (117M). The line curves steeply upward through GPT-2 (1.5B), GPT-3 (175B), ending at GPT-4 in 2023 at an estimated 1.7T parameters.",
    "screenshot_chromadb.png": "A screenshot of a Python code snippet showing how to initialize ChromaDB. It imports PersistentClient, creates a client pointing to a local directory, and calls get_or_create_collection('my_collection')."
}

def extract_image_summaries() -> list[Document]:
    """
    Simulates using a Vision LLM to describe images.
    In reality, you would load an image and prompt: "Describe this image in detail."
    Returns LangChain Documents where the content is the image description.
    """
    docs = []
    for filename, description in MOCK_IMAGES.items():
        doc = Document(
            page_content=f"Image Description: {description}",
            metadata={
                "source": filename,
                "type": "image",
                "content_type": "diagram/chart"
            }
        )
        docs.append(doc)
    return docs

@timer
def setup_multimodal_knowledge_base():
    """Index both text documents and image descriptions."""
    print_step(1, "Loading text documents...")
    text_docs = load_documents_from_directory()
    text_chunks = chunk_documents(text_docs, chunk_size=500, chunk_overlap=50)

    print_step(2, "Extracting descriptions from images (simulated Vision LLM)...")
    image_docs = extract_image_summaries()

    # Combine text chunks and image descriptions
    all_docs = text_chunks + image_docs

    print_step(3, "Indexing all modalities into vector store...")
    delete_collection(COLLECTION_NAME)
    create_vector_store(all_docs, collection_name=COLLECTION_NAME)
    print_result("Multimodal knowledge base ready", f"{len(text_chunks)} text chunks, {len(image_docs)} images")

def multimodal_query(question: str) -> str:
    """Run a query against the multimodal knowledge base."""
    print_step(4, f"Retrieving relevant content for: '{question}'")

    docs = similarity_search(question, collection_name=COLLECTION_NAME, k=3)

    # Separate retrieved content by type for display
    text_results = []
    image_results = []

    context_parts = []
    for d in docs:
        if d.metadata.get("type") == "image":
            image_results.append(d.metadata["source"])
            context_parts.append(f"[IMAGE: {d.metadata['source']}]\n{d.page_content}")
        else:
            text_results.append(d.metadata.get("source", "unknown"))
            context_parts.append(f"[TEXT: {d.metadata.get('source')}]\n{d.page_content}")

    print(f"  📄 Retrieved: {len(text_results)} texts, {len(image_results)} images")
    if image_results:
        print(f"  🖼️  Images found: {', '.join(image_results)}")

    context = "\n\n".join(context_parts)

    print_step(5, "Generating comprehensive answer...")
    answer = ask_llm_with_context(
        question=question,
        context=context,
        system_prompt="You are a helpful assistant. Answer based on the provided context, which includes both text excerpts and image descriptions. When referring to visual information, explicitly mention that it comes from an image/diagram."
    )

    return answer


def run():
    """Demo: Show Multimodal RAG handling text and images."""
    print_header("🖼️ MULTIMODAL RAG — Text + Images")

    print("""
    Multimodal RAG combines different types of media.
    In this approach:
    1. Images are described using a Vision LLM
    2. The descriptions are embedded and stored alongside text
    3. Retrieval finds BOTH text and images relevant to the query
    4. The generator weaves them together into one answer
    """)

    setup_multimodal_knowledge_base()

    questions = [
        "What is the architecture of a Transformer?", # Should pull text + diagram
        "How has the size of LLMs changed over time?", # Should pull the chart
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'═'*60}")
        print(f"  Question {i}: {question}")
        print(f"{'═'*60}")

        answer = multimodal_query(question)

        print(f"\n  🤖 Answer:\n  {answer}")

    print_header("📊 MULTIMODAL RAG — Key Takeaways")
    print("""
    Approaches to Multimodal RAG:
    1. Image-to-Text (What we did here):
       Extract text via Vision LLM → embed text → search text.
       (Standard, easy to build with existing text vector DBs)

    2. Native Multimodal Embeddings (e.g., CLIP):
       Embed image directly into vector space → search using image or text.
       (Better for visual similarity, harder to setup)

    Trade-offs:
    + Great for documents mixing media types (PDFs, slides)
    - More complex to build and requires more storage
    - Vision LLM extraction can be slow and expensive
    """)

if __name__ == "__main__":
    run()
