"""
shared/embeddings.py — Embedding Helpers
==========================================
Wraps Google Gemini's embedding model so every module can convert text
into vectors without repeating setup code.

KEY CONCEPT:
    Embeddings convert text into numerical vectors (lists of numbers).
    Similar texts produce similar vectors. This is how RAG "searches"
    for relevant documents — by comparing vector similarity.

    Think of it like GPS coordinates for meaning:
    - "dog" and "puppy" would be close together
    - "dog" and "airplane" would be far apart
"""

import os
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables from .env file
load_dotenv()


def get_embedding_model(model: str = "models/text-embedding-004"):
    """
    Create and return a Gemini embedding model instance.

    Args:
        model: Which embedding model to use.
               "models/text-embedding-004" is Google's latest text embedding model.

    Returns:
        A LangChain Embeddings instance.

    WHY THIS MODEL?
        - Free tier available
        - 768-dimensional vectors (good balance of quality vs. storage)
        - Supports up to 2048 tokens per input
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "❌ GOOGLE_API_KEY not found! See shared/llm.py for setup instructions."
        )

    return GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=api_key,
    )


def embed_text(text: str, model: str = "models/text-embedding-004") -> List[float]:
    """
    Convert a single text string into an embedding vector.

    This is used for embedding USER QUERIES at search time.

    Args:
        text:  The text to embed (usually a question).
        model: Which embedding model to use.

    Returns:
        A list of floats representing the text's position in semantic space.

    Example:
        >>> vec = embed_text("What is machine learning?")
        >>> print(f"Vector has {len(vec)} dimensions")
        "Vector has 768 dimensions"
    """
    embedding_model = get_embedding_model(model=model)
    return embedding_model.embed_query(text)


def embed_documents(
    texts: List[str], model: str = "models/text-embedding-004"
) -> List[List[float]]:
    """
    Convert multiple text strings into embedding vectors (batch operation).

    This is used during INDEXING — converting all your documents into
    searchable vectors at once.

    Args:
        texts: List of text strings to embed.
        model: Which embedding model to use.

    Returns:
        A list of embedding vectors (one per input text).

    WHY BATCH?
        Embedding one-by-one is slow due to network overhead.
        Batch embedding sends them all in fewer API calls.
    """
    embedding_model = get_embedding_model(model=model)
    return embedding_model.embed_documents(texts)
