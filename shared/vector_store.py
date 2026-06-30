"""
shared/vector_store.py — Vector Store Helpers
===============================================
Manages ChromaDB collections for storing and searching document embeddings.

KEY CONCEPT:
    A vector store is like a "search engine for meaning." Instead of
    matching keywords, it finds documents whose MEANING is closest to
    your query by comparing embedding vectors.

    ChromaDB is a lightweight, local vector database — perfect for learning.
    In production, you'd use Pinecone, Weaviate, or pgvector.

HOW IT WORKS:
    1. STORE: Convert documents → embeddings → save to ChromaDB
    2. SEARCH: Convert query → embedding → find nearest neighbors
    3. RETURN: Get back the original text of the closest matches
"""

import os
from typing import List, Optional, Dict, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from shared.embeddings import get_embedding_model

# ── Default directory for storing ChromaDB data on disk ──
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")


def create_vector_store(
    documents: List[Document],
    collection_name: str = "default",
    persist_directory: Optional[str] = None,
) -> Chroma:
    """
    Create a ChromaDB vector store from a list of LangChain Documents.

    This is the INDEXING step of RAG — it converts your documents into
    searchable vectors and stores them.

    Args:
        documents:         List of LangChain Document objects to index.
        collection_name:   Name for this collection (like a table name in SQL).
                           Use different names for different RAG demos.
        persist_directory: Where to save the database on disk.
                           If None, uses in-memory storage (lost on restart).

    Returns:
        A Chroma vector store instance ready for searching.

    Example:
        >>> from langchain_core.documents import Document
        >>> docs = [Document(page_content="AI is amazing", metadata={"source": "doc1"})]
        >>> store = create_vector_store(docs, collection_name="my_demo")
    """
    embedding_model = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=collection_name,
        persist_directory=persist_directory or CHROMA_PERSIST_DIR,
    )

    return vector_store


def get_vector_store(
    collection_name: str = "default",
    persist_directory: Optional[str] = None,
) -> Chroma:
    """
    Connect to an EXISTING ChromaDB collection (previously created).

    Use this when you've already indexed documents and want to search them
    without re-indexing.

    Args:
        collection_name:   Name of the collection to connect to.
        persist_directory: Where the database was saved.

    Returns:
        A Chroma vector store instance.
    """
    embedding_model = get_embedding_model()

    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_directory or CHROMA_PERSIST_DIR,
    )


def similarity_search(
    query: str,
    collection_name: str = "default",
    k: int = 4,
    persist_directory: Optional[str] = None,
) -> List[Document]:
    """
    Search for documents similar to a query.

    This is the RETRIEVAL step of RAG — finding the most relevant
    documents to answer the user's question.

    Args:
        query:             The search query (e.g., "What is deep learning?").
        collection_name:   Which collection to search.
        k:                 How many results to return (top-K).
        persist_directory: Where the database is stored.

    Returns:
        List of the K most similar Document objects, ordered by relevance.

    HOW "SIMILARITY" WORKS:
        1. Your query gets converted to an embedding vector
        2. ChromaDB compares it against all stored document vectors
        3. It returns the K documents with the highest cosine similarity
        (cosine similarity = how closely two vectors point in the same direction)
    """
    store = get_vector_store(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    return store.similarity_search(query, k=k)


def similarity_search_with_scores(
    query: str,
    collection_name: str = "default",
    k: int = 4,
    persist_directory: Optional[str] = None,
) -> List[tuple]:
    """
    Like similarity_search, but also returns the similarity score
    for each result.

    This is useful for:
    - AUTOCUT: Only keeping results above a score threshold
    - DEBUGGING: Understanding why certain documents were retrieved
    - RERANKING: Using scores as features for a reranker

    Returns:
        List of (Document, score) tuples. Lower score = MORE similar
        (ChromaDB uses distance, not similarity, so lower is better).
    """
    store = get_vector_store(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    return store.similarity_search_with_score(query, k=k)


def delete_collection(
    collection_name: str = "default",
    persist_directory: Optional[str] = None,
):
    """
    Delete a ChromaDB collection. Useful for cleaning up between demos.
    """
    import chromadb

    client = chromadb.PersistentClient(
        path=persist_directory or CHROMA_PERSIST_DIR
    )
    try:
        client.delete_collection(name=collection_name)
    except ValueError:
        pass  # Collection doesn't exist, that's fine
