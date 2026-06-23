"""
RAG Module — Knowledge base powered by ChromaDB + Sentence-Transformers.
Ingests markdown documents, chunks them, embeds locally, and retrieves relevant context.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from config import (
    KNOWLEDGE_BASE_DIR,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RAG_TOP_K,
)


def _get_embedding_function():
    """Get the Sentence-Transformers embedding function for ChromaDB."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks.
    Uses character-based chunking with overlap for context preservation.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # Skip empty chunks
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def _get_client() -> chromadb.PersistentClient:
    """Get or create a persistent ChromaDB client."""
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))


def _get_collection(client: chromadb.PersistentClient):
    """Get or create the knowledge base collection."""
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=_get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def init_knowledge_base(force_reload: bool = False) -> dict:
    """
    Load all markdown files from knowledge_base/, chunk them,
    embed with Sentence-Transformers, and store in ChromaDB.

    Args:
        force_reload: If True, delete existing collection and re-ingest.

    Returns:
        Summary dict with documents_loaded and total_chunks.
    """
    client = _get_client()

    if force_reload:
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
        except ValueError:
            pass  # Collection doesn't exist yet

    collection = _get_collection(client)

    # Check if already populated
    if collection.count() > 0 and not force_reload:
        return {
            "documents_loaded": 0,
            "total_chunks": collection.count(),
            "status": "already_populated",
        }

    # Load and process markdown files
    md_files = list(KNOWLEDGE_BASE_DIR.glob("*.md"))
    if not md_files:
        return {"documents_loaded": 0, "total_chunks": 0, "status": "no_documents_found"}

    all_chunks = []
    all_ids = []
    all_metadatas = []

    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        source_name = md_file.name
        chunks = _chunk_text(content)

        for i, chunk in enumerate(chunks):
            # Create a unique, deterministic ID for each chunk
            chunk_id = hashlib.md5(f"{source_name}_{i}_{chunk[:50]}".encode()).hexdigest()
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append({
                "source": source_name,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

    # Add to ChromaDB in batches
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch_end = min(i + batch_size, len(all_chunks))
        collection.add(
            documents=all_chunks[i:batch_end],
            ids=all_ids[i:batch_end],
            metadatas=all_metadatas[i:batch_end],
        )

    return {
        "documents_loaded": len(md_files),
        "total_chunks": len(all_chunks),
        "status": "success",
        "files": [f.name for f in md_files],
    }


def retrieve_context(query: str, n_results: int = RAG_TOP_K) -> list[dict]:
    """
    Retrieve relevant document chunks from the knowledge base.

    Args:
        query: Search query (e.g., candidate skills + role).
        n_results: Number of results to return.

    Returns:
        List of dicts with 'content', 'source', and 'distance' keys.
    """
    client = _get_client()
    collection = _get_collection(client)

    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
    )

    # Format results
    formatted = []
    if results["documents"] and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 0.0
            formatted.append({
                "content": doc,
                "source": metadata.get("source", "unknown"),
                "chunk_index": metadata.get("chunk_index", 0),
                "distance": round(distance, 4),
            })

    return formatted


def get_kb_stats() -> dict:
    """Get knowledge base statistics."""
    client = _get_client()
    collection = _get_collection(client)
    return {
        "total_chunks": collection.count(),
        "collection_name": CHROMA_COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
    }
