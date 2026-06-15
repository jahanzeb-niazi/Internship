"""Local embedding generation via ChromaDB's ONNX MiniLM model."""

from __future__ import annotations

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


class Embedder:
    """Uses ChromaDB's built-in all-MiniLM-L6-v2 ONNX embeddings (no PyTorch)."""

    def __init__(self) -> None:
        self._ef = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._ef(texts)

    def embed_query(self, query: str) -> list[float]:
        return self._ef([query])[0]
