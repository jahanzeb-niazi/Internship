"""ChromaDB persistent vector store."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings

from rag.chunker import DocumentChunk
from rag.embedder import Embedder


class ChromaStore:
    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = "rag_documents",
        embedder: Embedder | None = None,
    ) -> None:
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or Embedder()

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts)

        self.collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {"source": chunk.source, "page": chunk.page or 0}
                for chunk in chunks
            ],
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_embedding = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict] = []
        if not results["ids"] or not results["ids"][0]:
            return hits

        for doc_id, document, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append(
                {
                    "id": doc_id,
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )
        return hits
