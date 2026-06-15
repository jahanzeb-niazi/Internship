"""End-to-end RAG pipeline: ingest → retrieve → generate."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from rag.chunker import chunk_pdf
from rag.store import ChromaStore

load_dotenv()

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context.
Rules:
- Base your answer strictly on the context below.
- If the context does not contain enough information, say you cannot find the answer in the document.
- Cite the source page numbers when available.
- Be concise and accurate."""


class RAGPipeline:
    def __init__(
        self,
        chroma_dir: Path | None = None,
        collection_name: str = "rag_documents",
        llm_model: str | None = None,
    ) -> None:
        base = Path(__file__).resolve().parent.parent
        self.chroma_dir = chroma_dir or base / "chroma_db"
        self.store = ChromaStore(self.chroma_dir, collection_name=collection_name)
        self.llm_model = llm_model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "GEMINI_API_KEY is not set. Get a key from Google AI Studio "
                    "(https://aistudio.google.com/apikey) and add it to .env"
                )
            self._client = genai.Client(api_key=api_key)
        return self._client

    def ingest(self, pdf_path: Path, reset: bool = True) -> int:
        pdf_path = pdf_path.resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        chunks = chunk_pdf(pdf_path)
        if reset:
            self.store.reset()

        count = self.store.add_chunks(chunks)
        return count

    def retrieve(self, question: str, top_k: int = 3) -> list[dict]:
        return self.store.search(question, top_k=top_k)

    def _format_context(self, hits: list[dict]) -> str:
        parts: list[str] = []
        for index, hit in enumerate(hits, start=1):
            page = hit["metadata"].get("page")
            page_label = f" (page {page})" if page else ""
            parts.append(f"[Chunk {index}{page_label}]\n{hit['text']}")
        return "\n\n".join(parts)

    def _gemini_error_message(self, exc: errors.ClientError) -> str:
        code = exc.code
        if code == 429:
            return (
                "Gemini API quota exceeded. Wait a minute and retry, or switch model "
                "in .env: GEMINI_MODEL=gemini-2.5-flash-lite"
            )
        if code in (401, 403):
            return (
                "Invalid or unauthorized Gemini API key. "
                "Check GEMINI_API_KEY in .env (https://aistudio.google.com/apikey)."
            )
        if code == 404:
            return (
                f"Model '{self.llm_model}' not found. "
                "Set GEMINI_MODEL=gemini-2.5-flash in .env"
            )
        return f"Gemini API error ({code}): {exc}"

    def generate_answer(self, question: str, hits: list[dict]) -> str:
        if not hits:
            return "No relevant chunks were found in the document."

        context = self._format_context(hits)
        try:
            response = self.client.models.generate_content(
                model=self.llm_model,
                contents=f"Context:\n{context}\n\nQuestion: {question}",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                ),
            )
        except errors.ClientError as exc:
            return self._gemini_error_message(exc)

        return response.text or ""

    def ask(self, question: str, top_k: int = 3) -> dict:
        hits = self.retrieve(question, top_k=top_k)
        answer = self.generate_answer(question, hits)
        return {"question": question, "answer": answer, "sources": hits}
