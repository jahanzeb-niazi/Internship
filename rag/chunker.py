"""Efficient PDF text extraction and recursive chunking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


@dataclass
class DocumentChunk:
    text: str
    chunk_id: str
    source: str
    page: int | None


def extract_pdf_text(pdf_path: Path) -> list[tuple[str, int]]:
    """Extract text per page. Returns (text, 1-based page number) pairs."""
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[str, int]] = []

    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((text.strip(), index))

    return pages


def chunk_pdf(
    pdf_path: Path,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[DocumentChunk]:
    """
    Chunk a PDF using recursive character splitting.

    Uses paragraph → line → sentence → word boundaries for clean splits.
    Overlap preserves context at chunk boundaries.
    """
    pdf_path = pdf_path.resolve()
    pages = extract_pdf_text(pdf_path)

    if not pages:
        raise ValueError(f"No extractable text found in {pdf_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        is_separator_regex=False,
    )

    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for page_text, page_num in pages:
        for piece in splitter.split_text(page_text):
            if not piece.strip():
                continue
            chunks.append(
                DocumentChunk(
                    text=piece.strip(),
                    chunk_id=f"chunk_{chunk_index:04d}",
                    source=str(pdf_path),
                    page=page_num,
                )
            )
            chunk_index += 1

    return chunks
