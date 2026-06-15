"""Mini RAG system — ask questions grounded in a PDF document."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.pipeline import RAGPipeline

DATA_DIR = Path("data")


def list_pdfs(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"
    )


def resolve_pdf(pdf_arg: Path | None) -> Path:
    if pdf_arg is not None:
        pdf_path = pdf_arg.resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {pdf_path}")
        return pdf_path

    pdfs = list_pdfs(DATA_DIR)
    if not pdfs:
        raise FileNotFoundError(
            f"No PDF found in '{DATA_DIR}'. "
            "Add your document to data/ or pass --pdf path/to/file.pdf"
        )
    if len(pdfs) > 1:
        names = "\n  - ".join(p.name for p in pdfs)
        raise ValueError(
            f"Multiple PDFs in '{DATA_DIR}'. Choose one with --pdf:\n  - {names}"
        )
    return pdfs[0].resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mini RAG: chunk PDF → ChromaDB → retrieve top-3 → LLM answer"
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Path to a PDF (defaults to the only PDF in data/)",
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        help="Question to ask (interactive mode if omitted)",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-ingestion of the PDF",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks to retrieve (default: 3)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        pdf_path = resolve_pdf(args.pdf)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    pipeline = RAGPipeline()
    chroma_marker = pipeline.chroma_dir / "chroma.sqlite3"

    if args.reindex or not chroma_marker.exists():
        print(f"Ingesting {pdf_path} ...")
        count = pipeline.ingest(pdf_path)
        print(f"Indexed {count} chunks into ChromaDB at {pipeline.chroma_dir}")

    def run_query(question: str) -> None:
        print(f"\nQuestion: {question}\n")
        result = pipeline.ask(question, top_k=args.top_k)

        print("Retrieved chunks:")
        for index, hit in enumerate(result["sources"], start=1):
            page = hit["metadata"].get("page", "?")
            preview = hit["text"][:120].replace("\n", " ")
            print(f"  {index}. [page {page}] {preview}...")

        print("\nAnswer:")
        print(result["answer"])
        print()

    if args.question:
        run_query(args.question)
        return

    print(f"Mini RAG ready ({pdf_path.name}). Type a question (or 'quit' to exit).\n")
    while True:
        try:
            question = input("Ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break
        run_query(question)


if __name__ == "__main__":
    main()
