"""Quick smoke test for the RAG pipeline (ingest + retrieve)."""

from pathlib import Path

from rag.pipeline import RAGPipeline


def main() -> None:
    pipeline = RAGPipeline()
    count = pipeline.ingest(Path("data/Sample.pdf"))
    print(f"Indexed {count} chunks into {pipeline.chroma_dir}")

    question = "When was Chatgpt Launched?"
    hits = pipeline.retrieve(question, top_k=3)
    print(f"\nQuestion: {question}")
    print(f"Retrieved {len(hits)} chunks:\n")
    for index, hit in enumerate(hits, start=1):
        page = hit["metadata"].get("page")
        print(f"  {index}. [page {page}] {hit['text'][:100]}...")


if __name__ == "__main__":
    main()
