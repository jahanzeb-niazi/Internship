"""
07_context_distillation.py — Context Distillation
===================================================

WHAT IT IS:
    "Summarize before you search." (Also known as Contextual Retrieval).
    When a document is chunked, the chunks lose the global context of the document.
    "He announced the new product." -> Who is 'He'? What 'product'?

HOW IT WORKS:
    During indexing, before embedding a chunk, pass it to an LLM along with the
    FULL document. Ask the LLM to write a 1-2 sentence context bridging the chunk
    to the whole document. Prepend this context to the chunk, then embed it.

    "Context: Tim Cook at the 2024 Apple Keynote announcing the Vision Pro.
    Chunk: He announced the new product."
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.utils import print_header, print_step
from shared.llm import ask_llm

FULL_DOCUMENT = """
Title: Apple Q4 2024 Earnings Call Transcript
Speaker 1 (Tim Cook): We had a record quarter, driven heavily by iPhone 16 sales in emerging markets.
Speaker 2 (Luca Maestri): Margins were tight due to supply chain constraints, but services revenue offset that.
Speaker 1: Moving forward, we are heavily investing in Apple Intelligence. The new AI features will roll out next month. They are expected to drive significant upgrade cycles.
"""

ORPHAN_CHUNK = "The new AI features will roll out next month. They are expected to drive significant upgrade cycles."

def distill_context(full_doc: str, chunk: str) -> str:
    prompt = f"""You are an expert at preserving context. Look at this whole document, and a specific chunk from it.
Write a 1-2 sentence contextual summary that explains who/what/where the chunk is about, so it makes sense in isolation.

Document:
{full_doc}

Chunk:
{chunk}

Contextual Summary:"""

    return ask_llm(prompt, "You write concise context summaries.").strip()

def run():
    print_header("🧪 CONTEXT DISTILLATION")

    print_step(1, "The Problem: Orphaned Chunks")
    print(f"  Imagine this chunk was retrieved by itself:")
    print(f"  '{ORPHAN_CHUNK}'")
    print("  -> Who is rolling them out? What AI features? It's ambiguous.")

    print("\n" + "="*50)

    print_step(2, "The Solution: Generate Context Summary during Indexing")
    print("  We ask the LLM to summarize the chunk's role in the larger document.")
    context = distill_context(FULL_DOCUMENT, ORPHAN_CHUNK)
    print(f"\n  Generated Context: '{context}'")

    print("\n" + "="*50)

    print_step(3, "The Final Indexed Chunk")
    print("  We prepend the context to the original chunk and embed THIS string:")
    final_string = f"{context}\n\n{ORPHAN_CHUNK}"
    print(f"\n  ```\n  {final_string}\n  ```")
    print("\n  Now, if a user searches for 'Apple AI rollout', this chunk will match perfectly!")

if __name__ == "__main__":
    run()
