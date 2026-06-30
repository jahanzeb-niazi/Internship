"""
shared/utils.py — Common Utilities
=====================================
Pretty printing, timing, document loading, and other helpers
used across all RAG demos.
"""

import os
import time
import glob
from typing import List, Callable, Any
from functools import wraps
from langchain_core.documents import Document


# ── ANSI Color Codes for Terminal Output ──
# These make our demo output much easier to read
class Colors:
    HEADER = "\033[95m"      # Purple
    BLUE = "\033[94m"        # Blue
    CYAN = "\033[96m"        # Cyan
    GREEN = "\033[92m"       # Green
    WARNING = "\033[93m"     # Yellow
    FAIL = "\033[91m"        # Red
    ENDC = "\033[0m"         # Reset
    BOLD = "\033[1m"         # Bold
    DIM = "\033[2m"          # Dim


def print_header(title: str):
    """Print a styled section header."""
    print(f"\n{'='*60}")
    print(f"{Colors.HEADER}{Colors.BOLD}  {title}{Colors.ENDC}")
    print(f"{'='*60}\n")


def print_step(step_num: int, description: str):
    """Print a numbered step indicator."""
    print(f"{Colors.CYAN}[Step {step_num}]{Colors.ENDC} {description}")


def print_result(label: str, value: str, indent: int = 0):
    """Print a labeled result."""
    prefix = "  " * indent
    print(f"{prefix}{Colors.GREEN}✓ {label}:{Colors.ENDC} {value}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def print_error(message: str):
    """Print an error message."""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def print_comparison(label_a: str, text_a: str, label_b: str, text_b: str):
    """Print two results side by side for comparison."""
    print(f"\n{Colors.BLUE}── {label_a} ──{Colors.ENDC}")
    print(f"  {text_a}")
    print(f"\n{Colors.GREEN}── {label_b} ──{Colors.ENDC}")
    print(f"  {text_b}")
    print()


def print_documents(docs: List[Document], show_metadata: bool = True, max_chars: int = 200):
    """
    Pretty-print a list of retrieved documents.

    Args:
        docs:          List of LangChain Document objects.
        show_metadata: Whether to print metadata (source, page, etc.)
        max_chars:     Truncate content beyond this length.
    """
    if not docs:
        print_warning("No documents retrieved.")
        return

    for i, doc in enumerate(docs, 1):
        content = doc.page_content[:max_chars]
        if len(doc.page_content) > max_chars:
            content += "..."

        print(f"  {Colors.BOLD}[Doc {i}]{Colors.ENDC} {content}")
        if show_metadata and doc.metadata:
            meta_str = ", ".join(f"{k}={v}" for k, v in doc.metadata.items())
            print(f"         {Colors.DIM}Metadata: {meta_str}{Colors.ENDC}")
    print()


def timer(func: Callable) -> Callable:
    """
    Decorator that measures and prints execution time.

    Usage:
        @timer
        def my_slow_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  {Colors.DIM}⏱ {func.__name__} took {elapsed:.2f}s{Colors.ENDC}")
        return result
    return wrapper


def load_documents_from_directory(
    directory: str = None,
    glob_pattern: str = "*.txt",
) -> List[Document]:
    """
    Load all text files from a directory as LangChain Documents.

    Each file becomes one Document with metadata containing:
    - source: the filename
    - file_path: the full path

    Args:
        directory:    Path to the directory containing documents.
                      Defaults to the project's data/ folder.
        glob_pattern: File pattern to match (default: *.txt).

    Returns:
        List of Document objects.
    """
    if directory is None:
        directory = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data"
        )

    documents = []
    pattern = os.path.join(directory, glob_pattern)

    for filepath in sorted(glob.glob(pattern)):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        filename = os.path.basename(filepath)
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": filename,
                    "file_path": filepath,
                },
            )
        )

    if not documents:
        print_warning(f"No files matching '{glob_pattern}' found in {directory}")
        print_warning("Run 'python setup_data.py' first to create sample data.")

    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.

    WHY CHUNK?
        - LLMs have limited context windows
        - Smaller chunks = more precise retrieval
        - Overlap ensures we don't lose meaning at chunk boundaries

    Args:
        documents:     List of Documents to split.
        chunk_size:    Max characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of smaller Document objects (chunks).
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Split on these separators, in order of preference:
        # paragraphs → sentences → words → characters
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    return chunks
