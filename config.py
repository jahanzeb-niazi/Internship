"""
Central configuration for the AI Candidate Screening Assistant.
All tunable parameters live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "llm_calls.jsonl"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"
EVAL_DIR = BASE_DIR / "eval"
EVAL_RESULTS_DIR = EVAL_DIR / "eval_results"

# Ensure directories exist
LOG_DIR.mkdir(exist_ok=True)
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── LLM ────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = "gemini-3.1-flash-lite"

# ── Schema Validation ──────────────────────────────────────────────────────
MAX_RETRIES = 5

# ── Confidence ─────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.7  # Below this → needs_human_review

# ── RAG ────────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "hiring_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50     # overlap between chunks
RAG_TOP_K = 5          # number of results to retrieve

# ── Prompt Versions ────────────────────────────────────────────────────────
# Bump these when you change a prompt to track in logs
PROMPT_VERSIONS = {
    "parse_application": "v1.0",
    "qualify_candidate": "v1.0",
    "draft_outreach": "v1.0",
    "agent_reasoning": "v1.0",
}
