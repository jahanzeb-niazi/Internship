"""Load GEMINI_API_KEY from environment or .env file."""

from __future__ import annotations

import os
from pathlib import Path


def load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).parent / ".env")
    except ImportError:
        pass


def get_gemini_api_key() -> str | None:
    load_env()
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def get_gemini_model() -> str:
    load_env()
    return os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
