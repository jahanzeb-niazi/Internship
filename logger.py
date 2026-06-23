"""
LLM Call Logger — appends structured log entries to a JSONL file.
Tracks: function name, token counts, latency, prompt version, model, success/error.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from schemas import LLMCallLog
from config import LOG_FILE


_write_lock = threading.Lock()


def log_llm_call(entry: LLMCallLog) -> None:
    """
    Append a single LLM call log entry to the JSONL log file.
    Thread-safe via file lock.
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(entry.model_dump_json() + "\n")


def get_all_logs() -> list[LLMCallLog]:
    """Read all log entries from the log file."""
    if not LOG_FILE.exists():
        return []
    logs = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(LLMCallLog.model_validate_json(line))
    return logs


def get_call_stats() -> dict:
    """
    Compute summary statistics from the log file.
    Returns: total_calls, total_tokens, avg_latency_ms, success_rate, calls_by_function.
    """
    logs = get_all_logs()
    if not logs:
        return {
            "total_calls": 0,
            "total_tokens": 0,
            "avg_latency_ms": 0.0,
            "success_rate": 0.0,
            "calls_by_function": {},
        }

    total_tokens = sum(log.total_tokens for log in logs)
    avg_latency = sum(log.latency_ms for log in logs) / len(logs)
    success_count = sum(1 for log in logs if log.success)

    calls_by_function: dict[str, int] = {}
    for log in logs:
        calls_by_function[log.function_name] = calls_by_function.get(log.function_name, 0) + 1

    return {
        "total_calls": len(logs),
        "total_tokens": total_tokens,
        "avg_latency_ms": round(avg_latency, 2),
        "success_rate": round(success_count / len(logs) * 100, 2),
        "calls_by_function": calls_by_function,
    }


def clear_logs() -> None:
    """Clear the log file (useful for testing)."""
    if LOG_FILE.exists():
        LOG_FILE.unlink()
