"""Shared types for LLM backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Union, runtime_checkable


@dataclass
class ToolCall:
    name: str
    arguments: dict
    thought: str = ""


@dataclass
class FinalAnswer:
    text: str
    thought: str = ""


@dataclass
class ToolResult:
    """Normalized result from run_tool — success or error."""

    ok: bool
    data: dict
    error: str | None = None

    @classmethod
    def success(cls, data: dict) -> ToolResult:
        return cls(ok=True, data=data)

    @classmethod
    def failure(cls, error: str, data: dict | None = None) -> ToolResult:
        return cls(ok=False, data=data or {"error": error}, error=error)


LLMResponse = Union[ToolCall, FinalAnswer]


@runtime_checkable
class LLMBackend(Protocol):
    def reset(self, user_message: str) -> None: ...
    def decide(self) -> LLMResponse: ...
    def record_tool_result(self, name: str, result: dict) -> None: ...
    def record_denial(self, name: str, reason: str) -> None: ...
