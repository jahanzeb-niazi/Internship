<<<<<<< HEAD
"""Shared types for LLM backends."""
=======
"""Shared types for mock and Gemini LLM backends."""
>>>>>>> a110e738aaea2cef40a2e542d86a997a0e80769f

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


<<<<<<< HEAD
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


=======
>>>>>>> a110e738aaea2cef40a2e542d86a997a0e80769f
LLMResponse = Union[ToolCall, FinalAnswer]


@runtime_checkable
class LLMBackend(Protocol):
    def reset(self, user_message: str) -> None: ...
<<<<<<< HEAD
    def decide(self) -> LLMResponse: ...
    def record_tool_result(self, name: str, result: dict) -> None: ...
=======

    def decide(self) -> LLMResponse: ...

    def record_tool_result(self, name: str, result: dict) -> None: ...

>>>>>>> a110e738aaea2cef40a2e542d86a997a0e80769f
    def record_denial(self, name: str, reason: str) -> None: ...
