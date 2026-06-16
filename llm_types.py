"""Shared types for mock and Gemini LLM backends."""

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


LLMResponse = Union[ToolCall, FinalAnswer]


@runtime_checkable
class LLMBackend(Protocol):
    def reset(self, user_message: str) -> None: ...

    def decide(self) -> LLMResponse: ...

    def record_tool_result(self, name: str, result: dict) -> None: ...

    def record_denial(self, name: str, reason: str) -> None: ...
