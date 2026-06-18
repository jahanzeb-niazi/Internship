"""Offline mock LLM with keyword-based tool selection."""

from __future__ import annotations

import json
import re
from typing import List

from llm_types import FinalAnswer, LLMResponse, ToolCall


class MockLLM:
    def __init__(self) -> None:
        self.user_message = ""
        self.observations: List[str] = []

    def reset(self, user_message: str) -> None:
        self.user_message = user_message
        self.observations = []

    def decide(self) -> LLMResponse:
        return _decide(self.user_message, self.observations)

    def record_tool_result(self, name: str, result: dict) -> None:
        self.observations.append(f"{name} returned: {json.dumps(result)}")

    def record_denial(self, name: str, reason: str) -> None:
        self.observations.append(f"{name} was denied: {reason}")


def _decide(user_message: str, observations: List[str]) -> LLMResponse:
    msg = user_message.lower()

    if observations:
        return FinalAnswer(
            text=_summarize(user_message, observations),
            thought="I have tool results. Answering now.",
        )

    thought = f"Matching message to tool descriptions: '{msg[:50]}...'"

    if any(w in msg for w in ("weather", "temperature", "forecast", "rain", "sunny")):
        city = _extract_city(msg) or "London"
        return ToolCall(name="get_forecast", arguments={"city": city}, thought=thought)

    if any(w in msg for w in ("list", "show", "my tasks", "todo")) and ("task" in msg or "todo" in msg):
        return ToolCall(name="list_tasks", arguments={}, thought=thought)

    create_match = re.search(r"(?:add|create|new)\s+(?:task|todo)[:\s]+(.+)", msg)
    if create_match:
        return ToolCall(name="create_task", arguments={"title": create_match.group(1).strip().title()}, thought=thought)

    delete_match = re.search(r"(?:delete|remove)\s+task\s+(\d+)", msg)
    if delete_match:
        return ToolCall(name="delete_task", arguments={"task_id": int(delete_match.group(1))}, thought=thought)

    return FinalAnswer(
        text="Try: 'weather in Tokyo', 'list my tasks', 'add task: buy milk', 'delete task 1'.",
        thought=thought,
    )


def _extract_city(msg: str) -> str | None:
    for city in ("london", "tokyo", "new york", "paris"):
        if city in msg:
            return city.title()
    match = re.search(r"(?:in|for|at)\s+([a-z\s]+?)(?:\?|$|,|\.)", msg)
    return match.group(1).strip().title() if match else None


def _summarize(user_message: str, observations: List[str]) -> str:
    lines = [f"Answer for '{user_message}':"]
    lines.extend(f"  - {o}" for o in observations)
    return "\n".join(lines)
