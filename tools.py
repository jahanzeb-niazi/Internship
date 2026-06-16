"""
Step 2: Tools (what the agent CAN do)
-------------------------------------
Each tool has:
  - name: what the LLM calls
  - description: helps the LLM decide WHEN to use it
  - parameters: JSON-schema-like shape
  - requires_approval: human-in-the-loop gate for risky actions
  - handler: the actual function that runs
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from mock_apis import TASK_STORE, fetch_weather


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    requires_approval: bool
    handler: Callable[..., dict]


def _get_forecast(city: str) -> dict:
    return fetch_weather(city)


def _list_tasks() -> dict:
    return {"tasks": TASK_STORE.list_tasks()}


def _create_task(title: str) -> dict:
    return TASK_STORE.create_task(title)


def _delete_task(task_id: int) -> dict:
    return TASK_STORE.delete_task(task_id)


# Registry: all tools the agent may call
TOOLS: Dict[str, Tool] = {
    "get_forecast": Tool(
        name="get_forecast",
        description="Get current weather for a city. Use when user asks about weather or temperature.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name, e.g. London"}},
            "required": ["city"],
        },
        requires_approval=False,
        handler=_get_forecast,
    ),
    "list_tasks": Tool(
        name="list_tasks",
        description="List all tasks. Use when user wants to see their todo list or tasks.",
        parameters={"type": "object", "properties": {}, "required": []},
        requires_approval=False,
        handler=lambda: _list_tasks(),
    ),
    "create_task": Tool(
        name="create_task",
        description="Create a new task. Use when user wants to add something to their todo list.",
        parameters={
            "type": "object",
            "properties": {"title": {"type": "string", "description": "Task title"}},
            "required": ["title"],
        },
        requires_approval=True,  # writes need human approval
        handler=_create_task,
    ),
    "delete_task": Tool(
        name="delete_task",
        description="Delete a task by ID. Use when user wants to remove a task.",
        parameters={
            "type": "object",
            "properties": {"task_id": {"type": "integer", "description": "Task ID to delete"}},
            "required": ["task_id"],
        },
        requires_approval=True,  # destructive → approval gate
        handler=_delete_task,
    ),
}


def tools_for_llm() -> List[dict]:
    """OpenAI-style tool format (for reference / other providers)."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in TOOLS.values()
    ]


def tools_for_gemini() -> List[dict]:
    """Gemini function_declarations format (OpenAPI subset)."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        }
        for t in TOOLS.values()
    ]


def run_tool(name: str, arguments: Optional[dict] = None) -> dict:
    """Execute a tool by name with given arguments."""
    if name not in TOOLS:
        return {"error": f"Unknown tool: {name}"}
    tool = TOOLS[name]
    args = arguments or {}
    try:
        return tool.handler(**args)
    except TypeError as e:
        return {"error": f"Bad arguments for {name}: {e}"}


def pretty_tool_list() -> str:
    lines = ["Available tools:"]
    for t in TOOLS.values():
        gate = " [needs approval]" if t.requires_approval else ""
        lines.append(f"  - {t.name}: {t.description}{gate}")
    return "\n".join(lines)
