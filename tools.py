"""
Tool registry and safe execution with graceful error handling.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from llm_types import ToolResult
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
        requires_approval=True,
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
        requires_approval=True,
        handler=_delete_task,
    ),
}


def tools_for_gemini() -> List[dict]:
    return [
        {"name": t.name, "description": t.description, "parameters": t.parameters}
        for t in TOOLS.values()
    ]


def run_tool(name: str, arguments: Optional[dict] = None) -> ToolResult:
    """
    Execute a tool safely. Never raises — returns ToolResult with ok=True/False.
    """
    if name not in TOOLS:
        return ToolResult.failure(f"Unknown tool: '{name}'")

    tool = TOOLS[name]
    args = arguments or {}

    try:
        result = tool.handler(**args)
        return ToolResult.success(result)
    except TypeError as e:
        return ToolResult.failure(f"Bad arguments for '{name}': {e}")
    except ValueError as e:
        return ToolResult.failure(str(e))
    except Exception as e:
        return ToolResult.failure(
            f"Tool '{name}' failed: {e}",
            data={"error": str(e), "traceback": traceback.format_exc(limit=2)},
        )


def pretty_tool_list() -> str:
    lines = ["Available tools:"]
    for t in TOOLS.values():
        gate = " [needs approval]" if t.requires_approval else ""
        lines.append(f"  - {t.name}: {t.description}{gate}")
    return "\n".join(lines)
