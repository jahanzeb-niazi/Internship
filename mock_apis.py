"""Mock weather and task APIs (in-memory, no network)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List


WEATHER_DB: Dict[str, dict] = {
    "london": {"temp_c": 14, "condition": "cloudy", "humidity": 72},
    "tokyo": {"temp_c": 22, "condition": "sunny", "humidity": 55},
    "new york": {"temp_c": 18, "condition": "rainy", "humidity": 80},
    "paris": {"temp_c": 16, "condition": "partly cloudy", "humidity": 65},
}


def fetch_weather(city: str) -> dict:
    if not city or not city.strip():
        raise ValueError("City name cannot be empty")
    key = city.strip().lower()
    if key not in WEATHER_DB:
        return {
            "error": f"No weather data for '{city}'",
            "available_cities": list(WEATHER_DB.keys()),
        }
    data = WEATHER_DB[key]
    return {
        "city": city.title(),
        "temp_c": data["temp_c"],
        "condition": data["condition"],
        "humidity": data["humidity"],
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }


@dataclass
class Task:
    id: int
    title: str
    done: bool = False


@dataclass
class TaskStore:
    tasks: List[Task] = field(default_factory=lambda: [
        Task(id=1, title="Review tool-calling notes"),
        Task(id=2, title="Build ReAct loop demo"),
    ])
    _next_id: int = 3

    def list_tasks(self) -> List[dict]:
        return [{"id": t.id, "title": t.title, "done": t.done} for t in self.tasks]

    def create_task(self, title: str) -> dict:
        if not title or not title.strip():
            raise ValueError("Task title cannot be empty")
        task = Task(id=self._next_id, title=title.strip())
        self._next_id += 1
        self.tasks.append(task)
        return {"created": True, "task": {"id": task.id, "title": task.title, "done": task.done}}

    def delete_task(self, task_id: int) -> dict:
        if not isinstance(task_id, int) or task_id < 1:
            raise ValueError(f"Invalid task_id: {task_id}")
        for i, task in enumerate(self.tasks):
            if task.id == task_id:
                removed = self.tasks.pop(i)
                return {"deleted": True, "task": {"id": removed.id, "title": removed.title}}
        return {"deleted": False, "error": f"Task {task_id} not found"}


TASK_STORE = TaskStore()
