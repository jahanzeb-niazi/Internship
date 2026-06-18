"""
Action logger — records every tool call with input/output.
Logs to console (verbose mode) and to action_log.jsonl (append-only).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

LOG_FILE = Path(__file__).parent / "action_log.jsonl"

ActionStatus = Literal["pending", "approved", "denied", "success", "error"]


@dataclass
class ActionLogEntry:
    timestamp: str
    session_id: str
    step: int
    tool: str
    input: dict
    status: ActionStatus
    output: Optional[dict] = None
    error: Optional[str] = None
    approval_required: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


@dataclass
class ActionLogger:
    session_id: str
    verbose: bool = True
    log_file: Path = field(default_factory=lambda: LOG_FILE)
    _entries: list[ActionLogEntry] = field(default_factory=list)

    def log_tool_call(
        self,
        step: int,
        tool: str,
        input_args: dict,
        *,
        approval_required: bool = False,
        status: ActionStatus = "pending",
        output: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> ActionLogEntry:
        entry = ActionLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=self.session_id,
            step=step,
            tool=tool,
            input=input_args,
            status=status,
            output=output,
            error=error,
            approval_required=approval_required,
        )
        self._entries.append(entry)
        self._append_to_file(entry)
        if self.verbose:
            self._print_entry(entry)
        return entry

    def update_last(self, **kwargs: Any) -> None:
        if not self._entries:
            return
        entry = self._entries[-1]
        for key, value in kwargs.items():
            if hasattr(entry, key):
                setattr(entry, key, value)
        self._append_to_file(entry, update=True)
        if self.verbose:
            self._print_entry(entry)

    @property
    def entries(self) -> list[ActionLogEntry]:
        return list(self._entries)

    def summary(self) -> str:
        lines = [f"Session {self.session_id} - {len(self._entries)} action(s) logged"]
        for e in self._entries:
            icon = {"success": "OK", "error": "ERR", "denied": "DEN", "approved": "APR"}.get(
                e.status, e.status
            )
            lines.append(f"  [{icon}] step {e.step} {e.tool}({json.dumps(e.input)}) -> {e.status}")
        return "\n".join(lines)

    def _append_to_file(self, entry: ActionLogEntry, update: bool = False) -> None:
        try:
            with self.log_file.open("a", encoding="utf-8") as f:
                prefix = "UPDATE " if update else ""
                f.write(prefix + entry.to_json() + "\n")
        except OSError:
            if self.verbose:
                print(f"  [warn] Could not write to {self.log_file}")

    def _print_entry(self, entry: ActionLogEntry) -> None:
        print(f"  [LOG] {entry.tool} | status={entry.status} | in={json.dumps(entry.input)}", end="")
        if entry.output is not None:
            print(f" | out={json.dumps(entry.output)}", end="")
        if entry.error:
            print(f" | error={entry.error}", end="")
        print()
