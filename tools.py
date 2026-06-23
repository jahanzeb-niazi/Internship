"""
Tools — Simulated tool functions for the screening agent.
Each tool represents an action the agent can take.
State-modifying tools are tagged with requires_approval=True.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

from schemas import ToolCallResult


# ── Tool Registry ─────────────────────────────────────────────────────────
# Maps tool names to their metadata (function, requires_approval, description)

TOOL_REGISTRY: dict[str, dict[str, Any]] = {}


def register_tool(name: str, requires_approval: bool = False, description: str = ""):
    """Decorator to register a tool in the registry."""
    def decorator(func):
        TOOL_REGISTRY[name] = {
            "function": func,
            "requires_approval": requires_approval,
            "description": description,
        }
        func.tool_name = name
        func.requires_approval = requires_approval
        return func
    return decorator


# ── Tool Implementations ──────────────────────────────────────────────────

@register_tool(
    name="check_calendar_availability",
    requires_approval=False,
    description="Check interviewer calendar availability for scheduling. Does not modify state.",
)
def check_calendar_availability(
    candidate_name: str,
    date_range_days: int = 5,
) -> ToolCallResult:
    """
    Simulated: Check calendar availability for interview scheduling.
    Returns available time slots within the given date range.
    """
    try:
        base_date = datetime.now() + timedelta(days=1)
        available_slots = []
        for i in range(date_range_days):
            date = base_date + timedelta(days=i)
            if date.weekday() < 5:  # Weekdays only
                hour = random.choice([10, 11, 14, 15, 16])
                available_slots.append(
                    f"{date.strftime('%Y-%m-%d')} {hour}:00"
                )

        return ToolCallResult(
            tool_name="check_calendar_availability",
            success=True,
            result={
                "candidate": candidate_name,
                "available_slots": available_slots,
                "total_slots": len(available_slots),
            },
        )
    except Exception as e:
        return ToolCallResult(
            tool_name="check_calendar_availability",
            success=False,
            result={},
            error=str(e),
        )


@register_tool(
    name="schedule_interview",
    requires_approval=True,
    description="Schedule an interview for a candidate. MODIFIES STATE — requires human approval.",
)
def schedule_interview(
    candidate_name: str,
    interview_datetime: str = "TBD",
    interviewer: str = "Hiring Manager",
) -> ToolCallResult:
    """
    Simulated: Schedule an interview for the candidate.
    This is a state-modifying action and requires human approval.
    """
    try:
        return ToolCallResult(
            tool_name="schedule_interview",
            success=True,
            result={
                "candidate": candidate_name,
                "scheduled_datetime": interview_datetime,
                "interviewer": interviewer,
                "confirmation_id": f"INT-{random.randint(10000, 99999)}",
                "status": "confirmed",
            },
        )
    except Exception as e:
        return ToolCallResult(
            tool_name="schedule_interview",
            success=False,
            result={},
            error=str(e),
        )


@register_tool(
    name="send_candidate_notification",
    requires_approval=True,
    description="Send a notification/email to the candidate. MODIFIES STATE — requires human approval.",
)
def send_candidate_notification(
    candidate_name: str,
    email: str = "candidate@email.com",
    message: str = "",
) -> ToolCallResult:
    """
    Simulated: Send a notification to the candidate.
    This is a state-modifying action and requires human approval.
    """
    try:
        return ToolCallResult(
            tool_name="send_candidate_notification",
            success=True,
            result={
                "candidate": candidate_name,
                "email": email,
                "message_preview": message[:100] + "..." if len(message) > 100 else message,
                "status": "sent",
                "message_id": f"MSG-{random.randint(10000, 99999)}",
            },
        )
    except Exception as e:
        return ToolCallResult(
            tool_name="send_candidate_notification",
            success=False,
            result={},
            error=str(e),
        )


@register_tool(
    name="escalate_to_human",
    requires_approval=False,
    description="Escalate the candidate to a human reviewer. Read-only escalation.",
)
def escalate_to_human(
    candidate_name: str,
    reason: str,
) -> ToolCallResult:
    """
    Simulated: Escalate the candidate decision to a human reviewer.
    This is NOT a state-modifying action (just flags for review).
    """
    try:
        return ToolCallResult(
            tool_name="escalate_to_human",
            success=True,
            result={
                "candidate": candidate_name,
                "reason": reason,
                "escalation_id": f"ESC-{random.randint(10000, 99999)}",
                "status": "escalated_to_human_reviewer",
                "priority": "high" if "confidence" in reason.lower() else "normal",
            },
        )
    except Exception as e:
        return ToolCallResult(
            tool_name="escalate_to_human",
            success=False,
            result={},
            error=str(e),
        )


def get_tool(name: str) -> dict | None:
    """Get a tool by name from the registry."""
    return TOOL_REGISTRY.get(name)


def execute_tool(name: str, **kwargs) -> ToolCallResult:
    """
    Execute a tool by name with given arguments.
    Handles failures gracefully — never crashes.
    """
    tool_info = get_tool(name)
    if tool_info is None:
        return ToolCallResult(
            tool_name=name,
            success=False,
            result={},
            error=f"Unknown tool: {name}",
        )
    try:
        return tool_info["function"](**kwargs)
    except Exception as e:
        return ToolCallResult(
            tool_name=name,
            success=False,
            result={},
            error=f"Tool execution failed: {str(e)}",
        )


def list_tools() -> list[dict]:
    """List all available tools with their metadata."""
    return [
        {
            "name": name,
            "requires_approval": info["requires_approval"],
            "description": info["description"],
        }
        for name, info in TOOL_REGISTRY.items()
    ]
