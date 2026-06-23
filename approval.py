"""
Human-in-the-Loop Approval Gate — CLI-based.
State-modifying actions must pass through this gate before execution.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def request_approval(action_name: str, details: dict) -> bool:
    """
    Present a state-modifying action to the human for approval via CLI.

    Args:
        action_name: Name of the action to approve (e.g., "schedule_interview").
        details: Dictionary of action details to display.

    Returns:
        True if approved, False if rejected.
    """
    # Build the details table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")

    for key, value in details.items():
        table.add_row(str(key), str(value))

    # Display the approval panel
    console.print()
    console.print(Panel(
        table,
        title=f"🔒 APPROVAL REQUIRED: {action_name}",
        subtitle="This action modifies state and requires human approval",
        border_style="bold red",
    ))
    console.print()

    # Prompt for approval
    while True:
        response = console.input(
            "[bold yellow]Do you approve this action? (yes/no): [/]"
        ).strip().lower()

        if response in ("yes", "y"):
            console.print("[bold green]✅ Action APPROVED[/]")
            return True
        elif response in ("no", "n"):
            console.print("[bold red]❌ Action REJECTED[/]")
            return False
        else:
            console.print("[dim]Please enter 'yes' or 'no'[/]")


def request_review(candidate_name: str, reasoning: str, confidence: float) -> str:
    """
    Present a low-confidence decision for human review.

    Args:
        candidate_name: Name of the candidate.
        reasoning: Agent's reasoning for the decision.
        confidence: The confidence score.

    Returns:
        Human's decision: "qualified", "not_qualified", or "defer".
    """
    console.print()
    console.print(Panel(
        f"[bold]Candidate:[/] {candidate_name}\n"
        f"[bold]Confidence:[/] {confidence:.2f}\n"
        f"[bold]Reasoning:[/]\n{reasoning}",
        title="👤 HUMAN REVIEW REQUIRED",
        subtitle="Low confidence — please make the final decision",
        border_style="bold yellow",
    ))
    console.print()

    while True:
        response = console.input(
            "[bold yellow]Your decision (qualified / not_qualified / defer): [/]"
        ).strip().lower()

        if response in ("qualified", "not_qualified", "defer"):
            console.print(f"[bold green]Decision recorded: {response}[/]")
            return response
        else:
            console.print("[dim]Please enter 'qualified', 'not_qualified', or 'defer'[/]")
