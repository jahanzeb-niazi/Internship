"""
Main Entry Point — CLI for the AI Candidate Screening Assistant.
Supports interactive mode (paste application) and file input mode.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from config import CONFIDENCE_THRESHOLD
from rag import init_knowledge_base, get_kb_stats
from agent import run_screening_agent
from logger import get_call_stats
from schemas import ScreeningResult


console = Console()


# ── Sample Application (the Ayesha example from the brief) ────────────────

SAMPLE_APPLICATION = """Hi, I'm Ayesha. I have 3 years of experience building backend systems in Node.js and Python. I've worked on REST APIs, deployed on AWS, and recently started using LangChain for a personal project. I have a CS degree from FAST. I'm applying for the Backend Engineer role."""


def display_result(result: ScreeningResult) -> None:
    """Display the screening result in a formatted way."""

    console.print("\n")
    console.print(Panel(
        "[bold]📊 SCREENING RESULTS[/]",
        border_style="bold green",
    ))

    # ── Profile ───────────────────────────────────────────────────────────
    profile_table = Table(title="Candidate Profile", show_header=True)
    profile_table.add_column("Field", style="cyan", width=20)
    profile_table.add_column("Value", style="white")

    profile_table.add_row("Name", result.profile.name)
    profile_table.add_row("Skills", ", ".join(result.profile.skills))
    profile_table.add_row("Experience", f"{result.profile.years_of_experience} years")
    profile_table.add_row("Education", result.profile.education_level)
    console.print(profile_table)

    # ── Qualification ────────────────────────────────────────────────────
    console.print()
    decision = result.qualification

    decision_color = {
        "qualified": "green",
        "not_qualified": "red",
        "needs_human_review": "yellow",
    }.get(decision.decision, "white")

    qual_table = Table(title="Qualification Decision", show_header=True)
    qual_table.add_column("Field", style="cyan", width=20)
    qual_table.add_column("Value", style="white")

    qual_table.add_row("Decision", f"[bold {decision_color}]{decision.decision}[/]")
    qual_table.add_row("Confidence", f"{decision.confidence:.2f}")
    qual_table.add_row("Needs Review", str(decision.needs_human_review))
    qual_table.add_row("Sources", ", ".join(decision.sources))
    console.print(qual_table)

    console.print(Panel(
        decision.reasoning,
        title="Reasoning",
        border_style="dim",
    ))

    # ── Actions Taken ─────────────────────────────────────────────────────
    if result.actions_taken:
        console.print()
        action_table = Table(title="Actions Taken", show_header=True)
        action_table.add_column("#", style="cyan", width=5)
        action_table.add_column("Action", style="white")

        for i, action in enumerate(result.actions_taken, 1):
            action_table.add_row(str(i), action)
        console.print(action_table)

    # ── Tool Results ──────────────────────────────────────────────────────
    if result.tool_results:
        console.print()
        tool_table = Table(title="Tool Call Results", show_header=True)
        tool_table.add_column("Tool", style="cyan", width=30)
        tool_table.add_column("Success", style="white", width=10)
        tool_table.add_column("Details", style="dim")

        for tr in result.tool_results:
            status = "[green]✓[/]" if tr.success else "[red]✗[/]"
            details = json.dumps(tr.result, indent=0)[:80] if tr.result else (tr.error or "")
            tool_table.add_row(tr.tool_name, status, details)
        console.print(tool_table)

    # ── Outreach Email ───────────────────────────────────────────────────
    if result.outreach_email:
        console.print()
        console.print(Panel(
            f"[bold]Subject:[/] {result.outreach_email.subject}\n"
            f"[bold]Tone:[/] {result.outreach_email.tone}\n\n"
            f"{result.outreach_email.body}",
            title="✉️  Personalized Outreach Email",
            border_style="bold magenta",
        ))


def display_logs_summary() -> None:
    """Display a summary of LLM call logs."""
    stats = get_call_stats()
    console.print()
    log_table = Table(title="📝 LLM Call Logs Summary", show_header=True)
    log_table.add_column("Metric", style="cyan", width=25)
    log_table.add_column("Value", style="white")

    log_table.add_row("Total LLM Calls", str(stats["total_calls"]))
    log_table.add_row("Total Tokens Used", str(stats["total_tokens"]))
    log_table.add_row("Avg Latency (ms)", str(stats["avg_latency_ms"]))
    log_table.add_row("Success Rate", f"{stats['success_rate']}%")

    if stats["calls_by_function"]:
        for func, count in stats["calls_by_function"].items():
            log_table.add_row(f"  └ {func}", str(count))

    console.print(log_table)


def main():
    """Main CLI entry point."""
    console.print(Panel(
        "[bold]🤖 AI Candidate Screening Assistant[/]\n"
        f"Confidence Threshold: {CONFIDENCE_THRESHOLD}",
        border_style="bold blue",
    ))

    # ── Initialize Knowledge Base ─────────────────────────────────────────
    console.print("\n[bold]Initializing Knowledge Base...[/]")
    kb_result = init_knowledge_base()
    console.print(f"  Status: {kb_result['status']}")
    console.print(f"  Total Chunks: {kb_result['total_chunks']}")

    if kb_result.get("files"):
        console.print(f"  Documents: {', '.join(kb_result['files'])}")

    # ── Get Application Input ─────────────────────────────────────────────
    console.print("\n[bold yellow]Choose input mode:[/]")
    console.print("  1. Use sample application (Ayesha example)")
    console.print("  2. Enter application text manually")
    console.print("  3. Load from file")

    choice = console.input("\n[bold]Enter choice (1/2/3): [/]").strip()

    if choice == "1":
        raw_text = SAMPLE_APPLICATION
        console.print(Panel(raw_text, title="Input Application", border_style="dim"))
    elif choice == "2":
        console.print("[dim]Paste the application text (press Enter twice to finish):[/]")
        lines = []
        empty_count = 0
        while empty_count < 1:
            line = input()
            if line == "":
                empty_count += 1
            else:
                empty_count = 0
                lines.append(line)
        raw_text = "\n".join(lines)
    elif choice == "3":
        file_path = console.input("[bold]Enter file path: [/]").strip()
        raw_text = Path(file_path).read_text(encoding="utf-8")
    else:
        console.print("[yellow]Invalid choice, using sample application.[/]")
        raw_text = SAMPLE_APPLICATION

    # ── Run Screening Agent ──────────────────────────────────────────────
    result = run_screening_agent(raw_text)

    # ── Display Results ──────────────────────────────────────────────────
    display_result(result)
    display_logs_summary()

    console.print("\n[bold green]✅ Screening complete![/]\n")

    return result


if __name__ == "__main__":
    main()
