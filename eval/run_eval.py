"""
Custom Eval Report Script — Runs all test cases and generates an accuracy report.
Supports different prompt versions for A/B comparison.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config import CONFIDENCE_THRESHOLD, PROMPT_VERSIONS, EVAL_RESULTS_DIR
from rag import init_knowledge_base
from parser import parse_application
from qualifier import qualify_candidate

console = Console()


def load_test_cases() -> list[dict]:
    """Load test cases from JSON file."""
    test_file = Path(__file__).parent / "test_cases.json"
    with open(test_file, "r", encoding="utf-8") as f:
        return json.load(f)


def run_single_test(test_case: dict) -> dict:
    """Run a single test case and return the result."""
    start = time.time()
    result = {
        "id": test_case["id"],
        "description": test_case["description"],
        "difficulty": test_case["difficulty"],
        "expected_decision": test_case["expected_decision"],
        "actual_decision": None,
        "confidence": None,
        "profile_correct": False,
        "decision_correct": False,
        "error": None,
        "latency_s": 0.0,
    }

    try:
        # Parse profile
        profile = parse_application(test_case["input"])

        # Check profile correctness
        expected_profile = test_case["expected_profile"]
        profile_checks = []

        if expected_profile.get("name") and expected_profile["name"] != "Unknown":
            profile_checks.append(
                expected_profile["name"].lower() in profile.name.lower()
            )

        if expected_profile.get("skills_must_include"):
            profile_skills_lower = [s.lower() for s in profile.skills]
            for skill in expected_profile["skills_must_include"]:
                profile_checks.append(
                    any(skill.lower() in ps for ps in profile_skills_lower)
                )

        result["profile_correct"] = all(profile_checks) if profile_checks else True

        # Qualify candidate
        decision = qualify_candidate(profile, test_case["input"])
        result["actual_decision"] = decision.decision
        result["confidence"] = decision.confidence

        # Check decision correctness (with flexibility)
        expected = test_case["expected_decision"]
        acceptable = {expected}
        if expected in ("qualified", "not_qualified"):
            acceptable.add("needs_human_review")

        result["decision_correct"] = decision.decision in acceptable

    except Exception as e:
        result["error"] = str(e)

    result["latency_s"] = round(time.time() - start, 2)
    return result


def generate_report(results: list[dict], prompt_version_tag: str = "default") -> dict:
    """Generate an accuracy report from eval results."""
    total = len(results)
    errors = sum(1 for r in results if r["error"])
    valid_results = [r for r in results if not r["error"]]

    decision_correct = sum(1 for r in valid_results if r["decision_correct"])
    profile_correct = sum(1 for r in valid_results if r["profile_correct"])

    # By difficulty
    difficulties = {"easy": [], "medium": [], "hard": []}
    for r in valid_results:
        difficulties.get(r["difficulty"], []).append(r)

    difficulty_accuracy = {}
    for diff, cases in difficulties.items():
        if cases:
            correct = sum(1 for c in cases if c["decision_correct"])
            difficulty_accuracy[diff] = {
                "total": len(cases),
                "correct": correct,
                "accuracy": round(correct / len(cases) * 100, 1),
            }

    # Average confidence
    confidences = [r["confidence"] for r in valid_results if r["confidence"] is not None]
    avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

    report = {
        "timestamp": datetime.now().isoformat(),
        "prompt_version_tag": prompt_version_tag,
        "prompt_versions": PROMPT_VERSIONS,
        "total_cases": total,
        "errors": errors,
        "overall_accuracy": round(decision_correct / len(valid_results) * 100, 1) if valid_results else 0.0,
        "profile_accuracy": round(profile_correct / len(valid_results) * 100, 1) if valid_results else 0.0,
        "average_confidence": avg_confidence,
        "by_difficulty": difficulty_accuracy,
        "results": results,
    }

    return report


def display_report(report: dict) -> None:
    """Display the eval report in a formatted way."""
    console.print(Panel(
        f"[bold]Eval Report — {report['prompt_version_tag']}[/]\n"
        f"Timestamp: {report['timestamp']}",
        border_style="bold blue",
    ))

    # Overall stats
    stats_table = Table(title="Overall Results", show_header=True)
    stats_table.add_column("Metric", style="cyan", width=25)
    stats_table.add_column("Value", style="white")

    stats_table.add_row("Total Test Cases", str(report["total_cases"]))
    stats_table.add_row("Errors", str(report["errors"]))
    stats_table.add_row("Decision Accuracy", f"{report['overall_accuracy']}%")
    stats_table.add_row("Profile Accuracy", f"{report['profile_accuracy']}%")
    stats_table.add_row("Avg Confidence", str(report["average_confidence"]))
    console.print(stats_table)

    # By difficulty
    diff_table = Table(title="Accuracy by Difficulty", show_header=True)
    diff_table.add_column("Difficulty", style="cyan")
    diff_table.add_column("Total", style="white")
    diff_table.add_column("Correct", style="green")
    diff_table.add_column("Accuracy", style="yellow")

    for diff in ["easy", "medium", "hard"]:
        if diff in report["by_difficulty"]:
            d = report["by_difficulty"][diff]
            diff_table.add_row(diff, str(d["total"]), str(d["correct"]), f"{d['accuracy']}%")
    console.print(diff_table)

    # Individual results
    detail_table = Table(title="Individual Test Results", show_header=True)
    detail_table.add_column("ID", style="cyan", width=8)
    detail_table.add_column("Difficulty", width=8)
    detail_table.add_column("Expected", width=18)
    detail_table.add_column("Actual", width=18)
    detail_table.add_column("Conf.", width=6)
    detail_table.add_column("Decision", width=8)
    detail_table.add_column("Profile", width=8)

    for r in report["results"]:
        decision_status = "[green]✓[/]" if r["decision_correct"] else "[red]✗[/]"
        profile_status = "[green]✓[/]" if r["profile_correct"] else "[red]✗[/]"
        actual = r["actual_decision"] or "ERROR"
        conf = f"{r['confidence']:.2f}" if r["confidence"] else "N/A"

        if r["error"]:
            actual = f"[red]ERR[/]"
            decision_status = "[red]✗[/]"
            profile_status = "[red]✗[/]"

        detail_table.add_row(
            r["id"], r["difficulty"], r["expected_decision"],
            actual, conf, decision_status, profile_status,
        )

    console.print(detail_table)

    # Failure analysis
    failures = [r for r in report["results"] if not r["decision_correct"] and not r["error"]]
    if failures:
        console.print(Panel(
            "\n".join([
                f"• [{r['id']}] Expected '{r['expected_decision']}', "
                f"got '{r['actual_decision']}' (conf: {r['confidence']:.2f}) "
                f"— {r['description']}"
                for r in failures
            ]),
            title="❌ Failure Analysis",
            border_style="red",
        ))


def save_report(report: dict, filename: str | None = None) -> Path:
    """Save the report to the eval_results directory."""
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{report['prompt_version_tag']}_{timestamp}.json"

    filepath = EVAL_RESULTS_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    console.print(f"\n[dim]Report saved to: {filepath}[/]")
    return filepath


def main():
    """Run the full eval pipeline."""
    console.print(Panel(
        "[bold]🧪 AI Screening Assistant — Eval Harness[/]",
        border_style="bold blue",
    ))

    # Determine prompt version tag
    if len(sys.argv) > 1:
        version_tag = sys.argv[1]
    else:
        version_tag = "v1.0"

    # Initialize knowledge base
    console.print("\n[bold]Initializing knowledge base...[/]")
    init_knowledge_base()

    # Load and run test cases
    test_cases = load_test_cases()
    console.print(f"[bold]Running {len(test_cases)} test cases...[/]\n")

    results = []
    for i, tc in enumerate(test_cases, 1):
        console.print(f"  [{i}/{len(test_cases)}] {tc['id']}: {tc['description']}...", end=" ")
        result = run_single_test(tc)
        status = "[green]✓[/]" if result["decision_correct"] and not result["error"] else "[red]✗[/]"
        console.print(f"{status} ({result['latency_s']}s)")
        results.append(result)

    # Generate and display report
    report = generate_report(results, version_tag)
    console.print()
    display_report(report)

    # Save report
    save_report(report)

    console.print("\n[bold green]✅ Eval complete![/]\n")
    return report


if __name__ == "__main__":
    main()
