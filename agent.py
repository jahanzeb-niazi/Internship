"""
Screening Agent — LangGraph-based autonomous agent that:
  1. Parses raw application → CandidateProfile
  2. Qualifies candidate using RAG-grounded reasoning
  3. Decides next actions (qualify, schedule, notify, escalate)
  4. Executes tools with human approval for state-modifying actions
  5. Handles tool failures gracefully
"""

from __future__ import annotations

import json
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, END

from schemas import (
    CandidateProfile,
    QualificationDecision,
    ToolCallResult,
    ScreeningResult,
    OutreachEmail,
    AgentAction,
)
from parser import parse_application
from qualifier import qualify_candidate
from tools import execute_tool, get_tool, TOOL_REGISTRY
from approval import request_approval, request_review
from outreach import draft_outreach_email
from llm_service import call_llm
from config import CONFIDENCE_THRESHOLD

from rich.console import Console

console = Console()


# ── Agent State ───────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """State that flows through the LangGraph screening agent."""
    raw_text: str
    profile: CandidateProfile | None
    qualification: QualificationDecision | None
    actions_taken: list[str]
    tool_results: list[ToolCallResult]
    outreach_email: OutreachEmail | None
    current_step: str
    error: str | None
    should_continue: bool


# ── Node Functions ────────────────────────────────────────────────────────

def parse_node(state: AgentState) -> AgentState:
    """Node 1: Parse raw application text into structured profile."""
    console.print("\n[bold blue]📄 Step 1: Parsing Application...[/]")

    try:
        profile = parse_application(state["raw_text"])
        console.print(f"  [green]✓ Name:[/] {profile.name}")
        console.print(f"  [green]✓ Skills:[/] {', '.join(profile.skills)}")
        console.print(f"  [green]✓ Experience:[/] {profile.years_of_experience} years")
        console.print(f"  [green]✓ Education:[/] {profile.education_level}")

        state["profile"] = profile
        state["current_step"] = "qualify"
    except Exception as e:
        console.print(f"  [red]✗ Parsing failed: {e}[/]")
        state["error"] = f"Parsing failed: {str(e)}"
        state["should_continue"] = False

    return state


def qualify_node(state: AgentState) -> AgentState:
    """Node 2: Qualify the candidate using RAG-grounded reasoning."""
    console.print("\n[bold blue]🎯 Step 2: Qualifying Candidate...[/]")

    if state["profile"] is None:
        state["error"] = "No profile available for qualification"
        state["should_continue"] = False
        return state

    try:
        decision = qualify_candidate(state["profile"], state["raw_text"])
        console.print(f"  [green]✓ Decision:[/] {decision.decision}")
        console.print(f"  [green]✓ Confidence:[/] {decision.confidence:.2f}")
        console.print(f"  [green]✓ Sources:[/] {', '.join(decision.sources)}")
        console.print(f"  [green]✓ Needs Review:[/] {decision.needs_human_review}")
        console.print(f"  [dim]  Reasoning: {decision.reasoning[:150]}...[/]")

        state["qualification"] = decision
        state["current_step"] = "decide_action"
    except Exception as e:
        console.print(f"  [red]✗ Qualification failed: {e}[/]")
        state["error"] = f"Qualification failed: {str(e)}"
        state["should_continue"] = False

    return state


def decide_action_node(state: AgentState) -> AgentState:
    """Node 3: LLM decides the next action based on current state."""
    console.print("\n[bold blue]🤖 Step 3: Agent Deciding Next Action...[/]")

    if state["qualification"] is None:
        state["should_continue"] = False
        return state

    decision = state["qualification"]
    profile = state["profile"]
    actions_taken = state["actions_taken"]

    # Build action decision prompt
    available_tools = "\n".join(
        f"- {name}: {info['description']} (requires_approval: {info['requires_approval']})"
        for name, info in TOOL_REGISTRY.items()
    )

    prompt = f"""You are a screening agent. Based on the current state, decide the next action.

CANDIDATE: {profile.name}
QUALIFICATION DECISION: {decision.decision}
CONFIDENCE: {decision.confidence}
NEEDS HUMAN REVIEW: {decision.needs_human_review}
ACTIONS ALREADY TAKEN: {actions_taken if actions_taken else "None"}

AVAILABLE TOOLS:
{available_tools}

RULES:
1. If decision is "qualified" and confidence >= {CONFIDENCE_THRESHOLD}:
   - First check_calendar_availability (if not done)
   - Then schedule_interview (if not done)  
   - Then send_candidate_notification (if not done)
   - Then draft_outreach (if not done)
   - Then "done"
2. If decision is "needs_human_review" or confidence < {CONFIDENCE_THRESHOLD}:
   - escalate_to_human
   - Then "done"
3. If decision is "not_qualified":
   - send_candidate_notification with rejection info
   - Then "done"
4. If all needed actions are taken, use "done"

Decide the SINGLE next action to take. Choose from:
check_calendar_availability, schedule_interview, send_candidate_notification, escalate_to_human, draft_outreach, done
"""

    try:
        action = call_llm(
            prompt=prompt,
            function_name="agent_reasoning",
            response_model=AgentAction,
            temperature=0.3,
        )

        console.print(f"  [cyan]→ Action:[/] {action.action}")
        console.print(f"  [dim]  Reason: {action.reasoning}[/]")

        # Execute the decided action
        state = _execute_action(state, action)

    except Exception as e:
        console.print(f"  [red]✗ Agent reasoning failed: {e}[/]")
        state["error"] = f"Agent reasoning failed: {str(e)}"
        state["should_continue"] = False

    return state


def _execute_action(state: AgentState, action: AgentAction) -> AgentState:
    """Execute the decided action, including approval gates and tool calls."""
    profile = state["profile"]
    decision = state["qualification"]

    if action.action == "done":
        console.print("  [bold green]✅ Agent completed all actions.[/]")
        state["should_continue"] = False
        return state

    if action.action == "draft_outreach":
        # Draft outreach email
        console.print("\n[bold blue]✉️  Drafting Outreach Email...[/]")
        try:
            email = draft_outreach_email(profile, decision)
            state["outreach_email"] = email
            state["actions_taken"].append("draft_outreach")
            console.print(f"  [green]✓ Subject:[/] {email.subject}")
            console.print(f"  [green]✓ Tone:[/] {email.tone}")
        except Exception as e:
            console.print(f"  [red]✗ Outreach drafting failed: {e}[/]")
            state["tool_results"].append(ToolCallResult(
                tool_name="draft_outreach", success=False, error=str(e)
            ))
        return state

    if action.action == "escalate_to_human":
        # Handle human review for low-confidence decisions
        console.print("\n[bold yellow]👤 Escalating to Human...[/]")

        # First run the escalation tool
        result = execute_tool(
            "escalate_to_human",
            candidate_name=profile.name,
            reason=decision.reasoning,
        )
        state["tool_results"].append(result)
        state["actions_taken"].append("escalate_to_human")

        if result.success:
            console.print(f"  [green]✓ Escalated:[/] {result.result.get('escalation_id')}")

            # Ask human for their decision
            human_decision = request_review(
                profile.name, decision.reasoning, decision.confidence
            )
            if human_decision != "defer":
                decision.decision = human_decision
                decision.needs_human_review = False
                console.print(f"  [green]✓ Human decided:[/] {human_decision}")
        else:
            console.print(f"  [red]✗ Escalation failed:[/] {result.error}")

        return state

    tool_name = action.action
    tool_info = get_tool(tool_name)

    if tool_info is None:
        console.print(f"  [red]✗ Unknown tool: {tool_name}[/]")
        state["tool_results"].append(ToolCallResult(
            tool_name=tool_name, success=False, error=f"Unknown tool: {tool_name}"
        ))
        return state

    # Check approval gate for state-modifying tools
    if tool_info["requires_approval"]:
        # Build details for approval display
        details = {
            "candidate": profile.name,
            "action": tool_name,
            **action.parameters,
        }
        approved = request_approval(tool_name, details)
        if not approved:
            console.print(f"  [yellow]⚠ Action '{tool_name}' was rejected by human.[/]")
            state["actions_taken"].append(f"{tool_name}_rejected")
            state["tool_results"].append(ToolCallResult(
                tool_name=tool_name, success=False, error="Rejected by human"
            ))
            return state

    # Execute the tool
    console.print(f"\n[bold blue]🔧 Executing Tool: {tool_name}...[/]")

    # Build tool arguments
    tool_kwargs = {"candidate_name": profile.name, **action.parameters}

    result = execute_tool(tool_name, **tool_kwargs)
    state["tool_results"].append(result)

    if result.success:
        state["actions_taken"].append(tool_name)
        console.print(f"  [green]✓ Success:[/] {json.dumps(result.result, indent=2)}")
    else:
        console.print(f"  [red]✗ Failed:[/] {result.error}")
        # Agent doesn't crash on tool failure — it continues safely
        state["actions_taken"].append(f"{tool_name}_failed")

    return state


# ── Routing Functions ─────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """Determine if the agent should continue deciding actions or end."""
    if not state.get("should_continue", True):
        return "end"
    if state.get("error"):
        return "end"
    return "decide_action"


# ── Graph Construction ────────────────────────────────────────────────────

def build_screening_agent() -> StateGraph:
    """
    Build the LangGraph screening agent.

    Flow: parse → qualify → decide_action → (loop until done) → end
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("parse", parse_node)
    graph.add_node("qualify", qualify_node)
    graph.add_node("decide_action", decide_action_node)

    # Add edges
    graph.set_entry_point("parse")
    graph.add_edge("parse", "qualify")
    graph.add_edge("qualify", "decide_action")

    # Conditional edge: loop back to decide_action or end
    graph.add_conditional_edges(
        "decide_action",
        should_continue,
        {
            "decide_action": "decide_action",
            "end": END,
        },
    )

    return graph.compile()


def run_screening_agent(raw_text: str) -> ScreeningResult:
    """
    Run the full screening pipeline on a raw application text.

    Args:
        raw_text: Raw job application text from the candidate.

    Returns:
        Complete ScreeningResult with profile, decision, actions, and email.
    """
    console.print(Panel(
        "[bold]AI Candidate Screening Agent[/]\nProcessing application...",
        border_style="bold blue",
    ))

    # Initialize state
    initial_state: AgentState = {
        "raw_text": raw_text,
        "profile": None,
        "qualification": None,
        "actions_taken": [],
        "tool_results": [],
        "outreach_email": None,
        "current_step": "parse",
        "error": None,
        "should_continue": True,
    }

    # Run the agent graph
    agent = build_screening_agent()
    final_state = agent.invoke(initial_state)

    # Build the result
    result = ScreeningResult(
        profile=final_state["profile"] or CandidateProfile(
            name="Unknown", skills=[], years_of_experience=0, education_level="Unknown"
        ),
        qualification=final_state["qualification"] or QualificationDecision(
            decision="needs_human_review",
            confidence=0.0,
            reasoning="Processing failed",
            sources=[],
            needs_human_review=True,
        ),
        actions_taken=final_state["actions_taken"],
        outreach_email=final_state.get("outreach_email"),
        tool_results=final_state["tool_results"],
    )

    return result


# Import Panel for display
from rich.panel import Panel
