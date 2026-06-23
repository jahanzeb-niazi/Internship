"""
Pydantic v2 models for all structured data in the system.
Every LLM output is validated against these schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Candidate Profile (from input parsing) ─────────────────────────────────

class CandidateProfile(BaseModel):
    """Structured candidate data extracted from raw application text."""
    name: str = Field(description="Full name of the candidate")
    skills: list[str] = Field(description="Technical and relevant skills")
    years_of_experience: float = Field(description="Total years of professional experience")
    education_level: str = Field(description="Highest education level and institution")


# ── Qualification Decision ─────────────────────────────────────────────────

class QualificationDecision(BaseModel):
    """RAG-grounded qualification decision for a candidate."""
    decision: Literal["qualified", "not_qualified", "needs_human_review"] = Field(
        description="Qualification outcome"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    reasoning: str = Field(description="Explanation of why this decision was made")
    sources: list[str] = Field(
        description="Document names from the knowledge base that informed this decision"
    )
    needs_human_review: bool = Field(
        default=False,
        description="Whether this decision should be flagged for human review"
    )


# ── Outreach Email ─────────────────────────────────────────────────────────

class OutreachEmail(BaseModel):
    """Personalized outreach email for a qualified candidate."""
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Full email body text")
    tone: str = Field(description="Tone of the email, e.g., professional, friendly")


# ── Tool Call Result ───────────────────────────────────────────────────────

class ToolCallResult(BaseModel):
    """Result of executing a tool/function call."""
    tool_name: str
    success: bool
    result: dict = Field(default_factory=dict)
    error: Optional[str] = None


# ── Approval Request ──────────────────────────────────────────────────────

class ApprovalRequest(BaseModel):
    """Request for human approval before executing a state-modifying action."""
    action: str = Field(description="Name of the action to approve")
    details: dict = Field(description="Details of what will be executed")
    requires_approval: bool = Field(default=True)


# ── LLM Call Log Entry ────────────────────────────────────────────────────

class LLMCallLog(BaseModel):
    """Log entry for a single LLM API call."""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    function_name: str = Field(description="Name of the calling function")
    prompt_version: str = Field(description="Version of the prompt used")
    input_tokens: int = Field(default=0, description="Number of input tokens")
    output_tokens: int = Field(default=0, description="Number of output tokens")
    total_tokens: int = Field(default=0, description="Total tokens used")
    latency_ms: float = Field(description="Latency in milliseconds")
    model: str = Field(description="Model name used")
    success: bool = Field(default=True)
    error: Optional[str] = None


# ── Agent State ───────────────────────────────────────────────────────────

class AgentAction(BaseModel):
    """An action the agent decides to take."""
    action: Literal[
        "check_calendar_availability", "schedule_interview", "send_candidate_notification",
        "escalate_to_human", "draft_outreach", "done"
    ] = Field(description="The action to take")
    reasoning: str = Field(description="Why this action was chosen")
    parameters: dict = Field(default_factory=dict, description="Parameters for the action")


class ScreeningResult(BaseModel):
    """Complete result of screening a single candidate."""
    profile: CandidateProfile
    qualification: QualificationDecision
    actions_taken: list[str] = Field(default_factory=list)
    outreach_email: Optional[OutreachEmail] = None
    tool_results: list[ToolCallResult] = Field(default_factory=list)
