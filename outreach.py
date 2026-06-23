"""
Outreach Drafter — Generates personalized outreach emails for qualified candidates.
Uses the candidate's extracted profile to create role-specific, personalized emails.
"""

from __future__ import annotations

from schemas import CandidateProfile, QualificationDecision, OutreachEmail
from llm_service import call_llm


def draft_outreach_email(
    profile: CandidateProfile,
    decision: QualificationDecision,
) -> OutreachEmail:
    """
    Draft a personalized outreach email for a qualified candidate.

    The email references the candidate's specific skills, background,
    and the role they applied for — not a generic template.

    Args:
        profile: Structured candidate profile.
        decision: Qualification decision (includes sources which indicate the role).

    Returns:
        Validated OutreachEmail with subject, body, and tone.
    """
    skills_str = ", ".join(profile.skills)

    # Determine the role from the sources
    role = _infer_role_from_sources(decision.sources)

    prompt = f"""You are a friendly, professional recruiter. Write a personalized outreach email 
for a qualified candidate.

CANDIDATE PROFILE:
- Name: {profile.name}
- Skills: {skills_str}
- Years of Experience: {profile.years_of_experience}
- Education: {profile.education_level}

ROLE APPLIED FOR: {role}

QUALIFICATION: {decision.decision} (confidence: {decision.confidence:.2f})
REASONING SUMMARY: {decision.reasoning[:300]}

INSTRUCTIONS:
1. Address the candidate by name.
2. Reference SPECIFIC skills from their profile (e.g., "your experience with Node.js and AWS").
3. Mention the specific role they applied for.
4. Reference their background naturally (e.g., "with your CS degree from FAST and 3 years of backend experience").
5. Keep the tone professional yet warm and encouraging.
6. Include next steps (interview scheduling details).
7. This must NOT be a generic template — it should feel personally written for this candidate.

The email should be 150-250 words.
"""

    return call_llm(
        prompt=prompt,
        function_name="draft_outreach",
        response_model=OutreachEmail,
        temperature=0.7,
    )


def _infer_role_from_sources(sources: list[str]) -> str:
    """Infer the job role from the knowledge base source documents."""
    role_mapping = {
        "backend_engineer_jd.md": "Backend Engineer",
        "frontend_engineer_jd.md": "Frontend Engineer",
        "data_engineer_jd.md": "Data Engineer",
    }

    for source in sources:
        if source in role_mapping:
            return role_mapping[source]

    return "Software Engineer"
