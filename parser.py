"""
Input Parser — extracts a structured CandidateProfile from raw application text.
Uses Gemini via llm_service with Pydantic validation.
"""

from __future__ import annotations

from schemas import CandidateProfile
from llm_service import call_llm


def parse_application(raw_text: str) -> CandidateProfile:
    """
    Parse a raw job application text into a structured CandidateProfile.

    Args:
        raw_text: Free-form application text from a candidate.

    Returns:
        Validated CandidateProfile with name, skills, experience, education.

    Raises:
        RuntimeError: If parsing fails after all retries.
    """
    prompt = f"""You are an expert HR assistant. Extract structured candidate information 
from the following job application text.

APPLICATION TEXT:
\"\"\"
{raw_text}
\"\"\"

Extract the following fields:
- name: The candidate's full name
- skills: A list of technical and relevant skills mentioned (be comprehensive)
- years_of_experience: Number of years of professional experience (as a float, e.g., 3.0)
- education_level: Highest education level and institution (e.g., "CS degree, FAST University")

If a field is not explicitly mentioned, make a reasonable inference or use "Unknown" for strings 
and 0.0 for numbers.
"""
    return call_llm(
        prompt=prompt,
        function_name="parse_application",
        response_model=CandidateProfile,
        temperature=0.3,  # Low temperature for extraction accuracy
    )
