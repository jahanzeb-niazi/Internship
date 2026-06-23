"""
Qualifier — Makes RAG-grounded qualification decisions for candidates.
Retrieves relevant job descriptions, rubrics, and criteria from ChromaDB,
then uses Gemini to produce a structured decision with confidence score.
"""

from __future__ import annotations

from schemas import CandidateProfile, QualificationDecision
from llm_service import call_llm
from rag import retrieve_context
from config import CONFIDENCE_THRESHOLD


def qualify_candidate(
    profile: CandidateProfile,
    raw_text: str,
) -> QualificationDecision:
    """
    Qualify a candidate using RAG-grounded reasoning.

    Steps:
        1. Build a search query from the candidate's profile
        2. Retrieve relevant documents from the knowledge base
        3. Pass profile + context to Gemini for a grounded decision
        4. Enforce confidence threshold — low confidence → needs_human_review

    Args:
        profile: Structured candidate profile (from parser).
        raw_text: Original application text.

    Returns:
        Validated QualificationDecision with decision, confidence, reasoning, sources.
    """
    # Build RAG query from profile
    skills_str = ", ".join(profile.skills)
    query = (
        f"Job role qualifications requirements for candidate with skills: {skills_str}, "
        f"{profile.years_of_experience} years experience, "
        f"education: {profile.education_level}"
    )

    # Retrieve relevant context from knowledge base
    context_chunks = retrieve_context(query)

    # Format context for the prompt
    context_text = ""
    source_names = set()
    for chunk in context_chunks:
        context_text += f"\n--- Source: {chunk['source']} ---\n{chunk['content']}\n"
        source_names.add(chunk["source"])

    # Build qualification prompt
    prompt = f"""You are an expert hiring manager. Evaluate the following candidate based on 
the provided knowledge base documents (job descriptions, rubrics, and hiring criteria).

CANDIDATE PROFILE:
- Name: {profile.name}
- Skills: {skills_str}
- Years of Experience: {profile.years_of_experience}
- Education: {profile.education_level}

ORIGINAL APPLICATION:
\"\"\"{raw_text}\"\"\"

KNOWLEDGE BASE CONTEXT (use these documents to ground your decision):
{context_text}

INSTRUCTIONS:
1. Compare the candidate's profile against the job requirements and rubric.
2. Score the candidate using the rubric dimensions (technical skills, experience, education, 
   project relevance, communication).
3. Make a decision: "qualified", "not_qualified", or "needs_human_review".
4. Provide a confidence score between 0.0 and 1.0 based on how clear the evidence is.
5. Explain your reasoning, referencing specific criteria from the documents.
6. List which source documents you used in the "sources" field.

IMPORTANT:
- Use ONLY the document names from the knowledge base in the sources field.
  Available sources: {list(source_names)}
- The "needs_human_review" field should be set to true if confidence is below {CONFIDENCE_THRESHOLD}.
- Be fair and consistent. Evaluate based on skills and evidence, not assumptions.
"""

    decision = call_llm(
        prompt=prompt,
        function_name="qualify_candidate",
        response_model=QualificationDecision,
        temperature=0.3,
    )

    # Enforce confidence threshold — override if confidence is too low
    if decision.confidence < CONFIDENCE_THRESHOLD:
        decision.needs_human_review = True
        if decision.decision != "needs_human_review":
            decision.reasoning += (
                f" [AUTO-FLAG: Confidence {decision.confidence:.2f} is below "
                f"threshold {CONFIDENCE_THRESHOLD}. Flagged for human review.]"
            )

    return decision
