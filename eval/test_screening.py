"""
Pytest-based eval harness for the AI Candidate Screening Assistant.
Loads test cases from test_cases.json and runs parametrized tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIDENCE_THRESHOLD
from rag import init_knowledge_base
from parser import parse_application
from qualifier import qualify_candidate


# ── Load Test Cases ───────────────────────────────────────────────────────

TEST_CASES_FILE = Path(__file__).parent / "test_cases.json"


def load_test_cases() -> list[dict]:
    """Load test cases from JSON file."""
    with open(TEST_CASES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


TEST_CASES = load_test_cases()


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def setup_knowledge_base():
    """Initialize knowledge base once for all tests."""
    init_knowledge_base()


# ── Profile Extraction Tests ─────────────────────────────────────────────

@pytest.mark.parametrize(
    "test_case",
    TEST_CASES,
    ids=[tc["id"] for tc in TEST_CASES],
)
def test_profile_extraction(test_case):
    """Test that the parser extracts key profile fields correctly."""
    profile = parse_application(test_case["input"])
    expected = test_case["expected_profile"]

    # Check name (if expected)
    if expected.get("name") and expected["name"] != "Unknown":
        assert expected["name"].lower() in profile.name.lower(), (
            f"Expected name containing '{expected['name']}', got '{profile.name}'"
        )

    # Check that key skills are extracted
    if expected.get("skills_must_include"):
        profile_skills_lower = [s.lower() for s in profile.skills]
        for skill in expected["skills_must_include"]:
            assert any(
                skill.lower() in ps for ps in profile_skills_lower
            ), f"Expected skill '{skill}' in {profile.skills}"

    # Check experience is reasonable
    if expected.get("min_experience", 0) > 0:
        assert profile.years_of_experience >= expected["min_experience"] * 0.5, (
            f"Expected >= {expected['min_experience'] * 0.5} years, "
            f"got {profile.years_of_experience}"
        )


# ── Qualification Decision Tests ─────────────────────────────────────────

@pytest.mark.parametrize(
    "test_case",
    TEST_CASES,
    ids=[tc["id"] for tc in TEST_CASES],
)
def test_qualification_decision(test_case):
    """Test that the qualifier produces the expected decision."""
    profile = parse_application(test_case["input"])
    decision = qualify_candidate(profile, test_case["input"])

    expected = test_case["expected_decision"]

    # Allow some flexibility: needs_human_review is acceptable for any borderline case
    acceptable_decisions = {expected}
    if expected in ("qualified", "not_qualified"):
        acceptable_decisions.add("needs_human_review")  # conservative is OK

    assert decision.decision in acceptable_decisions, (
        f"[{test_case['id']}] Expected one of {acceptable_decisions}, "
        f"got '{decision.decision}' (confidence: {decision.confidence:.2f})"
    )

    # Verify confidence is in valid range
    assert 0.0 <= decision.confidence <= 1.0, (
        f"Confidence {decision.confidence} out of range [0, 1]"
    )

    # Verify sources are populated
    assert len(decision.sources) > 0, "Decision must cite at least one source document"


# ── Confidence Threshold Tests ───────────────────────────────────────────

@pytest.mark.parametrize(
    "test_case",
    [tc for tc in TEST_CASES if tc["expected_decision"] == "needs_human_review"],
    ids=[tc["id"] for tc in TEST_CASES if tc["expected_decision"] == "needs_human_review"],
)
def test_low_confidence_triggers_review(test_case):
    """Test that borderline cases get flagged for human review."""
    profile = parse_application(test_case["input"])
    decision = qualify_candidate(profile, test_case["input"])

    # If confidence is low, it should be flagged
    if decision.confidence < CONFIDENCE_THRESHOLD:
        assert decision.needs_human_review, (
            f"Low confidence ({decision.confidence:.2f}) should trigger needs_human_review"
        )
