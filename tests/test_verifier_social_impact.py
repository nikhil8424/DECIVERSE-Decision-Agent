"""Tests for SocialImpactModel-integrated Verifier and constraint evaluation."""

import pytest
from app.research.social_impact_model import SocialImpactResult
from app.agent.verifier import Verifier, VerificationState, ConstraintResult


def test_verifier_verified_state():
    """Verify VERIFIED status when all constraints are satisfied."""
    impact = SocialImpactResult(
        scenario_id="scen_pass",
        scenario_name="Optimal Transit Policy",
        acceptance_score=0.78,
        consensus_score=0.70,
        polarization_score=0.25,
        conflict_score=0.18,
        adoption_score=0.74,
        stability_score=0.80,
        overall_score=0.77,
        confidence=0.85,
    )

    constraints = {
        "polarization": "<0.35",
        "conflict": "<0.30",
        "adoption": ">0.60",
        "acceptance": ">0.70",
    }

    result: ConstraintResult = Verifier.evaluate(impact, constraints)

    assert result.status == VerificationState.VERIFIED
    assert len(result.satisfied_constraints) == 4
    assert len(result.violated_constraints) == 0
    assert len(result.uncertain_reasons) == 0
    assert "VERIFIED" in result.summary


def test_verifier_failed_state():
    """Verify FAILED status and exact violation tracking when constraints fail."""
    impact = SocialImpactResult(
        scenario_id="scen_fail",
        scenario_name="Contested Policy",
        acceptance_score=0.55,
        consensus_score=0.45,
        polarization_score=0.52,  # Violates < 0.35
        conflict_score=0.42,      # Violates < 0.30
        adoption_score=0.65,      # Satisfies > 0.60
        overall_score=0.58,
        confidence=0.85,
    )

    constraints = {
        "polarization": "<0.35",
        "conflict": "<0.30",
        "adoption": ">0.60",
    }

    result: ConstraintResult = Verifier.evaluate(impact, constraints)

    assert result.status == VerificationState.FAILED
    assert len(result.violated_constraints) == 2
    assert len(result.satisfied_constraints) == 1
    
    violated_str = " ".join(result.violated_constraints).lower()
    assert "polarization" in violated_str
    assert "conflict" in violated_str


def test_verifier_uncertain_state_on_empty_data():
    """Verify UNCERTAIN status when simulation metrics are empty or missing."""
    empty_impact = {
        "scenario_id": "scen_empty",
        "acceptance_score": 0.0,
        "polarization_score": 0.0,
        "overall_score": 0.0,
        "confidence": 0.0,
    }

    constraints = {"polarization": "<0.35", "adoption": ">0.60"}

    result: ConstraintResult = Verifier.evaluate(empty_impact, constraints)

    assert result.status == VerificationState.UNCERTAIN
    assert len(result.uncertain_reasons) > 0
    assert "UNCERTAIN" in result.summary


def test_verifier_constraint_parsing_operators():
    """Verify constraint rule parsing for various operators and expressions."""
    r1 = Verifier.parse_constraint("polarization", "<0.35")
    assert r1.operator == "<"
    assert r1.threshold == 0.35
    assert r1.dimension == "polarization_score"

    r2 = Verifier.parse_constraint("adoption", "adoption >= 0.60")
    assert r2.operator == ">="
    assert r2.threshold == 0.60
    assert r2.dimension == "adoption_score"

    r3 = Verifier.parse_constraint("conflict", "<= 0.30")
    assert r3.operator == "<="
    assert r3.threshold == 0.30
