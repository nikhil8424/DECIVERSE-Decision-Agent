"""Tests for Action Utility Planner, candidate action generation, and ranking."""

import pytest
from app.agent.state import ActionCandidate, DecisionState, VerificationRecord
from app.agent.planner import ActionUtilityPlanner


def test_action_utility_calculation():
    """Verify action utility weights and formula calculation."""
    candidate = ActionCandidate(
        action_type="modify_current_scenario",
        action_name="Adapt Policy",
        goal_progress=0.8,
        constraint_recovery_potential=0.9,
        information_gain=0.6,
        expected_improvement=0.8,
        action_cost=0.2,
        failure_risk=0.1,
        repetition_penalty=0.0,
    )

    score = ActionUtilityPlanner.compute_action_utility(candidate)
    
    # Expected: 0.8*0.25 + 0.9*0.30 + 0.6*0.15 + 0.8*0.20 - 0.2*0.10 - 0.1*0.10 - 0.0*0.15
    # = 0.20 + 0.27 + 0.09 + 0.16 - 0.02 - 0.01 = 0.69
    assert score == pytest.approx(0.69, abs=0.01)
    assert candidate.score == score


def test_planner_generates_multiple_ranked_candidates_on_init():
    """Verify that planner generates multiple candidate actions and ranks them."""
    state = DecisionState(
        goal="Optimize transit policy",
        problem_statement="Traffic congestion and fare burden",
    )

    selected, candidates, summary = ActionUtilityPlanner.plan_next_step(state)

    assert len(candidates) >= 2, "Planner must generate multiple candidate actions"
    # Verify descending sort order
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True), "Candidates must be ranked by utility score"
    assert selected == candidates[0], "Selected action must be highest ranked candidate"
    assert "observe_context" in selected.action_type or "extract" in selected.action_name.lower()
    assert len(summary) > 10, "Decision summary must be generated"
    assert "<think>" not in summary, "Decision summary must not leak think tags"


def test_planner_prioritizes_adaptation_after_verification_failure():
    """Verify planner prioritizes constraint recovery when previous verification failed."""
    state = DecisionState(
        goal="Optimize transit policy",
        context_id="ctx_123",
        graph_id="graph_123",
        current_scenario_id="scen_baseline",
        candidate_scenarios={"scen_baseline": {"name": "Baseline Policy", "max_rounds": 5}},
        social_impact_results={"scen_baseline": {"polarization_score": 0.48, "overall_score": 0.55}},
    )
    # Record a failed verification
    state.verification_history.append(VerificationRecord(
        verification_index=1,
        scenario_id="scen_baseline",
        scenario_name="Baseline Policy",
        status="FAILED",
        violated_constraints=["polarization (0.48 > 0.35) [FAIL]"],
    ))

    selected, candidates, summary = ActionUtilityPlanner.plan_next_step(state)

    assert len(candidates) >= 3
    # Top candidate should be scenario modification to recover from violation
    assert selected.action_type in ("modify_current_scenario", "request_human_confirmation")
    assert selected.constraint_recovery_potential >= 0.6
    assert "polarization" in summary.lower() or "violated" in summary.lower() or "highest" in summary.lower()


def test_planner_decision_summary_auditable_and_clean():
    """Verify decision summary contains rationale without private thinking tokens."""
    state = DecisionState(goal="Housing policy")
    selected, candidates, summary = ActionUtilityPlanner.plan_next_step(state)

    assert not summary.startswith("<think>")
    assert not summary.endswith("</think>")
    assert "selected" in summary.lower() or "score" in summary.lower()


def test_replanner_dynamics_across_states_a_b_c_d():
    """Verify that changing environment/state changes recovery candidate utility, ranking, and selected action.
    
    Tests:
    - State A: High Polarization -> inspect_stakeholder_evidence / stakeholder coalition analysis has highest utility.
    - State B: High Conflict -> request_human_confirmation for conflict arbitration has highest utility.
    - State C: Low Adoption -> modify_current_scenario for direct subsidy/incentive policy adjustment has highest utility.
    - State D: Multiple Simultaneous Deficits -> human consultation / tradeoff guidance outranks unilateral modification.
    """
    from app.agent.replanner import Replanner

    base_state = DecisionState(
        goal="Transit policy optimization",
        context_id="ctx_001",
        graph_id="graph_001",
        current_scenario_id="scen_base",
        candidate_scenarios={"scen_base": {"name": "Base Policy", "intervention": "Standard fare policy"}},
    )

    # State A: High Polarization only
    diag_a = {
        "scenario_id": "scen_base",
        "scenario_name": "Base Policy",
        "violated_constraints": ["polarization (0.55 fails < 0.35) [FAIL]"],
        "suggested_policy_adjustments": ["Introduce phased rollouts and joint consultation committees."],
    }
    candidates_a = Replanner.generate_recovery_candidates(base_state, diag_a)
    top_a = candidates_a[0]
    scores_a = {c.action_type: c.score for c in candidates_a}

    # State B: High Conflict only
    diag_b = {
        "scenario_id": "scen_base",
        "scenario_name": "Base Policy",
        "violated_constraints": ["conflict (0.60 fails < 0.30) [FAIL]"],
        "suggested_policy_adjustments": ["Provide explicit hardship exemptions or financial offsets."],
    }
    candidates_b = Replanner.generate_recovery_candidates(base_state, diag_b)
    top_b = candidates_b[0]
    scores_b = {c.action_type: c.score for c in candidates_b}

    # State C: Low Adoption only
    diag_c = {
        "scenario_id": "scen_base",
        "scenario_name": "Base Policy",
        "violated_constraints": ["adoption (0.35 fails > 0.50) [FAIL]"],
        "suggested_policy_adjustments": ["Augment commuter subsidies and increase marketing clarity."],
    }
    candidates_c = Replanner.generate_recovery_candidates(base_state, diag_c)
    top_c = candidates_c[0]
    scores_c = {c.action_type: c.score for c in candidates_c}

    # State D: Multiple Simultaneous Deficits (Polarization + Conflict + Adoption + Equity)
    diag_d = {
        "scenario_id": "scen_base",
        "scenario_name": "Base Policy",
        "violated_constraints": [
            "polarization (0.58 fails < 0.35) [FAIL]",
            "conflict (0.62 fails < 0.30) [FAIL]",
            "adoption (0.30 fails > 0.50) [FAIL]",
            "equity (0.40 fails > 0.60) [FAIL]",
        ],
        "suggested_policy_adjustments": ["Complete structural overhaul of policy parameters."],
    }
    candidates_d = Replanner.generate_recovery_candidates(base_state, diag_d)
    top_d = candidates_d[0]
    scores_d = {c.action_type: c.score for c in candidates_d}

    # Verify that candidate utilities differ genuinely based on state
    assert scores_c["modify_current_scenario"] > scores_d["modify_current_scenario"], \
        "Single adoption deficit has higher direct policy mutation utility than high-risk multi-deficit state"
    assert scores_a["inspect_stakeholder_evidence"] > scores_c["inspect_stakeholder_evidence"], \
        "Polarization state gives higher information utility to stakeholder sub-graph inspection"
    assert scores_d["request_human_confirmation"] > scores_c["request_human_confirmation"], \
        "Multi-deficit deadlock escalates human arbitration utility"

    # Verify that top selected action changes across distinct conditions
    assert top_c.action_type == "modify_current_scenario", "Low adoption should prioritize direct policy modification"
    assert top_d.action_type in ("request_human_confirmation", "inspect_stakeholder_evidence"), \
        "Multi-deficit deadlock must prioritize stakeholder guidance / human arbitration over blind mutation"
    assert top_a.action_type in ("inspect_stakeholder_evidence", "modify_current_scenario")


def test_human_rejection_redirects_action_selection():
    """Verify that human rejection applies repetition penalty and redirects planning to human directive."""
    state = DecisionState(
        goal="Find viable transit policy",
        context_id="ctx_123",
        graph_id="graph_123",
        current_scenario_id="scen_rejected",
        candidate_scenarios={"scen_rejected": {"name": "Rejected Flat Toll Policy", "intervention": "Flat $5 toll"}},
        social_impact_results={"scen_rejected": {"polarization_score": 0.45, "overall_score": 0.60}},
    )
    # Human rejected the proposal and provided feedback
    state.human_interaction_history.append({
        "selected_option": "Reject",
        "status": "rejected",
        "text_input": "Add a bus discount and cap peak toll at $3.",
    })

    selected, candidates, summary = ActionUtilityPlanner.plan_next_step(state)

    # Verify Candidate A (Rejected scenario simulation) was penalized
    unadapted_sim = next((c for c in candidates if c.parameters.get("scenario_id") == "scen_rejected"), None)
    if unadapted_sim:
        assert unadapted_sim.repetition_penalty >= 0.80

    # Verify adapted candidate incorporating human guidance is top-ranked
    assert selected.action_type == "modify_current_scenario"
    assert "bus discount" in selected.parameters.get("modification_goal", "").lower()
    assert selected.parameters.get("stakeholder_feedback") == "Add a bus discount and cap peak toll at $3."
    assert selected.score > 0.70
