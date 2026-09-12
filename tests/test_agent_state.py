"""Tests for DecisionState, audit records, serialization, and persistence."""

import json
from pathlib import Path
import pytest
from app.agent.state import (
    ActionCandidate,
    AgentStatus,
    DecisionState,
)


def test_decision_state_creation_and_defaults():
    """Verify default fields of DecisionState."""
    state = DecisionState(goal="Find best policy")
    assert state.goal == "Find best policy"
    assert state.status == AgentStatus.IDLE
    assert state.iteration_count == 0
    assert state.max_iterations == 5
    assert "polarization" in state.constraints
    assert state.run_id.startswith("agent_run_")


def test_decision_state_record_helpers():
    """Verify state event recording methods update audit history."""
    state = DecisionState(goal="Urban mobility")

    # Record observation
    state.record_observation("Identified 8 stakeholder groups")
    assert len(state.observations) == 1
    assert "Identified 8 stakeholder groups" in state.observations[0]

    # Record decision
    c1 = ActionCandidate(action_type="act1", action_name="Act 1", score=0.8)
    c2 = ActionCandidate(action_type="act2", action_name="Act 2", score=0.5)
    dec = state.record_decision("Initial hurdle", [c1, c2], c1, "Selected Act 1")
    assert dec.decision_index == 1
    assert len(state.decision_history) == 1
    assert state.decision_history[0].selected_action["action_type"] == "act1"

    # Record tool event
    state.record_tool_event("observe_context", {"problem": "test"}, {"stakeholders": 5}, success=True, duration_seconds=0.4)
    assert len(state.tool_events) == 1
    assert state.tool_events[0].tool_name == "observe_context"
    assert state.tool_events[0].duration_seconds == 0.4

    # Record verification
    v = state.record_verification(
        scenario_id="scen_1",
        scenario_name="Policy 1",
        status="VERIFIED",
        satisfied=["polarization < 0.35"],
        violated=[],
        uncertain=[],
        scores={"overall_score": 0.78, "polarization_score": 0.22},
        overall_score=0.78,
        confidence=0.9,
        summary="All verified",
    )
    assert len(state.verification_history) == 1
    assert state.best_score == 0.78
    assert state.best_scenario_id == "scen_1"

    # Record failure
    state.record_failure("llm_timeout", "Ollama timed out", provider="ollama", recovered=True, recovery_strategy="switch_provider")
    assert len(state.system_failures) == 1
    assert state.system_failures[0].failure_type == "llm_timeout"

    # Record disruption
    state.record_disruption("increase_polarization", "Surge in polarization", {"boost": 0.35})
    assert len(state.domain_disruptions) == 1
    assert state.domain_disruptions[0].disruption_type == "increase_polarization"


def test_decision_state_serialization_roundtrip():
    """Verify to_dict and from_dict produce equivalent state."""
    state = DecisionState(
        goal="Clean air initiative",
        problem_statement="Emissions in city center",
        constraints={"polarization": "<0.30", "adoption": ">0.70"},
        max_iterations=8,
        status=AgentStatus.PLANNING,
    )
    state.record_observation("Step 1 started")
    c = ActionCandidate(action_type="test", action_name="Test Action", score=0.9)
    state.record_decision("Hurdle 1", [c], c, "Test decision")

    state_dict = state.to_dict()
    reconstructed = DecisionState.from_dict(state_dict)

    assert reconstructed.run_id == state.run_id
    assert reconstructed.goal == "Clean air initiative"
    assert reconstructed.constraints == {"polarization": "<0.30", "adoption": ">0.70"}
    assert reconstructed.status == AgentStatus.PLANNING
    assert len(reconstructed.observations) == 1
    assert len(reconstructed.decision_history) == 1
    assert reconstructed.decision_history[0].selected_action["action_type"] == "test"


def test_decision_state_file_persistence(tmp_path: Path):
    """Verify save creates all required JSON audit files."""
    state = DecisionState(goal="Transit optimization")
    state.record_observation("Audit start")
    c = ActionCandidate(action_type="run", action_name="Run", score=0.85)
    state.record_decision("Issue", [c], c, "Summary")
    state.record_tool_event("test_tool", {}, {"res": True})
    state.record_verification("s1", "Policy 1", "VERIFIED", ["passed"], [], [], {}, 0.82, 0.95, "Verified")

    out_dir = str(tmp_path / "test_run_artifacts")
    paths = state.save(out_dir)

    assert set(paths.keys()) == {"state", "decisions", "tool_events", "failures", "verification", "final_result", "demo_trace"}

    for p in paths.values():
        assert Path(p).exists()
        assert Path(p).stat().st_size > 0

    with open(paths["final_result"], "r", encoding="utf-8") as f:
        final_json = json.load(f)
        assert final_json["best_score"] == 0.82
        assert final_json["best_scenario_name"] == "Policy 1"
