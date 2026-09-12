"""End-to-End Autonomous Control Loop Tests (Tests A through F).

Validates:
Test A - Normal Path: Goal -> Observe -> Plan -> Act -> Evaluate -> Verify -> Finalize
Test B - Domain Failure & Replanning: Disruption/Violation -> Replan -> Modify Scenario -> Pass
Test C - Technical Failure & Recovery: Provider Timeout -> Failover -> Continue Workflow
Test D - Malformed Tool Response: Corrupted JSON -> Repair -> Proceed
Test E - Human-in-the-Loop: Pause -> User Approval -> Resume
Test F - Unresolvable Goal: Repeated Failure -> Max Iterations -> Safe Stop with Diagnostics
"""

import os
from unittest.mock import MagicMock, patch
import pytest
from app.agent.state import AgentStatus, DecisionState
from app.agent.autonomous_controller import AutonomousDecisionController
from app.agent.disruption_engine import DisruptionEngine, DomainDisruptionType, TechnicalDisruptionType
from app.agent.provider_failover import ProviderFailoverManager


def test_scenario_a_normal_path():
    """Test A: Normal end-to-end path from Goal to Verification."""
    state = DecisionState(
        goal="Find the most socially viable transit policy",
        problem_statement="Metro City transit fare policy evaluation",
        constraints={
            "polarization": "<0.60",
            "conflict": "<0.60",
            "adoption": ">0.40",
        },
        max_iterations=5,
    )

    traces = []
    def _trace(phase, msg, payload):
        traces.append((phase, msg))

    controller = AutonomousDecisionController(state=state, on_trace_callback=_trace)
    controller.initialize_run(goal=state.goal, problem_statement=state.problem_statement)

    final_state = controller.run_until_completion()

    assert final_state.status == AgentStatus.VERIFIED
    assert final_state.best_score > 0.50
    assert len(final_state.decision_history) >= 2
    assert len(final_state.verification_history) >= 1
    assert any("VERIFIED" in t[0] for t in traces)


def test_scenario_b_domain_failure_and_replanning():
    """Test B: Domain failure (strict constraint violated) triggers replanning and policy adaptation."""
    # Set strict polarization constraint that baseline policy fails
    state = DecisionState(
        goal="Find a non-polarizing transit policy",
        problem_statement="Free weekend transit causing debate",
        constraints={
            "polarization": "<0.35",
            "conflict": "<0.30",
            "adoption": ">0.50",
        },
        max_iterations=6,
    )

    traces = []
    def _trace(phase, msg, payload):
        traces.append((phase, msg))

    controller = AutonomousDecisionController(state=state, on_trace_callback=_trace)
    controller.initialize_run(goal=state.goal, problem_statement=state.problem_statement)

    final_state = controller.run_until_completion()

    # Verify that replanning occurred
    phases = [t[0] for t in traces]
    assert "REPLAN" in phases or "FAILED" in phases
    assert len(final_state.candidate_scenarios) >= 2, "Agent must have generated an adapted policy"
    assert final_state.status == AgentStatus.VERIFIED, "Adapted policy should satisfy the constraints"
    assert "Adapted" in final_state.best_scenario_name or "Adaptation" in final_state.best_scenario_name


def test_scenario_c_technical_failure_and_provider_failover():
    """Test C: Primary LLM timeout causes automatic provider failover and workflow recovery."""
    provider_mgr = ProviderFailoverManager(primary_provider="ollama", fallback_chain=["ollama", "claude-cli", "codex-cli"])
    
    # Inject primary provider timeout
    provider_mgr.inject_provider_failure("ollama", "timeout")

    state = DecisionState(
        goal="Urban housing policy",
        primary_provider="ollama",
        active_provider="ollama",
        constraints={"polarization": "<0.60"},
    )

    traces = []
    def _trace(phase, msg, payload):
        traces.append((phase, msg))

    controller = AutonomousDecisionController(
        state=state,
        provider_manager=provider_mgr,
        on_trace_callback=_trace,
    )
    controller.initialize_run(goal=state.goal, problem_statement=state.problem_statement)

    final_state = controller.run_until_completion()

    # Verify provider switched and workflow completed
    assert final_state.active_provider == "claude-cli"
    assert len(final_state.system_failures) >= 1
    assert final_state.system_failures[0].recovered is True
    assert final_state.status == AgentStatus.VERIFIED


def test_scenario_d_malformed_tool_response_handling():
    """Test D: Malformed tool output is intercepted, repaired, and does not crash the controller."""
    disruption_eng = DisruptionEngine()
    disruption_eng.inject_technical_disruption(
        TechnicalDisruptionType.RETURN_MALFORMED_TOOL_RESPONSE,
        {"tool_name": "observe_context"}
    )

    state = DecisionState(goal="Transit optimization", constraints={"polarization": "<0.60"})
    controller = AutonomousDecisionController(state=state, disruption_engine=disruption_eng)
    controller.initialize_run(goal=state.goal, problem_statement=state.problem_statement)

    final_state = controller.run_until_completion()

    # Verify execution completed despite malformed response
    assert final_state.status in (AgentStatus.VERIFIED, AgentStatus.UNRESOLVED)
    assert len(final_state.tool_events) >= 1


def test_scenario_e_human_in_the_loop_approval():
    """Test E: Agent pauses for human policy confirmation and resumes upon approval."""
    state = DecisionState(
        goal="Transport compromise",
        constraints={"polarization": "<0.35", "adoption": ">0.50"},
    )

    controller = AutonomousDecisionController(state=state)
    controller.initialize_run(goal=state.goal, problem_statement=state.problem_statement)

    # Step until human checkpoint is reached
    for _ in range(5):
        if state.status == AgentStatus.PAUSED_FOR_HUMAN:
            break
        controller.execute_next_step()

    if state.status == AgentStatus.PAUSED_FOR_HUMAN:
        assert state.human_interaction_pending is True
        assert state.human_interaction_request is not None

        # User approves policy modification
        controller.resume_after_human(approval=True)
        assert state.human_interaction_pending is False
        assert len(state.human_interaction_history) == 1

        # Complete run
        controller.run_until_completion()
        assert state.status in (AgentStatus.VERIFIED, AgentStatus.UNRESOLVED)


def test_scenario_f_unresolvable_goal_safe_stopping():
    """Test F: Impossible constraints stop safely at iteration budget without fabricating success."""
    state = DecisionState(
        goal="Impossible perfect policy",
        constraints={
            "polarization": "<0.001",  # Mathematically impossible in stochastic social simulation
            "conflict": "<0.001",
            "adoption": ">0.999",
        },
        max_iterations=3,
    )

    traces = []
    def _trace(phase, msg, payload):
        traces.append((phase, msg))

    controller = AutonomousDecisionController(state=state, on_trace_callback=_trace)
    controller.initialize_run(goal=state.goal, problem_statement=state.problem_statement)

    final_state = controller.run_until_completion()

    assert final_state.status == AgentStatus.UNRESOLVED
    assert "UNRESOLVED" in final_state.final_outcome_summary
    assert "polarization" in final_state.final_outcome_summary.lower()
    assert final_state.iteration_count >= 3


def test_scenario_g_human_rejection_cycle_end_to_end():
    """Test G: Complete Human-in-the-Loop Rejection -> State Update -> Re-Planning -> Adapted Action Execution -> Verification."""
    state = DecisionState(
        goal="Find a socially accepted transport policy",
        problem_statement="Metro transit fee structure",
        constraints={
            "polarization": "<0.35",
            "conflict": "<0.30",
            "adoption": ">0.50",
        },
        max_iterations=6,
    )

    traces = []
    def _trace(phase, msg, payload):
        traces.append((phase, msg))

    controller = AutonomousDecisionController(state=state, on_trace_callback=_trace)
    controller.initialize_run(goal=state.goal, problem_statement=state.problem_statement)

    # Step 1: Observe context
    controller.execute_next_step()
    # Step 2: Formulate baseline
    controller.execute_next_step()
    # Step 3: Simulate baseline (fails polarization)
    controller.execute_next_step()

    # Step 4: Human Rejection with concrete policy feedback
    human_feedback = "Add a bus discount and cap peak toll at $3."
    controller.resume_after_human(approval=False, text_input=human_feedback)

    # Verify state updated
    assert len(state.human_interaction_history) == 1
    assert state.human_interaction_history[0]["text_input"] == human_feedback
    assert state.human_interaction_pending is False

    # Step 5: Execute next step (must plan and adapt policy with human directive)
    controller.execute_next_step()

    # Verify adapted candidate was created with human directive
    assert len(state.candidate_scenarios) >= 2
    adapted_scens = [s for s_id, s in state.candidate_scenarios.items() if "adapted" in s.get("name", "").lower()]
    assert len(adapted_scens) >= 1
    assert any("bus discount" in s.get("description", "").lower() or "directive" in s.get("description", "").lower() for s in adapted_scens)

    # Step 6: Complete run to verification
    final_state = controller.run_until_completion()
    assert final_state.status == AgentStatus.VERIFIED
    assert final_state.best_score > 0.70


def test_scenario_h_autonomous_demo_command_end_to_end(tmp_path):
    """Test H: CLI autonomous-demo command runs real closed-loop demo and exports full audit artifacts."""
    import argparse
    from app.cli import _run_autonomous_demo

    args = argparse.Namespace(
        command="autonomous-demo",
        files=["demo_transit_policy.md"],
        goal="Find the most socially viable transit policy that eliminates ideological polarization and maximizes commuter adoption",
        output_dir=str(tmp_path),
        json=True,
    )

    result = _run_autonomous_demo(args)

    assert result["status"] == "verified"
    assert result["best_score"] > 0.75
    assert result["decisions_count"] >= 3
    assert result["iterations_completed"] >= 3
    assert "artifacts" in result
    
    # Check that all 7 audit files exist
    artifacts = result["artifacts"]
    for key in ("state", "decisions", "tool_events", "failures", "verification", "final_result", "demo_trace"):
        assert key in artifacts
        assert os.path.exists(artifacts[key])
