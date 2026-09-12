"""Autonomous Decision Controller.

Central orchestrator executing the closed-loop autonomous decision cycle:
GOAL -> OBSERVE -> GENERATE ACTIONS -> RANK -> DECIDE -> ACT -> EVALUATE -> VERIFY -> REPLAN -> FINALIZE.
"""

from __future__ import annotations

from datetime import datetime
import os
import time
from typing import Any, Callable, Dict, List, Optional

from ..utils.logger import get_logger
from .disruption_engine import DisruptionEngine
from .failure_handler import FailureHandler, FailureSeverity, FailureType
from .human_interaction import HumanInteractionManager
from .planner import ActionUtilityPlanner
from .provider_failover import ProviderFailoverManager
from .replanner import Replanner
from .state import ActionCandidate, AgentStatus, DecisionState, VerificationRecord
from .tools import AgentToolRegistry, AgentToolResult
from .verifier import ConstraintResult, VerificationState, Verifier

logger = get_logger('mirofish.agent.controller')


class AutonomousDecisionController:
    """The central autonomous agent controller for MiroFish Community."""

    def __init__(
        self,
        state: Optional[DecisionState] = None,
        provider_manager: Optional[ProviderFailoverManager] = None,
        disruption_engine: Optional[DisruptionEngine] = None,
        human_manager: Optional[HumanInteractionManager] = None,
        on_trace_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    ):
        self.state = state or DecisionState()
        self.provider_manager = provider_manager or ProviderFailoverManager()
        self.disruption_engine = disruption_engine or DisruptionEngine()
        self.human_manager = human_manager or HumanInteractionManager()
        self.tool_registry = AgentToolRegistry(
            provider_manager=self.provider_manager,
            disruption_engine=self.disruption_engine,
        )
        self.planner = ActionUtilityPlanner()
        self.replanner = Replanner()
        self.on_trace_callback = on_trace_callback  # (phase, message, payload)
        
        logger.info(f"AutonomousDecisionController initialized for run {self.state.run_id}")

    def emit_trace(self, phase: str, message: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Emit real-time trace event to logger and registered UI callbacks."""
        log_msg = f"[{phase.upper()}] {message}"
        logger.info(log_msg)
        self.state.record_observation(f"{phase.upper()}: {message}")
        if self.on_trace_callback:
            try:
                self.on_trace_callback(phase, message, payload or {})
            except Exception as e:
                logger.warning(f"Error in trace callback: {e}")

    def initialize_run(
        self,
        goal: str,
        problem_statement: str,
        source_files: Optional[List[str]] = None,
        constraints: Optional[Dict[str, str]] = None,
        objectives: Optional[List[str]] = None,
        max_iterations: int = 5,
        primary_provider: Optional[str] = None,
    ) -> DecisionState:
        """Set up a new autonomous decision problem run."""
        self.state.goal = goal
        self.state.problem_statement = problem_statement or goal
        self.state.source_files = source_files or []
        if constraints:
            self.state.constraints = constraints
        if objectives:
            self.state.objectives = objectives
        self.state.max_iterations = max_iterations
        if primary_provider:
            self.state.primary_provider = primary_provider
            self.state.active_provider = primary_provider
            self.provider_manager.primary_provider = primary_provider

        self.state.status = AgentStatus.OBSERVING
        self.emit_trace("GOAL", f"Target goal: '{self.goal_summary()}' with {len(self.state.constraints)} constraint(s)")
        return self.state

    def goal_summary(self) -> str:
        return self.state.goal or self.state.problem_statement or "Socially viable policy optimization"

    def execute_next_step(self) -> AgentStatus:
        """Execute a single step of the autonomous control loop."""
        # 1. Check if paused for human interaction
        if self.state.status == AgentStatus.PAUSED_FOR_HUMAN:
            self.emit_trace("PAUSED", "Waiting for human approval or context input")
            return self.state.status

        # 2. Check if iteration budget exceeded
        if self.state.iteration_count >= self.state.max_iterations:
            self._handle_unresolved_stopping("Maximum iteration budget reached without full constraint verification.")
            return self.state.status

        self.state.iteration_count += 1
        self.emit_trace("OBSERVE", f"Starting Iteration #{self.state.iteration_count}/{self.state.max_iterations}")

        # 3. PLAN: Generate and rank multiple candidate actions
        self.state.status = AgentStatus.PLANNING
        selected_action, candidate_actions, decision_summary = self.planner.plan_next_step(self.state)
        
        # Record decision event
        self.state.record_decision(
            current_issue=self._get_current_issue(),
            candidate_actions=candidate_actions,
            selected_action=selected_action,
            decision_summary=decision_summary,
        )

        self.emit_trace(
            "PLAN",
            f"Evaluated {len(candidate_actions)} candidate action(s). Selected '{selected_action.action_name}' (Score: {selected_action.score:.2f})",
            {
                "candidates": [c.to_dict() for c in candidate_actions],
                "selected": selected_action.to_dict(),
                "summary": decision_summary,
            }
        )

        # 4. DECIDE & ACT: Execute the selected action
        self.state.status = AgentStatus.EXECUTING
        self.state.current_action = selected_action.action_name
        self._dispatch_action(selected_action)

        return self.state.status

    def _get_current_issue(self) -> str:
        """Summarize current hurdle for decision logging."""
        if not self.state.context_id:
            return "Community context and stakeholders not yet grounded."
        if not self.state.candidate_scenarios:
            return "Initial policy scenarios not yet created."
        if self.state.verification_history:
            last_v = self.state.verification_history[-1]
            if last_v.status == "FAILED":
                return f"Constraint violation(s): {', '.join(last_v.violated_constraints)}"
            if last_v.status == "UNCERTAIN":
                return f"Decision uncertainty: {', '.join(last_v.uncertain_aspects)}"
        return "Evaluating policy alternatives and social consequence tradeoffs."

    def _dispatch_action(self, candidate: ActionCandidate) -> None:
        """Route candidate action to corresponding tool and handle result."""
        a_type = candidate.action_type
        params = candidate.parameters

        if a_type == "observe_context":
            self.emit_trace("ACTION", "Running observe_context tool")
            result = self.tool_registry.execute_tool(
                "observe_context",
                problem_statement=params.get("problem_statement", self.state.problem_statement),
                source_files=params.get("source_files", self.state.source_files),
            )
            self._handle_observe_context_result(result)

        elif a_type == "inspect_stakeholder_evidence":
            self.emit_trace("ACTION", "Running get_stakeholders tool")
            result = self.tool_registry.execute_tool(
                "get_stakeholders",
                graph_id=self.state.graph_id or "graph_default",
                agent_count=20,
            )
            self._handle_stakeholders_result(result)

        elif a_type == "create_baseline_scenario":
            self.emit_trace("ACTION", "Running create_scenario tool for baseline")
            result = self.tool_registry.execute_tool(
                "create_scenario",
                name=params.get("name", "Baseline Policy"),
                description=params.get("description", ""),
                intervention=params.get("intervention", self.state.goal),
                platform=params.get("platform", "reddit"),
                max_rounds=params.get("max_rounds", 5),
                agent_count=params.get("agent_count", 20),
                context_id=self.state.context_id,
                problem_statement=self.state.goal,
            )
            self._handle_create_scenario_result(result)

        elif a_type == "run_simulation_and_eval":
            scenario_id = params.get("scenario_id")
            scenario_data = params.get("scenario_data", {})
            self.emit_trace("ACTION", f"Running simulation for '{scenario_data.get('name', scenario_id)}'")
            
            # Step 1: Run simulation
            sim_res = self.tool_registry.execute_tool(
                "run_simulation",
                scenario_id=scenario_id,
                scenario_data=scenario_data,
                rounds=params.get("rounds", 5),
            )
            
            if not sim_res.success:
                self._handle_tool_failure("run_simulation", sim_res)
                return

            sim_id = sim_res.data.get("simulation_id", "")
            self.state.simulation_results[scenario_id] = sim_res.data
            
            # Step 2: Calculate social impact
            self.emit_trace("EVALUATE", "Evaluating Emergent Behaviour & Social Impact")
            impact_res = self.tool_registry.execute_tool(
                "calculate_social_impact",
                scenario_id=scenario_id,
                scenario_name=scenario_data.get("name", scenario_id),
                simulation_id=sim_id,
            )
            
            if not impact_res.success:
                self._handle_tool_failure("calculate_social_impact", impact_res)
                return

            impact_data = impact_res.data.get("impact_result", {})
            self.state.social_impact_results[scenario_id] = impact_data
            
            # Step 3: Verify against constraints
            self.emit_trace("VERIFY", "Verifying social impact against constraints")
            self._execute_verification(scenario_id, scenario_data.get("name", scenario_id), impact_data)

        elif a_type == "modify_current_scenario":
            self.emit_trace("ACTION", "Running modify_scenario tool to adapt policy")
            result = self.tool_registry.execute_tool(
                "modify_scenario",
                base_scenario=params.get("base_scenario", {}),
                modification_goal=params.get("modification_goal", ""),
                violated_constraints=params.get("violated_constraints", []),
                stakeholder_feedback=params.get("stakeholder_feedback"),
            )
            self._handle_modify_scenario_result(result)

        elif a_type == "request_additional_simulation":
            scenario_id = params.get("scenario_id")
            scenario_name = params.get("scenario_name", "")
            scenario_data = params.get("scenario_data", {})
            self.emit_trace("ACTION", f"Sampling repeated simulations for '{scenario_name}' uncertainty")
            
            result = self.tool_registry.execute_tool(
                "request_additional_simulation",
                scenario_id=scenario_id,
                scenario_name=scenario_name,
                scenario_data=scenario_data,
                sample_runs=3,
            )
            if result.success:
                self.state.uncertainty_metrics = result.data
                self.emit_trace(
                    "OBSERVE",
                    f"Multi-run uncertainty quantified (Stability: {result.data.get('overall_stability', 0.0):.2f})",
                    result.data
                )

        elif a_type == "compare_candidates":
            self.emit_trace("ACTION", "Synthesizing tradeoffs across scenarios")
            result = self.tool_registry.execute_tool(
                "compare_scenarios",
                problem_id=self.state.run_id,
                problem_statement=self.state.goal,
                impact_results=params.get("impact_results", []),
            )
            if result.success:
                self.emit_trace("OBSERVE", "Multi-scenario decision intelligence synthesized", result.data)

        elif a_type == "request_human_confirmation":
            self.state.status = AgentStatus.PAUSED_FOR_HUMAN
            self.state.human_interaction_pending = True
            curr_data = self.state.candidate_scenarios.get(self.state.current_scenario_id or "", {})
            req = self.human_manager.create_policy_confirmation_request(
                current_scenario_name=curr_data.get("name", "Current Policy"),
                modified_scenario_name=f"{curr_data.get('name', 'Policy')} (Proposed Adaptation)",
                violated_constraints=self.state.verification_history[-1].violated_constraints if self.state.verification_history else [],
                proposed_modifications=["Compromise terms", "Stakeholder consultation committee"],
            )
            self.state.human_interaction_request = req.to_dict()
            self.emit_trace("HUMAN", "Agent paused for human confirmation", req.to_dict())

        elif a_type == "switch_provider":
            self.emit_trace("ACTION", "Executing LLM provider failover")
            curr_p = self.state.active_provider
            next_p = self.provider_manager.switch_to_next_available(curr_p, params.get("reason", "Provider failure"))
            if next_p:
                self.state.active_provider = next_p
                self.provider_manager.primary_provider = next_p
                # Mark failure as recovered
                if self.state.system_failures:
                    self.state.system_failures[-1].recovered = True
                    self.state.system_failures[-1].fallback_provider = next_p
                self.emit_trace("RECOVERY", f"Provider failover succeeded: Switched from {curr_p} to {next_p}")
            else:
                self._handle_unresolved_stopping("All LLM providers failed and were exhausted.")

        elif a_type == "finalize_run":
            self._finalize_success()

    # ---------------- Result Handlers ----------------

    def _handle_observe_context_result(self, result: AgentToolResult) -> None:
        self.state.record_tool_event("observe_context", {}, result.data, result.success, result.error, result.duration_seconds)
        if result.success:
            self.state.context_id = result.data.get("context_id")
            self.state.graph_id = result.data.get("graph_id")
            self.state.stakeholders = result.data.get("stakeholders", [])
            self.emit_trace(
                "OBSERVE",
                f"Context established with {result.data.get('stakeholder_count', 0)} stakeholders and {result.data.get('entity_count', 0)} entities",
                result.data
            )
        else:
            self._handle_tool_failure("observe_context", result)

    def _handle_stakeholders_result(self, result: AgentToolResult) -> None:
        self.state.record_tool_event("get_stakeholders", {}, result.data, result.success, result.error, result.duration_seconds)
        if result.success:
            self.state.stakeholders = result.data.get("stakeholders", [])
            self.emit_trace("OBSERVE", f"Identified {len(self.state.stakeholders)} digital twin stakeholder profiles", result.data)
        else:
            self._handle_tool_failure("get_stakeholders", result)

    def _handle_create_scenario_result(self, result: AgentToolResult) -> None:
        self.state.record_tool_event("create_scenario", {}, result.data, result.success, result.error, result.duration_seconds)
        if result.success:
            s_id = result.data.get("scenario_id")
            if s_id:
                self.state.candidate_scenarios[s_id] = result.data
                self.state.current_scenario_id = s_id
                self.emit_trace("OBSERVE", f"Created scenario '{result.data.get('name')}' ({s_id})", result.data)
        else:
            self._handle_tool_failure("create_scenario", result)

    def _handle_modify_scenario_result(self, result: AgentToolResult) -> None:
        self.state.record_tool_event("modify_scenario", {}, result.data, result.success, result.error, result.duration_seconds)
        if result.success:
            mod_data = result.data.get("modified_scenario", {})
            s_id = mod_data.get("scenario_id")
            if s_id:
                self.state.candidate_scenarios[s_id] = mod_data
                self.state.current_scenario_id = s_id
                self.emit_trace(
                    "REPLAN",
                    f"Policy adapted into '{mod_data.get('name')}' with modifications: {', '.join(result.data.get('modifications_applied', []))}",
                    result.data
                )
        else:
            self._handle_tool_failure("modify_scenario", result)

    def _execute_verification(self, scenario_id: str, scenario_name: str, impact_data: Dict[str, Any]) -> None:
        """Verify social impact metrics against constraints."""
        v_result: ConstraintResult = Verifier.evaluate(impact_data, self.state.constraints)
        
        # Record verification event in state
        record = self.state.record_verification(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            status=v_result.status.value,
            satisfied=v_result.satisfied_constraints,
            violated=v_result.violated_constraints,
            uncertain=v_result.uncertain_reasons,
            scores=v_result.impact_scores,
            overall_score=v_result.overall_score,
            confidence=v_result.confidence,
            summary=v_result.summary,
        )

        if v_result.status == VerificationState.VERIFIED:
            self.emit_trace(
                "VERIFIED",
                f"Scenario '{scenario_name}' satisfied ALL constraints! (Overall score: {v_result.overall_score:.2f})",
                v_result.to_dict()
            )
            self._finalize_success()

        elif v_result.status == VerificationState.FAILED:
            self.emit_trace(
                "FAILED",
                f"Verification FAILED for '{scenario_name}'. Violated constraints: {', '.join(v_result.violated_constraints)}",
                v_result.to_dict()
            )
            # Trigger Replanner analysis
            diagnostics = self.replanner.analyze_verification_failure(record, self.state)
            self.state.replanning_history.append(diagnostics)
            self.emit_trace("REPLAN", f"Replanner diagnosed root causes: {'; '.join(diagnostics['primary_deficits'])}", diagnostics)

        else: # UNCERTAIN
            self.emit_trace(
                "UNCERTAIN",
                f"Verification UNCERTAIN for '{scenario_name}': {'; '.join(v_result.uncertain_reasons)}",
                v_result.to_dict()
            )

    def _handle_tool_failure(self, tool_name: str, result: AgentToolResult) -> None:
        """Handle execution failures from tools."""
        analysis = FailureHandler.classify_exception(RuntimeError(result.error or "Tool failure"), context=tool_name)
        
        self.state.record_failure(
            failure_type=analysis.failure_type.value,
            error_message=analysis.error_message,
            provider=self.state.active_provider,
            recovered=False,
            recovery_strategy=analysis.suggested_recovery,
        )

        self.emit_trace(
            "FAILURE",
            f"Technical failure during '{tool_name}': {analysis.error_message} (Suggested recovery: {analysis.suggested_recovery})",
            {"analysis": analysis.__dict__}
        )

    def _finalize_success(self) -> None:
        """Finalize state upon successful constraint verification."""
        self.state.status = AgentStatus.VERIFIED
        self.state.final_outcome_summary = (
            f"Successfully verified candidate policy '{self.state.best_scenario_name}' "
            f"with overall social viability score of {self.state.best_score:.2f} satisfying all goal constraints."
        )
        self.emit_trace("FINALIZE", self.state.final_outcome_summary)

    def _handle_unresolved_stopping(self, reason: str) -> None:
        """Safely terminate without fabricating success when goal is unresolvable."""
        self.state.status = AgentStatus.UNRESOLVED
        
        all_violated = []
        for v in self.state.verification_history:
            for c in v.violated_constraints:
                if c not in all_violated:
                    all_violated.append(c)

        if all_violated:
            unresolved_info = f"Violated constraints across evaluations: {', '.join(all_violated)}"
        elif self.state.constraints:
            unresolved_info = f"Unsatisfied constraints: {', '.join(f'{k} {v}' for k, v in self.state.constraints.items())}"
        else:
            unresolved_info = "Constraints could not be fully verified."

        self.state.final_outcome_summary = (
            f"UNRESOLVED: {reason} Best observed candidate was '{self.state.best_scenario_name or 'None'}' "
            f"(Score: {self.state.best_score:.2f}). {unresolved_info}"
        )
        self.emit_trace("UNRESOLVED", self.state.final_outcome_summary)

    def run_until_completion(self, max_steps: Optional[int] = None) -> DecisionState:
        """Run the autonomous control loop until terminal state or maximum steps reached."""
        steps = 0
        limit = max_steps or (self.state.max_iterations * 3)

        while self.state.status not in (AgentStatus.VERIFIED, AgentStatus.UNRESOLVED, AgentStatus.FAILED, AgentStatus.STOPPED, AgentStatus.PAUSED_FOR_HUMAN):
            if steps >= limit:
                self._handle_unresolved_stopping("Execution step limit reached.")
                break
            steps += 1
            self.execute_next_step()

        return self.state

    def resume_after_human(self, approval: bool, text_input: Optional[str] = None) -> DecisionState:
        """Resume execution after receiving human response."""
        import uuid as _uuid
        option = "Approve" if approval else "Reject"
        resolved = self.human_manager.resolve_pending(option, text_input)
        self.state.human_interaction_pending = False
        self.state.human_interaction_request = None
        if resolved:
            self.state.human_interaction_history.append(resolved.to_dict())
        else:
            self.state.human_interaction_history.append({
                "interaction_id": f"human_{_uuid.uuid4().hex[:8]}",
                "interaction_type": "policy_directive",
                "selected_option": option,
                "status": "approved" if approval else "rejected",
                "text_input": text_input,
                "resolved_at": datetime.now().isoformat(),
            })

        if approval:
            self.emit_trace("HUMAN", f"Human approved policy adjustment. Resuming autonomous simulation.")
            self.state.status = AgentStatus.PLANNING
            # Execute step immediately
            self.execute_next_step()
        else:
            self.emit_trace("HUMAN", f"Human rejected policy adjustment. Replanning alternative.")
            self.state.status = AgentStatus.PLANNING

        return self.state
