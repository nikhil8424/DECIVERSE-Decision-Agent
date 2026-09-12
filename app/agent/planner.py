"""Action Utility Planner for Autonomous Decision Agent.

Implements transparent, multi-action generation, utility-based scoring and ranking,
and concise auditable decision summaries without leaking private chain-of-thought tokens.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logger import get_logger
from .state import ActionCandidate, DecisionRecord, DecisionState

logger = get_logger('mirofish.agent.planner')


class ActionUtilityPlanner:
    """Utility-driven autonomous planner that generates, scores, and ranks candidate actions."""

    # Weights for utility components
    WEIGHTS = {
        "goal_progress": 0.25,
        "constraint_recovery": 0.30,
        "information_gain": 0.15,
        "expected_improvement": 0.20,
        "action_cost": 0.10,
        "failure_risk": 0.10,
        "repetition_penalty": 0.15,
    }

    @classmethod
    def compute_action_utility(cls, candidate: ActionCandidate) -> float:
        """Calculate total Action Utility for a candidate."""
        raw_utility = (
            candidate.goal_progress * cls.WEIGHTS["goal_progress"]
            + candidate.constraint_recovery_potential * cls.WEIGHTS["constraint_recovery"]
            + candidate.information_gain * cls.WEIGHTS["information_gain"]
            + candidate.expected_improvement * cls.WEIGHTS["expected_improvement"]
            - candidate.action_cost * cls.WEIGHTS["action_cost"]
            - candidate.failure_risk * cls.WEIGHTS["failure_risk"]
            - candidate.repetition_penalty * cls.WEIGHTS["repetition_penalty"]
        )
        candidate.score = round(max(0.01, min(1.0, raw_utility)), 4)
        return candidate.score

    @classmethod
    def generate_candidate_actions(cls, state: DecisionState) -> List[ActionCandidate]:
        """Generate a pool of multiple feasible candidate actions given the current state."""
        candidates: List[ActionCandidate] = []
        recent_actions = [t.tool_name for t in state.tool_events[-3:]]
        
        # Determine current state condition
        has_context = bool(state.context_id)
        has_stakeholders = bool(state.stakeholders)
        has_scenarios = bool(state.candidate_scenarios)
        has_verification = bool(state.verification_history)
        last_verification = state.verification_history[-1] if has_verification else None
        last_status = last_verification.status if last_verification else None
        has_failures = bool(state.system_failures)
        unrecovered_failures = [f for f in state.system_failures if not f.recovered]

        # 1. Action: Observe Context / Gather Evidence
        if not has_context:
            candidates.append(ActionCandidate(
                action_type="observe_context",
                action_name="Extract Community Context & Stakeholders",
                parameters={"problem_statement": state.problem_statement or state.goal, "source_files": state.source_files},
                rationale="Required first step to ground decision space and extract community entities.",
                goal_progress=0.85,
                constraint_recovery_potential=0.10,
                information_gain=0.95,
                expected_improvement=0.50,
                action_cost=0.10,
                failure_risk=0.05,
                repetition_penalty=0.80 if "observe_context" in recent_actions else 0.0,
            ))
            candidates.append(ActionCandidate(
                action_type="create_baseline_scenario",
                action_name="Formulate Hypothetical Baseline Policy",
                parameters={
                    "name": "Hypothetical Baseline Policy",
                    "description": f"Initial policy draft for: {state.goal}",
                    "intervention": state.problem_statement or state.goal,
                    "platform": "reddit",
                    "max_rounds": 5,
                    "agent_count": 20,
                    "context_id": None,
                },
                rationale="Generate baseline scenario directly from input problem statement.",
                goal_progress=0.60,
                constraint_recovery_potential=0.15,
                information_gain=0.40,
                expected_improvement=0.40,
                action_cost=0.15,
                failure_risk=0.15,
                repetition_penalty=0.0,
            ))
            candidates.append(ActionCandidate(
                action_type="request_human_confirmation",
                action_name="Request Initial Goal & Constraints Clarification",
                parameters={"prompt": "Confirm initial constraints before beginning context observation"},
                rationale="Confirm policy goals with human stakeholder before initiating multi-agent run.",
                goal_progress=0.30,
                constraint_recovery_potential=0.10,
                information_gain=0.60,
                expected_improvement=0.30,
                action_cost=0.05,
                failure_risk=0.05,
                repetition_penalty=0.0,
            ))
        else:
            candidates.append(ActionCandidate(
                action_type="inspect_stakeholder_evidence",
                action_name="Inspect Detailed Stakeholder Dynamics",
                parameters={"graph_id": state.graph_id},
                rationale="Examine specific stakeholder group interests to inform compromise.",
                goal_progress=0.40,
                constraint_recovery_potential=0.55,
                information_gain=0.75,
                expected_improvement=0.45,
                action_cost=0.15,
                failure_risk=0.05,
                repetition_penalty=0.40 if "get_stakeholders" in recent_actions else 0.0,
            ))

        # 2. Action: Create Initial Scenarios (if context exists and no scenarios yet)
        if has_context and not has_scenarios:
            candidates.append(ActionCandidate(
                action_type="create_baseline_scenario",
                action_name="Formulate Baseline Policy Scenario",
                parameters={
                    "name": "Baseline Policy",
                    "description": f"Initial policy intervention for: {state.goal}",
                    "intervention": state.problem_statement or state.goal,
                    "platform": "reddit",
                    "max_rounds": 5,
                    "agent_count": 20,
                    "context_id": state.context_id,
                },
                rationale="Construct the baseline intervention model for multi-agent simulation.",
                goal_progress=0.80,
                constraint_recovery_potential=0.20,
                information_gain=0.70,
                expected_improvement=0.60,
                action_cost=0.15,
                failure_risk=0.05,
                repetition_penalty=0.70 if "create_scenario" in recent_actions else 0.0,
            ))

        # Check human feedback and rejection history in state
        latest_human_entry = state.human_interaction_history[-1] if state.human_interaction_history else None
        latest_feedback = ""
        latest_was_rejection = False
        if latest_human_entry:
            latest_feedback = latest_human_entry.get("text_input") or ""
            latest_was_rejection = (
                latest_human_entry.get("selected_option") == "Reject"
                or latest_human_entry.get("status") == "rejected"
            )

        # 3. Action: Run Simulation on Active Candidate Scenario
        unsimulated_scenarios = [
            s_id for s_id, s_data in state.candidate_scenarios.items()
            if s_id not in state.social_impact_results
        ]
        if unsimulated_scenarios:
            target_id = unsimulated_scenarios[0]
            target_data = state.candidate_scenarios[target_id]
            # If this scenario was specifically rejected by human and is unadapted, apply penalty
            is_unadapted_rejected = (
                latest_was_rejection
                and "adapted" not in target_data.get("name", "").lower()
                and not target_data.get("metadata", {}).get("parent_scenario_id")
            )
            sim_rep_penalty = 0.90 if is_unadapted_rejected else 0.0
            candidates.append(ActionCandidate(
                action_type="run_simulation_and_eval",
                action_name=f"Simulate & Evaluate '{target_data.get('name', target_id)}'",
                parameters={"scenario_id": target_id, "scenario_data": target_data, "rounds": target_data.get("max_rounds", 5)},
                rationale="Execute multi-agent simulation and measure emergent social impact metrics.",
                goal_progress=0.90,
                constraint_recovery_potential=0.85,
                information_gain=0.90,
                expected_improvement=0.85,
                action_cost=0.15,
                failure_risk=0.05,
                repetition_penalty=sim_rep_penalty,
            ))

        # 4. Action: Modify Scenario / Dynamic Replanning (if failed constraints or human rejection on evaluated scenario)
        if (
            (last_status in ("FAILED", "UNCERTAIN") or latest_was_rejection)
            and state.current_scenario_id
            and state.current_scenario_id in state.social_impact_results
        ):
            curr_data = state.candidate_scenarios.get(state.current_scenario_id, {})
            violated = last_verification.violated_constraints if last_verification else []
            
            if latest_was_rejection and latest_feedback:
                mod_goal = f"Incorporate human guidance: '{latest_feedback}'. Address deficits: {', '.join(violated) if violated else 'stakeholder compromise'}"
                act_name = f"Adapt Policy with Human Directive ('{latest_feedback[:32]}...')"
                ei_val = 0.95
                gp_val = 0.90
            else:
                mod_goal = f"Resolve violations: {', '.join(violated)}"
                act_name = f"Adapt Policy to Fix Violated Constraints ({len(violated)} failed)"
                ei_val = 0.85
                gp_val = 0.80

            candidates.append(ActionCandidate(
                action_type="modify_current_scenario",
                action_name=act_name,
                parameters={
                    "base_scenario": curr_data,
                    "modification_goal": mod_goal,
                    "violated_constraints": violated,
                    "stakeholder_feedback": latest_feedback if latest_was_rejection else None,
                },
                rationale="Generate an adapted policy specifically targeted at reducing violated constraints and incorporating human guidance.",
                goal_progress=gp_val,
                constraint_recovery_potential=0.95,
                information_gain=0.65,
                expected_improvement=ei_val,
                action_cost=0.15,
                failure_risk=0.05,
                repetition_penalty=0.30 if "modify_scenario" in recent_actions else 0.0,
            ))

        # 5. Action: Multi-Run Simulation for Uncertainty Reduction
        if state.current_scenario_id and state.current_scenario_id in state.social_impact_results:
            curr_data = state.candidate_scenarios.get(state.current_scenario_id, {})
            unc_level = state.uncertainty_metrics.get("overall_uncertainty", 0.20)
            unc_rep = 0.85 if latest_was_rejection else (0.60 if "request_additional_simulation" in recent_actions else 0.0)
            candidates.append(ActionCandidate(
                action_type="request_additional_simulation",
                action_name="Sample Repeated Runs for Uncertainty Quantification",
                parameters={"scenario_id": state.current_scenario_id, "scenario_name": curr_data.get("name", ""), "scenario_data": curr_data, "sample_runs": 3},
                rationale="Gather repeated stochastic samples to estimate confidence bounds and outcome variance.",
                goal_progress=0.50,
                constraint_recovery_potential=0.30,
                information_gain=0.85 + (0.10 if unc_level > 0.3 else 0.0),
                expected_improvement=0.40,
                action_cost=0.35,
                failure_risk=0.05,
                repetition_penalty=unc_rep,
            ))

        # 6. Action: Compare Candidate Scenarios
        if len(state.social_impact_results) >= 2:
            candidates.append(ActionCandidate(
                action_type="compare_candidates",
                action_name=f"Synthesize Tradeoffs across {len(state.social_impact_results)} Evaluated Scenarios",
                parameters={
                    "problem_id": state.run_id,
                    "problem_statement": state.goal,
                    "impact_results": list(state.social_impact_results.values()),
                },
                rationale="Compare multi-scenario metrics and derive comprehensive decision intelligence evidence.",
                goal_progress=0.80,
                constraint_recovery_potential=0.50,
                information_gain=0.80,
                expected_improvement=0.70,
                action_cost=0.15,
                failure_risk=0.05,
                repetition_penalty=0.50 if "compare_scenarios" in recent_actions else 0.0,
            ))

        # 7. Action: Request Human Confirmation (if large policy mutation or uncertain)
        if last_status == "FAILED" or len(state.candidate_scenarios) >= 2:
            # If human interaction is already pending or human just rejected/provided guidance, heavily penalize asking immediately again
            human_rep_penalty = 0.85 if (state.human_interaction_pending or latest_was_rejection) else 0.0
            candidates.append(ActionCandidate(
                action_type="request_human_confirmation",
                action_name="Request Human Stakeholder Confirmation",
                parameters={"prompt": "Confirm policy adjustment before committing next simulation"},
                rationale="Interactive check with human policymaker on proposed trade-offs.",
                goal_progress=0.60,
                constraint_recovery_potential=0.60,
                information_gain=0.70,
                expected_improvement=0.60,
                action_cost=0.10,
                failure_risk=0.05,
                repetition_penalty=human_rep_penalty,
            ))

        # 8. Action: Recover from Technical Failure / Provider Switch
        if unrecovered_failures:
            candidates.append(ActionCandidate(
                action_type="switch_provider",
                action_name="Switch to Fallback LLM Provider",
                parameters={"reason": unrecovered_failures[-1].error_message},
                rationale="Primary provider encountered failure; fallback provider required.",
                goal_progress=0.85,
                constraint_recovery_potential=0.90,
                information_gain=0.60,
                expected_improvement=0.90,
                action_cost=0.05,
                failure_risk=0.05,
                repetition_penalty=0.0,
            ))

        # Fallback candidate if pool is empty
        if not candidates:
            candidates.append(ActionCandidate(
                action_type="finalize_run",
                action_name="Finalize Autonomous Decision Run",
                parameters={},
                rationale="All viable actions explored.",
                goal_progress=1.0,
                action_cost=0.0,
            ))

        # Score and rank all candidate actions
        for candidate in candidates:
            cls.compute_action_utility(candidate)

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    @classmethod
    def plan_next_step(cls, state: DecisionState) -> Tuple[ActionCandidate, List[ActionCandidate], str]:
        """Generate, rank candidate actions, and select the highest utility action.
        
        Returns:
            Tuple of (selected_action, ranked_candidate_list, decision_summary)
        """
        candidates = cls.generate_candidate_actions(state)
        selected = candidates[0]

        # Formulate clean, concise decision summary (no think tags)
        reason_parts = [
            f"Selected action '{selected.action_name}' with highest utility score of {selected.score:.2f}."
        ]
        
        if selected.constraint_recovery_potential >= 0.7:
            reason_parts.append("Has highest potential to recover from violated constraints.")
        elif selected.information_gain >= 0.8:
            reason_parts.append("Provides highest information gain to reduce decision uncertainty.")
        elif selected.goal_progress >= 0.8:
            reason_parts.append("Directly advances core goal completion.")

        if len(candidates) > 1:
            runner_up = candidates[1]
            reason_parts.append(
                f"Preferred over '{runner_up.action_name}' (score: {runner_up.score:.2f}) due to better expected improvement."
            )

        decision_summary = " ".join(reason_parts)
        # Sanitize any accidental think tags
        decision_summary = re.sub(r'<think>[\s\S]*?</think>', '', decision_summary).strip()

        logger.info(f"Planner decision: {selected.action_type} (Score: {selected.score:.2f})")
        return selected, candidates, decision_summary
