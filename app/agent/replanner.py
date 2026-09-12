"""Replanner for Domain and Technical Failure Recovery.

Formulates targeted recovery strategies, generates candidate adaptations,
and computes recovery utilities when verification fails or disruptions occur.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..utils.logger import get_logger
from .state import ActionCandidate, DecisionState, VerificationRecord

logger = get_logger('mirofish.agent.replanner')


class Replanner:
    """Manages failure analysis, policy mutation strategies, and recovery replanning."""

    @classmethod
    def analyze_verification_failure(
        cls,
        verification: VerificationRecord,
        state: DecisionState,
    ) -> Dict[str, Any]:
        """Analyze root causes of a verification failure."""
        violated = verification.violated_constraints
        scores = verification.impact_scores
        
        diagnostics = {
            "scenario_id": verification.scenario_id,
            "scenario_name": verification.scenario_name,
            "violation_count": len(violated),
            "violated_constraints": violated,
            "primary_deficits": [],
            "suggested_policy_adjustments": [],
        }

        for v in violated:
            v_lower = v.lower()
            if "polarization" in v_lower:
                diagnostics["primary_deficits"].append("Severe ideological divergence between demographic groups.")
                diagnostics["suggested_policy_adjustments"].append(
                    "Introduce phased rollouts and joint municipal-business consultation committees."
                )
            elif "conflict" in v_lower:
                diagnostics["primary_deficits"].append("Intense disagreement and public opposition.")
                diagnostics["suggested_policy_adjustments"].append(
                    "Provide explicit hardship exemptions or financial offsets to contested groups."
                )
            elif "adoption" in v_lower or "acceptance" in v_lower:
                diagnostics["primary_deficits"].append("Low citizen uptake or resistance to intervention.")
                diagnostics["suggested_policy_adjustments"].append(
                    "Augment commuter subsidies, increase marketing clarity, or provide fare caps."
                )
            elif "equity" in v_lower:
                diagnostics["primary_deficits"].append("Unequal distribution of benefits across neighborhoods.")
                diagnostics["suggested_policy_adjustments"].append(
                    "Reallocate transit routes and frequencies toward underserved peripheral zones."
                )

        logger.info(f"Replanner diagnostics for {verification.scenario_id}: {len(diagnostics['primary_deficits'])} deficit(s) identified")
        return diagnostics

    @classmethod
    def generate_recovery_candidates(
        cls,
        state: DecisionState,
        diagnostics: Dict[str, Any],
    ) -> List[ActionCandidate]:
        """Generate ranked candidate recovery actions specifically addressing the diagnosed deficits."""
        candidates: List[ActionCandidate] = []
        base_scenario = state.candidate_scenarios.get(state.current_scenario_id or "", {})
        violated_list = [v.lower() for v in diagnostics.get("violated_constraints", [])]
        violation_count = len(violated_list)

        is_polarization = any("polarization" in v for v in violated_list)
        is_conflict = any("conflict" in v for v in violated_list)
        is_adoption = any("adoption" in v or "acceptance" in v for v in violated_list)
        
        uncertainty = state.uncertainty_metrics.get("overall_uncertainty", 0.15)
        recent_tools = [e.tool_name for e in state.tool_events[-3:]]

        # 1. Recovery Candidate: Targeted Policy Adaptation
        # If single deficit (e.g. adoption), direct modification has high recovery potential and low risk.
        # If multi-deficit (>=3 simultaneous), modifying without consensus has lower immediate recovery potential and higher risk.
        cr_mod = 0.95 if is_adoption and violation_count == 1 else (0.65 if violation_count >= 3 else 0.85)
        ei_mod = 0.90 if is_adoption else (0.60 if violation_count >= 3 else 0.75)
        fr_mod = round(min(0.50, 0.05 + 0.08 * max(1, violation_count)), 2)
        rp_mod = 0.40 if "modify_scenario" in recent_tools else 0.0

        candidates.append(ActionCandidate(
            action_type="modify_current_scenario",
            action_name=f"Synthesize Compromise Adaptation for '{base_scenario.get('name', 'Policy')}'",
            parameters={
                "base_scenario": base_scenario,
                "modification_goal": "; ".join(diagnostics.get("suggested_policy_adjustments", [])),
                "violated_constraints": diagnostics.get("violated_constraints", []),
            },
            rationale="Grounded modification to directly eliminate constraint deficits through stakeholder compromise.",
            goal_progress=0.80 if violation_count <= 2 else 0.65,
            constraint_recovery_potential=cr_mod,
            information_gain=0.55,
            expected_improvement=ei_mod,
            action_cost=0.20,
            failure_risk=fr_mod,
            repetition_penalty=rp_mod,
        ))

        # 2. Recovery Candidate: Inspect Specific Stakeholder Sub-Graphs
        # High value when polarization is present or uncertainty is elevated.
        cr_stk = 0.85 if is_polarization else (0.75 if violation_count >= 3 else 0.55)
        ig_stk = 0.90 if is_polarization or uncertainty > 0.3 else 0.70
        ei_stk = 0.70 if is_polarization else 0.50
        rp_stk = 0.40 if "get_stakeholders" in recent_tools else 0.0

        candidates.append(ActionCandidate(
            action_type="inspect_stakeholder_evidence",
            action_name="Audit Opposing Stakeholder Sub-Graphs",
            parameters={"graph_id": state.graph_id},
            rationale="Isolate opposing stakeholder concerns before committing further modifications.",
            goal_progress=0.50,
            constraint_recovery_potential=cr_stk,
            information_gain=ig_stk,
            expected_improvement=ei_stk,
            action_cost=0.12,
            failure_risk=0.05,
            repetition_penalty=rp_stk,
        ))

        # 3. Recovery Candidate: Human-in-the-loop consultation
        # Value scales with conflict intensity and multiple simultaneous deficits.
        cr_hum = 0.90 if (violation_count >= 3 or is_conflict) else 0.70
        ei_hum = 0.88 if (violation_count >= 3 or is_conflict) else 0.60
        rp_hum = 0.80 if state.human_interaction_pending else 0.0

        candidates.append(ActionCandidate(
            action_type="request_human_confirmation",
            action_name="Request Human Stakeholder Guidance on Tradeoffs",
            parameters={"prompt": f"Guidance requested on resolving: {', '.join(diagnostics.get('violated_constraints', []))}"},
            rationale="Escalate contested social tradeoff to human policymaker.",
            goal_progress=0.65,
            constraint_recovery_potential=cr_hum,
            information_gain=0.80,
            expected_improvement=ei_hum,
            action_cost=0.10,
            failure_risk=0.05,
            repetition_penalty=rp_hum,
        ))

        # 4. Calculate scores using standard utility formula
        from .planner import ActionUtilityPlanner
        for c in candidates:
            ActionUtilityPlanner.compute_action_utility(c)

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates
