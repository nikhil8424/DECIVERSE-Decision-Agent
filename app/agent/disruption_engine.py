"""Controlled Disruption and Failure Injection Framework.

Enables injecting both Domain Disruptions (stakeholder opposition, budget cuts, polarization)
and Technical Disruptions (provider timeouts, malformed responses, tool exceptions)
to test and demonstrate autonomous agent resilience and recovery.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger

logger = get_logger('mirofish.agent.disruption_engine')


class DomainDisruptionType(str, Enum):
    """Domain disruptions that affect simulated social dynamics and policy feasibility."""
    INCREASE_STAKEHOLDER_OPPOSITION = "increase_stakeholder_opposition"
    REDUCE_BUDGET = "reduce_budget"
    INCREASE_POLARIZATION = "increase_polarization"
    MAKE_SCENARIO_UNAVAILABLE = "make_scenario_unavailable"
    ADD_CONFLICTING_STAKEHOLDER = "add_conflicting_stakeholder"


class TechnicalDisruptionType(str, Enum):
    """Technical/system disruptions that test error handling, repair, and failover."""
    FORCE_LLM_TIMEOUT = "force_llm_timeout"
    FORCE_PROVIDER_UNAVAILABLE = "force_provider_unavailable"
    RETURN_MALFORMED_TOOL_RESPONSE = "return_malformed_tool_response"
    FORCE_OASIS_MALFORMED_OUTPUT = "force_oasis_malformed_output"
    RAISE_TOOL_EXCEPTION = "raise_tool_exception"


@dataclass
class ActiveDisruption:
    """An actively applied disruption rule."""
    disruption_id: str
    category: str  # domain or technical
    disruption_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    applied_count: int = 0
    max_occurrences: int = 1
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class DisruptionEngine:
    """Coordinates and applies runtime disruptions."""

    def __init__(self):
        self.active_disruptions: Dict[str, ActiveDisruption] = {}
        self.disruption_history: List[Dict[str, Any]] = []

    def inject_domain_disruption(
        self,
        disruption_type: DomainDisruptionType,
        parameters: Optional[Dict[str, Any]] = None,
        max_occurrences: int = 1,
    ) -> str:
        """Inject a domain-level disruption."""
        disruption_id = f"dom_dis_{datetime.now().strftime('%H%M%S')}"
        disruption = ActiveDisruption(
            disruption_id=disruption_id,
            category="domain",
            disruption_type=disruption_type.value,
            parameters=parameters or {},
            max_occurrences=max_occurrences,
        )
        self.active_disruptions[disruption_id] = disruption
        self.disruption_history.append({
            "disruption_id": disruption_id,
            "category": "domain",
            "type": disruption_type.value,
            "parameters": parameters or {},
            "injected_at": datetime.now().isoformat(),
        })
        logger.warning(f"Injected Domain Disruption: {disruption_type.value} ({parameters})")
        return disruption_id

    def inject_technical_disruption(
        self,
        disruption_type: TechnicalDisruptionType,
        parameters: Optional[Dict[str, Any]] = None,
        max_occurrences: int = 1,
    ) -> str:
        """Inject a technical/system-level disruption."""
        disruption_id = f"tech_dis_{datetime.now().strftime('%H%M%S')}"
        disruption = ActiveDisruption(
            disruption_id=disruption_id,
            category="technical",
            disruption_type=disruption_type.value,
            parameters=parameters or {},
            max_occurrences=max_occurrences,
        )
        self.active_disruptions[disruption_id] = disruption
        self.disruption_history.append({
            "disruption_id": disruption_id,
            "category": "technical",
            "type": disruption_type.value,
            "parameters": parameters or {},
            "injected_at": datetime.now().isoformat(),
        })
        logger.warning(f"Injected Technical Disruption: {disruption_type.value} ({parameters})")
        return disruption_id

    def clear_all(self) -> None:
        """Clear all active disruptions."""
        self.active_disruptions.clear()
        logger.info("Cleared all active disruptions")

    def check_and_apply_technical_disruption(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Check if an active technical disruption applies to the given tool.
        
        Returns:
            Dict indicating disruption action or raises exception if configured.
        """
        for d_id, disruption in list(self.active_disruptions.items()):
            if disruption.category != "technical":
                continue
            
            target_tool = disruption.parameters.get("tool_name")
            if target_tool and target_tool != tool_name:
                continue

            # Increment count
            disruption.applied_count += 1
            if disruption.applied_count >= disruption.max_occurrences:
                del self.active_disruptions[d_id]

            d_type = disruption.disruption_type
            if d_type == TechnicalDisruptionType.FORCE_LLM_TIMEOUT.value:
                raise TimeoutError(f"Injected LLM Timeout during tool '{tool_name}'")
            elif d_type == TechnicalDisruptionType.FORCE_PROVIDER_UNAVAILABLE.value:
                raise ConnectionRefusedError(f"Injected Provider Unavailable error during tool '{tool_name}'")
            elif d_type == TechnicalDisruptionType.RAISE_TOOL_EXCEPTION.value:
                msg = disruption.parameters.get("message", f"Controlled injected exception in {tool_name}")
                raise RuntimeError(msg)
            elif d_type == TechnicalDisruptionType.RETURN_MALFORMED_TOOL_RESPONSE.value:
                return {
                    "corrupted_raw_text": "{ acceptance: 0.72, polarization: 'ERROR_UNPARSED', invalid_trailing_comma, ",
                    "is_malformed": True,
                }
            elif d_type == TechnicalDisruptionType.FORCE_OASIS_MALFORMED_OUTPUT.value:
                return {
                    "oasis_status": "corrupted",
                    "actions": [],
                    "raw_output": "FATAL: OASIS simulation terminated unexpectedly with SIGSEGV",
                    "is_oasis_error": True,
                }

        return None

    def apply_domain_disruptions_to_metrics(
        self,
        scenario_id: str,
        metrics_dict: Dict[str, float],
    ) -> Dict[str, float]:
        """Apply active domain disruptions to calculated behaviour / impact metrics."""
        adjusted = dict(metrics_dict)
        
        for d_id, disruption in list(self.active_disruptions.items()):
            if disruption.category != "domain":
                continue
            
            target_scenario = disruption.parameters.get("scenario_id")
            if target_scenario and target_scenario != scenario_id:
                continue

            disruption.applied_count += 1
            if disruption.applied_count >= disruption.max_occurrences:
                del self.active_disruptions[d_id]

            d_type = disruption.disruption_type
            if d_type == DomainDisruptionType.INCREASE_POLARIZATION.value:
                boost = disruption.parameters.get("polarization_boost", 0.35)
                adjusted["polarization_score"] = min(1.0, adjusted.get("polarization_score", 0.0) + boost)
                adjusted["consensus_score"] = max(0.0, adjusted.get("consensus_score", 0.0) - boost * 0.5)
                logger.info(f"Domain disruption applied: polarization boosted to {adjusted['polarization_score']:.2f}")

            elif d_type == DomainDisruptionType.INCREASE_STAKEHOLDER_OPPOSITION.value:
                opp_penalty = disruption.parameters.get("opposition_penalty", 0.30)
                adjusted["acceptance_score"] = max(0.0, adjusted.get("acceptance_score", 0.0) - opp_penalty)
                adjusted["conflict_score"] = min(1.0, adjusted.get("conflict_score", 0.0) + opp_penalty)
                logger.info(f"Domain disruption applied: acceptance decreased to {adjusted['acceptance_score']:.2f}")

            elif d_type == DomainDisruptionType.REDUCE_BUDGET.value:
                budget_penalty = disruption.parameters.get("budget_penalty", 0.25)
                adjusted["adoption_score"] = max(0.0, adjusted.get("adoption_score", 0.0) - budget_penalty)
                adjusted["stability_score"] = max(0.0, adjusted.get("stability_score", 0.0) - budget_penalty * 0.5)
                logger.info(f"Domain disruption applied: budget cut reduced adoption to {adjusted['adoption_score']:.2f}")

        return adjusted
