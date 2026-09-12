"""Human-in-the-Loop Interaction Manager.

Supports pausing autonomous runs to request policy confirmation, additional
context documents, or technical failure resolution, directly updating state.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger

logger = get_logger('mirofish.agent.human_interaction')


class HumanInteractionType(str, Enum):
    """Types of human intervention checkpoints."""
    POLICY_CONFIRMATION = "policy_confirmation"
    ADDITIONAL_CONTEXT_REQUEST = "additional_context_request"
    TECHNICAL_ESCALATION = "technical_escalation"
    CONSTRAINT_RELAXATION = "constraint_relaxation"


@dataclass
class HumanInteractionRequest:
    """A structured request for human input."""
    request_id: str
    interaction_type: HumanInteractionType
    title: str
    prompt: str
    context_data: Dict[str, Any] = field(default_factory=dict)
    options: List[str] = field(default_factory=lambda: ["Approve", "Reject"])
    requires_text_input: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Human response
    resolved: bool = False
    selected_option: Optional[str] = None
    text_input: Optional[str] = None
    resolved_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["interaction_type"] = self.interaction_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HumanInteractionRequest":
        itype = HumanInteractionType(data.get("interaction_type", "policy_confirmation"))
        return cls(
            request_id=data["request_id"],
            interaction_type=itype,
            title=data.get("title", ""),
            prompt=data.get("prompt", ""),
            context_data=data.get("context_data", {}),
            options=data.get("options", ["Approve", "Reject"]),
            requires_text_input=data.get("requires_text_input", False),
            created_at=data.get("created_at", datetime.now().isoformat()),
            resolved=data.get("resolved", False),
            selected_option=data.get("selected_option"),
            text_input=data.get("text_input"),
            resolved_at=data.get("resolved_at"),
        )


class HumanInteractionManager:
    """Manages human checkpoints and pending state."""

    def __init__(self):
        self.pending_request: Optional[HumanInteractionRequest] = None
        self.interaction_history: List[HumanInteractionRequest] = []

    def create_policy_confirmation_request(
        self,
        current_scenario_name: str,
        modified_scenario_name: str,
        violated_constraints: List[str],
        proposed_modifications: List[str],
    ) -> HumanInteractionRequest:
        """Create a human confirmation request before simulating a modified policy."""
        req = HumanInteractionRequest(
            request_id=f"hitl_{uuid.uuid4().hex[:8]}",
            interaction_type=HumanInteractionType.POLICY_CONFIRMATION,
            title="Confirm Policy Modification",
            prompt=(
                f"Scenario '{current_scenario_name}' violated constraints ({', '.join(violated_constraints)}). "
                f"The agent generated modified scenario '{modified_scenario_name}'. "
                f"Do you approve proceeding with simulation of this modified policy?"
            ),
            context_data={
                "current_scenario_name": current_scenario_name,
                "modified_scenario_name": modified_scenario_name,
                "violated_constraints": violated_constraints,
                "proposed_modifications": proposed_modifications,
            },
            options=["Approve & Simulate", "Reject Modification"],
        )
        self.pending_request = req
        self.interaction_history.append(req)
        logger.info(f"Created human confirmation request: {req.request_id}")
        return req

    def create_missing_context_request(
        self,
        uncertain_aspect: str,
        suggested_information: str,
    ) -> HumanInteractionRequest:
        """Request additional evidence or context document from the human."""
        req = HumanInteractionRequest(
            request_id=f"hitl_{uuid.uuid4().hex[:8]}",
            interaction_type=HumanInteractionType.ADDITIONAL_CONTEXT_REQUEST,
            title="Additional Context Required",
            prompt=(
                f"The simulation analysis is uncertain regarding: {uncertain_aspect}. "
                f"Please provide additional background context or policy details to clarify."
            ),
            context_data={
                "uncertain_aspect": uncertain_aspect,
                "suggested_information": suggested_information,
            },
            options=["Submit Context", "Proceed Without Context"],
            requires_text_input=True,
        )
        self.pending_request = req
        self.interaction_history.append(req)
        logger.info(f"Created human missing context request: {req.request_id}")
        return req

    def create_technical_escalation_request(
        self,
        failure_summary: str,
        failed_providers: List[str],
    ) -> HumanInteractionRequest:
        """Escalate to human when all automated technical recovery mechanisms have failed."""
        req = HumanInteractionRequest(
            request_id=f"hitl_{uuid.uuid4().hex[:8]}",
            interaction_type=HumanInteractionType.TECHNICAL_ESCALATION,
            title="Technical Provider Escalation",
            prompt=(
                f"All configured providers failed ({', '.join(failed_providers)}). "
                f"Failure reason: {failure_summary}. How would you like to proceed?"
            ),
            context_data={
                "failure_summary": failure_summary,
                "failed_providers": failed_providers,
            },
            options=["Retry Primary Provider", "Use Cached/Fallback Data", "Stop Run"],
        )
        self.pending_request = req
        self.interaction_history.append(req)
        logger.warning(f"Created human technical escalation request: {req.request_id}")
        return req

    def resolve_pending(
        self,
        selected_option: str,
        text_input: Optional[str] = None,
    ) -> Optional[HumanInteractionRequest]:
        """Resolve currently pending human request."""
        if not self.pending_request:
            logger.warning("No pending human interaction request to resolve")
            return None

        self.pending_request.resolved = True
        self.pending_request.selected_option = selected_option
        self.pending_request.text_input = text_input
        self.pending_request.resolved_at = datetime.now().isoformat()
        
        resolved = self.pending_request
        self.pending_request = None
        logger.info(f"Resolved human interaction request: {resolved.request_id} -> {selected_option}")
        return resolved
