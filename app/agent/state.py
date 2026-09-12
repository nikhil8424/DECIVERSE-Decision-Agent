"""Persistent State Management for Autonomous Decision Agent.

Contains DecisionState and associated dataclasses for recording the entire
auditable execution lifecycle, observations, action candidates, utility scores,
decisions, tool events, failures, disruptions, and verification history.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentStatus(str, Enum):
    """Execution status of the autonomous agent."""
    IDLE = "idle"
    OBSERVING = "observing"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    VERIFYING = "verifying"
    REPLANNING = "replanning"
    PAUSED_FOR_HUMAN = "paused_for_human"
    VERIFIED = "verified"
    UNRESOLVED = "unresolved"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class ActionCandidate:
    """A candidate action evaluated by the planner."""
    action_type: str
    action_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    
    # Utility score breakdown (0.0 - 1.0 components)
    goal_progress: float = 0.0
    constraint_recovery_potential: float = 0.0
    information_gain: float = 0.0
    expected_improvement: float = 0.0
    action_cost: float = 0.0
    failure_risk: float = 0.0
    repetition_penalty: float = 0.0
    
    # Computed aggregate utility
    score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionCandidate":
        return cls(**data)


@dataclass
class DecisionRecord:
    """Auditable planner decision event."""
    decision_index: int
    current_issue: str
    candidate_actions: List[Dict[str, Any]] = field(default_factory=list)
    selected_action: Dict[str, Any] = field(default_factory=dict)
    decision_summary: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionRecord":
        return cls(**data)


@dataclass
class ActionRecord:
    """Auditable tool execution event."""
    action_id: str
    tool_name: str
    tool_args: Dict[str, Any] = field(default_factory=dict)
    tool_result: Optional[Dict[str, Any]] = None
    success: bool = True
    error: Optional[str] = None
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionRecord":
        return cls(**data)


@dataclass
class VerificationRecord:
    """Auditable verification outcome against objectives and constraints."""
    verification_index: int
    scenario_id: str
    scenario_name: str
    status: str  # VERIFIED, FAILED, UNCERTAIN
    satisfied_constraints: List[str] = field(default_factory=list)
    violated_constraints: List[str] = field(default_factory=list)
    uncertain_aspects: List[str] = field(default_factory=list)
    impact_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    confidence: float = 0.0
    summary: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationRecord":
        return cls(**data)


@dataclass
class DisruptionEvent:
    """Domain or contextual disruption injected or observed."""
    disruption_id: str
    disruption_type: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DisruptionEvent":
        return cls(**data)


@dataclass
class FailureEvent:
    """Technical or system failure event with recovery details."""
    failure_id: str
    failure_type: str  # LLM_TIMEOUT, PROVIDER_UNAVAILABLE, MALFORMED_RESPONSE, TOOL_EXCEPTION, OASIS_ERROR
    error_message: str
    provider: Optional[str] = None
    recovered: bool = False
    recovery_strategy: Optional[str] = None
    fallback_provider: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureEvent":
        return cls(**data)


@dataclass
class DecisionState:
    """Complete persistent state for an autonomous run."""
    run_id: str = field(default_factory=lambda: f"agent_run_{uuid.uuid4().hex[:12]}")
    goal: str = ""
    problem_statement: str = ""
    source_files: List[str] = field(default_factory=list)
    objectives: List[str] = field(default_factory=lambda: [
        "acceptance", "adoption", "consensus", "equity", "stability"
    ])
    # Constraints map constraint name -> operator + threshold, e.g. {"polarization": "<0.35", "adoption": ">0.60"}
    constraints: Dict[str, str] = field(default_factory=lambda: {
        "polarization": "<0.35",
        "conflict": "<0.30",
        "adoption": ">0.60",
    })
    
    # Execution status
    status: AgentStatus = AgentStatus.IDLE
    iteration_count: int = 0
    max_iterations: int = 5
    current_stage: str = "init"
    current_action: str = ""
    
    # Domain entities & Context
    context_id: Optional[str] = None
    graph_id: Optional[str] = None
    project_id: Optional[str] = None
    stakeholders: List[Dict[str, Any]] = field(default_factory=list)
    candidate_scenarios: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    current_scenario_id: Optional[str] = None
    
    # Audit History
    observations: List[str] = field(default_factory=list)
    decision_history: List[DecisionRecord] = field(default_factory=list)
    tool_events: List[ActionRecord] = field(default_factory=list)
    simulation_results: Dict[str, Any] = field(default_factory=dict)
    social_impact_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    verification_history: List[VerificationRecord] = field(default_factory=list)
    replanning_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Failures & Disruptions
    domain_disruptions: List[DisruptionEvent] = field(default_factory=list)
    system_failures: List[FailureEvent] = field(default_factory=list)
    
    # Providers
    primary_provider: str = "ollama"
    active_provider: str = "ollama"
    provider_fallback_chain: List[str] = field(default_factory=lambda: ["ollama", "claude-cli", "codex-cli"])
    provider_status_map: Dict[str, str] = field(default_factory=dict)
    
    # Best outcome tracking
    best_scenario_id: Optional[str] = None
    best_scenario_name: Optional[str] = None
    best_score: float = -1.0
    best_impact_result: Optional[Dict[str, Any]] = None
    final_outcome_summary: str = ""
    
    # Human in the loop
    human_interaction_pending: bool = False
    human_interaction_request: Optional[Dict[str, Any]] = None
    human_interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Multi-run uncertainty
    uncertainty_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def record_observation(self, observation: str) -> None:
        """Add a timestamped observation to the state."""
        self.observations.append(f"[{datetime.now().strftime('%H:%M:%S')}] {observation}")
        self.updated_at = datetime.now().isoformat()

    def record_decision(
        self,
        current_issue: str,
        candidate_actions: List[ActionCandidate],
        selected_action: ActionCandidate,
        decision_summary: str,
    ) -> DecisionRecord:
        """Record a planner decision event."""
        record = DecisionRecord(
            decision_index=len(self.decision_history) + 1,
            current_issue=current_issue,
            candidate_actions=[c.to_dict() for c in candidate_actions],
            selected_action=selected_action.to_dict(),
            decision_summary=decision_summary,
            timestamp=datetime.now().isoformat(),
        )
        self.decision_history.append(record)
        self.updated_at = datetime.now().isoformat()
        return record

    def record_tool_event(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None,
        duration_seconds: float = 0.0,
    ) -> ActionRecord:
        """Record an executed tool action."""
        record = ActionRecord(
            action_id=f"act_{uuid.uuid4().hex[:8]}",
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            success=success,
            error=error,
            duration_seconds=duration_seconds,
            timestamp=datetime.now().isoformat(),
        )
        self.tool_events.append(record)
        self.updated_at = datetime.now().isoformat()
        return record

    def record_verification(
        self,
        scenario_id: str,
        scenario_name: str,
        status: str,
        satisfied: List[str],
        violated: List[str],
        uncertain: List[str],
        scores: Dict[str, float],
        overall_score: float,
        confidence: float,
        summary: str,
    ) -> VerificationRecord:
        """Record a verification result."""
        record = VerificationRecord(
            verification_index=len(self.verification_history) + 1,
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            status=status,
            satisfied_constraints=satisfied,
            violated_constraints=violated,
            uncertain_aspects=uncertain,
            impact_scores=scores,
            overall_score=overall_score,
            confidence=confidence,
            summary=summary,
            timestamp=datetime.now().isoformat(),
        )
        self.verification_history.append(record)
        
        # Update best scenario if overall score improved
        if overall_score > self.best_score:
            self.best_score = overall_score
            self.best_scenario_id = scenario_id
            self.best_scenario_name = scenario_name
            self.best_impact_result = scores
            
        self.updated_at = datetime.now().isoformat()
        return record

    def record_failure(
        self,
        failure_type: str,
        error_message: str,
        provider: Optional[str] = None,
        recovered: bool = False,
        recovery_strategy: Optional[str] = None,
        fallback_provider: Optional[str] = None,
    ) -> FailureEvent:
        """Record a technical/system failure."""
        event = FailureEvent(
            failure_id=f"fail_{uuid.uuid4().hex[:8]}",
            failure_type=failure_type,
            error_message=error_message,
            provider=provider or self.active_provider,
            recovered=recovered,
            recovery_strategy=recovery_strategy,
            fallback_provider=fallback_provider,
            timestamp=datetime.now().isoformat(),
        )
        self.system_failures.append(event)
        self.updated_at = datetime.now().isoformat()
        return event

    def record_disruption(
        self,
        disruption_type: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> DisruptionEvent:
        """Record a domain/contextual disruption."""
        event = DisruptionEvent(
            disruption_id=f"disrupt_{uuid.uuid4().hex[:8]}",
            disruption_type=disruption_type,
            description=description,
            parameters=parameters or {},
            timestamp=datetime.now().isoformat(),
            active=True,
        )
        self.domain_disruptions.append(event)
        self.updated_at = datetime.now().isoformat()
        return event

    def to_dict(self) -> Dict[str, Any]:
        """Convert state into a comprehensive dictionary."""
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "problem_statement": self.problem_statement,
            "source_files": self.source_files,
            "objectives": self.objectives,
            "constraints": self.constraints,
            "status": self.status.value if isinstance(self.status, AgentStatus) else self.status,
            "iteration_count": self.iteration_count,
            "max_iterations": self.max_iterations,
            "current_stage": self.current_stage,
            "current_action": self.current_action,
            "context_id": self.context_id,
            "graph_id": self.graph_id,
            "project_id": self.project_id,
            "stakeholders": self.stakeholders,
            "candidate_scenarios": self.candidate_scenarios,
            "current_scenario_id": self.current_scenario_id,
            "observations": self.observations,
            "decision_history": [d.to_dict() for d in self.decision_history],
            "tool_events": [t.to_dict() for t in self.tool_events],
            "simulation_results": self.simulation_results,
            "social_impact_results": self.social_impact_results,
            "verification_history": [v.to_dict() for v in self.verification_history],
            "replanning_history": self.replanning_history,
            "domain_disruptions": [d.to_dict() for d in self.domain_disruptions],
            "system_failures": [f.to_dict() for f in self.system_failures],
            "primary_provider": self.primary_provider,
            "active_provider": self.active_provider,
            "provider_fallback_chain": self.provider_fallback_chain,
            "provider_status_map": self.provider_status_map,
            "best_scenario_id": self.best_scenario_id,
            "best_scenario_name": self.best_scenario_name,
            "best_score": self.best_score,
            "best_impact_result": self.best_impact_result,
            "final_outcome_summary": self.final_outcome_summary,
            "human_interaction_pending": self.human_interaction_pending,
            "human_interaction_request": self.human_interaction_request,
            "human_interaction_history": self.human_interaction_history,
            "uncertainty_metrics": self.uncertainty_metrics,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionState":
        """Reconstruct state from dictionary."""
        status_val = data.get("status", "idle")
        status = AgentStatus(status_val) if status_val in [s.value for s in AgentStatus] else AgentStatus.IDLE

        state = cls(
            run_id=data.get("run_id", ""),
            goal=data.get("goal", ""),
            problem_statement=data.get("problem_statement", ""),
            source_files=data.get("source_files", []),
            objectives=data.get("objectives", []),
            constraints=data.get("constraints", {}),
            status=status,
            iteration_count=data.get("iteration_count", 0),
            max_iterations=data.get("max_iterations", 5),
            current_stage=data.get("current_stage", "init"),
            current_action=data.get("current_action", ""),
            context_id=data.get("context_id"),
            graph_id=data.get("graph_id"),
            project_id=data.get("project_id"),
            stakeholders=data.get("stakeholders", []),
            candidate_scenarios=data.get("candidate_scenarios", {}),
            current_scenario_id=data.get("current_scenario_id"),
            observations=data.get("observations", []),
            simulation_results=data.get("simulation_results", {}),
            social_impact_results=data.get("social_impact_results", {}),
            replanning_history=data.get("replanning_history", []),
            primary_provider=data.get("primary_provider", "ollama"),
            active_provider=data.get("active_provider", "ollama"),
            provider_fallback_chain=data.get("provider_fallback_chain", ["ollama", "claude-cli", "codex-cli"]),
            provider_status_map=data.get("provider_status_map", {}),
            best_scenario_id=data.get("best_scenario_id"),
            best_scenario_name=data.get("best_scenario_name"),
            best_score=data.get("best_score", -1.0),
            best_impact_result=data.get("best_impact_result"),
            final_outcome_summary=data.get("final_outcome_summary", ""),
            human_interaction_pending=data.get("human_interaction_pending", False),
            human_interaction_request=data.get("human_interaction_request"),
            human_interaction_history=data.get("human_interaction_history", []),
            uncertainty_metrics=data.get("uncertainty_metrics", {}),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )

        state.decision_history = [
            DecisionRecord.from_dict(d) for d in data.get("decision_history", [])
        ]
        state.tool_events = [
            ActionRecord.from_dict(t) for t in data.get("tool_events", [])
        ]
        state.verification_history = [
            VerificationRecord.from_dict(v) for v in data.get("verification_history", [])
        ]
        state.domain_disruptions = [
            DisruptionEvent.from_dict(d) for d in data.get("domain_disruptions", [])
        ]
        state.system_failures = [
            FailureEvent.from_dict(f) for f in data.get("system_failures", [])
        ]

        return state

    def save(self, directory: str) -> Dict[str, str]:
        """Persist state and related audit JSON files into directory."""
        os.makedirs(directory, exist_ok=True)
        
        state_path = os.path.join(directory, "state.json")
        decisions_path = os.path.join(directory, "decisions.json")
        tool_events_path = os.path.join(directory, "tool_events.json")
        failures_path = os.path.join(directory, "failures.json")
        verification_path = os.path.join(directory, "verification.json")
        final_result_path = os.path.join(directory, "final_result.json")

        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        with open(decisions_path, "w", encoding="utf-8") as f:
            json.dump([d.to_dict() for d in self.decision_history], f, indent=2, ensure_ascii=False)

        with open(tool_events_path, "w", encoding="utf-8") as f:
            json.dump([t.to_dict() for t in self.tool_events], f, indent=2, ensure_ascii=False)

        with open(failures_path, "w", encoding="utf-8") as f:
            json.dump({
                "system_failures": [sf.to_dict() for sf in self.system_failures],
                "domain_disruptions": [dd.to_dict() for dd in self.domain_disruptions],
            }, f, indent=2, ensure_ascii=False)

        with open(verification_path, "w", encoding="utf-8") as f:
            json.dump([v.to_dict() for v in self.verification_history], f, indent=2, ensure_ascii=False)

        final_data = {
            "run_id": self.run_id,
            "status": self.status.value,
            "goal": self.goal,
            "best_scenario_id": self.best_scenario_id,
            "best_scenario_name": self.best_scenario_name,
            "best_score": self.best_score,
            "best_impact_result": self.best_impact_result,
            "final_outcome_summary": self.final_outcome_summary,
            "iteration_count": self.iteration_count,
            "max_iterations": self.max_iterations,
        }
        with open(final_result_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)

        # Also write demo_trace.md
        from ..run_artifacts import generate_demo_trace_markdown
        demo_trace_path = os.path.join(directory, "demo_trace.md")
        trace_md = generate_demo_trace_markdown(self.to_dict())
        with open(demo_trace_path, "w", encoding="utf-8") as f:
            f.write(trace_md)

        return {
            "state": state_path,
            "decisions": decisions_path,
            "tool_events": tool_events_path,
            "failures": failures_path,
            "verification": verification_path,
            "final_result": final_result_path,
            "demo_trace": demo_trace_path,
        }
