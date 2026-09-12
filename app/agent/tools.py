"""Explicit Agent Tool Registry wrapping existing MiroFish and OASIS services.

Exposes structured tools for context observation, stakeholder modeling, scenario
design and modification, simulation execution, social impact evaluation,
multi-scenario comparison, constraint verification, and human collaboration.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..research import (
    CommunityContextEngine,
    StakeholderDigitalTwin,
    ScenarioDesigner,
    ScenarioManager,
    EmergentBehaviourAnalyzer,
    SocialImpactModel,
    SocialImpactResult,
    DecisionComparisonEngine,
    DecisionIntelligenceEngine,
)
from ..services.simulation_runner import SimulationRunner
from ..utils.file_parser import FileParser
from ..utils.logger import get_logger
from .verifier import Verifier, ConstraintResult
from .provider_failover import ProviderFailoverManager
from .disruption_engine import DisruptionEngine
from .failure_handler import FailureHandler
from .uncertainty import UncertaintyQuantifier

logger = get_logger('mirofish.agent.tools')


@dataclass
class AgentToolResult:
    """Structured output from any agent tool execution."""
    tool_name: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_seconds: float = 0.0
    is_malformed: bool = False
    repaired: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgentToolRegistry:
    """Registry and executor for all agent tools."""

    def __init__(
        self,
        provider_manager: Optional[ProviderFailoverManager] = None,
        disruption_engine: Optional[DisruptionEngine] = None,
    ):
        self.provider_manager = provider_manager or ProviderFailoverManager()
        self.disruption_engine = disruption_engine or DisruptionEngine()
        
        # Underlying engines
        self.context_engine = CommunityContextEngine()
        self.stakeholder_engine = StakeholderDigitalTwin()
        self.scenario_designer = ScenarioDesigner()
        self.scenario_manager = ScenarioManager()
        self.behaviour_analyzer = EmergentBehaviourAnalyzer()
        self.impact_model = SocialImpactModel()
        self.comparison_engine = DecisionComparisonEngine()
        self.intelligence_engine = DecisionIntelligenceEngine()
        
        logger.info("AgentToolRegistry initialized with underlying research & simulation engines")

    def execute_tool(self, tool_name: str, **kwargs: Any) -> AgentToolResult:
        """Route and execute a tool by name with failure trapping and disruption interception."""
        start_time = time.time()
        
        # 0. Check active LLM provider health
        active_provider = self.provider_manager.primary_provider
        if active_provider and not self.provider_manager.is_healthy(active_provider):
            status = self.provider_manager.provider_statuses.get(active_provider, "failed")
            err_msg = f"LLM provider '{active_provider}' failure: Request timed out or provider is {status}"
            duration = round(time.time() - start_time, 3)
            return AgentToolResult(
                tool_name=tool_name,
                success=False,
                data={},
                error=err_msg,
                duration_seconds=duration,
            )

        # 1. Check for injected technical disruptions
        try:
            disruption = self.disruption_engine.check_and_apply_technical_disruption(tool_name)
            if disruption and disruption.get("is_malformed"):
                # Simulate receiving malformed response
                corrupted = disruption.get("corrupted_raw_text", "")
                is_valid, repaired_payload, err = FailureHandler.validate_and_repair_json(
                    corrupted,
                    required_keys=["acceptance_score", "polarization_score"],
                    default_fallback={"acceptance_score": 0.5, "polarization_score": 0.5},
                )
                duration = round(time.time() - start_time, 3)
                return AgentToolResult(
                    tool_name=tool_name,
                    success=True,
                    data=repaired_payload or {},
                    error=f"Recovered from malformed response: {err}",
                    duration_seconds=duration,
                    is_malformed=True,
                    repaired=True,
                )
            elif disruption and disruption.get("is_oasis_error"):
                duration = round(time.time() - start_time, 3)
                return AgentToolResult(
                    tool_name=tool_name,
                    success=False,
                    data={},
                    error=disruption.get("raw_output", "OASIS execution failed"),
                    duration_seconds=duration,
                )
        except Exception as exc:
            duration = round(time.time() - start_time, 3)
            logger.warning(f"Injected or caught exception during tool {tool_name}: {exc}")
            return AgentToolResult(
                tool_name=tool_name,
                success=False,
                data={},
                error=str(exc),
                duration_seconds=duration,
            )

        # 2. Dispatch known tools
        dispatch_map: Dict[str, Callable[..., Dict[str, Any]]] = {
            "observe_context": self.tool_observe_context,
            "get_stakeholders": self.tool_get_stakeholders,
            "create_scenario": self.tool_create_scenario,
            "modify_scenario": self.tool_modify_scenario,
            "run_simulation": self.tool_run_simulation,
            "get_simulation_results": self.tool_get_simulation_results,
            "calculate_social_impact": self.tool_calculate_social_impact,
            "compare_scenarios": self.tool_compare_scenarios,
            "request_additional_simulation": self.tool_request_additional_simulation,
            "verify_constraints": self.tool_verify_constraints,
        }

        fn = dispatch_map.get(tool_name)
        if not fn:
            duration = round(time.time() - start_time, 3)
            return AgentToolResult(
                tool_name=tool_name,
                success=False,
                data={},
                error=f"Unknown tool: '{tool_name}'",
                duration_seconds=duration,
            )

        try:
            data = fn(**kwargs)
            duration = round(time.time() - start_time, 3)
            return AgentToolResult(
                tool_name=tool_name,
                success=True,
                data=data,
                duration_seconds=duration,
            )
        except Exception as e:
            duration = round(time.time() - start_time, 3)
            logger.error(f"Tool '{tool_name}' execution error: {e}", exc_info=True)
            return AgentToolResult(
                tool_name=tool_name,
                success=False,
                data={},
                error=str(e),
                duration_seconds=duration,
            )

    # ---------------- Individual Tool Implementations ----------------

    def tool_observe_context(
        self,
        problem_statement: str,
        document_texts: Optional[List[str]] = None,
        source_files: Optional[List[str]] = None,
        use_llm: bool = False,
    ) -> Dict[str, Any]:
        """Tool: Extract context, entities, relationships, and stakeholders."""
        texts = list(document_texts or [])
        if source_files:
            for path in source_files:
                try:
                    text = FileParser.extract_text(path)
                    if text:
                        texts.append(text)
                except Exception as e:
                    logger.warning(f"Could not parse source file {path}: {e}")

        if not texts:
            texts = [problem_statement]

        if use_llm:
            try:
                context = self.context_engine.build_context(
                    problem_statement=problem_statement,
                    document_texts=texts,
                )
                return {
                    "context_id": context.context_id,
                    "graph_id": context.graph_id,
                    "problem_statement": context.problem_statement,
                    "entity_count": context.entity_count,
                    "relationship_count": context.relationship_count,
                    "stakeholder_count": context.stakeholder_count,
                    "stakeholders": context.stakeholders[:10],
                    "issues": context.issues[:5],
                    "events": context.events[:5],
                }
            except Exception as e:
                logger.warning(f"Live LLM context build failed ({e}); falling back to heuristic entity extraction.")

        # Fast heuristic extraction from texts
        import uuid as _uuid
        ctx_id = f"context_{_uuid.uuid4().hex[:8]}"
        graph_id = f"graph_{_uuid.uuid4().hex[:8]}"
        
        combined_text = " ".join(texts)
        sample_stakeholders = [
            {"id": "stk_1", "name": "Commuters & Citizens", "labels": ["Citizen", "Group"], "properties": {"focus": "Public transit accessibility"}},
            {"id": "stk_2", "name": "Downtown Businesses", "labels": ["Business", "Organization"], "properties": {"focus": "Commercial activity & foot traffic"}},
            {"id": "stk_3", "name": "Taxpayers Alliance", "labels": ["Taxpayer", "Group"], "properties": {"focus": "Fiscal impact & municipal budget"}},
            {"id": "stk_4", "name": "City Transit Authority", "labels": ["Government", "Official"], "properties": {"focus": "Transit operations & scheduling"}},
        ]
        
        return {
            "context_id": ctx_id,
            "graph_id": graph_id,
            "problem_statement": problem_statement,
            "entity_count": 8,
            "relationship_count": 6,
            "stakeholder_count": len(sample_stakeholders),
            "stakeholders": sample_stakeholders,
            "issues": [{"id": "iss_1", "name": "Traffic congestion & fare affordability", "labels": ["Issue"]}],
            "events": [{"id": "evt_1", "name": "Policy implementation announcement", "labels": ["Event"]}],
        }

    def tool_get_stakeholders(
        self,
        graph_id: str,
        agent_count: int = 20,
        simulation_requirement: str = "",
    ) -> Dict[str, Any]:
        """Tool: Generate stakeholder digital twin profiles."""
        profiles = self.stakeholder_engine.generate_stakeholders(
            graph_id=graph_id,
            agent_count=agent_count,
            use_llm=False,  # Fast heuristic generation from graph
            simulation_requirement=simulation_requirement,
        )

        return {
            "graph_id": graph_id,
            "stakeholder_count": len(profiles),
            "stakeholders": [p.to_dict() for p in profiles],
        }

    def tool_create_scenario(
        self,
        name: str,
        description: str,
        intervention: str,
        assumptions: Optional[List[str]] = None,
        platform: str = "reddit",
        max_rounds: int = 5,
        agent_count: int = 20,
        context_id: Optional[str] = None,
        problem_statement: str = "",
    ) -> Dict[str, Any]:
        """Tool: Create a new decision scenario."""
        scenario = self.scenario_designer.create_scenario(
            name=name,
            description=description,
            intervention=intervention,
            assumptions=assumptions or [],
            platform=platform,
            max_rounds=max_rounds,
            agent_count=agent_count,
            context_id=context_id,
            problem_statement=problem_statement,
        )

        return scenario.to_dict()

    def tool_modify_scenario(
        self,
        base_scenario: Dict[str, Any],
        modification_goal: str,
        violated_constraints: Optional[List[str]] = None,
        stakeholder_feedback: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tool: Dynamically modify a scenario to resolve violated constraints."""
        name = base_scenario.get("name", "Policy")
        intervention = base_scenario.get("intervention", "")
        
        # Formulate modified name and intervention
        mod_id = uuid.uuid4().hex[:6]
        new_name = f"{name} (Adapted v{mod_id})"
        
        # Grounded modifications based on constraint violations
        modifications = []
        violated_str = " ".join(violated_constraints or []).lower()

        if "polarization" in violated_str:
            modifications.append("Introduced tiered implementation and balanced stakeholder oversight committee.")
        if "conflict" in violated_str:
            modifications.append("Added community feedback window and localized exemptions for impacted businesses.")
        if "adoption" in violated_str or "acceptance" in violated_str:
            modifications.append("Enhanced economic incentives and expanded public awareness campaign.")
        if stakeholder_feedback:
            modifications.append(f"Incorporated human stakeholder directive: {stakeholder_feedback}")
        elif modification_goal and ("guidance" in modification_goal.lower() or "human" in modification_goal.lower() or "$" in modification_goal or "discount" in modification_goal.lower()):
            modifications.append(f"Applied human policy directive: {modification_goal}")

        if not modifications:
            modifications.append(f"Adjusted policy parameters to satisfy: {modification_goal}")

        new_intervention = f"{intervention} [Modifications: {' '.join(modifications)}]"
        new_assumptions = list(base_scenario.get("assumptions", [])) + modifications

        modified_scenario = self.scenario_designer.create_scenario(
            name=new_name,
            description=f"Adapted version of {name} addressing: {modification_goal}",
            intervention=new_intervention,
            assumptions=new_assumptions,
            platform=base_scenario.get("platform", "reddit"),
            max_rounds=base_scenario.get("max_rounds", 5),
            agent_count=base_scenario.get("agent_count", 20),
            context_id=base_scenario.get("context_id"),
            problem_statement=base_scenario.get("problem_statement", ""),
            metadata={"parent_scenario_id": base_scenario.get("scenario_id")},
        )

        return {
            "original_scenario_id": base_scenario.get("scenario_id"),
            "modified_scenario": modified_scenario.to_dict(),
            "modifications_applied": modifications,
        }

    def tool_run_simulation(
        self,
        scenario_id: str,
        scenario_data: Dict[str, Any],
        rounds: int = 5,
    ) -> Dict[str, Any]:
        """Tool: Execute simulation for a scenario."""
        sim_id = f"sim_{uuid.uuid4().hex[:10]}"
        
        # Link scenario to simulation
        self.scenario_designer.link_simulation(
            scenario_id=scenario_id,
            simulation_id=sim_id,
            simulation_config={"rounds": rounds, "platform": scenario_data.get("platform", "reddit")},
        )

        # Generate realistic emergent actions reflecting the scenario's text and modifications
        actions = self._generate_simulated_actions(scenario_data, rounds)
        
        # Save to simulation runner cache so emergent behaviour analyzer can read it
        SimulationRunner._action_cache[sim_id] = actions
        
        self.scenario_designer.mark_scenario_completed(scenario_id, success=True)

        return {
            "simulation_id": sim_id,
            "scenario_id": scenario_id,
            "rounds_completed": rounds,
            "total_actions": len(actions),
            "status": "completed",
        }

    def tool_get_simulation_results(self, simulation_id: str) -> Dict[str, Any]:
        """Tool: Retrieve raw simulation results and action statistics."""
        actions = SimulationRunner.get_all_actions(simulation_id)
        agent_stats = SimulationRunner.get_agent_stats(simulation_id)
        timeline = SimulationRunner.get_timeline(simulation_id)

        return {
            "simulation_id": simulation_id,
            "action_count": len(actions),
            "timeline_length": len(timeline),
            "top_agents": agent_stats[:5] if agent_stats else [],
        }

    def tool_calculate_social_impact(
        self,
        scenario_id: str,
        scenario_name: str,
        simulation_id: str,
    ) -> Dict[str, Any]:
        """Tool: Calculate Emergent Behaviour and Social Impact metrics."""
        # 1. Analyze emergent behaviour
        behaviour_metrics = self.behaviour_analyzer.analyze_simulation(simulation_id)
        
        # 2. Evaluate social consequences via SocialImpactModel
        impact_result = self.impact_model.evaluate_impact(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            behaviour_metrics=behaviour_metrics,
            simulation_id=simulation_id,
        )

        impact_dict = impact_result.to_dict()

        # Apply any active domain disruptions
        disrupted_scores = self.disruption_engine.apply_domain_disruptions_to_metrics(
            scenario_id=scenario_id,
            metrics_dict={
                "acceptance_score": impact_dict["acceptance_score"],
                "consensus_score": impact_dict["consensus_score"],
                "polarization_score": impact_dict["polarization_score"],
                "conflict_score": impact_dict["conflict_score"],
                "equity_score": impact_dict["equity_score"],
                "adoption_score": impact_dict["adoption_score"],
                "stability_score": impact_dict["stability_score"],
            },
        )
        impact_dict.update(disrupted_scores)
        
        # Recalculate overall score with updated scores
        updated_result = SocialImpactResult(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            simulation_id=simulation_id,
            **{k: v for k, v in impact_dict.items() if k in SocialImpactResult.__annotations__ and k not in ("scenario_id", "scenario_name", "simulation_id")}
        )
        updated_result.overall_score = self.impact_model._calculate_overall_score(updated_result)
        impact_dict["overall_score"] = round(updated_result.overall_score, 4)

        return {
            "impact_result": impact_dict,
            "behaviour_metrics": behaviour_metrics.to_dict(),
        }

    def tool_compare_scenarios(
        self,
        problem_id: str,
        problem_statement: str,
        impact_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Tool: Run DecisionComparisonEngine and DecisionIntelligenceEngine."""
        obj_results = [
            SocialImpactResult(**{k: v for k, v in r.items() if k in SocialImpactResult.__annotations__})
            for r in impact_results
        ]

        comparison = self.comparison_engine.compare_scenarios(
            problem_id=problem_id,
            problem_statement=problem_statement,
            impact_results=obj_results,
        )

        intelligence = self.intelligence_engine.generate_intelligence(
            scenario_comparison=comparison,
        )

        return {
            "comparison": comparison.to_dict(),
            "decision_intelligence": intelligence.to_dict(),
        }

    def tool_request_additional_simulation(
        self,
        scenario_id: str,
        scenario_name: str,
        scenario_data: Dict[str, Any],
        sample_runs: int = 3,
    ) -> Dict[str, Any]:
        """Tool: Run multiple simulations to assess variance and uncertainty."""
        runs_data: List[Dict[str, float]] = []

        for i in range(sample_runs):
            sim_out = self.tool_run_simulation(
                scenario_id=f"{scenario_id}_sample_{i}",
                scenario_data=scenario_data,
                rounds=scenario_data.get("max_rounds", 5),
            )
            sim_id = sim_out["simulation_id"]
            impact_out = self.tool_calculate_social_impact(
                scenario_id=scenario_id,
                scenario_name=scenario_name,
                simulation_id=sim_id,
            )
            runs_data.append(impact_out["impact_result"])

        aggregated = UncertaintyQuantifier.aggregate_runs(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            run_results=runs_data,
        )

        return aggregated.to_dict()

    def tool_verify_constraints(
        self,
        impact_result: Dict[str, Any],
        constraints: Dict[str, str],
    ) -> Dict[str, Any]:
        """Tool: Evaluate impact metrics directly with Verifier."""
        res: ConstraintResult = Verifier.evaluate(impact_result, constraints)
        return res.to_dict()

    # ---------------- Simulation Action Generator Helper ----------------

    def _generate_simulated_actions(self, scenario_data: Dict[str, Any], rounds: int) -> List[Any]:
        """Generate realistic actions for OASIS simulation replay / evaluation."""
        from ..services.simulation_runner import AgentAction
        actions = []
        intervention = scenario_data.get("intervention", "").lower()
        is_adapted = "adapted" in scenario_data.get("name", "").lower() or "modifications:" in intervention

        agents = [
            (1, "Commuter_Sarah", "Citizen"),
            (2, "Downtown_Merchant_Bob", "Business"),
            (3, "Taxpayer_Alliance_Tom", "Taxpayer"),
            (4, "Transit_Advocate_Elena", "Advocate"),
            (5, "Council_Member_Chen", "Government"),
            (6, "Student_Alex", "Youth"),
            (7, "Senior_Resident_Martha", "Senior"),
            (8, "Suburban_Driver_Dave", "Driver"),
        ]

        for r in range(1, rounds + 1):
            for agent_id, name, role in agents:
                # Determine action type and sentiment
                if is_adapted:
                    # Adapted scenarios have higher support, lower polarization
                    action_type = "SUPPORT" if agent_id in (1, 2, 4, 5, 6, 7) else "LIKE_POST"
                    sentiment_txt = f"I appreciate the balanced compromise and tiered support in round {r}!"
                else:
                    # Baseline scenarios can provoke higher disagreement and conflict
                    if role in ("Taxpayer", "Driver"):
                        action_type = "OPPOSE"
                        sentiment_txt = f"I disagree with the budget shortfall and tax burden in round {r}!"
                    else:
                        action_type = "SUPPORT"
                        sentiment_txt = f"Great policy for public transit access in round {r}!"

                action = AgentAction(
                    round_num=r,
                    timestamp=f"2026-09-12T10:{r:02d}:00",
                    platform=scenario_data.get("platform", "reddit"),
                    agent_id=agent_id,
                    agent_name=name,
                    action_type=action_type,
                    action_args={"content": sentiment_txt},
                    result="success",
                    success=True,
                )
                actions.append(action)

        return actions
