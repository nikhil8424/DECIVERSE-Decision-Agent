"""Tests for AgentToolRegistry and structured tool executions."""

import pytest
from app.agent.tools import AgentToolRegistry, AgentToolResult


@pytest.fixture
def registry():
    return AgentToolRegistry()


def test_tool_observe_context(registry: AgentToolRegistry):
    """Verify observe_context tool extracts community context and entities."""
    res = registry.execute_tool(
        "observe_context",
        problem_statement="Transit fare policy for Metro City",
        document_texts=["Mayor Jenkins announced free weekend bus policy for 6 months."],
    )

    assert isinstance(res, AgentToolResult)
    assert res.success is True
    assert "context_id" in res.data
    assert "graph_id" in res.data
    assert res.data["entity_count"] > 0
    assert res.duration_seconds >= 0.0


def test_tool_create_and_modify_scenario(registry: AgentToolRegistry):
    """Verify scenario creation and dynamic grounded modification."""
    # 1. Create scenario
    create_res = registry.execute_tool(
        "create_scenario",
        name="Policy A",
        description="Free weekend bus",
        intervention="Free buses on Saturday and Sunday",
    )
    assert create_res.success is True
    scenario_data = create_res.data
    assert scenario_data["name"] == "Policy A"
    assert "scenario_id" in scenario_data

    # 2. Modify scenario to address polarization
    mod_res = registry.execute_tool(
        "modify_scenario",
        base_scenario=scenario_data,
        modification_goal="Reduce polarization below 0.35",
        violated_constraints=["polarization < 0.35"],
    )
    assert mod_res.success is True
    mod_data = mod_res.data["modified_scenario"]
    assert "Adapted" in mod_data["name"]
    assert len(mod_res.data["modifications_applied"]) > 0
    assert "oversight" in mod_data["intervention"].lower() or "tiered" in mod_data["intervention"].lower() or "compromise" in mod_data["intervention"].lower()


def test_tool_simulation_and_social_impact_pipeline(registry: AgentToolRegistry):
    """Verify simulation execution and SocialImpactModel evaluation."""
    create_res = registry.execute_tool(
        "create_scenario",
        name="Policy B",
        description="Subsidized transit",
        intervention="Subsidized transit passes",
    )
    scen_id = create_res.data["scenario_id"]

    sim_res = registry.execute_tool(
        "run_simulation",
        scenario_id=scen_id,
        scenario_data=create_res.data,
        rounds=3,
    )
    assert sim_res.success is True
    sim_id = sim_res.data["simulation_id"]
    assert sim_res.data["rounds_completed"] == 3

    impact_res = registry.execute_tool(
        "calculate_social_impact",
        scenario_id=scen_id,
        scenario_name="Policy B",
        simulation_id=sim_id,
    )
    assert impact_res.success is True
    impact_data = impact_res.data["impact_result"]
    assert "acceptance_score" in impact_data
    assert "polarization_score" in impact_data
    assert "conflict_score" in impact_data
    assert "overall_score" in impact_data
    assert 0.0 <= impact_data["overall_score"] <= 1.0


def test_tool_additional_simulation_uncertainty(registry: AgentToolRegistry):
    """Verify multi-run uncertainty quantification tool."""
    create_res = registry.execute_tool(
        "create_scenario",
        name="Policy C",
        description="Fare cap",
        intervention="Fare cap for all commuters",
    )
    scen_id = create_res.data["scenario_id"]

    unc_res = registry.execute_tool(
        "request_additional_simulation",
        scenario_id=scen_id,
        scenario_name="Policy C",
        scenario_data=create_res.data,
        sample_runs=3,
    )
    assert unc_res.success is True
    assert unc_res.data["run_count"] == 3
    assert "dimension_stats" in unc_res.data
    assert "acceptance_score" in unc_res.data["dimension_stats"]
    assert "mean" in unc_res.data["dimension_stats"]["acceptance_score"]


def test_tool_unknown_name_handled_safely(registry: AgentToolRegistry):
    """Verify unknown tool call returns structured failure without crashing."""
    res = registry.execute_tool("non_existent_tool", arg="test")
    assert res.success is False
    assert "Unknown tool" in res.error
