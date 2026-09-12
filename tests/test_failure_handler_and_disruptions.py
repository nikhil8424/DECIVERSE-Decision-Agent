"""Tests for FailureHandler, response repair, and controlled DisruptionEngine."""

import pytest
from app.agent.failure_handler import FailureHandler, FailureType, FailureSeverity
from app.agent.disruption_engine import (
    DisruptionEngine,
    DomainDisruptionType,
    TechnicalDisruptionType,
)


def test_failure_classification_system_vs_domain():
    """Verify FailureHandler classifies exceptions correctly."""
    # 1. Timeout exception
    timeout_ana = FailureHandler.classify_exception(TimeoutError("Connection timed out after 30s"))
    assert timeout_ana.is_system_failure is True
    assert timeout_ana.failure_type == FailureType.LLM_TIMEOUT
    assert timeout_ana.suggested_recovery == "switch_provider"

    # 2. Connection refused
    conn_ana = FailureHandler.classify_exception(ConnectionRefusedError("Ollama is not reachable"))
    assert conn_ana.is_system_failure is True
    assert conn_ana.failure_type == FailureType.PROVIDER_UNAVAILABLE

    # 3. JSON decode error
    json_ana = FailureHandler.classify_exception(ValueError("Invalid JSON returned by LLM"))
    assert json_ana.is_system_failure is True
    assert json_ana.failure_type == FailureType.MALFORMED_RESPONSE


def test_validate_and_repair_json():
    """Verify malformed JSON payload validation and repair heuristics."""
    # 1. Markdown codeblock JSON
    md_text = "```json\n{\"acceptance_score\": 0.85, \"polarization_score\": 0.20}\n```"
    ok, parsed, err = FailureHandler.validate_and_repair_json(md_text, required_keys=["acceptance_score", "polarization_score"])
    assert ok is True
    assert parsed["acceptance_score"] == 0.85
    assert parsed["polarization_score"] == 0.20

    # 2. Embedded JSON with missing required key repaired via defaults
    partial_text = "The result is {\"acceptance_score\": 0.75} based on simulation."
    ok, parsed, err = FailureHandler.validate_and_repair_json(
        partial_text,
        required_keys=["acceptance_score", "polarization_score"],
        default_fallback={"acceptance_score": 0.5, "polarization_score": 0.5},
    )
    assert ok is True
    assert parsed["acceptance_score"] == 0.75
    assert parsed["polarization_score"] == 0.5  # Repaired from default fallback

    # 3. Corrupted JSON with regex heuristic extraction
    corrupted = '{"acceptance_score": 0.92, "polarization_score": 0.28, missing_closing_bracket'
    ok, parsed, err = FailureHandler.validate_and_repair_json(
        corrupted,
        required_keys=["acceptance_score", "polarization_score"],
    )
    assert ok is True
    assert parsed["acceptance_score"] == 0.92
    assert parsed["polarization_score"] == 0.28


def test_disruption_engine_domain_injections():
    """Verify domain disruptions modify actual metric outcomes."""
    engine = DisruptionEngine()
    
    # Inject polarization boost
    engine.inject_domain_disruption(DomainDisruptionType.INCREASE_POLARIZATION, {"polarization_boost": 0.30})
    
    base_metrics = {"acceptance_score": 0.75, "polarization_score": 0.25, "consensus_score": 0.70}
    adjusted = engine.apply_domain_disruptions_to_metrics("scen_1", base_metrics)

    assert adjusted["polarization_score"] == pytest.approx(0.55, abs=0.01)
    assert adjusted["consensus_score"] == pytest.approx(0.55, abs=0.01)


def test_disruption_engine_technical_injections():
    """Verify technical disruptions raise controlled exceptions."""
    engine = DisruptionEngine()
    
    # Force timeout
    engine.inject_technical_disruption(TechnicalDisruptionType.FORCE_LLM_TIMEOUT, {"tool_name": "run_simulation"})
    
    with pytest.raises(TimeoutError, match="Injected LLM Timeout"):
        engine.check_and_apply_technical_disruption("run_simulation")
