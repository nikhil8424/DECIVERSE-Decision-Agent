"""Tests for Provider Failover, timeout handling, and multi-provider resilience."""

from unittest.mock import MagicMock, patch
import pytest
from app.agent.provider_failover import ProviderFailoverManager, ProviderStatus


def test_provider_failover_initialization():
    """Verify fallback chain and active provider assignment."""
    mgr = ProviderFailoverManager(primary_provider="ollama", fallback_chain=["ollama", "claude-cli", "codex-cli"])
    assert mgr.active_provider == "ollama"
    assert mgr.fallback_chain == ["ollama", "claude-cli", "codex-cli"]
    assert mgr.provider_statuses["ollama"] == ProviderStatus.AVAILABLE


def test_provider_failover_on_timeout():
    """Verify failover switches to next provider upon simulated timeout."""
    mgr = ProviderFailoverManager(primary_provider="ollama", fallback_chain=["ollama", "claude-cli", "codex-cli"])
    
    # Inject forced timeout into primary provider (ollama)
    mgr.inject_provider_failure("ollama", "timeout")

    # Mock the LLM client call for fallback provider (claude-cli)
    with patch("app.agent.provider_failover.LLMClient") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.chat.return_value = "Response from Claude CLI"
        mock_client_cls.return_value = mock_instance

        result = mgr.chat([{"role": "user", "content": "Hello"}])

        assert result == "Response from Claude CLI"
        assert mgr.active_provider == "claude-cli"
        assert len(mgr.failover_history) == 1
        assert mgr.failover_history[0].failed_provider == "ollama"
        assert mgr.failover_history[0].target_provider == "claude-cli"
        assert "timeout" in mgr.failover_history[0].error_reason.lower()


def test_provider_failover_exhaustion_raises_error():
    """Verify failure of all providers in the chain raises classified error."""
    mgr = ProviderFailoverManager(primary_provider="ollama", fallback_chain=["ollama", "claude-cli"])
    
    # Disable all providers
    mgr.inject_provider_failure("ollama", "timeout")
    mgr.inject_provider_failure("claude-cli", "unavailable")

    with pytest.raises(RuntimeError, match="All providers"):
        mgr.chat([{"role": "user", "content": "Hello"}])
    
    assert mgr.provider_statuses["ollama"] == ProviderStatus.FAILED
    assert mgr.provider_statuses["claude-cli"] == ProviderStatus.FAILED
