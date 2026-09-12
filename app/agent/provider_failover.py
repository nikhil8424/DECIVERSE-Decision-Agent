"""LLM Provider Failover and Robustness Manager.

Provides multi-provider fallback hierarchy (e.g. Ollama -> Claude CLI -> Codex CLI),
with health tracking, timeout and error management, and auditable failover events.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger

logger = get_logger('mirofish.agent.provider_failover')


class ProviderStatus(str, Enum):
    """Health status of an LLM provider."""
    AVAILABLE = "available"
    DEGRADED = "degraded"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass
class FailoverEvent:
    """Record of a provider transition upon error."""
    failed_provider: str
    target_provider: str
    error_reason: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ProviderFailoverManager:
    """Manages multi-provider failover and resilience for the agent."""

    def __init__(
        self,
        primary_provider: Optional[str] = None,
        fallback_chain: Optional[List[str]] = None,
    ):
        self.primary_provider = (primary_provider or Config.LLM_PROVIDER or "ollama").lower()
        self.fallback_chain = fallback_chain or ["ollama", "claude-cli", "codex-cli"]
        
        # Ensure primary is at head of chain if in chain
        if self.primary_provider in self.fallback_chain:
            self.fallback_chain.remove(self.primary_provider)
        self.fallback_chain = [self.primary_provider] + self.fallback_chain
        
        self.active_provider = self.primary_provider
        self.provider_statuses: Dict[str, ProviderStatus] = {
            p: ProviderStatus.AVAILABLE for p in self.fallback_chain
        }
        self.failover_history: List[FailoverEvent] = []
        
        # Injectable failure overrides for testing/demo
        self._forced_timeouts: set[str] = set()
        self._forced_unavailables: set[str] = set()

    def get_status_summary(self) -> Dict[str, str]:
        """Return dict of provider statuses."""
        return {p: s.value for p, s in self.provider_statuses.items()}

    def inject_provider_failure(self, provider: str, failure_type: str = "timeout") -> None:
        """Inject controlled provider failure for demonstration/testing."""
        provider = provider.lower()
        if failure_type == "timeout":
            self._forced_timeouts.add(provider)
        elif failure_type in ("unavailable", "error"):
            self._forced_unavailables.add(provider)
        self.provider_statuses[provider] = ProviderStatus.DEGRADED
        logger.warning(f"Injected {failure_type} into provider {provider}")

    def is_healthy(self, provider: Optional[str] = None) -> bool:
        """Check if a provider is healthy and not forced to fail."""
        p = (provider or self.active_provider or self.primary_provider).lower()
        if p in self._forced_timeouts or p in self._forced_unavailables:
            return False
        return self.provider_statuses.get(p) not in (ProviderStatus.FAILED, ProviderStatus.DISABLED)

    def clear_injections(self) -> None:
        """Clear all injected failures."""
        self._forced_timeouts.clear()
        self._forced_unavailables.clear()
        for p in self.fallback_chain:
            self.provider_statuses[p] = ProviderStatus.AVAILABLE

    def switch_to_next_available(self, current_provider: str, error_reason: str) -> Optional[str]:
        """Switch active provider to the next available one in the fallback chain."""
        self.provider_statuses[current_provider] = ProviderStatus.FAILED
        
        current_idx = self.fallback_chain.index(current_provider) if current_provider in self.fallback_chain else -1
        next_providers = self.fallback_chain[current_idx + 1:]
        
        for candidate in next_providers:
            if self.provider_statuses.get(candidate) != ProviderStatus.FAILED:
                event = FailoverEvent(
                    failed_provider=current_provider,
                    target_provider=candidate,
                    error_reason=error_reason,
                )
                self.failover_history.append(event)
                self.active_provider = candidate
                logger.warning(
                    f"Provider failover: {current_provider} -> {candidate} (Reason: {error_reason})"
                )
                return candidate
        
        logger.error("All providers in fallback chain have failed!")
        return None

    def execute_with_failover(
        self,
        call_fn: Callable[[LLMClient], Any],
        max_chain_attempts: Optional[int] = None,
    ) -> Any:
        """Execute an LLM function with automatic failover across providers."""
        attempts = 0
        max_attempts = max_chain_attempts or len(self.fallback_chain)
        last_error = None

        while attempts < max_attempts:
            provider = self.active_provider
            attempts += 1
            
            # Check injected disruptions
            if provider in self._forced_timeouts:
                err_msg = f"Simulated timeout on provider {provider}"
                logger.warning(err_msg)
                next_p = self.switch_to_next_available(provider, err_msg)
                if not next_p:
                    raise RuntimeError(f"All providers failed. Last error: {err_msg}")
                continue

            if provider in self._forced_unavailables:
                err_msg = f"Simulated unavailable on provider {provider}"
                logger.warning(err_msg)
                next_p = self.switch_to_next_available(provider, err_msg)
                if not next_p:
                    raise RuntimeError(f"All providers failed. Last error: {err_msg}")
                continue

            try:
                client = LLMClient(provider=provider)
                result = call_fn(client)
                self.provider_statuses[provider] = ProviderStatus.AVAILABLE
                return result
            except Exception as e:
                err_msg = str(e)
                logger.warning(f"Provider {provider} execution failed: {err_msg}")
                last_error = e
                next_p = self.switch_to_next_available(provider, err_msg)
                if not next_p:
                    raise RuntimeError(f"All providers in fallback chain failed. Last error: {err_msg}") from last_error

        raise RuntimeError(f"Failed to execute LLM call after {attempts} provider attempts: {last_error}")

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """Send chat messages with failover support."""
        return self.execute_with_failover(
            lambda client: client.chat(messages=messages, temperature=temperature, max_tokens=max_tokens)
        )

    def chat_json(self, messages: List[Dict[str, str]], temperature: float = 0.3) -> Dict[str, Any]:
        """Send chat_json messages with failover and JSON parsing recovery."""
        return self.execute_with_failover(
            lambda client: client.chat_json(messages=messages, temperature=temperature)
        )
