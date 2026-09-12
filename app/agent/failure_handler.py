"""Failure Classification and Recovery Handler.

Distinguishes between Technical/System failures and Domain failures.
Provides malformed response validation, repair heuristics, and recovery suggestions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..utils.logger import get_logger

logger = get_logger('mirofish.agent.failure_handler')


class FailureType(str, Enum):
    """Categorized failure types."""
    # Technical / System Failures
    LLM_TIMEOUT = "llm_timeout"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    MALFORMED_RESPONSE = "malformed_response"
    TOOL_EXCEPTION = "tool_exception"
    OASIS_ERROR = "oasis_error"
    EMPTY_RESULT = "empty_result"
    MISSING_FIELD = "missing_field"

    # Domain / Contextual Failures
    CONSTRAINT_VIOLATION = "constraint_violation"
    STAKEHOLDER_OPPOSITION = "stakeholder_opposition"
    BUDGET_EXCEEDED = "budget_exceeded"
    SCENARIO_INFEASIBLE = "scenario_infeasible"
    HIGH_POLARIZATION = "high_polarization"
    HIGH_CONFLICT = "high_conflict"
    LOW_ADOPTION = "low_adoption"


class FailureSeverity(str, Enum):
    """Severity levels for failure handling."""
    LOW = "low"          # Recoverable via retry / normalization
    MEDIUM = "medium"    # Recoverable via replanning / provider failover
    HIGH = "high"        # Requires human-in-the-loop intervention
    CRITICAL = "critical"# Fatal unresolvable stop


@dataclass
class FailureAnalysis:
    """Detailed diagnosis of a failure."""
    is_system_failure: bool
    failure_type: FailureType
    severity: FailureSeverity
    error_message: str
    suggested_recovery: str
    repairable: bool = False
    repaired_payload: Optional[Dict[str, Any]] = None


class FailureHandler:
    """Classifies failures, repairs malformed tool responses, and suggests recovery."""

    @staticmethod
    def is_system_failure(failure_type: FailureType) -> bool:
        """Return True if failure is technical/system related."""
        return failure_type in (
            FailureType.LLM_TIMEOUT,
            FailureType.PROVIDER_UNAVAILABLE,
            FailureType.MALFORMED_RESPONSE,
            FailureType.TOOL_EXCEPTION,
            FailureType.OASIS_ERROR,
            FailureType.EMPTY_RESULT,
            FailureType.MISSING_FIELD,
        )

    @classmethod
    def classify_exception(cls, exc: Exception, context: str = "") -> FailureAnalysis:
        """Classify an unexpected exception into a structured failure analysis."""
        msg = str(exc)
        msg_lower = msg.lower()

        if "timeout" in msg_lower or "timed out" in msg_lower:
            return FailureAnalysis(
                is_system_failure=True,
                failure_type=FailureType.LLM_TIMEOUT,
                severity=FailureSeverity.MEDIUM,
                error_message=msg,
                suggested_recovery="switch_provider",
            )
        elif "not reachable" in msg_lower or "connection refused" in msg_lower or "failed" in msg_lower:
            return FailureAnalysis(
                is_system_failure=True,
                failure_type=FailureType.PROVIDER_UNAVAILABLE,
                severity=FailureSeverity.MEDIUM,
                error_message=msg,
                suggested_recovery="switch_provider",
            )
        elif "json" in msg_lower or "decode" in msg_lower or "malformed" in msg_lower:
            return FailureAnalysis(
                is_system_failure=True,
                failure_type=FailureType.MALFORMED_RESPONSE,
                severity=FailureSeverity.LOW,
                error_message=msg,
                suggested_recovery="repair_or_retry",
            )
        elif "oasis" in msg_lower:
            return FailureAnalysis(
                is_system_failure=True,
                failure_type=FailureType.OASIS_ERROR,
                severity=FailureSeverity.HIGH,
                error_message=msg,
                suggested_recovery="retry_simulation",
            )
        else:
            return FailureAnalysis(
                is_system_failure=True,
                failure_type=FailureType.TOOL_EXCEPTION,
                severity=FailureSeverity.MEDIUM,
                error_message=msg,
                suggested_recovery="retry_or_replan",
            )

    @classmethod
    def validate_and_repair_json(
        cls,
        raw_text: str,
        required_keys: Optional[List[str]] = None,
        default_fallback: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """Attempt to parse, validate, and heuristically repair malformed JSON payloads."""
        if not raw_text or not raw_text.strip():
            return False, default_fallback, "Empty response received"

        # 1. Strip think tags and codeblocks
        cleaned = re.sub(r'<think>[\s\S]*?</think>', '', raw_text).strip()
        code_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned, re.IGNORECASE)
        if code_match:
            cleaned = code_match.group(1).strip()

        # 2. Direct parse attempt
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                is_valid, err = cls._check_required_keys(parsed, required_keys)
                if is_valid:
                    return True, parsed, None
                # Try filling defaults
                repaired = cls._fill_missing_keys(parsed, required_keys, default_fallback)
                return True, repaired, f"Repaired missing fields: {err}"
        except json.JSONDecodeError:
            pass

        # 3. Outer boundary slice { ... }
        start = cleaned.find('{')
        end = cleaned.rfind('}')
        if start != -1 and end > start:
            snippet = cleaned[start:end + 1]
            try:
                parsed = json.loads(snippet)
                if isinstance(parsed, dict):
                    repaired = cls._fill_missing_keys(parsed, required_keys, default_fallback)
                    return True, repaired, "Repaired via boundary extraction"
            except json.JSONDecodeError:
                pass

        # 4. Regex key-value extraction heuristic
        recovered = {}
        if required_keys:
            for key in required_keys:
                # Match "key": value or "key": "value"
                pattern = rf'"{re.escape(key)}"\s*:\s*([^,\n\}}]+)'
                match = re.search(pattern, cleaned)
                if match:
                    val_str = match.group(1).strip().strip('"').strip("'")
                    try:
                        if val_str.lower() in ("true", "false"):
                            recovered[key] = (val_str.lower() == "true")
                        elif re.match(r'^-?\d+(\.\d+)?$', val_str):
                            recovered[key] = float(val_str) if '.' in val_str else int(val_str)
                        else:
                            recovered[key] = val_str
                    except Exception:
                        recovered[key] = val_str

        if recovered:
            repaired = cls._fill_missing_keys(recovered, required_keys, default_fallback)
            return True, repaired, "Repaired via regex heuristic extraction"

        if default_fallback:
            return False, default_fallback, "Unrecoverable payload, returned default fallback"

        return False, None, "Failed to parse or repair JSON payload"

    @staticmethod
    def _check_required_keys(payload: Dict[str, Any], required_keys: Optional[List[str]]) -> Tuple[bool, Optional[str]]:
        if not required_keys:
            return True, None
        missing = [k for k in required_keys if k not in payload]
        if missing:
            return False, f"Missing required keys: {', '.join(missing)}"
        return True, None

    @staticmethod
    def _fill_missing_keys(
        payload: Dict[str, Any],
        required_keys: Optional[List[str]],
        default_fallback: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        result = dict(payload)
        if required_keys:
            for key in required_keys:
                if key not in result:
                    if default_fallback and key in default_fallback:
                        result[key] = default_fallback[key]
                    else:
                        result[key] = None
        return result
