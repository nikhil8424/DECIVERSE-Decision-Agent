"""SocialImpactModel-Integrated Verifier.

Reconciles user-defined goal constraints directly against SocialImpactModel results.
Does NOT create a competing scoring system; directly consumes existing impact dimensions.
Supports VERIFIED, FAILED, and UNCERTAIN states.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..research.social_impact_model import SocialImpactResult
from ..utils.logger import get_logger

logger = get_logger('mirofish.agent.verifier')


class VerificationState(str, Enum):
    """Possible outcomes of a verification check."""
    VERIFIED = "VERIFIED"    # All critical constraints satisfied
    FAILED = "FAILED"        # One or more constraints violated with reliable data
    UNCERTAIN = "UNCERTAIN"  # Evidence missing or sample size / confidence too low


@dataclass
class ConstraintRule:
    """Parsed constraint rule."""
    dimension: str
    operator: str  # <, <=, >, >=, ==, !=
    threshold: float
    raw_expression: str


@dataclass
class ConstraintResult:
    """Detailed verification outcome."""
    status: VerificationState
    scenario_id: str
    scenario_name: str
    satisfied_constraints: List[str] = field(default_factory=list)
    violated_constraints: List[str] = field(default_factory=list)
    uncertain_reasons: List[str] = field(default_factory=list)
    impact_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    confidence: float = 0.0
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "satisfied_constraints": self.satisfied_constraints,
            "violated_constraints": self.violated_constraints,
            "uncertain_reasons": self.uncertain_reasons,
            "impact_scores": self.impact_scores,
            "overall_score": self.overall_score,
            "confidence": self.confidence,
            "summary": self.summary,
        }


class Verifier:
    """Evaluates SocialImpactModel results against user-defined constraints."""

    # Normalization map for common alias keys
    KEY_MAP = {
        "acceptance": "acceptance_score",
        "acceptance_score": "acceptance_score",
        "consensus": "consensus_score",
        "consensus_score": "consensus_score",
        "polarization": "polarization_score",
        "polarization_score": "polarization_score",
        "conflict": "conflict_score",
        "conflict_score": "conflict_score",
        "equity": "equity_score",
        "equity_score": "equity_score",
        "adoption": "adoption_score",
        "adoption_score": "adoption_score",
        "stability": "stability_score",
        "stability_score": "stability_score",
        "overall": "overall_score",
        "overall_score": "overall_score",
    }

    @classmethod
    def parse_constraint(cls, key: str, expr: str) -> Optional[ConstraintRule]:
        """Parse a constraint expression (e.g. key='polarization', expr='<0.35' or expr='polarization < 0.35')."""
        clean_expr = expr.strip()
        
        # Check if key is already inside expr (e.g. "polarization < 0.35")
        match = re.match(r'^([a-zA-Z_]+)\s*(<=|>=|<|>|==|!=)\s*([0-9\.]+)$', clean_expr)
        if match:
            dim_name, op, thresh_str = match.groups()
            norm_dim = cls.KEY_MAP.get(dim_name.lower(), dim_name.lower())
            return ConstraintRule(
                dimension=norm_dim,
                operator=op,
                threshold=float(thresh_str),
                raw_expression=clean_expr,
            )
        
        # Match only operator and number (e.g. "<0.35")
        op_match = re.match(r'^(<=|>=|<|>|==|!=)\s*([0-9\.]+)$', clean_expr)
        if op_match:
            op, thresh_str = op_match.groups()
            norm_dim = cls.KEY_MAP.get(key.lower(), key.lower())
            return ConstraintRule(
                dimension=norm_dim,
                operator=op,
                threshold=float(thresh_str),
                raw_expression=f"{key} {op} {thresh_str}",
            )

        logger.warning(f"Could not parse constraint rule: '{key}': '{expr}'")
        return None

    @classmethod
    def evaluate(
        cls,
        impact_result: SocialImpactResult | Dict[str, Any],
        constraints: Dict[str, str],
        min_confidence_threshold: float = 0.05,
    ) -> ConstraintResult:
        """Evaluate SocialImpactModel metrics against constraints.
        
        Args:
            impact_result: Output from SocialImpactModel
            constraints: User constraints dictionary
            min_confidence_threshold: Minimum confidence required before flagging as UNCERTAIN
            
        Returns:
            ConstraintResult with state (VERIFIED, FAILED, UNCERTAIN)
        """
        # Convert to dict if object
        if isinstance(impact_result, SocialImpactResult):
            data = impact_result.to_dict()
        else:
            data = dict(impact_result)

        scenario_id = data.get("scenario_id", "unknown_scenario")
        scenario_name = data.get("scenario_name", scenario_id)
        confidence = float(data.get("confidence", 0.0))
        overall_score = float(data.get("overall_score", 0.0))

        # Extract numeric dimension scores
        scores = {
            "acceptance_score": float(data.get("acceptance_score", 0.0)),
            "consensus_score": float(data.get("consensus_score", 0.0)),
            "polarization_score": float(data.get("polarization_score", 0.0)),
            "conflict_score": float(data.get("conflict_score", 0.0)),
            "equity_score": float(data.get("equity_score", 0.0)),
            "adoption_score": float(data.get("adoption_score", 0.0)),
            "stability_score": float(data.get("stability_score", 0.0)),
            "overall_score": overall_score,
        }

        satisfied: List[str] = []
        violated: List[str] = []
        uncertain_reasons: List[str] = []

        # Check for uncertain evidence conditions
        if not data or all(v == 0.0 for v in scores.values()):
            uncertain_reasons.append("Simulation data missing or empty; metrics could not be computed.")

        if confidence < min_confidence_threshold and len(uncertain_reasons) == 0:
            uncertain_reasons.append(f"Confidence is too low ({confidence:.2f} < {min_confidence_threshold:.2f}) due to insufficient sample size.")

        # Evaluate individual constraints
        for key, expr in constraints.items():
            rule = cls.parse_constraint(key, expr)
            if not rule:
                uncertain_reasons.append(f"Unparseable constraint rule: {key}={expr}")
                continue

            dim_val = scores.get(rule.dimension)
            if dim_val is None:
                uncertain_reasons.append(f"Metric '{rule.dimension}' not found in simulation impact results.")
                continue

            passed = cls._check_rule(dim_val, rule.operator, rule.threshold)
            short_dim = rule.dimension.replace("_score", "")
            
            if passed:
                satisfied.append(f"{short_dim} ({dim_val:.2f} {rule.operator} {rule.threshold:.2f}) [PASS]")
            else:
                violated.append(f"{short_dim} ({dim_val:.2f} fails {rule.operator} {rule.threshold:.2f}) [FAIL]")

        # Determine final status
        if uncertain_reasons:
            status = VerificationState.UNCERTAIN
            summary = f"UNCERTAIN: {len(uncertain_reasons)} uncertainty condition(s) detected: {'; '.join(uncertain_reasons)}"
        elif violated:
            status = VerificationState.FAILED
            summary = f"FAILED: {len(violated)} constraint(s) violated: {', '.join(violated)}"
        else:
            status = VerificationState.VERIFIED
            summary = f"VERIFIED: All {len(satisfied)} constraint(s) satisfied with overall viability score of {overall_score:.2f}."

        logger.info(f"Verification result for {scenario_id}: {status.value} (Satisfied={len(satisfied)}, Violated={len(violated)})")

        return ConstraintResult(
            status=status,
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            satisfied_constraints=satisfied,
            violated_constraints=violated,
            uncertain_reasons=uncertain_reasons,
            impact_scores=scores,
            overall_score=overall_score,
            confidence=confidence,
            summary=summary,
        )

    @staticmethod
    def _check_rule(val: float, op: str, threshold: float) -> bool:
        """Evaluate numeric relational operator."""
        if op == "<":
            return val < threshold
        elif op == "<=":
            return val <= threshold
        elif op == ">":
            return val > threshold
        elif op == ">=":
            return val >= threshold
        elif op == "==":
            return abs(val - threshold) < 1e-5
        elif op == "!=":
            return abs(val - threshold) >= 1e-5
        return False
