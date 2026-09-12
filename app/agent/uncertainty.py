"""Uncertainty Quantification and Multi-Run Simulation Aggregation.

Provides statistical aggregation across multiple simulation runs for a scenario,
including mean, variance, standard error, confidence intervals, and stability metrics.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger

logger = get_logger('mirofish.agent.uncertainty')


@dataclass
class DimensionStats:
    """Statistical summary for a single metric dimension."""
    mean: float = 0.0
    std_dev: float = 0.0
    variance: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    confidence_interval_95: List[float] = field(default_factory=lambda: [0.0, 0.0])
    is_stable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MultiRunMetrics:
    """Aggregated statistics for multiple simulation runs of a scenario."""
    scenario_id: str
    scenario_name: str
    run_count: int
    dimension_stats: Dict[str, DimensionStats] = field(default_factory=dict)
    overall_stability: float = 1.0  # 0 to 1, higher is more stable
    requires_more_simulations: bool = False
    recommendation: str = ""
    calculated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["dimension_stats"] = {k: v.to_dict() for k, v in self.dimension_stats.items()}
        return data


class UncertaintyQuantifier:
    """Calculates multi-run uncertainty and aggregation."""

    STABILITY_THRESHOLD_STDDEV = 0.15

    DIMENSIONS = [
        "acceptance_score",
        "consensus_score",
        "polarization_score",
        "conflict_score",
        "equity_score",
        "adoption_score",
        "stability_score",
        "overall_score",
    ]

    @classmethod
    def aggregate_runs(
        cls,
        scenario_id: str,
        scenario_name: str,
        run_results: List[Dict[str, float]],
    ) -> MultiRunMetrics:
        """Compute statistical aggregation across multiple simulation run results."""
        run_count = len(run_results)
        if run_count == 0:
            return MultiRunMetrics(
                scenario_id=scenario_id,
                scenario_name=scenario_name,
                run_count=0,
                requires_more_simulations=True,
                recommendation="No runs available. Simulation required.",
            )

        dimension_stats: Dict[str, DimensionStats] = {}
        high_variance_dims = []

        for dim in cls.DIMENSIONS:
            values = [r.get(dim, 0.0) for r in run_results if dim in r]
            if not values:
                continue

            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0.0
            var_val = statistics.variance(values) if len(values) > 1 else 0.0
            min_val = min(values)
            max_val = max(values)

            # 95% confidence interval using normal approximation
            se = (std_val / math.sqrt(len(values))) if len(values) > 1 else 0.0
            ci_low = max(0.0, mean_val - 1.96 * se)
            ci_high = min(1.0, mean_val + 1.96 * se)

            is_stable = std_val <= cls.STABILITY_THRESHOLD_STDDEV
            if not is_stable:
                high_variance_dims.append(dim)

            dimension_stats[dim] = DimensionStats(
                mean=round(mean_val, 4),
                std_dev=round(std_val, 4),
                variance=round(var_val, 4),
                min_val=round(min_val, 4),
                max_val=round(max_val, 4),
                confidence_interval_95=[round(ci_low, 4), round(ci_high, 4)],
                is_stable=is_stable,
            )

        # Overall stability is 1 minus average std_dev
        avg_std = statistics.mean([s.std_dev for s in dimension_stats.values()]) if dimension_stats else 0.0
        overall_stability = max(0.0, min(1.0, 1.0 - (avg_std * 2.0)))
        
        requires_more = run_count < 3 or len(high_variance_dims) >= 2

        if requires_more:
            recommendation = (
                f"Outcome exhibits high variance across {len(high_variance_dims)} dimension(s) "
                f"({', '.join(high_variance_dims[:2])}). Additional simulation runs recommended."
            )
        else:
            recommendation = (
                f"Outcome is statistically stable across {run_count} runs (stability: {overall_stability:.2f})."
            )

        logger.info(f"Uncertainty aggregated for {scenario_id}: {run_count} runs, stability={overall_stability:.2f}")

        return MultiRunMetrics(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            run_count=run_count,
            dimension_stats=dimension_stats,
            overall_stability=round(overall_stability, 3),
            requires_more_simulations=requires_more,
            recommendation=recommendation,
        )
