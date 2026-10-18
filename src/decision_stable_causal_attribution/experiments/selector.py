"""Experiment Selector and Value of Information (VOI) ranking.

Identifies which channels or segments to prioritize for randomized holdouts/experiments
by maximizing Expected Decision Regret Reduction minus Experimentation Cost.
Contrasts against heuristic baselines (widest interval, largest spend, random).
"""

import logging
from dataclasses import dataclass
from math import sqrt
from typing import Literal

import numpy as np

from decision_stable_causal_attribution.decision.optimizer import BudgetOptimizer, ChannelSpec
from decision_stable_causal_attribution.schemas.results import (
    ExperimentCandidate,
    MetricStatus,
    MetricValue,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentDesign:
    """Design parameters for a candidate experiment on a marketing channel."""

    channel_id: str
    sample_size: int
    cost_per_subject: float = 0.50
    segment_id: str | None = None


class ExperimentSelector:
    """Prioritizes experiments using Value of Information (VOI) analysis."""

    def __init__(
        self,
        optimizer: BudgetOptimizer,
        budget: float = 10_000.0,
    ) -> None:
        self.optimizer = optimizer
        self.budget = budget

    def rank_experiments(
        self,
        channels: list[ChannelSpec],
        channel_intervals: dict[str, tuple[float, float]],
        sample_size: int = 5_000,
        policy: Literal[
            "voi_optimal", "widest_interval", "largest_spend", "random"
        ] = "voi_optimal",
        random_seed: int = 42,
    ) -> list[ExperimentCandidate]:
        """Generate and rank candidate experiments across channels."""
        # 1. Base allocation and initial worst-case regret
        base_allocations = self.optimizer.optimize_point_estimate(channels)
        b_vec = np.array([a.budget for a in base_allocations])
        initial_regret = self.optimizer.compute_worst_case_regret(b_vec, channels)

        candidates: list[ExperimentCandidate] = []
        rng = np.random.default_rng(random_seed)

        for ch in channels:
            interval = channel_intervals.get(
                ch.channel_id, (ch.effect_estimate, ch.effect_estimate)
            )
            width = interval[1] - interval[0]

            exp_cost = sample_size * 0.10  # Experiment execution cost

            # Simulated post-experiment precision gain (variance scales as 1/N)
            shrunk_width = width / sqrt(max(sample_size / 500.0, 1.0))
            center = ch.effect_estimate
            shrunk_lower = center - shrunk_width / 2.0
            shrunk_upper = center + shrunk_width / 2.0

            # Evaluate expected regret with shrunk uncertainty for this channel
            hypothetical_channels = [
                ChannelSpec(
                    channel_id=c.channel_id,
                    effect_estimate=c.effect_estimate,
                    cost_per_exposure=c.cost_per_exposure,
                    saturation=c.saturation,
                    effect_lower=shrunk_lower if c.channel_id == ch.channel_id else c.effect_lower,
                    effect_upper=shrunk_upper if c.channel_id == ch.channel_id else c.effect_upper,
                )
                for c in channels
            ]

            post_regret = self.optimizer.compute_worst_case_regret(b_vec, hypothetical_channels)
            regret_reduction = max(initial_regret - post_regret, 0.0)

            # Convert regret reduction to dollar/conversion value
            gross_value = regret_reduction * 100.0
            voi = gross_value - exp_cost

            candidates.append(
                ExperimentCandidate(
                    experiment_id=f"exp-{ch.channel_id}-n{sample_size}",
                    channel_id=ch.channel_id,
                    segment_id=None,
                    sample_size=sample_size,
                    experiment_cost=round(exp_cost, 2),
                    expected_regret_reduction=MetricValue(
                        name="expected_regret_reduction",
                        value=round(regret_reduction, 4),
                        status=MetricStatus.AVAILABLE,
                        reason=None,
                        unit="regret_units",
                    ),
                    value_of_information=MetricValue(
                        name="net_value_of_information",
                        value=round(voi, 2),
                        status=MetricStatus.AVAILABLE,
                        reason=None,
                        unit="currency",
                    ),
                    selection_method=policy,
                )
            )

        # Apply sorting policy
        if policy == "voi_optimal":
            candidates.sort(
                key=lambda c: (
                    c.value_of_information.value
                    if c.value_of_information.value is not None
                    else 0.0
                ),
                reverse=True,
            )
        elif policy == "widest_interval":
            candidates.sort(
                key=lambda c: (
                    channel_intervals.get(c.channel_id, (0.0, 0.0))[1]
                    - channel_intervals.get(c.channel_id, (0.0, 0.0))[0]
                ),
                reverse=True,
            )
        elif policy == "largest_spend":
            spend_map = {a.channel_id: a.budget for a in base_allocations}
            candidates.sort(key=lambda c: spend_map.get(c.channel_id, 0.0), reverse=True)
        elif policy == "random":
            rng.shuffle(candidates)

        return candidates
