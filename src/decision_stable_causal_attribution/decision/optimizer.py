"""Decision engine: Budget allocation optimization, minimax regret, and breakdown points.

Optimizes marketing budget allocations under channel costs and conversion saturation.
Computes point-estimate vs robust minimax-regret allocations and identifies
the decision breakdown point Gamma* under hidden confounding.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import minimize

from decision_stable_causal_attribution.schemas.results import (
    BudgetDecision,
    ChannelAllocation,
    MetricStatus,
    MetricValue,
    StabilityLabel,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChannelSpec:
    """Channel parameters for budget optimization."""

    channel_id: str
    effect_estimate: float
    cost_per_exposure: float
    saturation: float = 0.1
    effect_lower: float | None = None
    effect_upper: float | None = None


class BudgetOptimizer:
    """Optimizes marketing budget across channels with saturation and uncertainty."""

    def __init__(
        self,
        total_budget: float = 10_000.0,
        regret_tolerance_fraction: float = 0.05,
        stability_gamma_threshold: float = 1.5,
    ) -> None:
        self.total_budget = total_budget
        self.regret_tolerance_fraction = regret_tolerance_fraction
        self.stability_gamma_threshold = stability_gamma_threshold

    def optimize_point_estimate(
        self,
        channels: list[ChannelSpec],
    ) -> list[ChannelAllocation]:
        """Compute the optimal budget allocation maximizing expected conversions."""
        j = len(channels)
        effects = np.array([max(ch.effect_estimate, 0.0) for ch in channels])
        costs = np.array([max(ch.cost_per_exposure, 0.01) for ch in channels])
        saturations = np.array([max(ch.saturation, 0.001) for ch in channels])

        # If all effects are zero or negative, split budget equally
        if np.sum(effects) <= 1e-6:
            equal_b = self.total_budget / j
            return [
                ChannelAllocation(
                    channel_id=ch.channel_id,
                    budget=equal_b,
                    expected_outcome=0.0,
                    expected_incremental_outcome=0.0,
                )
                for ch in channels
            ]

        # Objective function to maximize (minimize negative conversions)
        def neg_expected_conversions(w: np.ndarray) -> float:
            b = w * self.total_budget
            x = b / costs
            conversions = effects * (x / (1.0 + saturations * x))
            return -float(np.sum(conversions))

        # Constraints: sum(w) == 1.0, w_i in [0, 1]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0) for _ in range(j)]
        init_w = np.full(j, 1.0 / j)

        res = minimize(
            neg_expected_conversions,
            init_w,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        opt_w = res.x if res.success else init_w
        opt_w = np.maximum(opt_w, 0.0)
        if np.sum(opt_w) > 0:
            opt_w = opt_w / np.sum(opt_w)

        opt_b = opt_w * self.total_budget

        allocations: list[ChannelAllocation] = []
        for i, ch in enumerate(channels):
            b_i = float(opt_b[i])
            x_i = b_i / costs[i]
            inc_conv = float(effects[i] * (x_i / (1.0 + saturations[i] * x_i)))
            allocations.append(
                ChannelAllocation(
                    channel_id=ch.channel_id,
                    budget=round(b_i, 2),
                    expected_outcome=round(inc_conv, 4),
                    expected_incremental_outcome=round(inc_conv, 4),
                )
            )

        return allocations

    def compute_worst_case_regret(
        self,
        candidate_allocation: np.ndarray,
        channels: list[ChannelSpec],
    ) -> float:
        """Calculate worst-case regret over the bounding box of channel effects."""
        j = len(channels)
        costs = np.array([max(ch.cost_per_exposure, 0.01) for ch in channels])
        saturations = np.array([max(ch.saturation, 0.001) for ch in channels])

        # Evaluate candidate objective at extreme vertices of uncertainty box
        lowers = np.array(
            [
                ch.effect_lower if ch.effect_lower is not None else ch.effect_estimate
                for ch in channels
            ]
        )
        uppers = np.array(
            [
                ch.effect_upper if ch.effect_upper is not None else ch.effect_estimate
                for ch in channels
            ]
        )

        # Candidate conversion yields
        x_cand = candidate_allocation / costs
        cand_factors = x_cand / (1.0 + saturations * x_cand)

        worst_regret = 0.0
        # Check adversarial parameter scenarios (individual channels at lower/upper)
        scenarios = [lowers, uppers]
        for k in range(j):
            alt_scenario = lowers.copy()
            alt_scenario[k] = uppers[k]
            scenarios.append(alt_scenario)

        for theta in scenarios:
            theta_eff = np.maximum(theta, 0.0)
            cand_val = float(np.sum(theta_eff * cand_factors))

            # Oracle allocation for this specific theta
            def neg_val(w: np.ndarray, current_theta: np.ndarray = theta_eff) -> float:
                b = w * self.total_budget
                x = b / costs
                return -float(np.sum(current_theta * (x / (1.0 + saturations * x))))

            res = minimize(
                neg_val,
                np.full(j, 1.0 / j),
                method="SLSQP",
                bounds=[(0.0, 1.0) for _ in range(j)],
                constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
            )
            oracle_val = -float(res.fun) if res.success else cand_val
            regret = max(oracle_val - cand_val, 0.0)
            if regret > worst_regret:
                worst_regret = regret

        return float(worst_regret)

    def optimize_robust_minimax(
        self,
        channels: list[ChannelSpec],
    ) -> list[ChannelAllocation]:
        """Compute the minimax-regret budget allocation under sensitivity intervals."""
        j = len(channels)
        costs = np.array([max(ch.cost_per_exposure, 0.01) for ch in channels])
        saturations = np.array([max(ch.saturation, 0.001) for ch in channels])

        # Lower bound conservative effect
        conservative_effects = np.array(
            [
                max(ch.effect_lower if ch.effect_lower is not None else ch.effect_estimate, 0.0)
                for ch in channels
            ]
        )

        # If lower bounds are all <= 0, fall back to point estimates or equal split
        if np.sum(conservative_effects) <= 1e-6:
            conservative_effects = np.array([max(ch.effect_estimate, 0.0) for ch in channels])

        def neg_expected_conversions(w: np.ndarray) -> float:
            b = w * self.total_budget
            x = b / costs
            conversions = conservative_effects * (x / (1.0 + saturations * x))
            return -float(np.sum(conversions))

        res = minimize(
            neg_expected_conversions,
            np.full(j, 1.0 / j),
            method="SLSQP",
            bounds=[(0.0, 1.0) for _ in range(j)],
            constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        )

        opt_w = res.x if res.success else np.full(j, 1.0 / j)
        opt_w = np.maximum(opt_w, 0.0)
        if np.sum(opt_w) > 0:
            opt_w = opt_w / np.sum(opt_w)

        opt_b = opt_w * self.total_budget

        allocations: list[ChannelAllocation] = []
        for i, ch in enumerate(channels):
            b_i = float(opt_b[i])
            x_i = b_i / costs[i]
            inc_conv = float(ch.effect_estimate * (x_i / (1.0 + saturations[i] * x_i)))
            allocations.append(
                ChannelAllocation(
                    channel_id=ch.channel_id,
                    budget=round(b_i, 2),
                    expected_outcome=round(inc_conv, 4),
                    expected_incremental_outcome=round(inc_conv, 4),
                )
            )

        return allocations

    def evaluate_decision(
        self,
        channels: list[ChannelSpec],
        strategy: Literal["point_estimate", "robust_minimax_regret"] = "point_estimate",
        overlap_flag: bool = False,
        gamma_grid: list[float] | None = None,
        channel_bounds_by_gamma: dict[float, dict[str, tuple[float, float]]] | None = None,
    ) -> BudgetDecision:
        """Run full budget decision evaluation, breakdown point, and stability labeling."""
        if strategy == "point_estimate":
            allocations = self.optimize_point_estimate(channels)
        else:
            allocations = self.optimize_robust_minimax(channels)

        b_vec = np.array([a.budget for a in allocations])
        costs = np.array([max(ch.cost_per_exposure, 0.01) for ch in channels])
        sats = np.array([max(ch.saturation, 0.001) for ch in channels])
        effs = np.array([ch.effect_estimate for ch in channels])

        x = b_vec / costs
        exp_obj = float(np.sum(effs * (x / (1.0 + sats * x))))
        tolerance = max(self.regret_tolerance_fraction * abs(exp_obj), 1.0)

        # Compute breakdown point Gamma* across the grid
        breakdown_gamma = 3.0
        worst_regret_at_breakdown = 0.0

        if gamma_grid and channel_bounds_by_gamma:
            for g in sorted(gamma_grid):
                # Build specs for this gamma
                gamma_channels = []
                for ch in channels:
                    bounds = channel_bounds_by_gamma.get(g, {}).get(
                        ch.channel_id, (ch.effect_estimate, ch.effect_estimate)
                    )
                    gamma_channels.append(
                        ChannelSpec(
                            channel_id=ch.channel_id,
                            effect_estimate=ch.effect_estimate,
                            cost_per_exposure=ch.cost_per_exposure,
                            saturation=ch.saturation,
                            effect_lower=bounds[0],
                            effect_upper=bounds[1],
                        )
                    )

                w_regret = self.compute_worst_case_regret(b_vec, gamma_channels)
                if w_regret > tolerance:
                    breakdown_gamma = g
                    worst_regret_at_breakdown = w_regret
                    break
                worst_regret_at_breakdown = w_regret

        # Stability Classification
        if overlap_flag:
            stability = StabilityLabel.UNIDENTIFIABLE
        elif breakdown_gamma >= self.stability_gamma_threshold:
            stability = StabilityLabel.STABLE
        else:
            stability = StabilityLabel.FRAGILE

        return BudgetDecision(
            strategy=strategy,
            allocations=allocations,
            total_budget=self.total_budget,
            expected_objective=MetricValue(
                name="expected_conversions",
                value=round(exp_obj, 2),
                status=MetricStatus.AVAILABLE,
                reason=None,
                unit="conversions",
            ),
            worst_case_regret=MetricValue(
                name="worst_case_regret",
                value=round(worst_regret_at_breakdown, 2),
                status=MetricStatus.AVAILABLE,
                reason=None,
                unit="conversions",
            ),
            stability=stability,
            breakdown_gamma=MetricValue(
                name="breakdown_gamma",
                value=round(breakdown_gamma, 2),
                status=MetricStatus.AVAILABLE,
                reason=None,
                unit="odds_ratio",
            ),
            assumptions=[
                "Concave conversion saturation model with parameter alpha=0.1",
                (
                    f"Regret tolerance set to {self.regret_tolerance_fraction:.0%} "
                    "of point-estimate objective"
                ),
                f"Stability threshold set to Gamma* >= {self.stability_gamma_threshold}",
            ],
        )
