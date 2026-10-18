"""Tests for decision engine, budget allocation, and stability breakdown."""

import pytest

from decision_stable_causal_attribution.decision.optimizer import (
    BudgetOptimizer,
    ChannelSpec,
)
from decision_stable_causal_attribution.schemas.results import StabilityLabel


@pytest.fixture
def test_channels():
    return [
        ChannelSpec(
            channel_id="ch_search",
            effect_estimate=0.15,
            cost_per_exposure=1.5,
            saturation=0.1,
            effect_lower=0.10,
            effect_upper=0.20,
        ),
        ChannelSpec(
            channel_id="ch_social",
            effect_estimate=0.08,
            cost_per_exposure=0.8,
            saturation=0.12,
            effect_lower=0.03,
            effect_upper=0.12,
        ),
        ChannelSpec(
            channel_id="ch_display",
            effect_estimate=0.02,
            cost_per_exposure=0.5,
            saturation=0.15,
            effect_lower=-0.01,
            effect_upper=0.05,
        ),
    ]


def test_budget_optimizer_point_estimate(test_channels):
    """Verify point estimate allocation sums to total budget and prefers higher ROI."""
    optimizer = BudgetOptimizer(total_budget=10_000.0)
    allocations = optimizer.optimize_point_estimate(test_channels)

    assert len(allocations) == 3
    total_alloc = sum(a.budget for a in allocations)
    assert pytest.approx(total_alloc, abs=1.0) == 10_000.0

    alloc_map = {a.channel_id: a.budget for a in allocations}
    # ch_search has highest effect and good efficiency
    assert alloc_map["ch_search"] > alloc_map["ch_display"]


def test_decision_evaluation_and_stability(test_channels):
    """Verify decision evaluation returns valid BudgetDecision schema and stability."""
    optimizer = BudgetOptimizer(total_budget=5_000.0)
    gamma_grid = [1.0, 1.5, 2.0, 2.5]
    bounds_by_gamma = {
        g: {
            "ch_search": (0.15 - 0.03 * (g - 1), 0.15 + 0.03 * (g - 1)),
            "ch_social": (0.08 - 0.02 * (g - 1), 0.08 + 0.02 * (g - 1)),
            "ch_display": (0.02 - 0.02 * (g - 1), 0.02 + 0.02 * (g - 1)),
        }
        for g in gamma_grid
    }

    decision = optimizer.evaluate_decision(
        channels=test_channels,
        strategy="point_estimate",
        overlap_flag=False,
        gamma_grid=gamma_grid,
        channel_bounds_by_gamma=bounds_by_gamma,
    )

    assert decision.strategy == "point_estimate"
    assert decision.total_budget == 5_000.0
    assert decision.stability in [StabilityLabel.STABLE, StabilityLabel.FRAGILE]
    assert decision.breakdown_gamma.value is not None


def test_decision_unidentifiable_on_overlap_flag(test_channels):
    """Verify that overlap_flag=True forces stability to UNIDENTIFIABLE."""
    optimizer = BudgetOptimizer(total_budget=5_000.0)
    decision = optimizer.evaluate_decision(
        channels=test_channels,
        strategy="point_estimate",
        overlap_flag=True,
    )
    assert decision.stability == StabilityLabel.UNIDENTIFIABLE
