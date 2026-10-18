"""Tests for experiment selector and Value of Information ranking."""

import pytest

from decision_stable_causal_attribution.decision.optimizer import (
    BudgetOptimizer,
    ChannelSpec,
)
from decision_stable_causal_attribution.experiments.selector import (
    ExperimentSelector,
)


@pytest.fixture
def experiment_setup():
    optimizer = BudgetOptimizer(total_budget=10_000.0)
    channels = [
        ChannelSpec(
            channel_id="ch_search",
            effect_estimate=0.12,
            cost_per_exposure=1.5,
            saturation=0.1,
            effect_lower=0.08,
            effect_upper=0.16,
        ),
        ChannelSpec(
            channel_id="ch_social",
            effect_estimate=0.07,
            cost_per_exposure=0.8,
            saturation=0.1,
            effect_lower=0.02,
            effect_upper=0.12,
        ),
    ]
    intervals = {
        "ch_search": (0.08, 0.16),
        "ch_social": (0.02, 0.12),
    }
    return optimizer, channels, intervals


def test_experiment_ranking_voi(experiment_setup):
    """Verify experiment candidates are generated and ranked by VOI."""
    optimizer, channels, intervals = experiment_setup
    selector = ExperimentSelector(optimizer, budget=10_000.0)

    candidates = selector.rank_experiments(
        channels=channels,
        channel_intervals=intervals,
        sample_size=2_000,
        policy="voi_optimal",
    )

    assert len(candidates) == 2
    for c in candidates:
        assert c.sample_size == 2_000
        assert c.experiment_cost > 0.0
        assert c.value_of_information.value is not None
        assert c.selection_method == "voi_optimal"

    # Sorted descending by VOI
    assert candidates[0].value_of_information.value >= candidates[1].value_of_information.value
