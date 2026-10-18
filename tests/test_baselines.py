"""Tests for attribution and associational baselines."""

import pytest

from decision_stable_causal_attribution.baselines.attribution import (
    AssociationalBaseline,
    HeuristicAttribution,
)
from decision_stable_causal_attribution.simulator.config import (
    ChannelConfig,
    SimulationScenario,
    SimulatorConfig,
)
from decision_stable_causal_attribution.simulator.generator import (
    SyntheticJourneyGenerator,
)


@pytest.fixture
def sim_bundle():
    """Generate a small synthetic dataset bundle for baseline tests."""
    channels = (
        ChannelConfig(
            channel_id="ch_search",
            exposure_probability=0.4,
            treatment_effect=0.10,
            cost_per_exposure=1.0,
            saturation=0.1,
            confounding_sensitivity=0.5,
        ),
        ChannelConfig(
            channel_id="ch_display",
            exposure_probability=0.3,
            treatment_effect=0.05,
            cost_per_exposure=0.5,
            saturation=0.1,
            confounding_sensitivity=0.5,
        ),
    )
    cfg = SimulatorConfig(
        scenario=SimulationScenario.MILD_CONFOUNDING,
        random_seed=42,
        population_size=1000,
        hidden_intent_strength=0.5,
        proxy_quality=0.5,
        baseline_conversion_probability=0.05,
        overlap_strength=0.8,
        conversion_saturation=0.9,
        channels=channels,
    )
    return SyntheticJourneyGenerator(cfg).generate()


def test_heuristic_attribution_rules(sim_bundle):
    """Verify first-touch, last-touch, and linear MTA attribution sums."""
    attrib = HeuristicAttribution(sim_bundle)

    for rule in ["first_touch", "last_touch", "linear"]:
        shares = attrib.compute(rule)
        assert len(shares) == 2
        total_share = sum(s.attribution_share for s in shares)
        if shares[0].total_conversions > 0:
            assert pytest.approx(total_share, rel=1e-3) == 1.0


def test_associational_baseline(sim_bundle):
    """Verify naive difference-in-means baseline produces valid EffectEstimate."""
    baseline = AssociationalBaseline(sim_bundle)
    estimate = baseline.compute_naive_ate("ch_search")

    assert estimate.treatment_id == "ch_search"
    assert estimate.estimand == "ate"
    assert estimate.method == "naive_difference_in_means"
    assert estimate.standard_error > 0.0
    assert estimate.confidence_interval is not None
    assert estimate.confidence_interval[0] <= estimate.confidence_interval[1]
