"""Tests for AIPW causal treatment effect estimator."""

import pytest

from decision_stable_causal_attribution.estimators.aipw import AIPWEstimator
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
    """Generate a synthetic dataset bundle for AIPW tests."""
    channels = (
        ChannelConfig(
            channel_id="ch_search",
            exposure_probability=0.4,
            treatment_effect=0.10,
            cost_per_exposure=1.0,
            saturation=0.1,
            confounding_sensitivity=0.4,
        ),
        ChannelConfig(
            channel_id="ch_social",
            exposure_probability=0.3,
            treatment_effect=0.06,
            cost_per_exposure=0.7,
            saturation=0.1,
            confounding_sensitivity=0.4,
        ),
    )
    cfg = SimulatorConfig(
        scenario=SimulationScenario.MILD_CONFOUNDING,
        random_seed=42,
        population_size=1500,
        hidden_intent_strength=0.3,
        proxy_quality=0.7,
        baseline_conversion_probability=0.04,
        overlap_strength=0.8,
        conversion_saturation=0.9,
        channels=channels,
    )
    return SyntheticJourneyGenerator(cfg).generate()


def test_aipw_estimates_effects(sim_bundle):
    """Verify AIPW produces finite ATE, SE, and valid confidence intervals."""
    estimator = AIPWEstimator(n_splits=3, random_state=42)
    res = estimator.estimate_channel(sim_bundle, "ch_search")

    assert res.channel_id == "ch_search"
    assert -0.5 < res.ate < 0.5
    assert res.se > 0.0
    assert res.ci_lower < res.ci_upper
    assert len(res.propensity_scores) == 1500
    assert (res.propensity_scores >= 0.01).all()
    assert (res.propensity_scores <= 0.99).all()

    # Convert to schema
    effect_schema = estimator.to_effect_estimate(res)
    assert effect_schema.method == "aipw_doubly_robust"
    assert effect_schema.estimate == res.ate
