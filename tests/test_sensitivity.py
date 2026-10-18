"""Tests for sensitivity engine and bounds."""

import pytest

from decision_stable_causal_attribution.estimators.aipw import AIPWEstimator
from decision_stable_causal_attribution.sensitivity.bounds import SensitivityEngine
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
    channels = (
        ChannelConfig(
            channel_id="ch_search",
            exposure_probability=0.35,
            treatment_effect=0.08,
            cost_per_exposure=1.2,
            saturation=0.1,
            confounding_sensitivity=0.6,
        ),
        ChannelConfig(
            channel_id="ch_display",
            exposure_probability=0.25,
            treatment_effect=0.04,
            cost_per_exposure=0.6,
            saturation=0.1,
            confounding_sensitivity=0.4,
        ),
    )
    cfg = SimulatorConfig(
        scenario=SimulationScenario.MILD_CONFOUNDING,
        random_seed=42,
        population_size=1000,
        hidden_intent_strength=0.5,
        proxy_quality=0.6,
        baseline_conversion_probability=0.04,
        overlap_strength=0.75,
        conversion_saturation=0.85,
        channels=channels,
    )
    return SyntheticJourneyGenerator(cfg).generate()


def test_sensitivity_bounds_expand_with_gamma(sim_bundle):
    """Verify that sensitivity intervals expand monotonically as Gamma increases."""
    engine = SensitivityEngine(gamma_min=1.0, gamma_max=2.5, gamma_step=0.5)
    aipw = AIPWEstimator(n_splits=3, random_state=42)
    aipw_res = aipw.estimate_channel(sim_bundle, "ch_search")

    analysis = engine.compute_bounds(sim_bundle, "ch_search", aipw_res)
    assert len(analysis.points) == 4
    assert analysis.points[0].gamma == 1.0

    # At Gamma=1.0, lower == upper == estimate
    assert pytest.approx(analysis.points[0].lower_bound, rel=1e-3) == aipw_res.ate
    assert pytest.approx(analysis.points[0].upper_bound, rel=1e-3) == aipw_res.ate

    # Interval width strictly increases
    widths = [p.upper_bound - p.lower_bound for p in analysis.points]
    assert all(w2 > w1 for w1, w2 in zip(widths, widths[1:], strict=False))


def test_covariate_benchmarks(sim_bundle):
    """Verify covariate benchmarking produces estimates and positive implied Gamma."""
    engine = SensitivityEngine()
    benchmarks = engine.benchmark_covariates(sim_bundle, "ch_search")

    assert len(benchmarks) > 0
    for b in benchmarks:
        assert b.covariate_name in ["age_group", "device_mobile", "intent_proxy"]
        assert b.implied_gamma >= 1.0
