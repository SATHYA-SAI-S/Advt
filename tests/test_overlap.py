"""Tests for overlap and identifiability diagnostics."""

from decision_stable_causal_attribution.diagnostics.overlap import (
    OverlapDiagnosticEngine,
)
from decision_stable_causal_attribution.estimators.aipw import AIPWEstimator
from decision_stable_causal_attribution.simulator.config import (
    ChannelConfig,
    SimulationScenario,
    SimulatorConfig,
)
from decision_stable_causal_attribution.simulator.generator import (
    SyntheticJourneyGenerator,
)


def make_bundle(scenario: SimulationScenario, seed: int = 42):
    channels = (
        ChannelConfig(
            channel_id="ch_target",
            exposure_probability=0.3,
            treatment_effect=0.08,
            cost_per_exposure=1.0,
            saturation=0.1,
            confounding_sensitivity=0.9,
        ),
        ChannelConfig(
            channel_id="ch_other",
            exposure_probability=0.3,
            treatment_effect=0.05,
            cost_per_exposure=0.8,
            saturation=0.1,
            confounding_sensitivity=0.5,
        ),
    )
    cfg = SimulatorConfig(
        scenario=scenario,
        random_seed=seed,
        population_size=1000,
        hidden_intent_strength=0.8,
        proxy_quality=0.5,
        baseline_conversion_probability=0.03,
        overlap_strength=0.8 if scenario != SimulationScenario.POOR_OVERLAP else 0.05,
        conversion_saturation=0.8,
        channels=channels,
    )
    return SyntheticJourneyGenerator(cfg).generate()


def test_overlap_diagnostics_good_vs_poor():
    """Verify that poor overlap scenario triggers overlap_flag=True."""
    engine = OverlapDiagnosticEngine(extreme_threshold=0.05, min_ess_ratio=0.10)
    aipw = AIPWEstimator(n_splits=3, random_state=42)

    # 1. Mild confounding (Good overlap)
    mild_bundle = make_bundle(SimulationScenario.MILD_CONFOUNDING)
    mild_res = aipw.estimate_channel(mild_bundle, "ch_target")
    mild_diag = engine.evaluate(mild_bundle, "ch_target", mild_res.propensity_scores)
    assert mild_diag.effective_sample_size.value > 100
    assert mild_diag.overlap_flag is False

    # 2. Poor overlap scenario
    poor_bundle = make_bundle(SimulationScenario.POOR_OVERLAP)
    poor_res = aipw.estimate_channel(poor_bundle, "ch_target")
    poor_diag = engine.evaluate(poor_bundle, "ch_target", poor_res.propensity_scores)
    assert poor_diag.overlap_flag is True
    assert len(poor_diag.warnings) > 0
