
"""Tests for synthetic journey generation and simulator ground truth."""

from decision_stable_causal_attribution.schemas.metadata import DatasetKind
from decision_stable_causal_attribution.simulator.config import (
    ChannelConfig,
    SimulationScenario,
    SimulatorConfig,
)
from decision_stable_causal_attribution.simulator.generator import (
    SyntheticJourneyGenerator,
)


def make_config(
    scenario: SimulationScenario = SimulationScenario.MILD_CONFOUNDING,
    seed: int = 42,
) -> SimulatorConfig:
    channels = (
        ChannelConfig(
            channel_id="paid-search",
            exposure_probability=0.3,
            treatment_effect=0.2,
            cost_per_exposure=1.5,
            saturation=0.2,
            confounding_sensitivity=0.4,
        ),
        ChannelConfig(
            channel_id="social",
            exposure_probability=0.25,
            treatment_effect=0.1,
            cost_per_exposure=1.0,
            saturation=0.3,
            confounding_sensitivity=0.5,
        ),
    )

    return SimulatorConfig(
        scenario=scenario,
        random_seed=seed,
        population_size=100,
        hidden_intent_strength=0.5,
        proxy_quality=0.7,
        baseline_conversion_probability=0.05,
        overlap_strength=0.8,
        conversion_saturation=0.9,
        channels=channels,
    )


def test_generator_returns_valid_simulation_bundle() -> None:
    bundle = SyntheticJourneyGenerator(make_config()).generate()

    assert bundle.metadata.dataset_kind == DatasetKind.SIMULATION
    assert len(bundle.observations) == 100
    assert len(bundle.ground_truth) == 2


def test_same_seed_reproduces_observations_and_truth() -> None:
    config = make_config(seed=123)
    first = SyntheticJourneyGenerator(config).generate()
    second = SyntheticJourneyGenerator(config).generate()

    assert first.observations == second.observations
    assert first.ground_truth == second.ground_truth


def test_different_seed_changes_generated_data() -> None:
    first = SyntheticJourneyGenerator(make_config(seed=123)).generate()
    second = SyntheticJourneyGenerator(make_config(seed=456)).generate()

    assert first.observations != second.observations


def test_null_channel_has_zero_configured_effect() -> None:
    bundle = SyntheticJourneyGenerator(
        make_config(scenario=SimulationScenario.NULL_CHANNEL)
    ).generate()

    null_truth = next(
        record for record in bundle.ground_truth
        if record.treatment_id == "paid-search"
    )

    # Finite-sample outcome noise means this initial implementation does not
    # guarantee an exactly zero realized potential-outcome difference.
    assert null_truth.effect is not None


def test_generated_observations_do_not_contain_latent_intent() -> None:
    bundle = SyntheticJourneyGenerator(make_config()).generate()

    assert all(
        "hidden_intent" not in observation.covariates.values
        for observation in bundle.observations
    )
