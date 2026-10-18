
"""Tests for simulator configuration validation."""

import pytest
from pydantic import ValidationError

from decision_stable_causal_attribution.simulator.config import (
    ChannelConfig,
    SimulationScenario,
    SimulatorConfig,
)


def make_channel(channel_id: str, **overrides: object) -> ChannelConfig:
    values: dict[str, object] = {
        "channel_id": channel_id,
        "exposure_probability": 0.3,
        "treatment_effect": 0.02,
        "cost_per_exposure": 1.5,
        "saturation": 0.8,
        "confounding_sensitivity": 0.4,
    }
    values.update(overrides)
    return ChannelConfig.model_validate(values)


def make_config(**overrides: object) -> SimulatorConfig:
    values: dict[str, object] = {
        "scenario": SimulationScenario.MILD_CONFOUNDING,
        "random_seed": 42,
        "population_size": 1000,
        "hidden_intent_strength": 0.5,
        "proxy_quality": 0.7,
        "baseline_conversion_probability": 0.05,
        "overlap_strength": 0.8,
        "conversion_saturation": 0.9,
        "channels": (
            make_channel("paid-search"),
            make_channel("social"),
        ),
    }
    values.update(overrides)
    return SimulatorConfig.model_validate(values)


def test_valid_simulator_configuration_is_accepted() -> None:
    config = make_config()

    assert config.random_seed == 42
    assert config.population_size == 1000
    assert len(config.channels) == 2


def test_all_six_scenarios_are_defined() -> None:
    assert len(SimulationScenario) == 6


def test_population_size_below_minimum_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_config(population_size=99)


def test_negative_random_seed_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_config(random_seed=-1)


def test_duplicate_channel_ids_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make_config(
            channels=(
                make_channel("paid-search"),
                make_channel("paid-search"),
            )
        )


def test_invalid_exposure_probability_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_channel("paid-search", exposure_probability=1.0)


def test_negative_channel_cost_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_channel("paid-search", cost_per_exposure=-1.0)


def test_non_finite_channel_effect_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_channel("paid-search", treatment_effect=float("nan"))


def test_configuration_is_immutable() -> None:
    config = make_config()

    with pytest.raises(ValidationError):
        config.random_seed = 99  # type: ignore[misc]
