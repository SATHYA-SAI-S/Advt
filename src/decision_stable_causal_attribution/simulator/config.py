
"""Validated configuration for the synthetic causal attribution simulator."""

from enum import StrEnum
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SimulationScenario(StrEnum):
    """Required synthetic scenarios from the research specification."""

    MILD_CONFOUNDING = "mild_confounding"
    RETARGETING_TRAP = "retargeting_trap"
    POOR_OVERLAP = "poor_overlap"
    HIDDEN_INTENT_HEAVY = "hidden_intent_heavy"
    NULL_CHANNEL = "null_channel"
    HETEROGENEOUS_EFFECTS = "heterogeneous_effects"


class ChannelConfig(BaseModel):
    """Structural parameters for one marketing channel."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    channel_id: str = Field(min_length=1, max_length=128)
    exposure_probability: float = Field(gt=0, lt=1)
    treatment_effect: float
    cost_per_exposure: float = Field(ge=0)
    saturation: float = Field(gt=0, le=1)
    confounding_sensitivity: float = Field(ge=0, le=1)

    @field_validator(
        "exposure_probability",
        "treatment_effect",
        "cost_per_exposure",
        "saturation",
        "confounding_sensitivity",
    )
    @classmethod
    def require_finite_values(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("Channel configuration values must be finite.")
        return value

    @field_validator("channel_id")
    @classmethod
    def reject_blank_channel_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("channel_id cannot be blank.")
        return value.strip()


class SimulatorConfig(BaseModel):
    """Complete, reproducible configuration for one simulator scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    config_version: str = "1.0.0"
    scenario: SimulationScenario
    random_seed: int = Field(ge=0, le=2**32 - 1)
    population_size: int = Field(ge=100, le=10_000_000)

    # Latent purchase intent and observed proxy quality.
    hidden_intent_strength: float = Field(ge=0, le=1)
    proxy_quality: float = Field(ge=0, le=1)

    # Exposure assignment and positivity.
    baseline_conversion_probability: float = Field(gt=0, lt=1)
    overlap_strength: float = Field(ge=0, le=1)

    # Outcome generation.
    conversion_saturation: float = Field(gt=0, le=1)
    channels: tuple[ChannelConfig, ...] = Field(min_length=2)

    @field_validator("config_version")
    @classmethod
    def validate_config_version(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("config_version cannot be blank.")
        return value.strip()

    @field_validator(
        "hidden_intent_strength",
        "proxy_quality",
        "baseline_conversion_probability",
        "overlap_strength",
        "conversion_saturation",
    )
    @classmethod
    def require_finite_parameters(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("Simulator parameters must be finite.")
        return value

    @model_validator(mode="after")
    def validate_channels(self) -> "SimulatorConfig":
        channel_ids = [channel.channel_id for channel in self.channels]
        if len(channel_ids) != len(set(channel_ids)):
            raise ValueError("Channel identifiers must be unique.")
        return self
