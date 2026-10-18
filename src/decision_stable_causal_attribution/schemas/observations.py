"""Canonical observation contracts shared by simulator and data adapters."""

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ScalarValue = str | int | float | bool | None


class Covariates(BaseModel):
    """Pre-treatment covariates for one unit of analysis."""

    model_config = ConfigDict(extra="allow", frozen=True)

    values: dict[str, ScalarValue] = Field(default_factory=dict)

    @field_validator("values")
    @classmethod
    def validate_values(cls, values: dict[str, ScalarValue]) -> dict[str, ScalarValue]:
        """Reject empty feature names and non-finite numeric values."""
        import math

        for name, value in values.items():
            if not name.strip():
                raise ValueError("Covariate names cannot be empty.")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"Covariate {name!r} must be finite.")
        return values


class ExposureRecord(BaseModel):
    """Exposure to a channel at a specific time."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    channel_id: str = Field(min_length=1, max_length=128)
    exposed: bool
    timestamp: datetime
    cost: float | None = Field(default=None, ge=0)

    @field_validator("channel_id")
    @classmethod
    def normalize_channel_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("channel_id cannot be blank.")
        return value.strip()

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Exposure timestamps must include a timezone.")
        return value.astimezone(UTC)


class JourneyEvent(BaseModel):
    """An event associated with a unit's customer journey."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1, max_length=128)
    timestamp: datetime
    event_name: str = Field(min_length=1, max_length=128)
    channel_id: str | None = Field(default=None, max_length=128)
    attributes: dict[str, ScalarValue] = Field(default_factory=dict)

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Event timestamps must include a timezone.")
        return value.astimezone(UTC)

    @field_validator("event_name", "event_id")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Event identifiers and names cannot be blank.")
        return value.strip()


class ObservationRecord(BaseModel):
    """One unit's observed treatment, outcome, covariates, and journey."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_id: str = Field(min_length=1, max_length=256)
    covariates: Covariates = Field(default_factory=Covariates)
    treatment: int | None = Field(
        default=None,
        description="Binary treatment indicator where the adapter defines one.",
    )
    outcome: float | None = None
    outcome_observed: bool = True
    treatment_timestamp: datetime | None = None
    exposures: tuple[ExposureRecord, ...] = ()
    events: tuple[JourneyEvent, ...] = ()
    attributes: dict[str, ScalarValue] = Field(default_factory=dict)

    @field_validator("unit_id")
    @classmethod
    def reject_blank_unit_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("unit_id cannot be blank.")
        return value.strip()

    @field_validator("treatment")
    @classmethod
    def require_binary_treatment(cls, value: int | None) -> int | None:
        if value not in (None, 0, 1):
            raise ValueError("treatment must be 0, 1, or None.")
        return value

    @field_validator("outcome")
    @classmethod
    def require_finite_outcome(cls, value: float | None) -> float | None:
        import math

        if value is not None and not math.isfinite(value):
            raise ValueError("outcome must be finite when provided.")
        return value

    @field_validator("treatment_timestamp")
    @classmethod
    def normalize_treatment_timestamp(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("treatment_timestamp must include a timezone.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_outcome_observation(self) -> "ObservationRecord":
        if self.outcome_observed and self.outcome is None:
            raise ValueError(
                "An observed outcome must have a value; use outcome_observed=False "
                "when the outcome is unavailable."
            )
        if not self.outcome_observed and self.outcome is not None:
            raise ValueError("An unavailable outcome must be None, not a fabricated numeric value.")
        return self
