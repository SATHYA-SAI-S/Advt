"""Dataset provenance and capability metadata."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DatasetKind(StrEnum):
    """Origin and intended role of a dataset."""

    SIMULATION = "simulation"
    RANDOMIZED_BENCHMARK = "randomized_benchmark"
    OBSERVATIONAL = "observational"


class GroundTruthStatus(StrEnum):
    """Availability of known causal ground truth."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class DatasetMetadata(BaseModel):
    """Describes a dataset without conflating provenance and causal validity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1, max_length=128)
    dataset_kind: DatasetKind
    schema_version: Literal["1.0.0"] = "1.0.0"
    source: str = Field(min_length=1, max_length=256)
    description: str = Field(min_length=1, max_length=2000)
    unit_of_analysis: str = Field(min_length=1, max_length=128)
    treatment_definition: str = Field(min_length=1, max_length=1000)
    outcome_definition: str = Field(min_length=1, max_length=1000)
    ground_truth_status: GroundTruthStatus
    observational_caveat: str | None = None
    time_start: str | None = None
    time_end: str | None = None

    @field_validator(
        "dataset_id",
        "source",
        "description",
        "unit_of_analysis",
        "treatment_definition",
        "outcome_definition",
    )
    @classmethod
    def reject_whitespace_only(cls, value: str) -> str:
        """Reject strings that contain no meaningful content."""
        if not value.strip():
            raise ValueError("Value must contain non-whitespace characters.")
        return value.strip()

    @field_validator("observational_caveat")
    @classmethod
    def require_caveat_for_observational_data(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        """Keep observational limitations explicit in the dataset contract."""
        if info.data.get("dataset_kind") == DatasetKind.OBSERVATIONAL:
            if value is None or not value.strip():
                raise ValueError("Observational datasets require an explicit observational_caveat.")
            return value.strip()
        return value.strip() if value is not None else None

    @field_validator("ground_truth_status")
    @classmethod
    def prevent_observational_ground_truth(
        cls,
        value: GroundTruthStatus,
        info,
    ) -> GroundTruthStatus:
        """Observational datasets cannot claim known causal ground truth."""
        if (
            info.data.get("dataset_kind") == DatasetKind.OBSERVATIONAL
            and value == GroundTruthStatus.AVAILABLE
        ):
            raise ValueError("Observational datasets cannot claim known causal ground truth.")
        return value
