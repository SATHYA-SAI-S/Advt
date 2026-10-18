"""Validated dataset bundle shared by simulators and data adapters."""

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from decision_stable_causal_attribution.schemas.ground_truth import GroundTruthRecord
from decision_stable_causal_attribution.schemas.metadata import (
    DatasetKind,
    DatasetMetadata,
    GroundTruthStatus,
)
from decision_stable_causal_attribution.schemas.observations import ObservationRecord


class DatasetBundle(BaseModel):
    """Dataset metadata, unit observations, and optional known causal truth."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    metadata: DatasetMetadata
    observations: tuple[ObservationRecord, ...] = Field(min_length=1)
    ground_truth: tuple[GroundTruthRecord, ...] = ()

    @model_validator(mode="after")
    def validate_bundle_integrity(self) -> "DatasetBundle":
        """Enforce unit uniqueness and ground-truth provenance constraints."""
        unit_ids = [record.unit_id for record in self.observations]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("Observation unit_id values must be unique.")

        if self.metadata.dataset_kind == DatasetKind.OBSERVATIONAL and self.ground_truth:
            raise ValueError("Observational datasets cannot contain known causal ground truth.")

        if self.metadata.ground_truth_status == GroundTruthStatus.UNAVAILABLE and self.ground_truth:
            raise ValueError(
                "Dataset metadata declares ground truth unavailable, "
                "but ground-truth records were supplied."
            )

        if (
            self.metadata.ground_truth_status == GroundTruthStatus.AVAILABLE
            and not self.ground_truth
        ):
            raise ValueError(
                "Dataset metadata declares ground truth available, "
                "but no ground-truth records were supplied."
            )

        return self

    def observations_frame(self) -> pd.DataFrame:
        """Return unit-level data in a DataFrame for analysis pipelines."""
        rows = [
            {
                "unit_id": observation.unit_id,
                "treatment": observation.treatment,
                "outcome": observation.outcome,
                "outcome_observed": observation.outcome_observed,
                "treatment_timestamp": observation.treatment_timestamp,
                "covariates": observation.covariates.values,
                "attributes": observation.attributes,
                "exposures": [
                    exposure.model_dump(mode="json") for exposure in observation.exposures
                ],
                "events": [event.model_dump(mode="json") for event in observation.events],
            }
            for observation in self.observations
        ]
        return pd.DataFrame(rows)

    def ground_truth_frame(self) -> pd.DataFrame:
        """Return simulator truth separately from observed unit data."""
        return pd.DataFrame(record.model_dump(mode="json") for record in self.ground_truth)
