
"""Tests for dataset bundle integrity and DataFrame conversion."""

import pytest
from pydantic import ValidationError

from decision_stable_causal_attribution.schemas.dataset import DatasetBundle
from decision_stable_causal_attribution.schemas.ground_truth import (
    Estimand,
    GroundTruthRecord,
)
from decision_stable_causal_attribution.schemas.metadata import (
    DatasetKind,
    DatasetMetadata,
    GroundTruthStatus,
)
from decision_stable_causal_attribution.schemas.observations import (
    Covariates,
    ObservationRecord,
)


def make_metadata(
    kind: DatasetKind = DatasetKind.SIMULATION,
    truth_status: GroundTruthStatus = GroundTruthStatus.AVAILABLE,
) -> DatasetMetadata:
    return DatasetMetadata(
        dataset_id="test-dataset",
        dataset_kind=kind,
        source="pytest",
        description="Dataset bundle test",
        unit_of_analysis="visitor",
        treatment_definition="Binary treatment",
        outcome_definition="Binary conversion",
        ground_truth_status=truth_status,
        observational_caveat=(
            "Observational data do not establish causal effects."
            if kind == DatasetKind.OBSERVATIONAL
            else None
        ),
    )


def make_truth() -> GroundTruthRecord:
    return GroundTruthRecord(
        estimand=Estimand.ATE,
        treatment_id="paid-search",
        outcome_id="conversion",
        effect=0.02,
        population_id="all-visitors",
        replicate_id="seed-42",
    )


def test_valid_simulation_bundle_is_accepted() -> None:
    bundle = DatasetBundle(
        metadata=make_metadata(),
        observations=(
            ObservationRecord(unit_id="u1", treatment=1, outcome=1.0),
            ObservationRecord(unit_id="u2", treatment=0, outcome=0.0),
        ),
        ground_truth=(make_truth(),),
    )

    assert len(bundle.observations) == 2
    assert len(bundle.ground_truth) == 1


def test_duplicate_unit_ids_are_rejected() -> None:
    with pytest.raises(ValidationError):
        DatasetBundle(
            metadata=make_metadata(),
            observations=(
                ObservationRecord(unit_id="u1", treatment=1, outcome=1.0),
                ObservationRecord(unit_id="u1", treatment=0, outcome=0.0),
            ),
            ground_truth=(make_truth(),),
        )


def test_observational_bundle_rejects_ground_truth() -> None:
    with pytest.raises(ValidationError):
        DatasetBundle(
            metadata=make_metadata(
                DatasetKind.OBSERVATIONAL,
                GroundTruthStatus.UNAVAILABLE,
            ),
            observations=(
                ObservationRecord(unit_id="u1", treatment=1, outcome=1.0),
            ),
            ground_truth=(make_truth(),),
        )


def test_unavailable_ground_truth_must_be_empty() -> None:
    with pytest.raises(ValidationError):
        DatasetBundle(
            metadata=make_metadata(
                DatasetKind.RANDOMIZED_BENCHMARK,
                GroundTruthStatus.UNAVAILABLE,
            ),
            observations=(
                ObservationRecord(unit_id="u1", treatment=1, outcome=1.0),
            ),
            ground_truth=(make_truth(),),
        )


def test_available_ground_truth_requires_records() -> None:
    with pytest.raises(ValidationError):
        DatasetBundle(
            metadata=make_metadata(),
            observations=(
                ObservationRecord(unit_id="u1", treatment=1, outcome=1.0),
            ),
            ground_truth=(),
        )


def test_observations_frame_excludes_ground_truth() -> None:
    bundle = DatasetBundle(
        metadata=make_metadata(),
        observations=(
            ObservationRecord(
                unit_id="u1",
                treatment=1,
                outcome=1.0,
                covariates=Covariates(values={"device": "desktop"}),
            ),
        ),
        ground_truth=(make_truth(),),
    )

    frame = bundle.observations_frame()

    assert list(frame["unit_id"]) == ["u1"]
    assert "effect" not in frame.columns


def test_ground_truth_frame_is_separate() -> None:
    bundle = DatasetBundle(
        metadata=make_metadata(),
        observations=(
            ObservationRecord(unit_id="u1", treatment=1, outcome=1.0),
        ),
        ground_truth=(make_truth(),),
    )

    truth_frame = bundle.ground_truth_frame()

    assert "effect" in truth_frame.columns
    assert truth_frame.iloc[0]["effect"] == 0.02
