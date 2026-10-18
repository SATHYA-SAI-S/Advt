"""Tests for canonical observation and dataset metadata contracts."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

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
    ExposureRecord,
    ObservationRecord,
)


def make_metadata(
    *,
    kind: DatasetKind = DatasetKind.SIMULATION,
    truth: GroundTruthStatus = GroundTruthStatus.AVAILABLE,
    caveat: str | None = None,
) -> DatasetMetadata:
    """Build valid metadata with overridable capability fields."""
    return DatasetMetadata(
        dataset_id="test-dataset",
        dataset_kind=kind,
        source="pytest",
        description="Schema contract test dataset",
        unit_of_analysis="visitor",
        treatment_definition="Binary channel treatment",
        outcome_definition="Binary conversion outcome",
        ground_truth_status=truth,
        observational_caveat=caveat,
    )


def test_valid_observation_is_accepted() -> None:
    observation = ObservationRecord(
        unit_id="visitor-001",
        covariates=Covariates(values={"device": "desktop", "prior_visits": 2}),
        treatment=1,
        outcome=1.0,
    )

    assert observation.unit_id == "visitor-001"
    assert observation.treatment == 1
    assert observation.outcome == 1.0


@pytest.mark.parametrize("treatment", [-1, 2, 10])
def test_non_binary_treatment_is_rejected(treatment: int) -> None:
    with pytest.raises(ValidationError):
        ObservationRecord(
            unit_id="visitor-001",
            treatment=treatment,
            outcome=0.0,
        )


def test_unavailable_outcome_must_be_explicit() -> None:
    observation = ObservationRecord(
        unit_id="visitor-002",
        outcome=None,
        outcome_observed=False,
    )

    assert observation.outcome is None
    assert observation.outcome_observed is False


def test_missing_observed_outcome_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ObservationRecord(
            unit_id="visitor-003",
            outcome=None,
            outcome_observed=True,
        )


def test_unavailable_outcome_cannot_contain_numeric_value() -> None:
    with pytest.raises(ValidationError):
        ObservationRecord(
            unit_id="visitor-004",
            outcome=0.0,
            outcome_observed=False,
        )


def test_exposure_timestamp_is_normalized_to_utc() -> None:
    exposure = ExposureRecord(
        channel_id="paid-search",
        exposed=True,
        timestamp=datetime.fromisoformat("2026-10-18T12:00:00+05:30"),
        cost=2.5,
    )

    assert exposure.timestamp.tzinfo == UTC
    assert exposure.timestamp.hour == 6
    assert exposure.timestamp.minute == 30


def test_timezone_naive_exposure_timestamp_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ExposureRecord(
            channel_id="paid-search",
            exposed=True,
            timestamp=datetime(2026, 10, 18, 12, 0),
        )


def test_negative_exposure_cost_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ExposureRecord(
            channel_id="paid-search",
            exposed=True,
            timestamp=datetime.now(UTC),
            cost=-1.0,
        )


def test_covariates_reject_non_finite_numbers() -> None:
    with pytest.raises(ValidationError):
        Covariates(values={"score": float("nan")})


def test_simulator_can_declare_known_ground_truth() -> None:
    metadata = make_metadata()

    assert metadata.dataset_kind == DatasetKind.SIMULATION
    assert metadata.ground_truth_status == GroundTruthStatus.AVAILABLE


def test_observational_dataset_requires_caveat() -> None:
    with pytest.raises(ValidationError):
        make_metadata(
            kind=DatasetKind.OBSERVATIONAL,
            truth=GroundTruthStatus.UNAVAILABLE,
        )


def test_observational_dataset_cannot_claim_known_ground_truth() -> None:
    with pytest.raises(ValidationError):
        make_metadata(
            kind=DatasetKind.OBSERVATIONAL,
            truth=GroundTruthStatus.AVAILABLE,
            caveat="Causal effects are not known from this observational dataset.",
        )


def test_valid_observational_metadata_is_accepted() -> None:
    metadata = make_metadata(
        kind=DatasetKind.OBSERVATIONAL,
        truth=GroundTruthStatus.UNAVAILABLE,
        caveat="Observational associations do not establish causal effects.",
    )

    assert metadata.ground_truth_status == GroundTruthStatus.UNAVAILABLE
    assert metadata.observational_caveat is not None


def make_ground_truth(**overrides: object) -> GroundTruthRecord:
    values: dict[str, object] = {
        "estimand": Estimand.ATE,
        "treatment_id": "paid-search",
        "outcome_id": "conversion",
        "effect": 0.025,
        "population_id": "all-visitors",
        "replicate_id": "seed-42",
    }
    values.update(overrides)
    return GroundTruthRecord.model_validate(values)


def test_ground_truth_accepts_known_simulator_effect() -> None:
    truth = make_ground_truth()

    assert truth.estimand == Estimand.ATE
    assert truth.effect == 0.025


def test_ground_truth_rejects_non_finite_effect() -> None:
    with pytest.raises(ValidationError):
        make_ground_truth(effect=float("nan"))


def test_cate_requires_subgroup_identifier() -> None:
    with pytest.raises(ValidationError):
        make_ground_truth(estimand=Estimand.CATE)


def test_cate_accepts_subgroup_identifier() -> None:
    truth = make_ground_truth(
        estimand=Estimand.CATE,
        subgroup_id="mobile-visitors",
    )

    assert truth.subgroup_id == "mobile-visitors"


def test_ground_truth_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        make_ground_truth(unverified=True)
