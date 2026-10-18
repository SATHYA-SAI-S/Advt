"""Tests for the versioned frontend result contract."""

import pytest
from pydantic import ValidationError

from decision_stable_causal_attribution.schemas.results import (
    EffectEstimate,
    MetricStatus,
    MetricValue,
    ResearchResult,
    SensitivityAnalysis,
    SensitivityPoint,
)


def test_available_metric_accepts_zero() -> None:
    metric = MetricValue(
        name="conversion_lift",
        value=0.0,
        status=MetricStatus.AVAILABLE,
    )

    assert metric.value == 0.0


def test_unavailable_metric_requires_reason_and_null_value() -> None:
    metric = MetricValue(
        name="causal_regret",
        value=None,
        status=MetricStatus.UNAVAILABLE,
        reason="Observational data do not provide causal ground truth.",
    )

    assert metric.value is None


def test_unavailable_metric_cannot_be_zero() -> None:
    with pytest.raises(ValidationError):
        MetricValue(
            name="causal_regret",
            value=0.0,
            status=MetricStatus.UNAVAILABLE,
            reason="Ground truth is unavailable.",
        )


def test_available_metric_requires_value() -> None:
    with pytest.raises(ValidationError):
        MetricValue(name="rmse", status=MetricStatus.AVAILABLE)


def test_cate_estimate_requires_subgroup() -> None:
    with pytest.raises(ValidationError):
        EffectEstimate(
            treatment_id="paid-search",
            outcome_id="conversion",
            estimand="cate",
            estimate=0.03,
            method="cross-fitted-aipw",
        )


def test_effect_estimate_accepts_valid_interval() -> None:
    estimate = EffectEstimate(
        treatment_id="paid-search",
        outcome_id="conversion",
        estimand="ate",
        estimate=0.03,
        confidence_interval=(0.01, 0.05),
        method="cross-fitted-aipw",
    )

    assert estimate.confidence_interval == (0.01, 0.05)


def test_sensitivity_gamma_values_must_increase() -> None:
    with pytest.raises(ValidationError):
        SensitivityAnalysis(
            treatment_id="paid-search",
            outcome_id="conversion",
            method="marginal-sensitivity-model",
            calibration_description="Calibrated against observed covariates.",
            points=[
                SensitivityPoint(
                    gamma=1.5,
                    lower_bound=-0.01,
                    upper_bound=0.05,
                    includes_null=True,
                ),
                SensitivityPoint(
                    gamma=1.2,
                    lower_bound=-0.02,
                    upper_bound=0.06,
                    includes_null=True,
                ),
            ],
        )


def test_result_serializes_with_schema_version() -> None:
    result = ResearchResult(
        run_id="run-001",
        dataset_id="simulator-mild-confounding",
        generated_at="2026-10-18T12:00:00Z",
    )

    payload = result.model_dump(mode="json")

    assert payload["schema_version"] == "1.0.0"
    assert payload["run_id"] == "run-001"
    assert payload["effect_estimates"] == []


def test_result_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ResearchResult.model_validate(
            {
                "run_id": "run-001",
                "dataset_id": "simulator-mild-confounding",
                "generated_at": "2026-10-18T12:00:00Z",
                "unexpected_field": True,
            }
        )
