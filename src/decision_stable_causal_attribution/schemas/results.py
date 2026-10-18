
"""Versioned result contracts shared by the research backend and frontend."""

from enum import StrEnum
from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MetricStatus(StrEnum):
    """Availability of a reported metric."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


class StabilityLabel(StrEnum):
    """Decision stability classification."""

    STABLE = "stable"
    FRAGILE = "fragile"
    UNIDENTIFIABLE = "unidentifiable"


class MetricValue(BaseModel):
    """A metric with explicit availability semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    value: float | None = None
    status: MetricStatus
    reason: str | None = None
    unit: str | None = Field(default=None, max_length=64)

    @field_validator("name", "reason", "unit")
    @classmethod
    def normalize_optional_strings(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Text fields cannot be blank.")
        return value.strip() if value is not None else None

    @field_validator("value")
    @classmethod
    def require_finite_value(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("Metric values must be finite.")
        return value

    @model_validator(mode="after")
    def validate_availability(self) -> "MetricValue":
        if self.status == MetricStatus.AVAILABLE and self.value is None:
            raise ValueError("Available metrics must have a numeric value.")
        if self.status != MetricStatus.AVAILABLE and self.value is not None:
            raise ValueError(
                "Unavailable or inapplicable metrics must use value=None."
            )
        if self.status != MetricStatus.AVAILABLE and self.reason is None:
            raise ValueError(
                "Unavailable or inapplicable metrics require a reason."
            )
        return self


class EffectEstimate(BaseModel):
    """Estimated treatment effect and its uncertainty."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    treatment_id: str = Field(min_length=1, max_length=128)
    outcome_id: str = Field(min_length=1, max_length=128)
    estimand: Literal["ate", "att", "cate"]
    estimate: float
    standard_error: float | None = Field(default=None, ge=0)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    confidence_interval: tuple[float, float] | None = None
    subgroup_id: str | None = None
    method: str = Field(min_length=1, max_length=128)

    @field_validator("estimate", "standard_error")
    @classmethod
    def require_finite_numbers(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("Estimates and standard errors must be finite.")
        return value

    @field_validator("confidence_interval")
    @classmethod
    def validate_interval(
        cls,
        value: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        if value is not None:
            lower, upper = value
            if not isfinite(lower) or not isfinite(upper) or lower > upper:
                raise ValueError("Confidence interval bounds must be finite and ordered.")
        return value

    @model_validator(mode="after")
    def validate_estimand(self) -> "EffectEstimate":
        if self.estimand == "cate" and not self.subgroup_id:
            raise ValueError("CATE estimates require a subgroup_id.")
        return self


class OverlapDiagnostics(BaseModel):
    """Diagnostics for treatment overlap and weighting quality."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    treatment_id: str = Field(min_length=1, max_length=128)
    propensity_min: float | None = Field(default=None, ge=0, le=1)
    propensity_max: float | None = Field(default=None, ge=0, le=1)
    extreme_propensity_share: MetricValue
    effective_sample_size: MetricValue
    maximum_weight_share: MetricValue
    covariate_balance: list[MetricValue] = Field(default_factory=list)
    overlap_flag: bool
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_propensity_range(self) -> "OverlapDiagnostics":
        if (self.propensity_min is None) != (self.propensity_max is None):
            raise ValueError(
                "Propensity minimum and maximum must either both be set or both be absent."
            )
        if (
            self.propensity_min is not None
            and self.propensity_max is not None
            and self.propensity_min > self.propensity_max
        ):
            raise ValueError("propensity_min cannot exceed propensity_max.")
        return self


class SensitivityPoint(BaseModel):
    """Sensitivity result for one value of the confounding parameter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gamma: float = Field(ge=1)
    lower_bound: float
    upper_bound: float
    includes_null: bool

    @model_validator(mode="after")
    def validate_bounds(self) -> "SensitivityPoint":
        if not isfinite(self.gamma) or not isfinite(self.lower_bound):
            raise ValueError("Sensitivity values must be finite.")
        if not isfinite(self.upper_bound) or self.lower_bound > self.upper_bound:
            raise ValueError("Sensitivity bounds must be finite and ordered.")
        return self


class SensitivityAnalysis(BaseModel):
    """Sensitivity bounds and their methodological provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    treatment_id: str = Field(min_length=1, max_length=128)
    outcome_id: str = Field(min_length=1, max_length=128)
    method: str = Field(min_length=1, max_length=256)
    calibration_description: str = Field(min_length=1, max_length=1000)
    points: list[SensitivityPoint] = Field(min_length=1)

    @model_validator(mode="after")
    def require_increasing_gamma(self) -> "SensitivityAnalysis":
        gammas = [point.gamma for point in self.points]
        if any(
            left >= right
            for left, right in zip(gammas, gammas[1:], strict=False)
        ):
            raise ValueError("Sensitivity gamma values must be strictly increasing.")
        return self


class ChannelAllocation(BaseModel):
    """Budget allocation for one decisio
    n channel."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    channel_id: str = Field(min_length=1, max_length=128)
    budget: float = Field(ge=0)
    expected_outcome: float | None = None
    expected_incremental_outcome: float | None = None

    @field_validator("budget", "expected_outcome", "expected_incremental_outcome")
    @classmethod
    def require_finite_values(cls, value: float | None) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("Allocation values must be finite.")
        return value


class BudgetDecision(BaseModel):
    """Point-estimate or robust budget allocation and stability assessment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: Literal["point_estimate", "robust_minimax_regret"]
    allocations: list[ChannelAllocation]
    total_budget: float = Field(ge=0)
    expected_objective: MetricValue
    worst_case_regret: MetricValue
    stability: StabilityLabel
    breakdown_gamma: MetricValue
    assumptions: list[str] = Field(default_factory=list)


class ExperimentCandidate(BaseModel):
    """Candidate experiment and its estimated decision value."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_id: str = Field(min_length=1, max_length=128)
    channel_id: str = Field(min_length=1, max_length=128)
    segment_id: str | None = None
    sample_size: int = Field(ge=1)
    experiment_cost: float = Field(ge=0)
    expected_regret_reduction: MetricValue
    value_of_information: MetricValue
    selection_method: str = Field(min_length=1, max_length=128)


class EvaluationSummary(BaseModel):
    """Evaluation metrics across scenarios or repeated simulation seeds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1, max_length=128)
    repetitions: int = Field(ge=1)
    metrics: list[MetricValue] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ResearchResult(BaseModel):
    """Versioned top-level result document consumed by the frontend."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = "1.0.0"
    run_id: str = Field(min_length=1, max_length=128)
    dataset_id: str = Field(min_length=1, max_length=128)
    generated_at: str = Field(min_length=1)
    effect_estimates: list[EffectEstimate] = Field(default_factory=list)
    overlap_diagnostics: list[OverlapDiagnostics] = Field(default_factory=list)
    sensitivity_analyses: list[SensitivityAnalysis] = Field(default_factory=list)
    budget_decisions: list[BudgetDecision] = Field(default_factory=list)
    experiment_candidates: list[ExperimentCandidate] = Field(default_factory=list)
    evaluation: EvaluationSummary | None = None
    warnings: list[str] = Field(default_factory=list)
