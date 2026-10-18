
"""Ground-truth contracts for simulator-controlled causal effects."""

from enum import StrEnum
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Estimand(StrEnum):
    """Causal quantity represented by a ground-truth record."""

    ATE = "ate"
    ATT = "att"
    CATE = "cate"


class GroundTruthRecord(BaseModel):
    """Known causal effect for a defined treatment, outcome, and population."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    estimand: Estimand
    treatment_id: str = Field(min_length=1, max_length=128)
    outcome_id: str = Field(min_length=1, max_length=128)
    effect: float
    population_id: str = Field(min_length=1, max_length=128)
    subgroup_id: str | None = Field(default=None, max_length=128)
    sample_size: int | None = Field(default=None, ge=1)
    replicate_id: str = Field(min_length=1, max_length=128)

    @field_validator(
        "treatment_id",
        "outcome_id",
        "population_id",
        "replicate_id",
        "subgroup_id",
    )
    @classmethod
    def reject_blank_identifiers(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Identifiers cannot be blank.")
        return value.strip() if value is not None else None

    @field_validator("effect")
    @classmethod
    def require_finite_effect(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("Ground-truth effects must be finite.")
        return value

    @model_validator(mode="after")
    def validate_estimand_scope(self) -> "GroundTruthRecord":
        """Require subgroup identity for conditional treatment effects."""
        if self.estimand == Estimand.CATE and self.subgroup_id is None:
            raise ValueError("CATE ground truth requires a subgroup_id.")
        return self
