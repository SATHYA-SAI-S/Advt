"""Dataset adapters for Criteo uplift benchmark and Google Analytics 4 (GA4)."""

from decision_stable_causal_attribution.adapters.ga4 import (
    extract_ga4_cohort_to_parquet,
    load_ga4_dataset_bundle,
)

__all__ = [
    "extract_ga4_cohort_to_parquet",
    "load_ga4_dataset_bundle",
]
