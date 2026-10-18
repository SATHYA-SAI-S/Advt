"""Tests for Criteo and GA4 dataset adapters."""

from pathlib import Path

import pytest

from decision_stable_causal_attribution.adapters.criteo import (
    build_criteo_dataset_bundle,
)
from decision_stable_causal_attribution.adapters.ga4 import (
    load_ga4_dataset_bundle,
)
from decision_stable_causal_attribution.schemas.metadata import (
    DatasetKind,
    GroundTruthStatus,
)


def test_criteo_adapter():
    """Verify Criteo adapter reads gzip file and creates valid benchmark bundle."""
    criteo_path = Path("data/raw/criteo-uplift-v2.1.csv.gz")
    if not criteo_path.exists():
        pytest.skip("Criteo raw data file not found.")

    bundle = build_criteo_dataset_bundle(
        file_path=criteo_path,
        sample_size=100,
        random_seed=42,
    )

    assert bundle.metadata.dataset_kind == DatasetKind.RANDOMIZED_BENCHMARK
    assert bundle.metadata.ground_truth_status == GroundTruthStatus.AVAILABLE
    assert len(bundle.observations) == 100
    assert len(bundle.ground_truth) == 1
    assert bundle.ground_truth[0].treatment_id == "criteo_display"
    assert bundle.observations[0].treatment in [0, 1]


def test_ga4_adapter():
    """Verify GA4 adapter loads from Parquet and strictly enforces observational schema."""
    parquet_path = Path("data/processed/ga4_cohort.parquet")
    if not parquet_path.exists():
        pytest.skip("GA4 parquet file not found.")

    bundle = load_ga4_dataset_bundle(
        parquet_path=parquet_path,
        max_records=100,
    )

    assert bundle.metadata.dataset_kind == DatasetKind.OBSERVATIONAL
    assert bundle.metadata.ground_truth_status == GroundTruthStatus.UNAVAILABLE
    assert bundle.metadata.observational_caveat is not None
    assert len(bundle.ground_truth) == 0
    assert len(bundle.observations) == 100
    assert bundle.observations[0].treatment is None
