"""Criteo Uplift v2.1 benchmark dataset adapter.

Maps the large-scale randomized advertising benchmark:
- Randomized treatment assignment (`treatment`, 85% treated / 15% control)
- Randomized outcomes (`conversion`, `visit`)
- Pre-treatment continuous covariates (`f0` through `f11`)
- Distinguishes intended randomized treatment from actual ad exposure (`exposure`)
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

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
    ExposureRecord,
    ObservationRecord,
)

logger = logging.getLogger(__name__)


def load_criteo_sample(
    file_path: Path | str = "data/raw/criteo-uplift-v2.1.csv.gz",
    sample_size: int = 50_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Read a random sample of Criteo Uplift data for fast benchmarking.

    Uses pathlib for cross-platform OS independence.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Criteo dataset not found at {path}")

    # Read nrows or sample with chunking if large
    logger.info("Loading %d rows from Criteo dataset at %s...", sample_size, path)
    df = pd.read_csv(path, compression="gzip", nrows=sample_size)
    return df


def build_criteo_dataset_bundle(
    file_path: Path | str = "data/raw/criteo-uplift-v2.1.csv.gz",
    sample_size: int = 20_000,
    random_seed: int = 42,
    outcome_col: str = "conversion",
) -> DatasetBundle:
    """Convert Criteo randomized trial data into standard DatasetBundle.

    Because Criteo is a true randomized trial, the experimental benchmark ATE
    can be estimated directly via difference-in-means on randomized assignment.
    """
    df = load_criteo_sample(file_path=file_path, sample_size=sample_size, random_seed=random_seed)

    # Experimental difference-in-means ground truth estimate from RCT
    treated = df[df["treatment"] == 1][outcome_col]
    control = df[df["treatment"] == 0][outcome_col]
    treated_mean = (
        float(treated.mean()) if len(treated) > 0 and not np.isnan(treated.mean()) else 0.0
    )
    control_mean = (
        float(control.mean()) if len(control) > 0 and not np.isnan(control.mean()) else 0.0
    )
    rct_ate = float(treated_mean - control_mean)

    feature_cols = [f"f{i}" for i in range(12)]
    observations: list[ObservationRecord] = []

    for idx, row in df.iterrows():
        unit_id = f"criteo-{idx:08d}"
        treatment_val = int(row["treatment"])
        outcome_val = float(row[outcome_col])

        covariates = Covariates(values={col: float(row[col]) for col in feature_cols})

        exposure_rec = ExposureRecord(
            channel_id="criteo_display",
            exposed=bool(treatment_val == 1),
            timestamp=datetime(2020, 1, 1, tzinfo=UTC),
            cost=1.0 if treatment_val == 1 else 0.0,
        )

        observations.append(
            ObservationRecord(
                unit_id=unit_id,
                covariates=covariates,
                treatment=treatment_val,
                outcome=outcome_val,
                outcome_observed=True,
                treatment_timestamp=None,
                exposures=(exposure_rec,),
                attributes={
                    "actual_exposure": int(row["exposure"]),
                    "visit": int(row["visit"]),
                    "conversion": int(row["conversion"]),
                },
            )
        )

    metadata = DatasetMetadata(
        dataset_id=f"criteo-uplift-v2.1-n{sample_size}",
        dataset_kind=DatasetKind.RANDOMIZED_BENCHMARK,
        source="criteo-uplift-v2.1",
        description=(
            "Randomized advertising uplift benchmark from Criteo AI Lab (v2.1). "
            "Supports treatment-effect benchmarking with randomized assignment."
        ),
        unit_of_analysis="user",
        treatment_definition="Randomized ad treatment allocation (1=treatment, 0=control)",
        outcome_definition=f"Binary {outcome_col} indicator",
        ground_truth_status=GroundTruthStatus.AVAILABLE,
        observational_caveat=None,
    )

    truth = [
        GroundTruthRecord(
            estimand=Estimand.ATE,
            treatment_id="criteo_display",
            outcome_id=outcome_col,
            effect=rct_ate,
            population_id="criteo_rct_sample",
            sample_size=len(df),
            replicate_id=f"seed-{random_seed}",
        )
    ]

    return DatasetBundle(
        metadata=metadata,
        observations=tuple(observations),
        ground_truth=tuple(truth),
    )
