"""Tabular conversion and data utilities for DatasetBundle."""

import pandas as pd

from decision_stable_causal_attribution.schemas.dataset import DatasetBundle


def bundle_to_tabular_frame(bundle: DatasetBundle) -> pd.DataFrame:
    """Convert DatasetBundle into a flattened tabular DataFrame.

    Flattens:
    - covariates into `covariate_<key>` columns
    - exposures into `exposure_<channel_id>` and `cost_<channel_id>` columns
    - preserves unit_id, treatment, outcome
    """
    rows = []
    for obs in bundle.observations:
        row = {
            "unit_id": obs.unit_id,
            "treatment": obs.treatment,
            "outcome": obs.outcome,
            "outcome_observed": obs.outcome_observed,
        }

        # Flatten covariates
        for k, v in obs.covariates.values.items():
            row[f"covariate_{k}"] = v

        # Flatten exposures
        for exp in obs.exposures:
            row[f"exposure_{exp.channel_id}"] = 1 if exp.exposed else 0
            row[f"cost_{exp.channel_id}"] = exp.cost

        # Include attributes
        for k, v in obs.attributes.items():
            row[f"attr_{k}"] = v

        rows.append(row)

    return pd.DataFrame(rows)
