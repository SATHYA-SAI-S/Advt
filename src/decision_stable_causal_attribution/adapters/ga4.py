"""Google Analytics 4 (GA4) BigQuery sample ecommerce dataset adapter.

Extracts a 30-day first-visit cohort from:
`bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
Covering 2020-11-01 through 2021-01-31 (cohort index window: Nov 1 - Dec 31, 2020;
follow-up window through Jan 31, 2021).
"""

from datetime import UTC, datetime
import logging
from pathlib import Path
from typing import Any

from google.cloud import bigquery
import pandas as pd

from decision_stable_causal_attribution.schemas.dataset import DatasetBundle
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

GA4_COHORT_QUERY = """
WITH first_visits AS (
  SELECT
    user_pseudo_id,
    TIMESTAMP_MICROS(MIN(event_timestamp)) AS first_visit_time,
    ANY_VALUE(traffic_source.source) AS acquisition_source,
    ANY_VALUE(traffic_source.medium) AS acquisition_medium,
    ANY_VALUE(device.category) AS device_category,
    ANY_VALUE(device.operating_system) AS operating_system,
    ANY_VALUE(device.web_info.browser) AS browser,
    ANY_VALUE(geo.country) AS country,
    ANY_VALUE(geo.continent) AS continent
  FROM
    `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE
    _TABLE_SUFFIX BETWEEN '20201101' AND '20201231'
    AND event_name = 'first_visit'
  GROUP BY
    user_pseudo_id
),
purchases AS (
  SELECT
    user_pseudo_id,
    MIN(TIMESTAMP_MICROS(event_timestamp)) AS first_purchase_time,
    COUNT(1) AS purchase_count,
    SUM(ecommerce.purchase_revenue_in_usd) AS total_revenue
  FROM
    `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE
    _TABLE_SUFFIX BETWEEN '20201101' AND '20210131'
    AND event_name = 'purchase'
  GROUP BY
    user_pseudo_id
)
SELECT
  fv.user_pseudo_id,
  fv.first_visit_time,
  CONCAT(IFNULL(fv.acquisition_source, 'direct'), ' / ', IFNULL(fv.acquisition_medium, 'none')) AS acquisition_channel,
  IFNULL(fv.device_category, 'unknown') AS device_category,
  IFNULL(fv.operating_system, 'unknown') AS operating_system,
  IFNULL(fv.browser, 'unknown') AS browser,
  IFNULL(fv.country, 'unknown') AS country,
  IFNULL(fv.continent, 'unknown') AS continent,
  CASE
    WHEN p.first_purchase_time IS NOT NULL 
     AND TIMESTAMP_DIFF(p.first_purchase_time, fv.first_visit_time, DAY) <= 30
    THEN 1.0
    ELSE 0.0
  END AS converted_30d,
  CASE
    WHEN p.first_purchase_time IS NOT NULL 
     AND TIMESTAMP_DIFF(p.first_purchase_time, fv.first_visit_time, DAY) <= 30
    THEN IFNULL(p.total_revenue, 0.0)
    ELSE 0.0
  END AS revenue_30d
FROM
  first_visits fv
LEFT JOIN
  purchases p
ON
  fv.user_pseudo_id = p.user_pseudo_id
"""


def extract_ga4_cohort_to_parquet(
    project_id: str = "rime-hackathon-mcp",
    output_path: Path | str = "data/processed/ga4_cohort.parquet",
    limit: int | None = None,
) -> Path:
    """Execute BigQuery cohort query and save results to Parquet.
    
    Uses pathlib for cross-platform OS independence.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    client = bigquery.Client(project=project_id)
    query = GA4_COHORT_QUERY
    if limit is not None:
        query = f"{query}\nLIMIT {limit}"

    logger.info("Submitting GA4 cohort extraction query to BigQuery (project=%s)...", project_id)
    query_job = client.query(query)
    df: pd.DataFrame = query_job.to_dataframe()
    
    logger.info("Query returned %d cohort records. Writing to %s", len(df), dest)
    df.to_parquet(dest, index=False)
    return dest


def load_ga4_dataset_bundle(
    parquet_path: Path | str = "data/processed/ga4_cohort.parquet",
    top_channels: list[str] | None = None,
    max_records: int | None = None,
) -> DatasetBundle:
    """Load GA4 cohort Parquet and convert to standard DatasetBundle.
    
    Conforms strictly to the observational schema:
    - ground_truth_status = UNAVAILABLE
    - observational_caveat must be provided
    - ground_truth tuple is empty
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"GA4 parquet file not found at {path}. Run extraction first.")

    df = pd.read_parquet(path)
    if max_records is not None and len(df) > max_records:
        df = df.iloc[:max_records]

    if top_channels is None:
        # Default top decision channels with sufficient volume
        counts = df["acquisition_channel"].value_counts()
        top_channels = list(counts.head(5).index)

    observations: list[ObservationRecord] = []
    timestamp = datetime(2020, 11, 1, tzinfo=UTC)

    for idx, row in df.iterrows():
        unit_id = f"ga4-{row['user_pseudo_id']}"
        channel = str(row["acquisition_channel"])

        # Per-channel binary exposure
        exposures = tuple(
            ExposureRecord(
                channel_id=ch,
                exposed=(channel == ch),
                timestamp=timestamp,
                cost=0.0,
            )
            for ch in top_channels
        )

        covariates = Covariates(
            values={
                "is_mobile": 1.0 if row["device_category"] == "mobile" else 0.0,
                "is_desktop": 1.0 if row["device_category"] == "desktop" else 0.0,
                "is_americas": 1.0 if row["continent"] == "Americas" else 0.0,
                "is_chrome": 1.0 if "Chrome" in str(row["browser"]) else 0.0,
                "is_mac": 1.0 if "Macintosh" in str(row["operating_system"]) else 0.0,
                "is_windows": 1.0 if "Windows" in str(row["operating_system"]) else 0.0,
            }
        )

        observations.append(
            ObservationRecord(
                unit_id=unit_id,
                covariates=covariates,
                treatment=None,
                outcome=float(row["converted_30d"]),
                outcome_observed=True,
                treatment_timestamp=None,
                exposures=exposures,
                attributes={
                    "acquisition_channel": channel,
                    "revenue_30d": float(row["revenue_30d"]),
                },
            )
        )

    metadata = DatasetMetadata(
        dataset_id="ga4-public-ecommerce-cohort",
        dataset_kind=DatasetKind.OBSERVATIONAL,
        source="bigquery-public-data.ga4_obfuscated_sample_ecommerce",
        description=(
            "30-day first-visit observational visitor cohort from Google Analytics 4 "
            "obfuscated sample ecommerce dataset (Nov 1 - Dec 31, 2020 first visits, "
            "follow-up through Jan 31, 2021)."
        ),
        unit_of_analysis="visitor (user_pseudo_id)",
        treatment_definition="First-user acquisition channel derived from traffic_source",
        outcome_definition="Binary purchase event within 30 days of first visit",
        ground_truth_status=GroundTruthStatus.UNAVAILABLE,
        observational_caveat=(
            "GA4 observational data contains unmeasured user purchase intent, self-selection, "
            "and non-randomized channel assignment. Ground truth causal effects cannot be known "
            "without randomized interventions."
        ),
    )

    return DatasetBundle(
        metadata=metadata,
        observations=tuple(observations),
        ground_truth=(),
    )
