"""Attribution baselines: First-Touch, Last-Touch, Linear MTA, and Associational Logit.

Provides benchmark heuristic and associational models to contrast against
causal-effect estimates and demonstrate confounding bias.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from decision_stable_causal_attribution.data_utils import bundle_to_tabular_frame
from decision_stable_causal_attribution.schemas.dataset import DatasetBundle
from decision_stable_causal_attribution.schemas.results import EffectEstimate


@dataclass(frozen=True)
class AttributionShare:
    """Attributed conversions and share for a single channel under a heuristic rule."""

    channel_id: str
    rule: str  # first_touch, last_touch, linear
    attributed_conversions: float
    attribution_share: float
    total_conversions: float


class HeuristicAttribution:
    """Calculates heuristic multi-touch attribution (First-touch, Last-touch, Linear)."""

    def __init__(self, bundle: DatasetBundle) -> None:
        self.bundle = bundle
        self.channel_ids = self._extract_channel_ids()

    def _extract_channel_ids(self) -> list[str]:
        if not self.bundle.observations:
            return []
        first_obs = self.bundle.observations[0]
        return [exp.channel_id for exp in first_obs.exposures]

    def compute(
        self, rule: Literal["first_touch", "last_touch", "linear"]
    ) -> list[AttributionShare]:
        """Compute attribution shares for converting customer journeys."""
        channel_conversions: dict[str, float] = {ch: 0.0 for ch in self.channel_ids}
        total_converting_journeys = 0.0

        for obs in self.bundle.observations:
            if obs.outcome != 1.0:
                continue

            # Identify exposed channels with timestamps
            exposed_channels = [exp for exp in obs.exposures if exp.exposed]
            if not exposed_channels:
                continue

            total_converting_journeys += 1.0

            if rule == "first_touch":
                # Earliest timestamp or first in list
                first_exp = min(
                    exposed_channels,
                    key=lambda e: e.timestamp or pd.Timestamp.min,
                )
                channel_conversions[first_exp.channel_id] += 1.0

            elif rule == "last_touch":
                # Latest timestamp or last in list
                last_exp = max(
                    exposed_channels,
                    key=lambda e: e.timestamp or pd.Timestamp.max,
                )
                channel_conversions[last_exp.channel_id] += 1.0

            elif rule == "linear":
                # Equal credit across all exposed touchpoints
                credit_per_channel = 1.0 / len(exposed_channels)
                for exp in exposed_channels:
                    channel_conversions[exp.channel_id] += credit_per_channel

        total_attributed = sum(channel_conversions.values())
        shares = []
        for ch in self.channel_ids:
            conv = channel_conversions[ch]
            share = (conv / total_attributed) if total_attributed > 0 else 0.0
            shares.append(
                AttributionShare(
                    channel_id=ch,
                    rule=rule,
                    attributed_conversions=conv,
                    attribution_share=share,
                    total_conversions=total_converting_journeys,
                )
            )
        return shares


class AssociationalBaseline:
    """Standard naive associational baseline (difference-in-means & regression)."""

    def __init__(self, bundle: DatasetBundle) -> None:
        self.bundle = bundle
        self.df = bundle_to_tabular_frame(bundle)
        self.channel_ids = (
            [exp.channel_id for exp in bundle.observations[0].exposures]
            if bundle.observations
            else []
        )

    def compute_naive_ate(self, channel_id: str) -> EffectEstimate:
        """Calculate naive unadjusted conversion difference between exposed and unexposed."""
        # Find exposure indicator
        exp_col = f"exposure_{channel_id}"
        if exp_col in self.df.columns:
            exposed = self.df[self.df[exp_col] == 1]["outcome"]
            unexposed = self.df[self.df[exp_col] == 0]["outcome"]
        else:
            # Fallback to general treatment column if single channel
            exposed = self.df[self.df["treatment"] == 1]["outcome"]
            unexposed = self.df[self.df["treatment"] == 0]["outcome"]

        p_exposed = float(exposed.mean()) if len(exposed) > 0 else 0.0
        p_unexposed = float(unexposed.mean()) if len(unexposed) > 0 else 0.0
        naive_effect = p_exposed - p_unexposed

        # Standard error of difference in proportions
        n1 = len(exposed)
        n0 = len(unexposed)
        se = float(
            np.sqrt(
                p_exposed * (1 - p_exposed) / max(n1, 1)
                + p_unexposed * (1 - p_unexposed) / max(n0, 1)
            )
        )

        return EffectEstimate(
            treatment_id=channel_id,
            outcome_id="conversion",
            estimand="ate",
            estimate=naive_effect,
            standard_error=se,
            confidence_level=0.95,
            confidence_interval=(
                naive_effect - 1.96 * se,
                naive_effect + 1.96 * se,
            ),
            subgroup_id=None,
            method="naive_difference_in_means",
        )
