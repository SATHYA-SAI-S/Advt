"""Overlap and identifiability diagnostics for observational causal inference.

Evaluates empirical positivity, propensity score concentration, effective sample size (ESS),
weight share, and covariate balance (SMD).
Flags channels failing overlap thresholds as observationally unidentifiable.
"""

import logging
from math import isfinite, sqrt

import numpy as np

from decision_stable_causal_attribution.data_utils import bundle_to_tabular_frame
from decision_stable_causal_attribution.schemas.dataset import DatasetBundle
from decision_stable_causal_attribution.schemas.results import (
    MetricStatus,
    MetricValue,
    OverlapDiagnostics,
)

logger = logging.getLogger(__name__)


class OverlapDiagnosticEngine:
    """Computes comprehensive overlap and identifiability diagnostics for a channel."""

    def __init__(
        self,
        extreme_threshold: float = 0.05,
        min_ess_ratio: float = 0.10,
        max_weight_share_threshold: float = 0.05,
        max_smd_threshold: float = 0.25,
    ) -> None:
        self.extreme_threshold = extreme_threshold
        self.min_ess_ratio = min_ess_ratio
        self.max_weight_share_threshold = max_weight_share_threshold
        self.max_smd_threshold = max_smd_threshold

    def evaluate(
        self,
        bundle: DatasetBundle,
        channel_id: str,
        propensity_scores: np.ndarray,
    ) -> OverlapDiagnostics:
        """Evaluate overlap quality and return validated OverlapDiagnostics schema."""
        df = bundle_to_tabular_frame(bundle)
        n = len(df)

        exp_col = f"exposure_{channel_id}"
        if exp_col in df.columns:
            t = df[exp_col].to_numpy(dtype=float)
        elif "treatment" in df.columns and df["treatment"].notna().all():
            t = df["treatment"].to_numpy(dtype=float)
        else:
            raise ValueError(f"Channel {channel_id} not found in exposures or treatment.")

        e = np.asarray(propensity_scores, dtype=float)
        p_min = float(np.min(e))
        p_max = float(np.max(e))

        # 1. Extreme propensity share (< threshold or > 1 - threshold)
        is_extreme = (e < self.extreme_threshold) | (e > (1.0 - self.extreme_threshold))
        extreme_share = float(np.mean(is_extreme))

        # 2. IPW weights and Effective Sample Size
        # w_i = T / e + (1 - T) / (1 - e)
        w = (t / np.maximum(e, 1e-4)) + ((1.0 - t) / np.maximum(1.0 - e, 1e-4))
        sum_w = float(np.sum(w))
        sum_w2 = float(np.sum(w**2))
        ess = (sum_w**2) / sum_w2 if sum_w2 > 0 else 0.0
        ess_ratio = ess / n if n > 0 else 0.0

        # 3. Maximum single weight share
        max_w = float(np.max(w))
        max_weight_share = (max_w / sum_w) if sum_w > 0 else 0.0

        # 4. Covariate Balance (Standardized Mean Difference, SMD)
        covariate_cols = [c for c in df.columns if c.startswith("covariate_")]
        balance_metrics: list[MetricValue] = []
        max_post_smd = 0.0

        for col in covariate_cols:
            x_var = df[col].to_numpy(dtype=float)
            x1 = x_var[t == 1]
            x0 = x_var[t == 0]

            var1 = np.var(x1, ddof=1) if len(x1) > 1 else 0.0
            var0 = np.var(x0, ddof=1) if len(x0) > 1 else 0.0
            pooled_sd = sqrt(0.5 * (var1 + var0)) if (var1 + var0) > 0 else 1.0

            # Weighted SMD after IPW
            w1 = w[t == 1]
            w0 = w[t == 0]
            w_mean1 = float(np.average(x1, weights=w1)) if len(x1) > 0 and np.sum(w1) > 0 else 0.0
            w_mean0 = float(np.average(x0, weights=w0)) if len(x0) > 0 and np.sum(w0) > 0 else 0.0
            raw_diff = abs(w_mean1 - w_mean0)
            weighted_smd = float(raw_diff / pooled_sd) if pooled_sd > 1e-6 else 0.0
            if not isfinite(weighted_smd):
                weighted_smd = 0.0
            max_post_smd = max(max_post_smd, weighted_smd)

            balance_metrics.append(
                MetricValue(
                    name=f"smd_{col}",
                    value=round(weighted_smd, 4),
                    status=MetricStatus.AVAILABLE,
                    reason=None,
                    unit="standardized_diff",
                )
            )

        # 5. Overlap flag & warning synthesis
        warnings: list[str] = []
        overlap_failed = False

        if extreme_share > self.extreme_threshold or p_min <= 0.005 or p_max >= 0.995:
            overlap_failed = True
            warnings.append(
                "Severe propensity concentration / positivity violation: "
                f"propensity range [{p_min:.3f}, {p_max:.3f}], "
                f"{extreme_share:.1%} of units have extreme propensity "
                f"(<{self.extreme_threshold} or >{1.0 - self.extreme_threshold})."
            )

        if ess_ratio < self.min_ess_ratio:
            overlap_failed = True
            warnings.append(
                f"Effective sample size ratio ({ess_ratio:.2%}) is below "
                f"minimum threshold ({self.min_ess_ratio:.0%})."
            )

        if max_weight_share > self.max_weight_share_threshold:
            warnings.append(
                f"High weight concentration: single observation carries "
                f"{max_weight_share:.2%} of total weight."
            )

        if max_post_smd > self.max_smd_threshold:
            warnings.append(
                f"Residual covariate imbalance detected: maximum post-weighting "
                f"SMD is {max_post_smd:.3f} (> {self.max_smd_threshold})."
            )

        return OverlapDiagnostics(
            treatment_id=channel_id,
            propensity_min=p_min,
            propensity_max=p_max,
            extreme_propensity_share=MetricValue(
                name="extreme_propensity_share",
                value=extreme_share,
                status=MetricStatus.AVAILABLE,
                reason=None,
                unit="fraction",
            ),
            effective_sample_size=MetricValue(
                name="effective_sample_size",
                value=ess,
                status=MetricStatus.AVAILABLE,
                reason=None,
                unit="units",
            ),
            maximum_weight_share=MetricValue(
                name="maximum_weight_share",
                value=max_weight_share,
                status=MetricStatus.AVAILABLE,
                reason=None,
                unit="fraction",
            ),
            covariate_balance=balance_metrics,
            overlap_flag=overlap_failed,
            warnings=warnings,
        )
