"""Augmented Inverse Probability Weighting (AIPW / Doubly Robust) Causal Estimator.

Implements cross-fitted AIPW estimation for channel Average Treatment Effects (ATE).
Guarantees double robustness: consistent if EITHER the propensity model OR the outcome
model is correctly specified.
"""

import logging
from dataclasses import dataclass
from math import sqrt

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold

from decision_stable_causal_attribution.data_utils import bundle_to_tabular_frame
from decision_stable_causal_attribution.schemas.dataset import DatasetBundle
from decision_stable_causal_attribution.schemas.results import EffectEstimate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AIPWPointResult:
    """Detailed intermediate results from an AIPW estimation run."""

    channel_id: str
    ate: float
    se: float
    ci_lower: float
    ci_upper: float
    propensity_scores: np.ndarray
    mu1: np.ndarray
    mu0: np.ndarray
    gamma1: np.ndarray
    gamma0: np.ndarray


class AIPWEstimator:
    """Doubly robust AIPW estimator for Average Treatment Effects."""

    def __init__(
        self,
        n_splits: int = 5,
        clip_epsilon: float = 0.01,
        random_state: int = 42,
    ) -> None:
        self.n_splits = n_splits
        self.clip_epsilon = clip_epsilon
        self.random_state = random_state

    def estimate_channel(
        self,
        bundle: DatasetBundle,
        channel_id: str,
    ) -> AIPWPointResult:
        """Estimate the ATE of a specific marketing channel using cross-fitting."""
        df = bundle_to_tabular_frame(bundle)
        n = len(df)

        exp_col = f"exposure_{channel_id}"
        if exp_col in df.columns:
            t = df[exp_col].to_numpy(dtype=float)
        elif "treatment" in df.columns and df["treatment"].notna().all():
            t = df["treatment"].to_numpy(dtype=float)
        else:
            raise ValueError(f"Channel {channel_id} not found in exposures or treatment.")

        y = df["outcome"].to_numpy(dtype=float)

        # Extract numeric covariate columns
        covariate_cols = [c for c in df.columns if c.startswith("covariate_")]
        if not covariate_cols:
            x = np.ones((n, 1), dtype=float)
        else:
            x = df[covariate_cols].to_numpy(dtype=float)

        # Cross-fitting arrays
        e_hat = np.zeros(n, dtype=float)
        mu1_hat = np.zeros(n, dtype=float)
        mu0_hat = np.zeros(n, dtype=float)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        for train_idx, test_idx in kf.split(x):
            x_train, x_test = x[train_idx], x[test_idx]
            t_train = t[train_idx]
            y_train = y[train_idx]

            # 1. Propensity model e(X) = P(T=1 | X)
            prop_model = LogisticRegression(C=1.0, max_iter=500)
            if len(np.unique(t_train)) < 2:
                # Degenerate treatment assignment in fold
                p_train = float(np.mean(t_train))
                e_test = np.full(len(test_idx), p_train)
            else:
                prop_model.fit(x_train, t_train)
                e_test = prop_model.predict_proba(x_test)[:, 1]

            # Clip propensities to avoid extreme weights
            e_test = np.clip(e_test, self.clip_epsilon, 1.0 - self.clip_epsilon)
            e_hat[test_idx] = e_test

            # 2. Outcome models mu1(X) = E[Y | T=1, X] and mu0(X) = E[Y | T=0, X]
            idx_1 = t_train == 1
            idx_0 = t_train == 0

            # Fit mu1
            if np.sum(idx_1) > 5 and len(np.unique(y_train[idx_1])) > 1:
                model_1 = Ridge(alpha=1.0)
                model_1.fit(x_train[idx_1], y_train[idx_1])
                mu1_test = model_1.predict(x_test)
            else:
                mu1_test = np.full(
                    len(test_idx), np.mean(y_train[idx_1]) if np.sum(idx_1) > 0 else 0.0
                )

            # Fit mu0
            if np.sum(idx_0) > 5 and len(np.unique(y_train[idx_0])) > 1:
                model_0 = Ridge(alpha=1.0)
                model_0.fit(x_train[idx_0], y_train[idx_0])
                mu0_test = model_0.predict(x_test)
            else:
                mu0_test = np.full(
                    len(test_idx), np.mean(y_train[idx_0]) if np.sum(idx_0) > 0 else 0.0
                )

            mu1_hat[test_idx] = mu1_test
            mu0_hat[test_idx] = mu0_test

        # 3. Individual Doubly Robust pseudo-outcomes
        gamma1 = mu1_hat + (t / e_hat) * (y - mu1_hat)
        gamma0 = mu0_hat + ((1.0 - t) / (1.0 - e_hat)) * (y - mu0_hat)
        tau_i = gamma1 - gamma0

        ate = float(np.mean(tau_i))
        se = float(np.std(tau_i, ddof=1) / sqrt(n))
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se

        return AIPWPointResult(
            channel_id=channel_id,
            ate=ate,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            propensity_scores=e_hat,
            mu1=mu1_hat,
            mu0=mu0_hat,
            gamma1=gamma1,
            gamma0=gamma0,
        )

    def to_effect_estimate(self, result: AIPWPointResult) -> EffectEstimate:
        """Convert result into standard validated EffectEstimate schema."""
        return EffectEstimate(
            treatment_id=result.channel_id,
            outcome_id="conversion",
            estimand="ate",
            estimate=result.ate,
            standard_error=result.se,
            confidence_level=0.95,
            confidence_interval=(result.ci_lower, result.ci_upper),
            subgroup_id=None,
            method="aipw_doubly_robust",
        )
