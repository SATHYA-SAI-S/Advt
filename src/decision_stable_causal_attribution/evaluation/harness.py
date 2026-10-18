"""Evaluation harness for Decision-Stable Causal Attribution.

Coordinates end-to-end execution across synthetic scenarios, randomized benchmarks,
and observational real data. Evaluates bias, RMSE, coverage, regret, and exports
validated ResearchResult JSON for Person 2's D3.js frontend.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from decision_stable_causal_attribution.decision.optimizer import (
    BudgetOptimizer,
    ChannelSpec,
)
from decision_stable_causal_attribution.diagnostics.overlap import (
    OverlapDiagnosticEngine,
)
from decision_stable_causal_attribution.estimators.aipw import AIPWEstimator
from decision_stable_causal_attribution.experiments.selector import (
    ExperimentSelector,
)
from decision_stable_causal_attribution.schemas.dataset import DatasetBundle
from decision_stable_causal_attribution.schemas.metadata import GroundTruthStatus
from decision_stable_causal_attribution.schemas.results import (
    EvaluationSummary,
    MetricStatus,
    MetricValue,
    ResearchResult,
)
from decision_stable_causal_attribution.sensitivity.bounds import SensitivityEngine

logger = logging.getLogger(__name__)


class ScenarioPipeline:
    """Executes the full causal estimation and decision pipeline for a single dataset."""

    def __init__(
        self,
        n_splits: int = 3,
        total_budget: float = 10_000.0,
        random_seed: int = 42,
    ) -> None:
        self.aipw = AIPWEstimator(n_splits=n_splits, random_state=random_seed)
        self.diagnostics = OverlapDiagnosticEngine()
        self.sensitivity = SensitivityEngine(gamma_min=1.0, gamma_max=3.0, gamma_step=0.2)
        self.optimizer = BudgetOptimizer(total_budget=total_budget)
        self.exp_selector = ExperimentSelector(self.optimizer, budget=total_budget)

    def run(self, bundle: DatasetBundle, run_id: str | None = None) -> ResearchResult:
        """Execute all pipeline steps and assemble validated ResearchResult."""
        now_iso = datetime.now(UTC).isoformat()
        dataset_id = bundle.metadata.dataset_id
        if run_id is None:
            run_id = f"run-{dataset_id}-{int(datetime.now(UTC).timestamp())}"

        # 1. Extract channel IDs from exposures
        channel_ids = (
            [exp.channel_id for exp in bundle.observations[0].exposures]
            if bundle.observations
            else []
        )

        effect_estimates = []
        overlap_diagnostics = []
        sensitivity_analyses = []
        channel_specs: list[ChannelSpec] = []
        channel_bounds_by_gamma: dict[float, dict[str, tuple[float, float]]] = {
            g: {} for g in self.sensitivity.gammas
        }
        any_overlap_failed = False
        warnings = []

        # 2. Estimate effects and sensitivity per channel
        for ch in channel_ids:
            try:
                aipw_res = self.aipw.estimate_channel(bundle, ch)
                effect_est = self.aipw.to_effect_estimate(aipw_res)
                effect_estimates.append(effect_est)

                # Overlap diagnostics
                diag = self.diagnostics.evaluate(bundle, ch, aipw_res.propensity_scores)
                overlap_diagnostics.append(diag)
                if diag.overlap_flag:
                    any_overlap_failed = True
                    warnings.append(f"Channel {ch} flagged with poor overlap.")

                # Sensitivity bounds
                sens = self.sensitivity.compute_bounds(bundle, ch, aipw_res)
                sensitivity_analyses.append(sens)

                for pt in sens.points:
                    channel_bounds_by_gamma[pt.gamma][ch] = (pt.lower_bound, pt.upper_bound)

                # Channel costs from exposures
                obs_with_exp = [
                    e
                    for obs in bundle.observations
                    for e in obs.exposures
                    if e.channel_id == ch and e.cost > 0
                ]
                cost = obs_with_exp[0].cost if obs_with_exp else 1.0

                channel_specs.append(
                    ChannelSpec(
                        channel_id=ch,
                        effect_estimate=aipw_res.ate,
                        cost_per_exposure=cost,
                        saturation=0.1,
                    )
                )
            except Exception as e:
                logger.error("Error processing channel %s: %s", ch, e)
                warnings.append(f"Failed to estimate channel {ch}: {str(e)}")

        # 3. Budget decisions (point-estimate and robust minimax)
        budget_decisions = []
        if channel_specs:
            point_decision = self.optimizer.evaluate_decision(
                channels=channel_specs,
                strategy="point_estimate",
                overlap_flag=any_overlap_failed,
                gamma_grid=self.sensitivity.gammas,
                channel_bounds_by_gamma=channel_bounds_by_gamma,
            )
            robust_decision = self.optimizer.evaluate_decision(
                channels=channel_specs,
                strategy="robust_minimax_regret",
                overlap_flag=any_overlap_failed,
                gamma_grid=self.sensitivity.gammas,
                channel_bounds_by_gamma=channel_bounds_by_gamma,
            )
            budget_decisions.extend([point_decision, robust_decision])

        # 4. Experiment candidates
        experiment_candidates = []
        if channel_specs:
            intervals_at_breakdown = {
                ch: channel_bounds_by_gamma.get(1.5, {}).get(ch, (0.0, 0.0)) for ch in channel_ids
            }
            experiment_candidates = self.exp_selector.rank_experiments(
                channels=channel_specs,
                channel_intervals=intervals_at_breakdown,
                sample_size=5_000,
                policy="voi_optimal",
            )

        # 5. Evaluation metrics against ground truth (if available)
        eval_metrics = []
        truth_map = {rec.treatment_id: rec.effect for rec in bundle.ground_truth}
        if bundle.metadata.ground_truth_status == GroundTruthStatus.AVAILABLE and truth_map:
            biases = []
            coverages = []
            for est in effect_estimates:
                if est.treatment_id in truth_map:
                    true_ate = truth_map[est.treatment_id]
                    bias = est.estimate - true_ate
                    biases.append(bias)

                    # Check 95% CI coverage
                    if est.confidence_interval:
                        lo, hi = est.confidence_interval
                        coverages.append(1.0 if lo <= true_ate <= hi else 0.0)

            rmse = float(np.sqrt(np.mean(np.array(biases) ** 2))) if biases else 0.0
            mean_bias = float(np.mean(biases)) if biases else 0.0
            cov_rate = float(np.mean(coverages)) if coverages else 0.0

            eval_metrics.extend(
                [
                    MetricValue(
                        name="mean_bias",
                        value=round(mean_bias, 4),
                        status=MetricStatus.AVAILABLE,
                        reason=None,
                        unit="effect_units",
                    ),
                    MetricValue(
                        name="rmse",
                        value=round(rmse, 4),
                        status=MetricStatus.AVAILABLE,
                        reason=None,
                        unit="effect_units",
                    ),
                    MetricValue(
                        name="confidence_interval_coverage",
                        value=round(cov_rate, 4),
                        status=MetricStatus.AVAILABLE,
                        reason=None,
                        unit="rate",
                    ),
                ]
            )
        else:
            eval_metrics.append(
                MetricValue(
                    name="mean_bias",
                    value=None,
                    status=MetricStatus.UNAVAILABLE,
                    reason="Ground truth counterfactuals are not available for observational data.",
                    unit=None,
                )
            )

        eval_summary = EvaluationSummary(
            scenario_id=bundle.metadata.dataset_id,
            repetitions=1,
            metrics=eval_metrics,
            notes=[f"Dataset source: {bundle.metadata.source}"],
        )

        return ResearchResult(
            schema_version="1.0.0",
            run_id=run_id,
            dataset_id=dataset_id,
            generated_at=now_iso,
            effect_estimates=effect_estimates,
            overlap_diagnostics=overlap_diagnostics,
            sensitivity_analyses=sensitivity_analyses,
            budget_decisions=budget_decisions,
            experiment_candidates=experiment_candidates,
            evaluation=eval_summary,
            warnings=warnings,
        )


def export_research_result_json(
    result: ResearchResult,
    output_path: Path | str = "results/research_result.json",
) -> Path:
    """Serialize ResearchResult to JSON matching the shared frontend schema.

    Guarantees OS-independent path handling and UTF-8 encoding.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    json_data = result.model_dump_json(indent=2)
    with dest.open("w", encoding="utf-8") as f:
        f.write(json_data)

    logger.info("Successfully exported ResearchResult to %s", dest)
    return dest
