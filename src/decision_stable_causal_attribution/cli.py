"""Command-line interface for Decision-Stable Causal Attribution.

Provides unified, OS-independent commands for simulation, estimation, benchmarking,
and JSON result generation for Person 2's D3.js frontend.
"""

import argparse
import logging
from pathlib import Path

from decision_stable_causal_attribution.adapters.criteo import build_criteo_dataset_bundle
from decision_stable_causal_attribution.adapters.ga4 import (
    extract_ga4_cohort_to_parquet,
    load_ga4_dataset_bundle,
)
from decision_stable_causal_attribution.evaluation.harness import (
    ScenarioPipeline,
    export_research_result_json,
)
from decision_stable_causal_attribution.simulator.config import (
    ChannelConfig,
    SimulationScenario,
    SimulatorConfig,
)
from decision_stable_causal_attribution.simulator.generator import (
    SyntheticJourneyGenerator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("causal_attribution_cli")


def _get_default_scenario_config(
    scenario: SimulationScenario,
    population_size: int = 5_000,
    seed: int = 42,
) -> SimulatorConfig:
    """Construct a default SimulatorConfig for named scenarios."""
    channels = (
        ChannelConfig(
            channel_id="paid_search",
            exposure_probability=0.35,
            treatment_effect=0.12,
            cost_per_exposure=1.50,
            saturation=0.10,
            confounding_sensitivity=0.80,
        ),
        ChannelConfig(
            channel_id="social_display",
            exposure_probability=0.25,
            treatment_effect=0.08,
            cost_per_exposure=0.80,
            saturation=0.12,
            confounding_sensitivity=0.60,
        ),
        ChannelConfig(
            channel_id="retargeting",
            exposure_probability=0.20,
            treatment_effect=0.04,
            cost_per_exposure=2.00,
            saturation=0.15,
            confounding_sensitivity=0.95,
        ),
    )

    return SimulatorConfig(
        config_version="1.0.0",
        scenario=scenario,
        random_seed=seed,
        population_size=population_size,
        hidden_intent_strength=0.70,
        proxy_quality=0.50,
        baseline_conversion_probability=0.03,
        overlap_strength=0.70 if scenario != SimulationScenario.POOR_OVERLAP else 0.10,
        conversion_saturation=0.80,
        channels=channels,
    )


def cmd_simulate(args: argparse.Namespace) -> None:
    """Generate synthetic journey data and run full estimation pipeline."""
    scenario_enum = SimulationScenario(args.scenario)
    logger.info("Generating scenario: %s (N=%d, seed=%d)", scenario_enum.value, args.n, args.seed)

    cfg = _get_default_scenario_config(scenario_enum, population_size=args.n, seed=args.seed)
    gen = SyntheticJourneyGenerator(cfg)
    bundle = gen.generate()

    logger.info("Running causal estimation, diagnostics, sensitivity, and decision engine...")
    pipeline = ScenarioPipeline(total_budget=args.budget, random_seed=args.seed)
    result = pipeline.run(bundle)

    output_path = Path(args.output)
    export_research_result_json(result, output_path)
    logger.info("Simulation complete! ResearchResult exported to %s", output_path)


def cmd_download_ga4(args: argparse.Namespace) -> None:
    """Extract GA4 30-day cohort from BigQuery."""
    logger.info("Extracting GA4 sample ecommerce cohort from BigQuery...")
    path = extract_ga4_cohort_to_parquet(
        project_id=args.project,
        output_path=args.output,
        limit=args.limit,
    )
    logger.info("GA4 cohort successfully extracted to: %s", path)


def cmd_run_ga4(args: argparse.Namespace) -> None:
    """Run pipeline on the extracted GA4 observational cohort."""
    logger.info("Loading GA4 dataset bundle from %s...", args.input)
    bundle = load_ga4_dataset_bundle(parquet_path=args.input, max_records=args.max_records)

    logger.info("Running observational diagnostics and sensitivity analysis on GA4...")
    pipeline = ScenarioPipeline(total_budget=args.budget)
    result = pipeline.run(bundle)

    output_path = Path(args.output)
    export_research_result_json(result, output_path)
    logger.info("GA4 evaluation complete! ResearchResult exported to %s", output_path)


def cmd_run_criteo(args: argparse.Namespace) -> None:
    """Run pipeline on the Criteo uplift randomized benchmark."""
    logger.info("Loading Criteo uplift sample from %s (N=%d)...", args.input, args.sample_size)
    bundle = build_criteo_dataset_bundle(
        file_path=args.input,
        sample_size=args.sample_size,
        random_seed=args.seed,
    )

    logger.info("Running causal pipeline on Criteo randomized benchmark...")
    pipeline = ScenarioPipeline(total_budget=args.budget, random_seed=args.seed)
    result = pipeline.run(bundle)

    output_path = Path(args.output)
    export_research_result_json(result, output_path)
    logger.info("Criteo benchmark complete! ResearchResult exported to %s", output_path)


def cmd_evaluate_all(args: argparse.Namespace) -> None:
    """Run all 6 scenarios across multiple seeds and print summary evaluation."""
    scenarios = list(SimulationScenario)
    logger.info("Evaluating all %d scenarios with N=%d across seeds...", len(scenarios), args.n)

    summary_rows = []
    for sc in scenarios:
        for seed in [42, 101, 2024]:
            cfg = _get_default_scenario_config(sc, population_size=args.n, seed=seed)
            gen = SyntheticJourneyGenerator(cfg)
            bundle = gen.generate()

            pipeline = ScenarioPipeline(total_budget=args.budget, random_seed=seed)
            res = pipeline.run(bundle)

            # Extract metrics
            metric_map = {m.name: m.value for m in res.evaluation.metrics} if res.evaluation else {}
            decision = res.budget_decisions[0] if res.budget_decisions else None

            summary_rows.append(
                {
                    "scenario": sc.value,
                    "seed": seed,
                    "rmse": metric_map.get("rmse"),
                    "ci_coverage": metric_map.get("confidence_interval_coverage"),
                    "stability": decision.stability.value if decision else "unknown",
                    "breakdown_gamma": decision.breakdown_gamma.value if decision else None,
                }
            )

    import pandas as pd

    df_sum = pd.DataFrame(summary_rows)
    print("\n=== MULTI-SCENARIO EVALUATION SUMMARY ===")
    print(df_sum.to_string(index=False))


def main() -> None:
    """CLI entry point dispatch."""
    parser = argparse.ArgumentParser(
        description="Decision-Stable Causal Attribution research backend CLI (Person 1)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # simulate
    p_sim = subparsers.add_parser("simulate", help="Run synthetic simulation scenario.")
    p_sim.add_argument(
        "--scenario",
        default="mild_confounding",
        choices=[s.value for s in SimulationScenario],
        help="Named simulation scenario.",
    )
    p_sim.add_argument("--n", type=int, default=5_000, help="Sample population size.")
    p_sim.add_argument("--seed", type=int, default=42, help="Random seed.")
    p_sim.add_argument("--budget", type=float, default=10_000.0, help="Total marketing budget.")
    p_sim.add_argument(
        "--output",
        default="results/research_result.json",
        help="Path for exported ResearchResult JSON.",
    )
    p_sim.set_defaults(func=cmd_simulate)

    # download-ga4
    p_dl = subparsers.add_parser("download-ga4", help="Extract GA4 cohort from BigQuery.")
    p_dl.add_argument("--project", default="rime-hackathon-mcp", help="GCP project ID.")
    p_dl.add_argument(
        "--output",
        default="data/processed/ga4_cohort.parquet",
        help="Target Parquet path.",
    )
    p_dl.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    p_dl.set_defaults(func=cmd_download_ga4)

    # run-ga4
    p_ga4 = subparsers.add_parser("run-ga4", help="Run pipeline on GA4 observational cohort.")
    p_ga4.add_argument(
        "--input",
        default="data/processed/ga4_cohort.parquet",
        help="Input GA4 Parquet path.",
    )
    p_ga4.add_argument("--max-records", type=int, default=10_000, help="Max records to analyze.")
    p_ga4.add_argument("--budget", type=float, default=10_000.0, help="Total budget.")
    p_ga4.add_argument(
        "--output",
        default="results/ga4_research_result.json",
        help="Output JSON path.",
    )
    p_ga4.set_defaults(func=cmd_run_ga4)

    # run-criteo
    p_crit = subparsers.add_parser(
        "run-criteo", help="Run pipeline on Criteo randomized benchmark."
    )
    p_crit.add_argument(
        "--input",
        default="data/raw/criteo-uplift-v2.1.csv.gz",
        help="Input Criteo file path.",
    )
    p_crit.add_argument("--sample-size", type=int, default=20_000, help="Sample size.")
    p_crit.add_argument("--seed", type=int, default=42, help="Random seed.")
    p_crit.add_argument("--budget", type=float, default=10_000.0, help="Total budget.")
    p_crit.add_argument(
        "--output",
        default="results/criteo_research_result.json",
        help="Output JSON path.",
    )
    p_crit.set_defaults(func=cmd_run_criteo)

    # evaluate-all
    p_eval = subparsers.add_parser("evaluate-all", help="Run multi-scenario batch evaluation.")
    p_eval.add_argument("--n", type=int, default=3_000, help="Sample size per scenario.")
    p_eval.add_argument("--budget", type=float, default=10_000.0, help="Total budget.")
    p_eval.set_defaults(func=cmd_evaluate_all)

    parsed_args = parser.parse_args()
    parsed_args.func(parsed_args)


if __name__ == "__main__":
    main()
