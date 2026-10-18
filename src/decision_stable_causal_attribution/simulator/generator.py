
"""Synthetic customer journey generation with intervention-based ground truth."""

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from numpy.random import Generator, SeedSequence, default_rng
from scipy.special import expit, logit

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
from decision_stable_causal_attribution.simulator.config import (
    SimulationScenario,
    SimulatorConfig,
)


@dataclass(frozen=True)
class _GeneratedPopulation:
    """Internal arrays used to generate observations and causal ground truth."""

    age_group: np.ndarray
    device_mobile: np.ndarray
    observed_intent_proxy: np.ndarray
    hidden_intent: np.ndarray
    exposures: np.ndarray
    outcomes: np.ndarray


class SyntheticJourneyGenerator:
    """Generate reproducible synthetic customer journeys.

    The outcome model is logistic. Channel effects are additive on the
    log-odds scale, with diminishing returns when multiple channels are
    exposed. Ground truth is the average marginal effect of each channel,
    toggling that channel while holding all other observed exposures fixed.

    Hidden intent influences both exposure assignment and conversion, but
    is deliberately excluded from the observed covariates.
    """

    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config

    def generate(self) -> DatasetBundle:
        """Generate a complete dataset and its simulation ground truth."""
        rng = default_rng(SeedSequence(self.config.random_seed))
        population = self._generate_population(rng)
        observations = self._build_observations(population)
        truth = self._build_ground_truth(population)

        return DatasetBundle(
            metadata=DatasetMetadata(
                dataset_id=(
                    f"simulator-{self.config.scenario.value}"
                    f"-seed-{self.config.random_seed}"
                ),
                dataset_kind=DatasetKind.SIMULATION,
                source="synthetic_journey_generator",
                description=(
                    "Synthetic customer journeys generated from a structural "
                    "data-generating process. Ground truth contains expected "
                    "average marginal effects under channel interventions."
                ),
                unit_of_analysis="customer",
                treatment_definition=(
                    "Binary exposure indicator for each configured marketing "
                    "channel, recorded separately in each customer's exposures."
                ),
                outcome_definition=(
                    "Binary conversion sampled from the structural outcome "
                    "probability determined by customer covariates and channel "
                    "exposures."
                ),
                ground_truth_status=GroundTruthStatus.AVAILABLE,
                observational_caveat=None,
            ),
            observations=tuple(observations),
            ground_truth=tuple(truth),
        )

    def _generate_population(self, rng: Generator) -> _GeneratedPopulation:
        """Generate covariates, exposures, and observed outcomes."""
        n = self.config.population_size
        channels = self.config.channels
        scenario = self.config.scenario

        hidden_intent = rng.normal(loc=0.0, scale=1.0, size=n)
        age_group = rng.integers(0, 3, size=n)
        device_mobile = rng.binomial(1, 0.4, size=n)

        proxy_noise = rng.normal(loc=0.0, scale=1.0, size=n)
        proxy_quality = self.config.proxy_quality
        observed_intent_proxy = (
            np.sqrt(proxy_quality) * hidden_intent
            + np.sqrt(1.0 - proxy_quality) * proxy_noise
        )

        exposures = np.zeros((n, len(channels)), dtype=np.int8)

        for j, channel in enumerate(channels):
            assignment_logit = float(logit(channel.exposure_probability))
            assignment_strength = (
                self.config.hidden_intent_strength
                * channel.confounding_sensitivity
            )
            assignment_logit += assignment_strength * hidden_intent

            if scenario == SimulationScenario.RETARGETING_TRAP:
                assignment_logit += 1.5 * hidden_intent
            elif scenario == SimulationScenario.POOR_OVERLAP:
                assignment_logit += 5.0 * hidden_intent
            elif scenario == SimulationScenario.HIDDEN_INTENT_HEAVY:
                assignment_logit += 2.5 * hidden_intent

            # Reduce assignment discrimination when overlap_strength is high.
            # The dedicated poor-overlap scenario intentionally overrides this.
            if scenario != SimulationScenario.POOR_OVERLAP:
                assignment_logit *= (
                    1.0 + 3.0 * (1.0 - self.config.overlap_strength)
                )

            assignment_probability = expit(assignment_logit)
            exposures[:, j] = rng.binomial(
                1,
                assignment_probability,
            ).astype(np.int8)

        outcome_probability = self._outcome_probability(
            age_group=age_group,
            device_mobile=device_mobile,
            hidden_intent=hidden_intent,
            exposures=exposures,
        )
        outcomes = rng.binomial(1, outcome_probability).astype(np.int8)

        return _GeneratedPopulation(
            age_group=age_group,
            device_mobile=device_mobile,
            observed_intent_proxy=observed_intent_proxy,
            hidden_intent=hidden_intent,
            exposures=exposures,
            outcomes=outcomes,
        )

    def _outcome_probability(
        self,
        age_group: np.ndarray,
        device_mobile: np.ndarray,
        hidden_intent: np.ndarray,
        exposures: np.ndarray,
    ) -> np.ndarray:
        """Calculate conversion probabilities for an exposure matrix."""
        baseline_logit = float(logit(self.config.baseline_conversion_probability))
        baseline_logit_array = (
            baseline_logit
            + 0.7 * hidden_intent
            + 0.2 * age_group
            - 0.25 * device_mobile
        )

        exposed_channel_count = exposures.sum(axis=1).astype(float)
        linear_predictor = baseline_logit_array.copy()

        for j, channel in enumerate(self.config.channels):
            effect = channel.treatment_effect

            if (
                self.config.scenario == SimulationScenario.NULL_CHANNEL
                and j == 0
            ):
                effect = 0.0

            if self.config.scenario == SimulationScenario.HETEROGENEOUS_EFFECTS:
                effect = effect * (0.5 + age_group / 2.0)

            # A channel's contribution is reduced when other channels are
            # also present. With only one exposed channel, no co-exposure
            # discount is applied.
            co_exposure_count = np.maximum(exposed_channel_count - 1.0, 0.0)
            saturation_denominator = 1.0 + channel.saturation * co_exposure_count

            contribution = (
                effect
                * self.config.conversion_saturation
                * exposures[:, j]
                / saturation_denominator
            )
            linear_predictor += contribution

        return np.asarray(expit(linear_predictor), dtype=float)

    def _build_observations(
        self,
        population: _GeneratedPopulation,
    ) -> list[ObservationRecord]:
        """Convert internal arrays to validated observation records."""
        records: list[ObservationRecord] = []
        channels = self.config.channels
        timestamp = datetime(2026, 1, 1, tzinfo=UTC)

        for i in range(self.config.population_size):
            exposure_records = tuple(
                ExposureRecord(
                    channel_id=channel.channel_id,
                    exposed=bool(population.exposures[i, j]),
                    timestamp=timestamp,
                    cost=(
                        channel.cost_per_exposure
                        if population.exposures[i, j] == 1
                        else 0.0
                    ),
                )
                for j, channel in enumerate(channels)
            )

            records.append(
                ObservationRecord(
                    unit_id=f"customer-{i:08d}",
                    covariates=Covariates(
                        values={
                            "age_group": int(population.age_group[i]),
                            "device_mobile": int(population.device_mobile[i]),
                            "intent_proxy": float(
                                population.observed_intent_proxy[i]
                            ),
                        }
                    ),
                    # Treatment is represented per channel in `exposures`.
                    # A single scalar treatment would be ambiguous here.
                    treatment=None,
                    outcome=float(population.outcomes[i]),
                    outcome_observed=True,
                    treatment_timestamp=None,
                    exposures=exposure_records,
                    attributes={
                        "scenario": self.config.scenario.value,
                        "random_seed": self.config.random_seed,
                    },
                )
            )

        return records

    def _build_ground_truth(
        self,
        population: _GeneratedPopulation,
    ) -> list[GroundTruthRecord]:
        """Compute expected average marginal effects for each channel.

        For each channel, compare the expected outcome when that channel is
        switched on versus off, holding all other channels at their observed
        values. The same population-level latent variables are used for both
        counterfactual predictions, avoiding Monte Carlo noise in the effect.
        """
        truth: list[GroundTruthRecord] = []

        for j, channel in enumerate(self.config.channels):
            exposures_off = population.exposures.copy()
            exposures_on = population.exposures.copy()

            exposures_off[:, j] = 0
            exposures_on[:, j] = 1

            probability_off = self._outcome_probability(
                age_group=population.age_group,
                device_mobile=population.device_mobile,
                hidden_intent=population.hidden_intent,
                exposures=exposures_off,
            )
            probability_on = self._outcome_probability(
                age_group=population.age_group,
                device_mobile=population.device_mobile,
                hidden_intent=population.hidden_intent,
                exposures=exposures_on,
            )

            effect = float(np.mean(probability_on - probability_off))

            truth.append(
                GroundTruthRecord(
                    estimand=Estimand.ATE,
                    treatment_id=channel.channel_id,
                    outcome_id="conversion",
                    effect=effect,
                    population_id="simulated_population",
                    sample_size=self.config.population_size,
                    replicate_id=f"seed-{self.config.random_seed}",
                )
            )

        return truth
