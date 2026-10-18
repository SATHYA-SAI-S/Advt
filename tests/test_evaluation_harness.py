"""Tests for evaluation harness and JSON schema compliance."""

import json

import pytest

from decision_stable_causal_attribution.evaluation.harness import (
    ScenarioPipeline,
    export_research_result_json,
)
from decision_stable_causal_attribution.schemas.results import ResearchResult
from decision_stable_causal_attribution.simulator.config import (
    ChannelConfig,
    SimulationScenario,
    SimulatorConfig,
)
from decision_stable_causal_attribution.simulator.generator import (
    SyntheticJourneyGenerator,
)


@pytest.fixture
def sim_bundle():
    channels = (
        ChannelConfig(
            channel_id="ch_search",
            exposure_probability=0.35,
            treatment_effect=0.10,
            cost_per_exposure=1.5,
            saturation=0.1,
            confounding_sensitivity=0.6,
        ),
        ChannelConfig(
            channel_id="ch_display",
            exposure_probability=0.25,
            treatment_effect=0.05,
            cost_per_exposure=0.8,
            saturation=0.1,
            confounding_sensitivity=0.5,
        ),
    )
    cfg = SimulatorConfig(
        scenario=SimulationScenario.MILD_CONFOUNDING,
        random_seed=42,
        population_size=500,
        hidden_intent_strength=0.5,
        proxy_quality=0.6,
        baseline_conversion_probability=0.04,
        overlap_strength=0.8,
        conversion_saturation=0.9,
        channels=channels,
    )
    return SyntheticJourneyGenerator(cfg).generate()


def test_pipeline_end_to_end(sim_bundle, tmp_path):
    """Verify ScenarioPipeline runs end-to-end and exports valid ResearchResult JSON."""
    pipeline = ScenarioPipeline(n_splits=2, total_budget=5_000.0, random_seed=42)
    result = pipeline.run(sim_bundle)

    assert isinstance(result, ResearchResult)
    assert result.schema_version == "1.0.0"
    assert len(result.effect_estimates) == 2
    assert len(result.overlap_diagnostics) == 2
    assert len(result.sensitivity_analyses) == 2
    assert len(result.budget_decisions) == 2
    assert len(result.experiment_candidates) > 0
    assert result.evaluation is not None

    # Test JSON export and round-trip load
    export_file = tmp_path / "test_research_result.json"
    exported_path = export_research_result_json(result, export_file)
    assert exported_path.exists()

    with exported_path.open("r", encoding="utf-8") as f:
        loaded_dict = json.load(f)

    # Re-validate with Pydantic
    reloaded_obj = ResearchResult.model_validate(loaded_dict)
    assert reloaded_obj.dataset_id == result.dataset_id
