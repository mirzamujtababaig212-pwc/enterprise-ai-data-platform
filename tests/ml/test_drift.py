from __future__ import annotations

import pytest

from ml.monitoring.drift import (
    PSI_EPSILON,
    calculate_population_stability_index,
)


def test_psi_is_zero_for_identical_distributions() -> None:
    result = calculate_population_stability_index(
        reference_counts=(25, 25, 25, 25),
        observation_counts=(25, 25, 25, 25),
    )

    assert result == pytest.approx(0.0)


def test_psi_is_positive_for_shifted_distribution() -> None:
    result = calculate_population_stability_index(
        reference_counts=(25, 25, 25, 25),
        observation_counts=(10, 10, 40, 40),
    )

    assert result > 0.0


def test_psi_is_deterministic() -> None:
    reference_counts = (50, 30, 15, 5)
    observation_counts = (40, 30, 20, 10)

    first = calculate_population_stability_index(
        reference_counts=reference_counts,
        observation_counts=observation_counts,
    )
    second = calculate_population_stability_index(
        reference_counts=reference_counts,
        observation_counts=observation_counts,
    )

    assert first == second


def test_psi_handles_zero_reference_bins() -> None:
    result = calculate_population_stability_index(
        reference_counts=(100, 0),
        observation_counts=(50, 50),
    )

    assert result > 0.0
    assert result != float("inf")


def test_psi_handles_zero_observation_bins() -> None:
    result = calculate_population_stability_index(
        reference_counts=(50, 50),
        observation_counts=(100, 0),
    )

    assert result > 0.0
    assert result != float("inf")


def test_psi_uses_configured_epsilon() -> None:
    assert PSI_EPSILON > 0.0
    assert PSI_EPSILON < 1.0


@pytest.mark.parametrize(
    ("reference_counts", "observation_counts"),
    [
        ((10, 20), (10,)),
        ((10,), (10, 20)),
        ((), (10,)),
        ((10,), ()),
    ],
)
def test_psi_rejects_mismatched_or_empty_distributions(
    reference_counts: tuple[int, ...],
    observation_counts: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        calculate_population_stability_index(
            reference_counts=reference_counts,
            observation_counts=observation_counts,
        )


@pytest.mark.parametrize(
    ("reference_counts", "observation_counts"),
    [
        ((10, -1), (5, 6)),
        ((5, 6), (10, -1)),
    ],
)
def test_psi_rejects_negative_counts(
    reference_counts: tuple[int, ...],
    observation_counts: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="counts must be non-negative"):
        calculate_population_stability_index(
            reference_counts=reference_counts,
            observation_counts=observation_counts,
        )


def test_psi_rejects_distributions_with_no_observations() -> None:
    with pytest.raises(ValueError, match="must contain at least one observation"):
        calculate_population_stability_index(
            reference_counts=(0, 0),
            observation_counts=(0, 0),
        )


def test_psi_is_zero_when_distributions_are_proportional() -> None:
    result = calculate_population_stability_index(
        reference_counts=(10, 20, 30),
        observation_counts=(100, 200, 300),
    )

    assert result == pytest.approx(0.0)


def test_psi_zero_bin_smoothing_does_not_produce_infinite_result() -> None:
    result = calculate_population_stability_index(
        reference_counts=(100, 0, 100),
        observation_counts=(0, 100, 100),
    )

    assert result > 0.0
    assert result < float("inf")


def test_feature_drift_calculates_psi_for_aligned_observation_window() -> None:
    from datetime import UTC, datetime

    import pandas as pd

    from ml.contracts import FeatureContract, FeatureDefinition
    from ml.monitoring.drift import calculate_feature_drift
    from ml.monitoring.observation_window import build_observation_window
    from ml.training.reference_distribution import build_reference_distribution

    contract = FeatureContract(
        name="vehicle-risk",
        version="v1",
        features=(
            FeatureDefinition(name="feature_a", dtype="float"),
            FeatureDefinition(name="feature_b", dtype="int"),
        ),
    )

    training = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [10, 20, 30, 40],
        }
    )

    observations = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 5.0],
            "feature_b": [10, 20, 30, 40],
            "risk": [0, 0, 1, 1],
            "risk_probability": [0.1, 0.2, 0.8, 0.9],
            "model_name": ["TestModel"] * 4,
            "model_version": ["2"] * 4,
            "model_alias": ["champion"] * 4,
            "training_run_id": ["run-123"] * 4,
        }
    )

    reference = build_reference_distribution(
        training,
        contract,
        dataset_name="test-dataset",
        dataset_version="v1",
        training_run_id="run-123",
        model_name="TestModel",
    )

    observation = build_observation_window(
        observations,
        contract,
        window_id="window-001",
        window_start=datetime(2026, 9, 25, tzinfo=UTC),
        window_end=datetime(2026, 9, 26, tzinfo=UTC),
        model_name="TestModel",
        model_version="2",
        model_alias="champion",
        training_run_id="run-123",
        reference_distribution=reference,
    )

    result = calculate_feature_drift(
        reference_distribution=reference,
        observation_window=observation,
    )

    assert result["feature_a"] > 0.0
    assert result["feature_b"] == pytest.approx(0.0)


def test_feature_drift_rejects_mismatched_training_run() -> None:
    from datetime import UTC, datetime

    import pandas as pd

    from ml.contracts import FeatureContract, FeatureDefinition
    from ml.monitoring.drift import calculate_feature_drift
    from ml.monitoring.observation_window import build_observation_window
    from ml.training.reference_distribution import build_reference_distribution

    contract = FeatureContract(
        name="vehicle-risk",
        version="v1",
        features=(FeatureDefinition(name="feature_a", dtype="float"),),
    )

    training = pd.DataFrame({"feature_a": [1.0, 2.0, 3.0, 4.0]})
    observations = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "risk": [0, 0, 1, 1],
            "model_name": ["TestModel"] * 4,
            "model_version": ["2"] * 4,
            "model_alias": ["champion"] * 4,
            "training_run_id": ["run-999"] * 4,
        }
    )

    reference = build_reference_distribution(
        training,
        contract,
        dataset_name="test-dataset",
        dataset_version="v1",
        training_run_id="run-123",
        model_name="TestModel",
    )

    observation = build_observation_window(
        observations,
        contract,
        window_id="window-001",
        window_start=datetime(2026, 9, 25, tzinfo=UTC),
        window_end=datetime(2026, 9, 26, tzinfo=UTC),
        model_name="TestModel",
        model_version="2",
        model_alias="champion",
        training_run_id="run-999",
    )

    with pytest.raises(ValueError, match="training_run_id"):
        calculate_feature_drift(
            reference_distribution=reference,
            observation_window=observation,
        )


def test_feature_drift_rejects_mismatched_model_name() -> None:
    from datetime import UTC, datetime

    import pandas as pd

    from ml.contracts import FeatureContract, FeatureDefinition
    from ml.monitoring.drift import calculate_feature_drift
    from ml.monitoring.observation_window import build_observation_window
    from ml.training.reference_distribution import build_reference_distribution

    contract = FeatureContract(
        name="vehicle-risk",
        version="v1",
        features=(FeatureDefinition(name="feature_a", dtype="float"),),
    )

    training = pd.DataFrame({"feature_a": [1.0, 2.0, 3.0, 4.0]})
    observations = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "risk": [0, 0, 1, 1],
            "model_name": ["OtherModel"] * 4,
            "model_version": ["2"] * 4,
            "model_alias": ["champion"] * 4,
            "training_run_id": ["run-123"] * 4,
        }
    )

    reference = build_reference_distribution(
        training,
        contract,
        dataset_name="test-dataset",
        dataset_version="v1",
        training_run_id="run-123",
        model_name="TestModel",
    )

    observation = build_observation_window(
        observations,
        contract,
        window_id="window-001",
        window_start=datetime(2026, 9, 25, tzinfo=UTC),
        window_end=datetime(2026, 9, 26, tzinfo=UTC),
        model_name="OtherModel",
        model_version="2",
        model_alias="champion",
        training_run_id="run-123",
    )

    with pytest.raises(ValueError, match="model_name"):
        calculate_feature_drift(
            reference_distribution=reference,
            observation_window=observation,
        )


def test_feature_drift_rejects_mismatched_feature_contract() -> None:
    from datetime import UTC, datetime

    import pandas as pd

    from ml.contracts import FeatureContract, FeatureDefinition
    from ml.monitoring.drift import calculate_feature_drift
    from ml.monitoring.observation_window import build_observation_window
    from ml.training.reference_distribution import build_reference_distribution

    reference_contract = FeatureContract(
        name="vehicle-risk",
        version="v1",
        features=(FeatureDefinition(name="feature_a", dtype="float"),),
    )
    observation_contract = FeatureContract(
        name="vehicle-risk",
        version="v2",
        features=(FeatureDefinition(name="feature_a", dtype="float"),),
    )

    training = pd.DataFrame({"feature_a": [1.0, 2.0, 3.0, 4.0]})
    observations = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "risk": [0, 0, 1, 1],
            "model_name": ["TestModel"] * 4,
            "model_version": ["2"] * 4,
            "model_alias": ["champion"] * 4,
            "training_run_id": ["run-123"] * 4,
        }
    )

    reference = build_reference_distribution(
        training,
        reference_contract,
        dataset_name="test-dataset",
        dataset_version="v1",
        training_run_id="run-123",
        model_name="TestModel",
    )

    observation = build_observation_window(
        observations,
        observation_contract,
        window_id="window-001",
        window_start=datetime(2026, 9, 25, tzinfo=UTC),
        window_end=datetime(2026, 9, 26, tzinfo=UTC),
        model_name="TestModel",
        model_version="2",
        model_alias="champion",
        training_run_id="run-123",
    )

    with pytest.raises(ValueError, match="feature contract"):
        calculate_feature_drift(
            reference_distribution=reference,
            observation_window=observation,
        )


def test_feature_drift_rejects_mismatched_feature_names() -> None:
    from datetime import UTC, datetime

    import pandas as pd

    from ml.contracts import FeatureContract, FeatureDefinition
    from ml.monitoring.drift import calculate_feature_drift
    from ml.monitoring.observation_window import build_observation_window
    from ml.training.reference_distribution import build_reference_distribution

    reference_contract = FeatureContract(
        name="vehicle-risk",
        version="v1",
        features=(
            FeatureDefinition(name="feature_a", dtype="float"),
            FeatureDefinition(name="feature_b", dtype="float"),
        ),
    )
    observation_contract = FeatureContract(
        name="vehicle-risk",
        version="v1",
        features=(
            FeatureDefinition(name="feature_a", dtype="float"),
            FeatureDefinition(name="feature_c", dtype="float"),
        ),
    )

    training = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [10.0, 20.0, 30.0, 40.0],
        }
    )
    observations = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_c": [10.0, 20.0, 30.0, 40.0],
            "risk": [0, 0, 1, 1],
            "model_name": ["TestModel"] * 4,
            "model_version": ["2"] * 4,
            "model_alias": ["champion"] * 4,
            "training_run_id": ["run-123"] * 4,
        }
    )

    reference = build_reference_distribution(
        training,
        reference_contract,
        dataset_name="test-dataset",
        dataset_version="v1",
        training_run_id="run-123",
        model_name="TestModel",
    )

    observation = build_observation_window(
        observations,
        observation_contract,
        window_id="window-001",
        window_start=datetime(2026, 9, 25, tzinfo=UTC),
        window_end=datetime(2026, 9, 26, tzinfo=UTC),
        model_name="TestModel",
        model_version="2",
        model_alias="champion",
        training_run_id="run-123",
    )

    with pytest.raises(ValueError, match="features"):
        calculate_feature_drift(
            reference_distribution=reference,
            observation_window=observation,
        )


def test_feature_drift_skips_feature_with_empty_reference_histogram() -> None:
    from datetime import UTC, datetime

    import pandas as pd

    from ml.contracts import FeatureContract, FeatureDefinition
    from ml.monitoring.drift import calculate_feature_drift
    from ml.monitoring.observation_window import build_observation_window
    from ml.training.reference_distribution import build_reference_distribution

    contract = FeatureContract(
        name="vehicle-risk",
        version="v1",
        features=(FeatureDefinition(name="feature_a", dtype="float"),),
    )

    training = pd.DataFrame({"feature_a": [float("nan")] * 4})
    observations = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "risk": [0, 0, 1, 1],
            "model_name": ["TestModel"] * 4,
            "model_version": ["2"] * 4,
            "model_alias": ["champion"] * 4,
            "training_run_id": ["run-123"] * 4,
        }
    )

    reference = build_reference_distribution(
        training,
        contract,
        dataset_name="test-dataset",
        dataset_version="v1",
        training_run_id="run-123",
        model_name="TestModel",
    )

    observation = build_observation_window(
        observations,
        contract,
        window_id="window-001",
        window_start=datetime(2026, 9, 25, tzinfo=UTC),
        window_end=datetime(2026, 9, 26, tzinfo=UTC),
        model_name="TestModel",
        model_version="2",
        model_alias="champion",
        training_run_id="run-123",
    )

    result = calculate_feature_drift(
        reference_distribution=reference,
        observation_window=observation,
    )

    assert result == {}


def test_feature_drift_skips_feature_with_empty_observation_histogram() -> None:
    from datetime import UTC, datetime

    import pandas as pd

    from ml.contracts import FeatureContract, FeatureDefinition
    from ml.monitoring.drift import calculate_feature_drift
    from ml.monitoring.observation_window import build_observation_window
    from ml.training.reference_distribution import build_reference_distribution

    contract = FeatureContract(
        name="vehicle-risk",
        version="v1",
        features=(FeatureDefinition(name="feature_a", dtype="float"),),
    )

    training = pd.DataFrame({"feature_a": [1.0, 2.0, 3.0, 4.0]})
    observations = pd.DataFrame(
        {
            "feature_a": [float("nan")] * 4,
            "risk": [0, 0, 1, 1],
            "model_name": ["TestModel"] * 4,
            "model_version": ["2"] * 4,
            "model_alias": ["champion"] * 4,
            "training_run_id": ["run-123"] * 4,
        }
    )

    reference = build_reference_distribution(
        training,
        contract,
        dataset_name="test-dataset",
        dataset_version="v1",
        training_run_id="run-123",
        model_name="TestModel",
    )

    observation = build_observation_window(
        observations,
        contract,
        window_id="window-001",
        window_start=datetime(2026, 9, 25, tzinfo=UTC),
        window_end=datetime(2026, 9, 26, tzinfo=UTC),
        model_name="TestModel",
        model_version="2",
        model_alias="champion",
        training_run_id="run-123",
    )

    result = calculate_feature_drift(
        reference_distribution=reference,
        observation_window=observation,
    )

    assert result == {}


def test_build_drift_evaluation_captures_lineage_and_feature_psi() -> None:
    from datetime import UTC, datetime

    import pandas as pd

    from ml.contracts import FeatureContract, FeatureDefinition
    from ml.monitoring.drift import (
        build_drift_evaluation,
    )
    from ml.monitoring.observation_window import build_observation_window
    from ml.training.reference_distribution import build_reference_distribution

    contract = FeatureContract(
        name="vehicle-risk",
        version="v1",
        features=(
            FeatureDefinition(name="feature_a", dtype="float"),
            FeatureDefinition(name="feature_b", dtype="int"),
        ),
    )

    training = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [10, 20, 30, 40],
        }
    )

    observations = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 5.0],
            "feature_b": [10, 20, 30, 40],
            "risk": [0, 0, 1, 1],
            "risk_probability": [0.1, 0.2, 0.8, 0.9],
            "model_name": ["TestModel"] * 4,
            "model_version": ["2"] * 4,
            "model_alias": ["champion"] * 4,
            "training_run_id": ["run-123"] * 4,
        }
    )

    reference = build_reference_distribution(
        training,
        contract,
        dataset_name="test-dataset",
        dataset_version="v1",
        training_run_id="run-123",
        model_name="TestModel",
    )

    observation = build_observation_window(
        observations,
        contract,
        window_id="window-001",
        window_start=datetime(2026, 9, 25, tzinfo=UTC),
        window_end=datetime(2026, 9, 26, tzinfo=UTC),
        model_name="TestModel",
        model_version="2",
        model_alias="champion",
        training_run_id="run-123",
        reference_distribution=reference,
    )

    evaluation = build_drift_evaluation(
        reference_distribution=reference,
        observation_window=observation,
        evaluation_id="drift-001",
    )

    assert evaluation.schema_version == "v1"
    assert evaluation.evaluation_id == "drift-001"
    assert evaluation.model_name == "TestModel"
    assert evaluation.model_version == "2"
    assert evaluation.model_alias == "champion"
    assert evaluation.training_run_id == "run-123"
    assert evaluation.observation_window_id == "window-001"
    assert evaluation.feature_contract_name == "vehicle-risk"
    assert evaluation.feature_contract_version == "v1"
    assert evaluation.window_start == observation.window_start
    assert evaluation.window_end == observation.window_end
    assert evaluation.feature_names == ("feature_a", "feature_b")
    assert evaluation.feature_psi["feature_a"] > 0.0
    assert evaluation.feature_psi["feature_b"] == pytest.approx(0.0)


def test_drift_evaluation_serializes_to_json_compatible_dict() -> None:
    from datetime import UTC, datetime

    import pandas as pd

    from ml.contracts import FeatureContract, FeatureDefinition
    from ml.monitoring.drift import build_drift_evaluation
    from ml.monitoring.observation_window import build_observation_window
    from ml.training.reference_distribution import build_reference_distribution

    contract = FeatureContract(
        name="vehicle-risk",
        version="v1",
        features=(FeatureDefinition(name="feature_a", dtype="float"),),
    )

    training = pd.DataFrame({"feature_a": [1.0, 2.0, 3.0, 4.0]})
    observations = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 5.0],
            "risk": [0, 0, 1, 1],
            "model_name": ["TestModel"] * 4,
            "model_version": ["2"] * 4,
            "model_alias": ["champion"] * 4,
            "training_run_id": ["run-123"] * 4,
        }
    )

    reference = build_reference_distribution(
        training,
        contract,
        dataset_name="test-dataset",
        dataset_version="v1",
        training_run_id="run-123",
        model_name="TestModel",
    )

    observation = build_observation_window(
        observations,
        contract,
        window_id="window-001",
        window_start=datetime(2026, 9, 25, tzinfo=UTC),
        window_end=datetime(2026, 9, 26, tzinfo=UTC),
        model_name="TestModel",
        model_version="2",
        model_alias="champion",
        training_run_id="run-123",
        reference_distribution=reference,
    )

    evaluation = build_drift_evaluation(
        reference_distribution=reference,
        observation_window=observation,
        evaluation_id="drift-001",
    )

    payload = evaluation.as_dict()

    assert payload["schema_version"] == "v1"
    assert payload["evaluation_id"] == "drift-001"
    assert payload["model_name"] == "TestModel"
    assert payload["model_version"] == "2"
    assert payload["model_alias"] == "champion"
    assert payload["training_run_id"] == "run-123"
    assert payload["observation_window_id"] == "window-001"
    assert payload["feature_contract_name"] == "vehicle-risk"
    assert payload["feature_contract_version"] == "v1"
    assert payload["window_start"] == "2026-09-25T00:00:00+00:00"
    assert payload["window_end"] == "2026-09-26T00:00:00+00:00"
    assert payload["feature_names"] == ["feature_a"]
    assert payload["feature_psi"]["feature_a"] > 0.0
