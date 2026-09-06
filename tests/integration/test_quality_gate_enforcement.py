from ml.training.schemas import TrainingConfig


def test_quality_gate_enforcement_defaults_to_true() -> None:
    config = TrainingConfig()

    assert config.enforce_quality_gate is True


def test_quality_gate_enforcement_can_be_disabled() -> None:
    config = TrainingConfig(
        experiment_name="test-experiment",
        run_name="test-run",
        enforce_quality_gate=False,
    )

    assert config.enforce_quality_gate is False
