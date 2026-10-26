from __future__ import annotations

import pandas as pd

from ml.models.vehicle_risk import MODEL_NAME as VEHICLE_MODEL_NAME
from ml.registry import ModelRegistryManager
from ml.scripts.register_vehicle_risk import vehicle_training_dataframe
from ml.training import TrainingConfig, VehicleRiskTrainer
from ml.training.customer_churn import CustomerChurnTrainer
from ml.training.loan_default import LoanDefaultTrainer


def customer_churn_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "tenure_months": 6 + (i % 30),
                "monthly_charges": 50.0 + (i % 8) * 8.0,
                "total_charges": 300.0 + i * 75.0,
                "support_tickets": i % 7,
                "usage_hours": 20.0 + (i % 10) * 5.0,
                "payment_failures": i % 4,
            }
            for i in range(40)
        ]
    )


def loan_default_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "income": 30000.0 + (i % 10) * 5000.0,
                "age": 25 + (i % 35),
                "credit_score": 520 + (i % 20) * 15,
                "loan_amount": 10000.0 + (i % 12) * 5000.0,
                "employment_years": 1 + (i % 15),
                "debt_to_income": 0.20 + (i % 8) * 0.05,
            }
            for i in range(500)
        ]
    )


def main() -> None:
    registry = ModelRegistryManager()

    cases = [
        (
            VEHICLE_MODEL_NAME,
            VehicleRiskTrainer(),
            vehicle_training_dataframe(),
            "ci-bootstrap-vehicle-risk",
        ),
        (
            "CustomerChurnModel",
            CustomerChurnTrainer(),
            customer_churn_dataframe(),
            "ci-bootstrap-customer-churn",
        ),
        (
            "LoanDefaultModel",
            LoanDefaultTrainer(),
            loan_default_dataframe(),
            "ci-bootstrap-loan-default",
        ),
    ]

    for model_name, trainer, dataframe, run_name in cases:
        print(f"\n=== Bootstrapping {model_name} ===")

        result = trainer.train(
            dataframe,
            TrainingConfig(
                experiment_name="enterprise-ai-platform-ci-bootstrap",
                run_name=run_name,
                random_state=42,
                test_size=0.3,
                dataset_name=f"synthetic-ci-{model_name}",
                dataset_version="1",
                enforce_quality_gate=True,
            ),
        )

        if result.metadata is None:
            raise RuntimeError(f"{model_name}: trainer returned no model metadata")

        registered = registry.register_model(
            model_uri=result.model_uri,
            model_name=model_name,
            run_id=result.run_id,
            metadata=result.metadata,
        )

        registry.promote_to_champion(
            model_name=model_name,
            version=registered.version,
        )

        champion = registry.get_champion(model_name)

        if str(champion.version) != registered.version:
            raise RuntimeError(
                f"{model_name}: champion version mismatch; "
                f"registered={registered.version}, champion={champion.version}"
            )

        if champion.tags.get("validation_status") != "PASSED":
            raise RuntimeError(f"{model_name}: champion did not pass validation")

        if champion.tags.get("deployment_status") != "CHAMPION":
            raise RuntimeError(f"{model_name}: deployment status is not CHAMPION")

        print(f"Registered version: {registered.version}")
        print(f"Champion URI: {registry.get_champion_uri(model_name)}")
        print("Quality gate: PASSED")
        print("Registry lifecycle: PASSED")


if __name__ == "__main__":
    main()
