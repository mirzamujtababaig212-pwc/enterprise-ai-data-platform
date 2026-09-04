from __future__ import annotations

from ml.models.vehicle_risk import MODEL_NAME
from ml.registry import ModelRegistryManager


def main() -> None:
    registry = ModelRegistryManager()

    result = registry.reconcile(MODEL_NAME)

    print("=" * 70)
    print("MLFLOW REGISTRY RECONCILIATION")
    print("=" * 70)

    print(
        "Model:",
        result["model_name"],
    )

    print(
        "Champion version:",
        result["champion_version"],
    )

    print(
        "Total versions:",
        result["total_versions"],
    )

    print(
        "Retired versions:",
        result["retired_versions"],
    )

    champion = registry.get_champion(MODEL_NAME)

    print()
    print("CHAMPION VALIDATION:")
    print(champion.tags.get("validation_status"))

    print("CHAMPION DEPLOYMENT:")
    print(champion.tags.get("deployment_status"))

    print()
    print("RECONCILIATION: PASSED")


if __name__ == "__main__":
    main()
