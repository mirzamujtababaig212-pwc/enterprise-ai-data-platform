from __future__ import annotations

from ml.models.vehicle_risk import MODEL_NAME
from ml.registry import ModelRegistryManager


def main() -> None:
    registry = ModelRegistryManager()

    versions = registry.list_versions(MODEL_NAME)

    if len(versions) < 2:
        raise RuntimeError("Rollback test requires at least two registered model versions")

    champion = registry.get_champion(MODEL_NAME)

    original_champion = str(champion.version)

    validated_versions = [
        version for version in versions if version.tags.get("validation_status") == "PASSED"
    ]

    if len(validated_versions) < 2:
        raise RuntimeError("Rollback test requires at least two validated model versions")

    rollback_candidates = [
        version for version in validated_versions if str(version.version) != original_champion
    ]

    if not rollback_candidates:
        raise RuntimeError("No alternative validated version available for rollback")

    rollback_target = sorted(
        rollback_candidates,
        key=lambda item: int(item.version),
    )[0]

    rollback_version = str(rollback_target.version)

    print("=" * 70)
    print("VEHICLE RISK MODEL ROLLBACK TEST")
    print("=" * 70)

    print(
        "Original champion:",
        original_champion,
    )

    print(
        "Rollback target:",
        rollback_version,
    )

    print()
    print("STEP 1: Promoting rollback target")

    registry.promote_to_champion(
        model_name=MODEL_NAME,
        version=rollback_version,
    )

    champion_after_promotion = registry.get_champion(MODEL_NAME)

    if str(champion_after_promotion.version) != rollback_version:
        raise AssertionError("Rollback target was not promoted to champion")

    print(
        "Champion after promotion:",
        champion_after_promotion.version,
    )

    print()
    print("STEP 2: Restoring original champion")

    registry.rollback_to_version(
        model_name=MODEL_NAME,
        version=original_champion,
    )

    restored_champion = registry.get_champion(MODEL_NAME)

    if str(restored_champion.version) != original_champion:
        raise AssertionError("Original champion was not restored")

    print(
        "Restored champion:",
        restored_champion.version,
    )

    print()
    print("STEP 3: Verifying registry invariants")

    versions_after = registry.list_versions(MODEL_NAME)

    logical_champions = [
        version for version in versions_after if version.tags.get("deployment_status") == "CHAMPION"
    ]

    if len(logical_champions) != 1:
        raise AssertionError("Registry must contain exactly one logical CHAMPION version")

    only_champion = logical_champions[0]

    if str(only_champion.version) != original_champion:
        raise AssertionError("Logical champion does not match champion alias")

    if only_champion.tags.get("validation_status") != "PASSED":
        raise AssertionError("Champion is not validation-approved")

    print(
        "Logical champion:",
        only_champion.version,
    )

    print(
        "Validation status:",
        only_champion.tags.get("validation_status"),
    )

    print(
        "Deployment status:",
        only_champion.tags.get("deployment_status"),
    )

    print()
    print("=" * 70)
    print("ROLLBACK TEST: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
