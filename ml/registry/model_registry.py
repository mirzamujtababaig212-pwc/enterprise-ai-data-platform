from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
import os
import mlflow
from mlflow import MlflowClient


@dataclass(frozen=True)
class RegisteredModelResult:
    model_name: str
    version: str
    run_id: str
    model_uri: str
    alias: str


class ModelRegistryManager:
    """
    Production wrapper around the MLflow Model Registry.

    Responsibilities:

    - Register validated model versions.
    - Maintain candidate and champion aliases.
    - Enforce validation before promotion.
    - Maintain exactly one logical CHAMPION version.
    - Retire the previous champion during promotion.
    - Support safe rollback to another validated version.
    - Provide registry reconciliation and inspection helpers.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        registry_uri: str | None = None,
    ) -> None:

        effective_tracking_uri = (
            tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or "http://mlflow:5000"
        )

        registry_uri = registry_uri or os.getenv("MLFLOW_REGISTRY_URI")

        mlflow.set_tracking_uri(effective_tracking_uri)

        if registry_uri:
            mlflow.set_registry_uri(registry_uri)

        self.client = MlflowClient()

    # ------------------------------------------------------------------
    # MODEL REGISTRATION
    # ------------------------------------------------------------------

    def register_model(
        self,
        model_uri: str,
        model_name: str,
        run_id: str,
        evaluation_passed: bool,
    ) -> RegisteredModelResult:
        """
        Register a validated model version.

        Newly registered models become candidates.
        They do not automatically become champion here.
        """

        if not evaluation_passed:
            raise ValueError("Model cannot be registered because evaluation failed")

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )

        version = str(model_version.version)

        self._wait_for_model_version(
            model_name=model_name,
            version=version,
        )

        # Version-level metadata.
        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="validation_status",
            value="PASSED",
        )

        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="source_run_id",
            value=run_id,
        )

        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="model_type",
            value="vehicle_risk_classifier",
        )

        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="deployment_status",
            value="CANDIDATE",
        )

        # Registered-model metadata.
        self.client.set_registered_model_tag(
            name=model_name,
            key="model_type",
            value="vehicle_risk_classifier",
        )

        self.client.set_registered_model_tag(
            name=model_name,
            key="framework",
            value="scikit-learn",
        )

        self.client.set_registered_model_tag(
            name=model_name,
            key="managed_by",
            value="enterprise-ai-platform",
        )

        # Candidate alias.
        self.client.set_registered_model_alias(
            name=model_name,
            alias="candidate",
            version=version,
        )

        return RegisteredModelResult(
            model_name=model_name,
            version=version,
            run_id=run_id,
            model_uri=(f"models:/{model_name}/{version}"),
            alias="candidate",
        )

    # ------------------------------------------------------------------
    # CHAMPION MANAGEMENT
    # ------------------------------------------------------------------

    def promote_to_champion(
        self,
        model_name: str,
        version: str,
    ) -> None:
        """
        Promote a validated model version to champion.

        Invariants:

        - Target version must exist.
        - Target must have validation_status=PASSED.
        - Previous champion is retired.
        - Target receives deployment_status=CHAMPION.
        - champion alias points to target.
        - Exactly one version retains logical CHAMPION status.
        """

        target = self.client.get_model_version(
            name=model_name,
            version=version,
        )

        validation_status = target.tags.get("validation_status")

        if validation_status != "PASSED":
            raise ValueError("Only validated model versions can become champion")

        current_champion = self._get_current_champion(model_name)

        # If the requested version is already champion,
        # simply repair the alias/tag invariants.
        if current_champion is not None and str(current_champion.version) == str(version):
            self.client.set_registered_model_alias(
                name=model_name,
                alias="champion",
                version=str(version),
            )

            self.client.set_model_version_tag(
                name=model_name,
                version=str(version),
                key="deployment_status",
                value="CHAMPION",
            )

            return

        # Retire the existing champion first.
        if current_champion is not None:

            current_version = str(current_champion.version)

            self.client.set_model_version_tag(
                name=model_name,
                version=current_version,
                key="deployment_status",
                value="RETIRED",
            )

        # Promote the target.
        self.client.set_model_version_tag(
            name=model_name,
            version=str(version),
            key="deployment_status",
            value="CHAMPION",
        )

        self.client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=str(version),
        )

        # Candidate and champion may temporarily refer
        # to the same version. This is intentional.
        #
        # Candidate means "latest evaluated candidate".
        # Champion means "currently deployed logical version".

    def rollback_to_version(
        self,
        model_name: str,
        version: str,
    ) -> None:
        """
        Roll back production to a previously validated version.

        Rollback is deliberately implemented through the same
        promotion lifecycle so the registry invariants remain
        identical to a normal promotion.
        """

        target = self.client.get_model_version(
            name=model_name,
            version=version,
        )

        validation_status = target.tags.get("validation_status")

        if validation_status != "PASSED":
            raise ValueError("Only validated model versions can be used for rollback")

        self.promote_to_champion(
            model_name=model_name,
            version=str(version),
        )

    # ------------------------------------------------------------------
    # CHAMPION DISCOVERY
    # ------------------------------------------------------------------

    def _get_current_champion(
        self,
        model_name: str,
    ) -> Any | None:
        """
        Return the logical champion based on deployment_status.

        The deployment tag is intentionally used rather than relying
        only on the MLflow alias so that registry state can be audited
        and reconciled independently.
        """

        versions = self.list_versions(model_name)

        champions = [
            version for version in versions if version.tags.get("deployment_status") == "CHAMPION"
        ]

        if not champions:
            return None

        if len(champions) > 1:
            raise RuntimeError(
                "Registry invariant violated: " f"{len(champions)} versions are marked CHAMPION"
            )

        return champions[0]

    def get_candidate(
        self,
        model_name: str,
    ) -> Any:
        """
        Return the model version referenced by candidate alias.
        """

        return self.client.get_model_version_by_alias(
            name=model_name,
            alias="candidate",
        )

    def get_champion(
        self,
        model_name: str,
    ) -> Any:
        """
        Return the model version referenced by champion alias.
        """

        return self.client.get_model_version_by_alias(
            name=model_name,
            alias="champion",
        )

    def get_champion_uri(
        self,
        model_name: str,
    ) -> str:
        """
        Return the MLflow alias URI for the current champion.
        """

        self.get_champion(model_name)

        return f"models:/{model_name}@champion"

    # ------------------------------------------------------------------
    # VERSION INSPECTION
    # ------------------------------------------------------------------

    def list_versions(
        self,
        model_name: str,
    ) -> list[Any]:
        """
        Return all registered versions for a model.
        """

        return list(self.client.search_model_versions(filter_string=(f"name='{model_name}'")))

    # ------------------------------------------------------------------
    # RECONCILIATION
    # ------------------------------------------------------------------

    def reconcile(
        self,
        model_name: str,
    ) -> dict[str, Any]:
        """
        Reconcile logical registry tags with the champion alias.

        The method ensures that exactly one version is marked
        CHAMPION and that the champion alias points to it.

        If no logical champion exists but the alias does, the alias
        target is promoted to champion only if it passed validation.
        """

        versions = self.list_versions(model_name)

        if not versions:
            raise RuntimeError(f"No registered versions found for {model_name}")

        logical_champions = [
            version for version in versions if version.tags.get("deployment_status") == "CHAMPION"
        ]

        # More than one logical champion is an invariant violation.
        if len(logical_champions) > 1:

            raise RuntimeError(
                "Registry reconciliation failed: " "multiple CHAMPION versions detected"
            )

        # If there is no logical champion, try to recover from alias.
        if len(logical_champions) == 0:

            try:
                alias_champion = self.get_champion(model_name)
            except Exception as exc:
                raise RuntimeError(
                    "Registry reconciliation failed: " "no logical champion and no champion alias"
                ) from exc

            alias_validation = alias_champion.tags.get("validation_status")

            if alias_validation != "PASSED":
                raise RuntimeError("Champion alias points to an " "unvalidated model version")

            self.client.set_model_version_tag(
                name=model_name,
                version=str(alias_champion.version),
                key="deployment_status",
                value="CHAMPION",
            )

            logical_champions = [
                self.client.get_model_version(
                    name=model_name,
                    version=str(alias_champion.version),
                )
            ]

        champion = logical_champions[0]

        champion_version = str(champion.version)

        # Make sure the alias agrees with logical state.
        try:
            alias_champion = self.get_champion(model_name)

            alias_version = str(alias_champion.version)

        except Exception:
            alias_version = None

        if alias_version != champion_version:

            self.client.set_registered_model_alias(
                name=model_name,
                alias="champion",
                version=champion_version,
            )

        # Retire every other version that might carry a stale
        # CHAMPION tag.
        retired_versions: list[str] = []

        for version in versions:

            current_version = str(version.version)

            if current_version == champion_version:
                continue

            if version.tags.get("deployment_status") == "CHAMPION":

                self.client.set_model_version_tag(
                    name=model_name,
                    version=current_version,
                    key="deployment_status",
                    value="RETIRED",
                )

                retired_versions.append(current_version)

        return {
            "model_name": model_name,
            "champion_version": champion_version,
            "total_versions": len(versions),
            "retired_versions": retired_versions,
        }

    # ------------------------------------------------------------------
    # ASYNC MODEL VERSION CREATION
    # ------------------------------------------------------------------

    def _wait_for_model_version(
        self,
        model_name: str,
        version: str,
        timeout_seconds: int = 60,
    ) -> None:
        """
        Wait until MLflow finishes creating the model version.
        """

        deadline = time.time() + timeout_seconds

        while time.time() < deadline:

            model_version = self.client.get_model_version(
                name=model_name,
                version=version,
            )

            status = getattr(
                model_version,
                "status",
                None,
            )

            if status in (
                None,
                "READY",
            ):
                return

            time.sleep(1)

        raise TimeoutError("Timed out waiting for MLflow model " f"{model_name} version {version}")
