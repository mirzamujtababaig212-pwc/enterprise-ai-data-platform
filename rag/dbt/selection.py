from __future__ import annotations

from rag.dbt.models import DbtModel


class DbtModelSelectionPolicy:
    """Determines which dbt models are eligible for RAG ingestion."""

    def select(self, model: DbtModel) -> bool:
        if self._is_test_model(model):
            return False

        if self._has_exclude_tag(model):
            return False

        if self._has_include_tag(model):
            return True

        return self._is_business_model(model)

    @staticmethod
    def _is_test_model(model: DbtModel) -> bool:
        name = model.name.lower()
        return "test" in name or "integration" in model.tags or "test" in model.tags

    @staticmethod
    def _has_exclude_tag(model: DbtModel) -> bool:
        return "ai_exclude" in model.tags

    @staticmethod
    def _has_include_tag(model: DbtModel) -> bool:
        return "ai_knowledge" in model.tags

    @staticmethod
    def _is_business_model(model: DbtModel) -> bool:
        name = model.name.lower()

        return name.startswith("rpt_") or name.startswith("fct_") or name.startswith("dim_")
