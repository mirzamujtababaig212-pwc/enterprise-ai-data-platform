from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from rag.contracts import Retriever
from rag.evaluation.external.models import (
    ExternalEvaluationRequest,
    ExternalEvaluationResult,
)
from rag.evaluation.external.workflow import RAGGenerationEvaluationWorkflow
from rag.evaluation.models import RetrievalEvaluationCase


class ExternalEvaluationDispatcher(Protocol):
    async def evaluate(
        self,
        request: ExternalEvaluationRequest,
        *,
        cases: Sequence[RetrievalEvaluationCase],
        retriever: Retriever,
    ) -> ExternalEvaluationResult: ...


class RagasExternalEvaluationDispatcher:
    """Dispatch supported RAGAS evaluations to their concrete workflow."""

    _SUPPORTED_PROVIDER = "ragas"
    _SUPPORTED_EVALUATOR = "faithfulness"

    def __init__(
        self,
        *,
        faithfulness_workflow: RAGGenerationEvaluationWorkflow | None = None,
        faithfulness_workflow_factory: Callable[[], RAGGenerationEvaluationWorkflow] | None = None,
    ) -> None:
        if faithfulness_workflow is None and faithfulness_workflow_factory is None:
            raise ValueError(
                "Either faithfulness_workflow or " "faithfulness_workflow_factory is required"
            )

        if faithfulness_workflow is not None and faithfulness_workflow_factory is not None:
            raise ValueError(
                "faithfulness_workflow and faithfulness_workflow_factory " "are mutually exclusive"
            )

        self._faithfulness_workflow = faithfulness_workflow
        self._faithfulness_workflow_factory = faithfulness_workflow_factory

    def _get_faithfulness_workflow(self) -> RAGGenerationEvaluationWorkflow:
        if self._faithfulness_workflow is None:
            assert self._faithfulness_workflow_factory is not None
            self._faithfulness_workflow = self._faithfulness_workflow_factory()

        return self._faithfulness_workflow

    async def evaluate(
        self,
        request: ExternalEvaluationRequest,
        *,
        cases: Sequence[RetrievalEvaluationCase],
        retriever: Retriever,
    ) -> ExternalEvaluationResult:
        if (
            request.provider == self._SUPPORTED_PROVIDER
            and request.evaluator == self._SUPPORTED_EVALUATOR
        ):
            workflow = self._get_faithfulness_workflow()

            return await workflow.evaluate(
                cases,
                retriever=retriever,
            )

        raise ValueError(
            "Unsupported external evaluation: " f"{request.provider}/{request.evaluator}"
        )
