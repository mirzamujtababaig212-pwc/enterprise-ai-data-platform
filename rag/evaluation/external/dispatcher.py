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
    """Dispatch supported RAGAS evaluations to concrete workflows."""

    _SUPPORTED_PROVIDER = "ragas"
    _FAITHFULNESS_EVALUATOR = "faithfulness"
    _ANSWER_RELEVANCY_EVALUATOR = "answer_relevancy"

    def __init__(
        self,
        *,
        faithfulness_workflow: RAGGenerationEvaluationWorkflow | None = None,
        faithfulness_workflow_factory: Callable[[], RAGGenerationEvaluationWorkflow] | None = None,
        answer_relevancy_workflow: RAGGenerationEvaluationWorkflow | None = None,
        answer_relevancy_workflow_factory: (
            Callable[[], RAGGenerationEvaluationWorkflow] | None
        ) = None,
    ) -> None:
        if faithfulness_workflow is not None and faithfulness_workflow_factory is not None:
            raise ValueError(
                "faithfulness_workflow and faithfulness_workflow_factory " "are mutually exclusive"
            )

        if answer_relevancy_workflow is not None and answer_relevancy_workflow_factory is not None:
            raise ValueError(
                "answer_relevancy_workflow and "
                "answer_relevancy_workflow_factory are mutually exclusive"
            )

        if (
            faithfulness_workflow is None
            and faithfulness_workflow_factory is None
            and answer_relevancy_workflow is None
            and answer_relevancy_workflow_factory is None
        ):
            raise ValueError(
                "At least one RAGAS evaluation workflow or workflow factory " "is required"
            )

        self._faithfulness_workflow = faithfulness_workflow
        self._faithfulness_workflow_factory = faithfulness_workflow_factory
        self._answer_relevancy_workflow = answer_relevancy_workflow
        self._answer_relevancy_workflow_factory = answer_relevancy_workflow_factory

    def _get_faithfulness_workflow(self) -> RAGGenerationEvaluationWorkflow:
        if self._faithfulness_workflow is None:
            assert self._faithfulness_workflow_factory is not None
            self._faithfulness_workflow = self._faithfulness_workflow_factory()

        return self._faithfulness_workflow

    def _get_answer_relevancy_workflow(
        self,
    ) -> RAGGenerationEvaluationWorkflow:
        if self._answer_relevancy_workflow is None:
            assert self._answer_relevancy_workflow_factory is not None
            self._answer_relevancy_workflow = self._answer_relevancy_workflow_factory()

        return self._answer_relevancy_workflow

    async def evaluate(
        self,
        request: ExternalEvaluationRequest,
        *,
        cases: Sequence[RetrievalEvaluationCase],
        retriever: Retriever,
    ) -> ExternalEvaluationResult:
        if (
            request.provider == self._SUPPORTED_PROVIDER
            and request.evaluator == self._FAITHFULNESS_EVALUATOR
        ):
            workflow = self._get_faithfulness_workflow()
        elif (
            request.provider == self._SUPPORTED_PROVIDER
            and request.evaluator == self._ANSWER_RELEVANCY_EVALUATOR
        ):
            workflow = self._get_answer_relevancy_workflow()
        else:
            raise ValueError(
                "Unsupported external evaluation: " f"{request.provider}/{request.evaluator}"
            )

        return await workflow.evaluate(
            cases,
            retriever=retriever,
        )
