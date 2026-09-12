from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag.contracts import Retriever
from rag.evaluation.external.models import (
    ExternalEvaluationRequest,
    ExternalEvaluationResult,
)
from rag.evaluation.models import RetrievalEvaluationCase


class ExternalEvaluationDispatcher(Protocol):
    async def evaluate(
        self,
        request: ExternalEvaluationRequest,
        *,
        cases: Sequence[RetrievalEvaluationCase],
        retriever: Retriever,
    ) -> ExternalEvaluationResult: ...
