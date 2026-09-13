from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rag.contracts import EmbeddingService, Retriever
from rag.evaluation.dataset import RetrievalEvaluationDataset
from rag.evaluation.datasets.vehicle import (
    VEHICLE_EMBEDDING_IDENTITY,
    VehicleBenchmarkEmbeddingService,
    vehicle_benchmark_chunks,
    vehicle_evaluation_cases,
)
from rag.evaluation.lineage import RetrievalEvaluationArtifact
from rag.models import EmbeddingIdentity
from rag.retrieval import SemanticRetriever
from rag.stores import VectorStoreFactory


class EvaluationDatasetDefinition(Protocol):
    """Application-neutral definition of an executable evaluation dataset."""

    name: str
    version: str

    def build_dataset(self) -> RetrievalEvaluationDataset: ...

    def build_embedding_service(self) -> EmbeddingService: ...

    def build_embedding_identity(self) -> EmbeddingIdentity: ...

    def build_retrieval_artifact(self) -> RetrievalEvaluationArtifact: ...

    async def build_retriever(self) -> Retriever: ...


@dataclass(frozen=True)
class VehicleRetrievalEvaluationDatasetDefinition:
    """Executable definition for the deterministic vehicle retrieval dataset."""

    name: str = "vehicle-retrieval"
    version: str = "v2"
    vector_store_backend: str = "in_memory"

    def build_dataset(self) -> RetrievalEvaluationDataset:
        return RetrievalEvaluationDataset.from_cases(
            self.name,
            vehicle_evaluation_cases(),
            version=self.version,
        )

    def build_embedding_service(self) -> EmbeddingService:
        return VehicleBenchmarkEmbeddingService()

    def build_embedding_identity(self) -> EmbeddingIdentity:
        return VEHICLE_EMBEDDING_IDENTITY

    async def build_retriever(self) -> Retriever:
        vector_store = VectorStoreFactory.create(
            backend=self.vector_store_backend,
        )
        await vector_store.upsert(vehicle_benchmark_chunks())

        return SemanticRetriever(
            embedding_service=self.build_embedding_service(),
            vector_store=vector_store,
        )

    def build_retrieval_artifact(self) -> RetrievalEvaluationArtifact:
        return RetrievalEvaluationArtifact(
            retriever_type="SemanticRetriever",
            vector_store_type=VectorStoreFactory.type_name(
                self.vector_store_backend,
            ),
        )


class EvaluationDatasetRegistry:
    """
    Explicit registry of executable evaluation datasets.

    Dataset selection is deterministic and versioned. No implicit/latest
    dataset resolution is performed.
    """

    _definitions: dict[tuple[str, str], EvaluationDatasetDefinition] = {
        ("vehicle-retrieval", "v2"): VehicleRetrievalEvaluationDatasetDefinition(),
    }

    @classmethod
    def get(
        cls,
        *,
        name: str,
        version: str,
    ) -> EvaluationDatasetDefinition:
        key = (name, version)

        try:
            return cls._definitions[key]
        except KeyError as exc:
            raise ValueError(
                f"evaluation dataset not found: name={name!r}, version={version!r}"
            ) from exc
