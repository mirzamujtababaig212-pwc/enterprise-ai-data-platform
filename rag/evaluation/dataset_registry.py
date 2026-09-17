from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rag.contracts import EmbeddingService, Retriever
from rag.evaluation.dataset import RetrievalEvaluationDataset
from rag.evaluation.datasets.enterprise_policy import (
    ENTERPRISE_POLICY_EMBEDDING_IDENTITY,
    EnterprisePolicyBenchmarkEmbeddingService,
    enterprise_policy_benchmark_chunks,
    enterprise_policy_evaluation_cases,
)
from rag.evaluation.datasets.vehicle import (
    VEHICLE_EMBEDDING_IDENTITY,
    VehicleBenchmarkEmbeddingService,
    vehicle_benchmark_chunks,
    vehicle_evaluation_cases,
)
from rag.evaluation.datasets.vehicle_retrieval_quality import (
    VEHICLE_QUALITY_EMBEDDING_IDENTITY,
    VehicleQualityBenchmarkEmbeddingService,
    vehicle_quality_benchmark_chunks,
    vehicle_quality_evaluation_cases,
)
from rag.evaluation.lineage import (
    HybridRetrievalConfiguration,
    RetrievalEvaluationArtifact,
)
from rag.models import EmbeddingIdentity
from rag.retrieval import (
    HybridRetriever,
    InMemoryLexicalRetriever,
    SemanticRetriever,
)
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
class EnterprisePolicyRetrievalEvaluationDatasetDefinition:
    """Executable definition for the enterprise policy retrieval evaluation dataset."""

    name: str = "enterprise-policy-retrieval"
    version: str = "v1"

    def build_dataset(self) -> RetrievalEvaluationDataset:
        return RetrievalEvaluationDataset.from_cases(
            self.name,
            enterprise_policy_evaluation_cases(),
            version=self.version,
        )

    def build_embedding_service(self) -> EmbeddingService:
        return EnterprisePolicyBenchmarkEmbeddingService()

    def build_embedding_identity(self) -> EmbeddingIdentity:
        return ENTERPRISE_POLICY_EMBEDDING_IDENTITY

    async def build_retriever(self) -> Retriever:
        vector_store = VectorStoreFactory.create(
            backend="in_memory",
        )
        await vector_store.upsert(enterprise_policy_benchmark_chunks())

        semantic_retriever = SemanticRetriever(
            embedding_service=self.build_embedding_service(),
            vector_store=vector_store,
        )

        lexical_retriever = InMemoryLexicalRetriever(
            [item.chunk for item in enterprise_policy_benchmark_chunks()]
        )

        return HybridRetriever(
            semantic_retriever=semantic_retriever,
            lexical_retriever=lexical_retriever,
            candidate_k=5,
            rrf_k=60,
            semantic_weight=1.0,
            lexical_weight=0.5,
        )

    def build_retrieval_artifact(self) -> RetrievalEvaluationArtifact:
        return RetrievalEvaluationArtifact(
            retriever_type="HybridRetriever",
            vector_store_type="InMemoryVectorStore",
            hybrid_configuration=HybridRetrievalConfiguration(
                candidate_k=5,
                rrf_k=60,
                semantic_weight=1.0,
                lexical_weight=0.5,
            ),
        )


@dataclass(frozen=True)
class VehicleRetrievalQualityEvaluationDatasetDefinition:
    """Executable definition for the experimental vehicle retrieval-quality benchmark."""

    name: str = "vehicle-retrieval-quality"
    version: str = "v1"

    def build_dataset(self) -> RetrievalEvaluationDataset:
        return RetrievalEvaluationDataset.from_cases(
            self.name,
            vehicle_quality_evaluation_cases(),
            version=self.version,
        )

    def build_embedding_service(self) -> EmbeddingService:
        return VehicleQualityBenchmarkEmbeddingService()

    def build_embedding_identity(self) -> EmbeddingIdentity:
        return VEHICLE_QUALITY_EMBEDDING_IDENTITY

    async def build_retriever(self) -> Retriever:
        vector_store = VectorStoreFactory.create(
            backend="in_memory",
        )
        await vector_store.upsert(vehicle_quality_benchmark_chunks())

        semantic_retriever = SemanticRetriever(
            embedding_service=self.build_embedding_service(),
            vector_store=vector_store,
        )

        lexical_retriever = InMemoryLexicalRetriever(
            [item.chunk for item in vehicle_quality_benchmark_chunks()]
        )

        return HybridRetriever(
            semantic_retriever=semantic_retriever,
            lexical_retriever=lexical_retriever,
            candidate_k=5,
            rrf_k=60,
            semantic_weight=1.0,
            lexical_weight=0.5,
        )

    def build_retrieval_artifact(self) -> RetrievalEvaluationArtifact:
        return RetrievalEvaluationArtifact(
            retriever_type="HybridRetriever",
            vector_store_type="InMemoryVectorStore",
            hybrid_configuration=HybridRetrievalConfiguration(
                candidate_k=5,
                rrf_k=60,
                semantic_weight=1.0,
                lexical_weight=0.5,
            ),
        )


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
        (
            "vehicle-retrieval-quality",
            "v1",
        ): VehicleRetrievalQualityEvaluationDatasetDefinition(),
        (
            "enterprise-policy-retrieval",
            "v1",
        ): EnterprisePolicyRetrievalEvaluationDatasetDefinition(),
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
