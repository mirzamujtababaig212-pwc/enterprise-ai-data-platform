from __future__ import annotations

from collections.abc import Mapping, Sequence

from rag.contracts import EmbeddingService, VectorStore
from rag.governance import GovernancePolicy
from rag.models import RetrievalResult


class SemanticRetriever:
    """
    Query-to-vector retrieval service.

    The retriever knows about embedding, vector-store, and governance
    contracts, but knows nothing about a specific embedding provider,
    database, or enterprise data source.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
        metadata_filter: Mapping[str, object] | None = None,
        governance_policy: GovernancePolicy | None = None,
    ) -> Sequence[RetrievalResult]:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if min_score is not None and not -1.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between -1.0 and 1.0.")

        effective_metadata_filter = self._build_metadata_filter(
            metadata_filter=metadata_filter,
            governance_policy=governance_policy,
        )

        embedding = await self.embedding_service.embed(query)

        results = await self.vector_store.search(
            embedding,
            top_k=top_k,
            metadata_filter=effective_metadata_filter,
        )

        if min_score is None:
            return results

        return [result for result in results if result.score >= min_score]

    @staticmethod
    def _build_metadata_filter(
        *,
        metadata_filter: Mapping[str, object] | None,
        governance_policy: GovernancePolicy | None,
    ) -> dict[str, object] | None:
        if metadata_filter is None and governance_policy is None:
            return None

        effective_filter = dict(metadata_filter or {})

        if governance_policy is None:
            return effective_filter

        policy_filter = governance_policy.to_metadata_filter()

        for key, policy_value in policy_filter.items():
            if key in effective_filter and effective_filter[key] != policy_value:
                raise ValueError(
                    f"Metadata filter conflicts with governance policy for key '{key}'."
                )

            effective_filter[key] = policy_value

        return effective_filter
