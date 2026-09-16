from __future__ import annotations

import asyncio
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence

from sqlalchemy import column, func, select
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Session

from app.control_plane.persistence.database import SessionLocal
from app.control_plane.persistence.models import RAGChunkRecord
from rag.governance import GovernancePolicy
from rag.models import DocumentChunk, RetrievalResult

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")
_MIN_SCORE = 0.0


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in _TOKEN_PATTERN.findall(text))


class InMemoryLexicalRetriever:
    """
    Deterministic lexical retriever over an in-memory collection of chunks.

    The implementation uses BM25-style term weighting and returns scores
    normalized to [0, 1] for the current query. It is intentionally
    backend-independent so the retrieval orchestration can be validated
    before introducing an external lexical search engine.
    """

    def __init__(self, chunks: Sequence[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("chunks must not be empty.")

        ids = [chunk.id for chunk in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("chunks must not contain duplicate IDs.")

        self._chunks = tuple(chunks)
        self._tokens = {chunk.id: _tokenize(chunk.content) for chunk in self._chunks}
        self._document_frequency = self._build_document_frequency()
        self._average_document_length = sum(len(tokens) for tokens in self._tokens.values()) / len(
            self._tokens
        )

    def _build_document_frequency(self) -> Counter[str]:
        document_frequency: Counter[str] = Counter()

        for tokens in self._tokens.values():
            document_frequency.update(set(tokens))

        return document_frequency

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

    @staticmethod
    def _matches_metadata(
        chunk: DocumentChunk,
        metadata_filter: Mapping[str, object] | None,
    ) -> bool:
        if metadata_filter is None:
            return True

        return all(
            chunk.metadata.get(key) == expected_value
            for key, expected_value in metadata_filter.items()
        )

    def _score(
        self,
        query_tokens: Sequence[str],
        document_tokens: Sequence[str],
    ) -> float:
        if not query_tokens or not document_tokens:
            return 0.0

        document_length = len(document_tokens)
        term_frequency = Counter(document_tokens)

        k1 = 1.2
        b = 0.75

        score = 0.0

        for term in set(query_tokens):
            frequency = term_frequency.get(term, 0)
            if frequency == 0:
                continue

            document_frequency = self._document_frequency.get(term, 0)
            if document_frequency == 0:
                continue

            idf = math.log(
                1.0 + (len(self._chunks) - document_frequency + 0.5) / (document_frequency + 0.5)
            )

            denominator = frequency + k1 * (
                1.0 - b + b * document_length / self._average_document_length
            )

            score += idf * (frequency * (k1 + 1.0) / denominator)

        return score

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

        query_tokens = _tokenize(query)

        candidates: list[tuple[DocumentChunk, float]] = []

        for chunk in self._chunks:
            if not self._matches_metadata(chunk, effective_metadata_filter):
                continue

            raw_score = self._score(
                query_tokens,
                self._tokens[chunk.id],
            )

            if raw_score > _MIN_SCORE:
                candidates.append((chunk, raw_score))

        if not candidates:
            return []

        max_score = max(score for _, score in candidates)

        results = [
            RetrievalResult(
                chunk=chunk,
                score=score / max_score if max_score > 0.0 else 0.0,
            )
            for chunk, score in candidates
        ]

        filtered = [result for result in results if min_score is None or result.score >= min_score]

        return sorted(
            filtered,
            key=lambda result: (-result.score, result.chunk.id),
        )[:top_k]


class PostgreSQLLexicalRetriever:
    """
    PostgreSQL full-text-search retriever over the authoritative rag_chunks table.

    PostgreSQL ranks matching chunks with ts_rank_cd and normalizes the
    resulting scores to [0, 1] for the current query, matching the score
    contract of InMemoryLexicalRetriever.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session] = SessionLocal,
    ) -> None:
        self._session_factory = session_factory

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

        return await asyncio.to_thread(
            self._retrieve_sync,
            query,
            top_k,
            min_score,
            effective_metadata_filter,
        )

    def _retrieve_sync(
        self,
        query: str,
        top_k: int,
        min_score: float | None,
        metadata_filter: Mapping[str, object] | None,
    ) -> list[RetrievalResult]:
        session: Session = self._session_factory()

        try:
            content_tsv = column("content_tsv", TSVECTOR())
            tsquery = func.websearch_to_tsquery("simple", query)
            rank = func.ts_rank_cd(content_tsv, tsquery)

            statement = select(RAGChunkRecord, rank.label("rank")).where(
                content_tsv.op("@@")(tsquery)
            )

            if metadata_filter:
                for key, value in metadata_filter.items():
                    statement = statement.where(
                        RAGChunkRecord.chunk_metadata[key].as_string() == str(value)
                    )

            rows = session.execute(statement).all()

            if not rows:
                return []

            max_rank = max(float(rank_value) for _, rank_value in rows)

            if max_rank <= 0.0:
                return []

            results = [
                RetrievalResult(
                    chunk=DocumentChunk(
                        id=record.chunk_id,
                        document_id=record.document_id,
                        content=record.content,
                        metadata=dict(record.chunk_metadata),
                        chunk_index=record.chunk_index,
                    ),
                    score=float(rank_value) / max_rank,
                )
                for record, rank_value in rows
            ]

            filtered = [
                result for result in results if min_score is None or result.score >= min_score
            ]

            return sorted(
                filtered,
                key=lambda result: (-result.score, result.chunk.id),
            )[:top_k]

        finally:
            session.close()
