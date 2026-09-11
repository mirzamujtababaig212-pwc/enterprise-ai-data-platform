from __future__ import annotations

import math
from collections.abc import Sequence


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be greater than zero.")


def recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    k: int,
) -> float:
    """
    Calculate Recall@K.

    Recall = relevant retrieved documents / total relevant documents.
    """
    _validate_k(k)

    relevant = set(relevant_chunk_ids)
    if not relevant:
        return 0.0

    retrieved = set(retrieved_chunk_ids[:k])
    return len(retrieved & relevant) / len(relevant)


def precision_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
    k: int,
) -> float:
    """
    Calculate Precision@K.

    Precision = relevant retrieved documents / retrieved documents considered.
    """
    _validate_k(k)

    retrieved = list(retrieved_chunk_ids[:k])
    if not retrieved:
        return 0.0

    relevant = set(relevant_chunk_ids)
    return sum(chunk_id in relevant for chunk_id in retrieved) / len(retrieved)


def reciprocal_rank(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str],
) -> float:
    """
    Calculate reciprocal rank of the first relevant retrieved chunk.
    """
    relevant = set(relevant_chunk_ids)

    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevance_grades: dict[str, float],
    k: int,
) -> float:
    """
    Calculate NDCG@K using graded relevance.

    DCG uses:
        (2^relevance - 1) / log2(rank + 1)

    IDCG is calculated from the ideal descending relevance ordering.
    """
    _validate_k(k)

    retrieved = retrieved_chunk_ids[:k]

    if not relevance_grades:
        return 0.0

    dcg = 0.0

    for rank, chunk_id in enumerate(retrieved, start=1):
        relevance = relevance_grades.get(chunk_id, 0.0)
        dcg += (2**relevance - 1.0) / math.log2(rank + 1)

    ideal_grades = sorted(
        relevance_grades.values(),
        reverse=True,
    )[:k]

    idcg = sum(
        (2**relevance - 1.0) / math.log2(rank + 1)
        for rank, relevance in enumerate(ideal_grades, start=1)
    )

    if idcg == 0.0:
        return 0.0

    return dcg / idcg
