from __future__ import annotations

import re
from dataclasses import dataclass, field

GROUNDING_METHOD = "lexical_sentence_support_v1"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)*")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?(?:%|[A-Za-z]+)?(?![A-Za-z0-9])")

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "here",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "me",
        "might",
        "more",
        "most",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "ours",
        "she",
        "should",
        "so",
        "some",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "too",
        "under",
        "up",
        "us",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
    }
)


@dataclass(frozen=True)
class GroundingClaimAttribution:
    """Transient attribution of one answer claim to retrieved RAG sources."""

    claim_index: int
    claim_text: str
    supported: bool
    supporting_source_indexes: tuple[int, ...] = ()
    supporting_source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentGroundingEvaluation:
    """Deterministic textual-support assessment for a RAG-generated answer."""

    evaluated: bool
    supported: bool | None
    support_ratio: float | None
    supported_sources_total: int
    source_candidates_total: int
    method: str = GROUNDING_METHOD
    attributions: tuple[GroundingClaimAttribution, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.supported_sources_total < 0:
            raise ValueError("supported_sources_total must be non-negative.")

        if self.source_candidates_total < 0:
            raise ValueError("source_candidates_total must be non-negative.")

        if self.support_ratio is not None and not 0.0 <= self.support_ratio <= 1.0:
            raise ValueError("support_ratio must be between 0.0 and 1.0.")

        if not self.method.strip():
            raise ValueError("method must not be blank.")

    def as_dict(self) -> dict[str, object]:
        return {
            "evaluated": self.evaluated,
            "supported": self.supported,
            "support_ratio": self.support_ratio,
            "supported_sources_total": self.supported_sources_total,
            "source_candidates_total": self.source_candidates_total,
            "method": self.method,
        }


class AgentGroundingEvaluator:
    """Evaluate detectable textual support without an LLM judge."""

    @staticmethod
    def evaluate(
        *,
        answer_text: str | None,
        source_texts: list[str] | tuple[str, ...] = (),
        sources: list[object] | tuple[object, ...] = (),
    ) -> AgentGroundingEvaluation:
        candidates = [
            source.strip() for source in source_texts if isinstance(source, str) and source.strip()
        ]

        structured_sources = [
            source
            for source in sources
            if hasattr(source, "content")
            and isinstance(source.content, str)
            and source.content.strip()
        ]

        if structured_sources:
            candidates = [source.content.strip() for source in structured_sources]

        if answer_text is None or not answer_text.strip() or not candidates:
            return AgentGroundingEvaluation(
                evaluated=False,
                supported=None,
                support_ratio=None,
                supported_sources_total=0,
                source_candidates_total=len(candidates),
            )

        answer_sentences = AgentGroundingEvaluator._split_sentences(answer_text)

        if not answer_sentences:
            return AgentGroundingEvaluation(
                evaluated=False,
                supported=None,
                support_ratio=None,
                supported_sources_total=0,
                source_candidates_total=len(candidates),
            )

        source_sentences = [
            AgentGroundingEvaluator._split_sentences(source) for source in candidates
        ]

        supported_sentences = 0
        supporting_sources: set[int] = set()
        attributions: list[GroundingClaimAttribution] = []

        claim_index = 0

        for sentence in answer_sentences:
            sentence_tokens = AgentGroundingEvaluator._content_tokens(sentence)

            if not sentence_tokens:
                continue

            claim_source_indexes: set[int] = set()

            for source_index, source_units in enumerate(source_sentences):
                for source_unit in source_units:
                    source_tokens = AgentGroundingEvaluator._content_tokens(source_unit)

                    if not source_tokens:
                        continue

                    overlap = sentence_tokens & source_tokens

                    if len(overlap) < 2:
                        continue

                    coverage = len(overlap) / len(sentence_tokens)

                    if coverage < 0.35:
                        continue

                    if AgentGroundingEvaluator._has_numeric_contradiction(
                        sentence,
                        source_unit,
                    ):
                        continue

                    claim_source_indexes.add(source_index)

            if claim_source_indexes:
                supported_sentences += 1
                supporting_sources.update(claim_source_indexes)

            supporting_source_ids_list: list[str] = []

            if structured_sources:
                for index in sorted(claim_source_indexes):
                    if index >= len(structured_sources):
                        continue

                    source = structured_sources[index]
                    chunk_id = getattr(source, "chunk_id", None)

                    if isinstance(chunk_id, str) and chunk_id:
                        supporting_source_ids_list.append(chunk_id)

            supporting_source_ids = tuple(supporting_source_ids_list)

            attributions.append(
                GroundingClaimAttribution(
                    claim_index=claim_index,
                    claim_text=sentence,
                    supported=bool(claim_source_indexes),
                    supporting_source_indexes=tuple(sorted(claim_source_indexes)),
                    supporting_source_ids=supporting_source_ids,
                )
            )
            claim_index += 1

        evaluated_sentence_count = sum(
            1 for sentence in answer_sentences if AgentGroundingEvaluator._content_tokens(sentence)
        )

        if evaluated_sentence_count == 0:
            return AgentGroundingEvaluation(
                evaluated=False,
                supported=None,
                support_ratio=None,
                supported_sources_total=0,
                source_candidates_total=len(candidates),
            )

        support_ratio = supported_sentences / evaluated_sentence_count

        return AgentGroundingEvaluation(
            evaluated=True,
            supported=support_ratio == 1.0,
            support_ratio=support_ratio,
            supported_sources_total=len(supporting_sources),
            source_candidates_total=len(candidates),
            attributions=tuple(attributions),
        )

    @staticmethod
    def _split_sentences(value: str) -> list[str]:
        normalized = re.sub(r"\\r?\\n+", "\\n", value.strip())
        normalized = re.sub(r"(?m)^\\s*[-*+]\\s+", "", normalized)

        units: list[str] = []

        for line in normalized.splitlines():
            units.extend(
                sentence.strip()
                for sentence in _SENTENCE_SPLIT_RE.split(line.strip())
                if sentence.strip()
            )

        return units

    @staticmethod
    def _content_tokens(value: str) -> set[str]:
        return {
            token.casefold()
            for token in _TOKEN_RE.findall(value)
            if len(token) >= 3 and token.casefold() not in _STOPWORDS
        }

    @staticmethod
    def _has_numeric_contradiction(
        answer_sentence: str,
        source_text: str,
    ) -> bool:
        answer_numbers = {value.casefold() for value in _NUMBER_RE.findall(answer_sentence)}
        source_numbers = {value.casefold() for value in _NUMBER_RE.findall(source_text)}

        if not answer_numbers or not source_numbers:
            return False

        return answer_numbers.isdisjoint(source_numbers)
