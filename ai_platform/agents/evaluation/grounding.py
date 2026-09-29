from __future__ import annotations

import re
from dataclasses import dataclass

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
class AgentGroundingEvaluation:
    """Deterministic textual-support assessment for a RAG-generated answer."""

    evaluated: bool
    supported: bool | None
    support_ratio: float | None
    supported_sources_total: int
    source_candidates_total: int
    method: str = GROUNDING_METHOD

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
        source_texts: list[str] | tuple[str, ...],
    ) -> AgentGroundingEvaluation:
        candidates = [
            source.strip() for source in source_texts if isinstance(source, str) and source.strip()
        ]

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

        source_tokens = [AgentGroundingEvaluator._content_tokens(source) for source in candidates]

        supported_sentences = 0
        supporting_sources: set[int] = set()

        for sentence in answer_sentences:
            sentence_tokens = AgentGroundingEvaluator._content_tokens(sentence)

            if not sentence_tokens:
                continue

            sentence_supported = False

            for source_index, tokens in enumerate(source_tokens):
                if not tokens:
                    continue

                overlap = sentence_tokens & tokens

                if len(overlap) < 2:
                    continue

                coverage = len(overlap) / len(sentence_tokens)

                if coverage < 0.35:
                    continue

                if AgentGroundingEvaluator._has_numeric_contradiction(
                    sentence,
                    candidates[source_index],
                ):
                    continue

                sentence_supported = True
                supporting_sources.add(source_index)

            if sentence_supported:
                supported_sentences += 1

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
        )

    @staticmethod
    def _split_sentences(value: str) -> list[str]:
        return [
            sentence.strip()
            for sentence in _SENTENCE_SPLIT_RE.split(value.strip())
            if sentence.strip()
        ]

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
