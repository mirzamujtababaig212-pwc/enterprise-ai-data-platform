from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMRequest:
    messages: list[dict[str, Any]]
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None


@dataclass
class LLMResponse:
    provider: str
    model: str
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier."""
        raise NotImplementedError

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a completion."""
        raise NotImplementedError
