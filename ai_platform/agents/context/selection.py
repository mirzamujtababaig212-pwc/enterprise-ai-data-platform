from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ai_platform.agents.llm_messages import (
    AgentMessage,
    AgentMessageRole,
    assistant_tool_calls_from_message,
)


class ContextWindowExceededError(RuntimeError):
    """
    Raised when mandatory context cannot fit within the configured
    input-token capacity.
    """


@dataclass(frozen=True)
class ContextSelectionPolicy:
    """
    Provider-neutral context-window selection policy.

    The policy controls the amount of input context available to the
    LLM after reserving space for the model's output.
    """

    context_window_tokens: int
    reserved_output_tokens: int
    policy_id: str = "default"
    policy_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.context_window_tokens <= 0:
            raise ValueError("context_window_tokens must be greater than zero.")

        if self.reserved_output_tokens < 0:
            raise ValueError("reserved_output_tokens must not be negative.")

        if self.reserved_output_tokens >= self.context_window_tokens:
            raise ValueError("reserved_output_tokens must be less than context_window_tokens.")

        if not self.policy_id.strip():
            raise ValueError("policy_id must not be empty.")

        if not self.policy_version.strip():
            raise ValueError("policy_version must not be empty.")

    @property
    def input_token_limit(self) -> int:
        """Return the maximum estimated input tokens available."""
        return self.context_window_tokens - self.reserved_output_tokens


@dataclass(frozen=True)
class ContextSelectionResult:
    """Selected LLM messages plus bounded, content-free diagnostics."""

    messages: tuple[AgentMessage, ...]
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class _SelectionUnit:
    """Atomic unit considered by the context selector."""

    indexes: tuple[int, ...]
    messages: tuple[AgentMessage, ...]
    mandatory: bool

    @property
    def estimated_tokens(self) -> int:
        return sum(ContextSelector._estimate_tokens(message.content) for message in self.messages)


class ContextSelector:
    """
    Select a bounded LLM-facing projection from canonical agent messages.

    The selector never mutates the supplied canonical messages. It treats
    system messages and the current user message as mandatory. The latest
    tool-call/result group is also mandatory, while older tool groups remain
    atomic but may be dropped. Older ordinary conversation messages are
    removed first.
    """

    def __init__(self, policy: ContextSelectionPolicy) -> None:
        if not isinstance(policy, ContextSelectionPolicy):
            raise TypeError("Context policy must be a ContextSelectionPolicy.")

        self.policy = policy

    def select(
        self,
        messages: Sequence[AgentMessage],
    ) -> ContextSelectionResult:
        """
        Select messages that fit the configured input-token capacity.

        Selection is deterministic and preserves the original message order.
        """

        canonical_messages = tuple(messages)

        if any(not isinstance(message, AgentMessage) for message in canonical_messages):
            raise TypeError("Context messages must contain AgentMessage instances.")

        if not canonical_messages:
            return ContextSelectionResult(
                messages=(),
                diagnostics=self._diagnostics(
                    original_estimated_tokens=0,
                    selected_estimated_tokens=0,
                    dropped_message_count=0,
                    selection_applied=False,
                ),
            )

        units = self._build_units(canonical_messages)

        mandatory_units = [unit for unit in units if unit.mandatory]

        mandatory_indexes = {index for unit in mandatory_units for index in unit.indexes}

        mandatory_tokens = sum(unit.estimated_tokens for unit in mandatory_units)

        if mandatory_tokens > self.policy.input_token_limit:
            raise ContextWindowExceededError(
                "Mandatory context exceeds the configured input-token capacity: "
                f"mandatory_tokens={mandatory_tokens}, "
                f"input_token_limit={self.policy.input_token_limit}, "
                f"context_window_tokens={self.policy.context_window_tokens}, "
                f"reserved_output_tokens={self.policy.reserved_output_tokens}."
            )

        selected_indexes = set(mandatory_indexes)
        remaining_tokens = self.policy.input_token_limit - mandatory_tokens

        optional_units = [unit for unit in units if not unit.mandatory]

        # Prefer the newest optional messages. Selection happens in reverse
        # order, but the final projection is reconstructed in canonical order.
        for unit in reversed(optional_units):
            if unit.estimated_tokens <= remaining_tokens:
                selected_indexes.update(unit.indexes)
                remaining_tokens -= unit.estimated_tokens

        selected_messages = tuple(
            message for index, message in enumerate(canonical_messages) if index in selected_indexes
        )

        original_estimated_tokens = self._estimate_messages(canonical_messages)
        selected_estimated_tokens = self._estimate_messages(selected_messages)

        selection_applied = selected_messages != canonical_messages

        return ContextSelectionResult(
            messages=selected_messages,
            diagnostics=self._diagnostics(
                original_estimated_tokens=original_estimated_tokens,
                selected_estimated_tokens=selected_estimated_tokens,
                dropped_message_count=(len(canonical_messages) - len(selected_messages)),
                selection_applied=selection_applied,
            ),
        )

    @staticmethod
    def _build_units(
        messages: tuple[AgentMessage, ...],
    ) -> list[_SelectionUnit]:
        """
        Convert canonical messages into selection units.

        Tool-call messages and their immediately following tool-result
        messages form atomic units. Only the latest tool-call/result group
        is mandatory; older groups may be dropped as a whole. System
        messages and the final user message are always mandatory.
        """

        final_user_index = ContextSelector._find_final_user_index(messages)

        units: list[_SelectionUnit] = []
        tool_groups: list[tuple[int, int]] = []

        index = 0

        while index < len(messages):
            message = messages[index]

            if message.role is AgentMessageRole.ASSISTANT:
                tool_calls = ContextSelector._try_parse_tool_calls(message)

                if tool_calls is not None:
                    end_index = index + 1

                    while (
                        end_index < len(messages)
                        and messages[end_index].role is AgentMessageRole.TOOL
                    ):
                        end_index += 1

                    tool_groups.append((index, end_index))
                    index = end_index
                    continue

            index += 1

        active_tool_group = tool_groups[-1] if tool_groups else None
        tool_group_by_start = {start: end for start, end in tool_groups}

        index = 0

        while index < len(messages):
            if index in tool_group_by_start:
                end_index = tool_group_by_start[index]

                units.append(
                    _SelectionUnit(
                        indexes=tuple(range(index, end_index)),
                        messages=messages[index:end_index],
                        mandatory=(active_tool_group == (index, end_index)),
                    )
                )

                index = end_index
                continue

            if any(start < index < end for start, end in tool_groups):
                index += 1
                continue

            message = messages[index]

            units.append(
                _SelectionUnit(
                    indexes=(index,),
                    messages=(message,),
                    mandatory=(
                        message.role is AgentMessageRole.SYSTEM or index == final_user_index
                    ),
                )
            )

            index += 1

        return units

    @staticmethod
    def _find_final_user_index(
        messages: tuple[AgentMessage, ...],
    ) -> int | None:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].role is AgentMessageRole.USER:
                return index

        return None

    @staticmethod
    def _try_parse_tool_calls(
        message: AgentMessage,
    ):
        try:
            return assistant_tool_calls_from_message(message)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _estimate_messages(
        messages: tuple[AgentMessage, ...],
    ) -> int:
        return sum(ContextSelector._estimate_tokens(message.content) for message in messages)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Provider-neutral token estimate matching AgentContextAssembly.

        This is intentionally an estimate rather than a provider tokenizer.
        """
        if not text:
            return 0

        return max(1, (len(text) + 3) // 4)

    def _diagnostics(
        self,
        *,
        original_estimated_tokens: int,
        selected_estimated_tokens: int,
        dropped_message_count: int,
        selection_applied: bool,
    ) -> dict[str, Any]:
        return {
            "policy_id": self.policy.policy_id,
            "policy_version": self.policy.policy_version,
            "context_window_tokens": self.policy.context_window_tokens,
            "reserved_output_tokens": self.policy.reserved_output_tokens,
            "input_token_limit": self.policy.input_token_limit,
            "original_estimated_tokens": original_estimated_tokens,
            "selected_estimated_tokens": selected_estimated_tokens,
            "dropped_message_count": dropped_message_count,
            "selection_applied": selection_applied,
            "token_estimation": {
                "method": "chars_per_4",
            },
        }
