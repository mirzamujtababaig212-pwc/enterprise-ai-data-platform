from __future__ import annotations

from dataclasses import dataclass
import json
from enum import StrEnum
from typing import Any, Protocol

from ai_platform.agents.llm_messages import AgentMessage


class AgentCheckpointPosition(StrEnum):
    """
    Execution boundaries at which an agent checkpoint may be captured.
    """

    BEFORE_LLM_REQUEST = "before_llm_request"
    AFTER_LLM_RESPONSE = "after_llm_response"
    AFTER_TOOL_EXECUTION = "after_tool_execution"


class AgentCheckpointHandler(Protocol):
    """
    Receives provider-neutral execution checkpoints.
    """

    async def save(
        self,
        checkpoint: AgentExecutionCheckpoint,
    ) -> None: ...


@dataclass(frozen=True)
class AgentExecutionCheckpoint:
    """
    Immutable, provider-neutral snapshot of resumable agent execution state.

    Runtime dependencies such as the LLM client, tool registry, memory
    implementation, observers, and database sessions are intentionally
    excluded. They must be reconstructed by the runtime when execution
    resumes.
    """

    schema_version: int
    run_id: str
    agent_name: str
    session_id: str | None
    user_id: str | None
    messages: tuple[AgentMessage, ...]
    tool_round: int
    position: AgentCheckpointPosition
    metadata: dict[str, Any]

    CURRENT_SCHEMA_VERSION = 1

    def __post_init__(self) -> None:
        if self.schema_version != self.CURRENT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported checkpoint schema_version: {self.schema_version}.")

        if not isinstance(self.position, AgentCheckpointPosition):
            raise TypeError("Checkpoint position must be an AgentCheckpointPosition.")

        if self.position is not AgentCheckpointPosition.AFTER_TOOL_EXECUTION:
            raise ValueError("Checkpoints are currently supported only after tool execution.")
        if not self.run_id.strip():
            raise ValueError("Checkpoint run_id must not be empty.")

        if not self.agent_name.strip():
            raise ValueError("Checkpoint agent_name must not be empty.")

        if self.session_id is not None and not self.session_id.strip():
            raise ValueError("Checkpoint session_id must not be empty.")

        if self.user_id is not None and not self.user_id.strip():
            raise ValueError("Checkpoint user_id must not be empty.")

        if self.tool_round < 0:
            raise ValueError("Checkpoint tool_round must not be negative.")

        for message in self.messages:
            if not isinstance(message, AgentMessage):
                raise TypeError("Checkpoint messages must contain AgentMessage instances.")

        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """
        Return a JSON-compatible representation of the checkpoint.
        """
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "messages": [
                {
                    "role": message.role.value,
                    "content": message.content,
                }
                for message in self.messages
            ],
            "tool_round": self.tool_round,
            "position": self.position.value,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """
        Serialize the checkpoint deterministically to JSON.
        """
        return json.dumps(
            self.to_dict(),
            default=str,
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AgentExecutionCheckpoint:
        """
        Reconstruct a checkpoint from its JSON-compatible representation.
        """
        if not isinstance(payload, dict):
            raise TypeError("Checkpoint payload must be a dictionary.")

        from ai_platform.agents.llm_messages import AgentMessageRole

        raw_messages = payload.get("messages", [])

        if not isinstance(raw_messages, list):
            raise TypeError("Checkpoint messages must be a list.")

        messages = []

        for raw_message in raw_messages:
            if not isinstance(raw_message, dict):
                raise TypeError("Checkpoint message must be a dictionary.")

            try:
                role = AgentMessageRole(raw_message["role"])
                content = raw_message["content"]
            except KeyError as exc:
                raise ValueError(f"Checkpoint message is missing field: {exc.args[0]}") from exc

            messages.append(
                AgentMessage(
                    role=role,
                    content=content,
                )
            )

        metadata = payload.get("metadata", {})

        if not isinstance(metadata, dict):
            raise TypeError("Checkpoint metadata must be a dictionary.")

        return cls(
            schema_version=payload["schema_version"],
            run_id=payload["run_id"],
            agent_name=payload["agent_name"],
            session_id=payload.get("session_id"),
            user_id=payload.get("user_id"),
            messages=tuple(messages),
            tool_round=payload.get("tool_round", 0),
            position=AgentCheckpointPosition(payload["position"]),
            metadata=metadata,
        )

    @classmethod
    def from_json(cls, payload: str) -> AgentExecutionCheckpoint:
        """
        Reconstruct a checkpoint from serialized JSON.
        """
        if not isinstance(payload, str):
            raise TypeError("Checkpoint JSON payload must be a string.")

        decoded = json.loads(payload)

        if not isinstance(decoded, dict):
            raise TypeError("Checkpoint JSON must contain an object.")

        return cls.from_dict(decoded)
