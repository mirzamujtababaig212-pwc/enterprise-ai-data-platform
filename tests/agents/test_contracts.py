from __future__ import annotations

from ai_platform.agents.contracts import AgentExecutionContract


class LifecycleAgent:
    async def prepare_context(self, context) -> None:
        pass

    async def evaluate_pre_execution(self, context) -> None:
        pass

    async def orchestrate_step(self, context) -> None:
        pass

    async def execute_boundary(self, context):
        pass

    async def evaluate_post_execution(self, context, response):
        return response


class LegacyAgent:
    async def run(self, context):
        return None


def test_lifecycle_agent_implements_execution_contract() -> None:
    assert isinstance(LifecycleAgent(), AgentExecutionContract)


def test_legacy_agent_does_not_require_lifecycle_contract() -> None:
    assert not isinstance(LegacyAgent(), AgentExecutionContract)
