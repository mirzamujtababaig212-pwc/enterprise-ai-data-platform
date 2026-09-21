from __future__ import annotations

from ai_platform.agents.orchestration import (
    OrchestrationPlan,
    OrchestrationStep,
    OrchestrationStepCompletionPolicy,
    OrchestrationStepStatus,
)

ENTERPRISE_RAG_ANALYST_AGENT = "enterprise-rag-analyst"


def build_enterprise_rag_analyst_plan() -> OrchestrationPlan:
    """
    Build the deterministic logical execution plan for the enterprise
    RAG analyst workflow.

    The plan describes business-level execution stages. It does not
    execute tools or LLM calls and does not assign tool rounds.
    """
    return OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="retrieve_evidence",
                step_index=0,
                name="Retrieve enterprise evidence",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE,
                metadata={
                    "phase": "evidence_retrieval",
                },
            ),
            OrchestrationStep(
                step_id="analyze_evidence",
                step_index=1,
                name="Analyze retrieved evidence",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE,
                metadata={
                    "phase": "analysis",
                },
            ),
            OrchestrationStep(
                step_id="produce_answer",
                step_index=2,
                name="Produce grounded answer",
                status=OrchestrationStepStatus.PENDING,
                completion_policy=OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE,
                metadata={
                    "phase": "response_generation",
                },
            ),
        )
    )


def build_agent_orchestration_plan(agent_name: str) -> OrchestrationPlan:
    """
    Return the application-owned orchestration plan for an agent.

    Agents without an explicit orchestration plan are intentionally not
    assigned one implicitly yet. This keeps orchestration opt-in while
    the runtime integration is being introduced.
    """
    if not isinstance(agent_name, str) or not agent_name.strip():
        raise ValueError("Agent name must not be empty.")

    if agent_name == ENTERPRISE_RAG_ANALYST_AGENT:
        return build_enterprise_rag_analyst_plan()

    raise LookupError(f"No orchestration plan is registered for agent '{agent_name}'.")
