from __future__ import annotations

from dataclasses import dataclass

from ai_platform.agents.evaluation.grounding import GroundingClaimAttribution
from ai_platform.agents.evaluation.run import AgentEvaluationRun


@dataclass(frozen=True)
class AgentEvaluationDiagnostics:
    """Transient diagnostics produced while evaluating an agent run.

    Diagnostics are intentionally excluded from the durable evaluation
    artifact and therefore are not persisted or exposed by the standard
    evaluation API response.
    """

    grounding_attributions: tuple[GroundingClaimAttribution, ...] = ()


@dataclass(frozen=True)
class AgentEvaluationResult:
    """Evaluation artifact plus transient diagnostics."""

    evaluation_run: AgentEvaluationRun
    diagnostics: AgentEvaluationDiagnostics
