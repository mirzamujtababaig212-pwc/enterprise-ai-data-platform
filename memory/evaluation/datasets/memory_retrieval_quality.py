from __future__ import annotations

from datetime import datetime, timedelta, timezone

from memory.evaluation.models import MemoryRetrievalEvaluationCase
from memory.models import MemoryItem, MemoryType

MEMORY_RETRIEVAL_QUALITY_NAMESPACE = "memory-retrieval-quality"
PROJECT_A_NAMESPACE = "project-a"
PROJECT_B_NAMESPACE = "project-b"

_BASE_TIME = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _timestamp(minutes: int) -> datetime:
    return _BASE_TIME + timedelta(minutes=minutes)


def _item(
    memory_id: str,
    content: str,
    *,
    namespace: str = MEMORY_RETRIEVAL_QUALITY_NAMESPACE,
    memory_type: MemoryType = "semantic",
    minutes: int = 0,
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        memory_type=memory_type,
        content=content,
        namespace=namespace,
        created_at=_timestamp(minutes),
    )


MEMORY_RETRIEVAL_QUALITY_ITEMS = (
    _item(
        "memory-deployment-approval",
        "Production deployment requires approval from the release owner.",
        minutes=10,
    ),
    _item(
        "memory-deployment-configuration",
        "Deployment configuration must be validated before production release.",
        minutes=20,
    ),
    _item(
        "memory-deployment-monitoring",
        "Production deployments should be monitored after release for service health.",
        minutes=30,
    ),
    _item(
        "memory-invoice-validation",
        "Invoices must be validated against supplier records and purchase orders.",
        minutes=40,
    ),
    _item(
        "memory-purchase-order",
        "Purchase orders require approval before supplier commitments are made.",
        minutes=50,
    ),
    _item(
        "memory-vendor-onboarding",
        "New vendors require onboarding and compliance documentation before use.",
        minutes=60,
    ),
    _item(
        "memory-access-review",
        "Access reviews confirm that employee permissions remain appropriate.",
        minutes=70,
    ),
    _item(
        "memory-privileged-access",
        "Privileged access requires explicit authorization from the security authority.",
        minutes=80,
    ),
    _item(
        "memory-incident-escalation",
        "Critical security incidents must be escalated to security leadership.",
        minutes=90,
    ),
    _item(
        "memory-release-deployment-distractor",
        "Deployment release monitoring should continue after production rollout.",
        minutes=100,
    ),
    _item(
        "memory-release-deployment-relevant",
        "The deployment release requires approval before production rollout.",
        minutes=10,
    ),
    _item(
        "memory-project-a-semantic",
        "Project Atlas uses the approved deployment configuration for production releases.",
        namespace=PROJECT_A_NAMESPACE,
        minutes=110,
    ),
    _item(
        "memory-project-a-episodic",
        "The release team approved the Atlas deployment configuration during the launch review.",
        namespace=PROJECT_A_NAMESPACE,
        memory_type="episodic",
        minutes=120,
    ),
    _item(
        "memory-project-a-working",
        "Current deployment checklist includes configuration validation and smoke testing.",
        namespace=PROJECT_A_NAMESPACE,
        memory_type="working",
        minutes=130,
    ),
    _item(
        "memory-project-b-distractor",
        "Project Atlas uses the approved deployment configuration for production releases.",
        namespace=PROJECT_B_NAMESPACE,
        minutes=140,
    ),
    _item(
        "memory-unrelated-finance",
        "Quarterly revenue reporting requires reconciliation of financial statements.",
        minutes=150,
    ),
    _item(
        "memory-unrelated-hiring",
        "Engineering hiring plans include interviews and candidate evaluation.",
        minutes=160,
    ),
)


MEMORY_RETRIEVAL_QUALITY_CASES = (
    MemoryRetrievalEvaluationCase(
        query="deployment configuration",
        namespace=MEMORY_RETRIEVAL_QUALITY_NAMESPACE,
        relevant_memory_ids=(
            "memory-deployment-configuration",
            "memory-deployment-approval",
        ),
    ),
    MemoryRetrievalEvaluationCase(
        query="production release validation",
        namespace=MEMORY_RETRIEVAL_QUALITY_NAMESPACE,
        relevant_memory_ids=(
            "memory-deployment-configuration",
            "memory-deployment-monitoring",
        ),
        relevance_grades={
            "memory-deployment-configuration": 3.0,
            "memory-deployment-monitoring": 1.0,
        },
    ),
    MemoryRetrievalEvaluationCase(
        query="invoice purchase order",
        namespace=MEMORY_RETRIEVAL_QUALITY_NAMESPACE,
        relevant_memory_ids=(
            "memory-invoice-validation",
            "memory-purchase-order",
        ),
    ),
    MemoryRetrievalEvaluationCase(
        query="vendor onboarding",
        namespace=MEMORY_RETRIEVAL_QUALITY_NAMESPACE,
        relevant_memory_ids=("memory-vendor-onboarding",),
    ),
    MemoryRetrievalEvaluationCase(
        query="access review",
        namespace=MEMORY_RETRIEVAL_QUALITY_NAMESPACE,
        memory_type="semantic",
        relevant_memory_ids=("memory-access-review",),
    ),
    MemoryRetrievalEvaluationCase(
        query="privileged access authorization",
        namespace=MEMORY_RETRIEVAL_QUALITY_NAMESPACE,
        memory_type="semantic",
        relevant_memory_ids=("memory-privileged-access",),
    ),
    MemoryRetrievalEvaluationCase(
        query="incident escalation",
        namespace=MEMORY_RETRIEVAL_QUALITY_NAMESPACE,
        relevant_memory_ids=("memory-incident-escalation",),
    ),
    MemoryRetrievalEvaluationCase(
        query="Atlas deployment configuration",
        namespace=PROJECT_A_NAMESPACE,
        memory_type="semantic",
        relevant_memory_ids=("memory-project-a-semantic",),
    ),
    MemoryRetrievalEvaluationCase(
        query="Atlas deployment configuration",
        namespace=PROJECT_A_NAMESPACE,
        memory_type="episodic",
        relevant_memory_ids=("memory-project-a-episodic",),
    ),
    MemoryRetrievalEvaluationCase(
        query="deployment checklist",
        namespace=PROJECT_A_NAMESPACE,
        memory_type="working",
        relevant_memory_ids=("memory-project-a-working",),
    ),
    MemoryRetrievalEvaluationCase(
        query="deployment release",
        namespace=MEMORY_RETRIEVAL_QUALITY_NAMESPACE,
        relevant_memory_ids=("memory-release-deployment-relevant",),
    ),
)
