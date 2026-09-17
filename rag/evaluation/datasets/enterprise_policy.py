from __future__ import annotations

from dataclasses import dataclass

from rag.evaluation.models import RetrievalEvaluationCase
from rag.models import (
    DocumentChunk,
    EmbeddedChunk,
    EmbeddingIdentity,
    EmbeddingResult,
)

ENTERPRISE_POLICY_QUERY_TAXONOMY = {
    "Who can approve a high-value supplier payment?": "semantic",
    "vendor payment approval threshold": "lexical",
    "What approval is required before paying a new supplier?": "mixed",
    "What must be checked before an invoice is paid?": "semantic",
    "invoice validation purchase order matching": "lexical",
    "How is a new supplier formally established?": "semantic",
    "vendor onboarding required documentation": "lexical",
    "What controls prevent one person from requesting and approving access?": "semantic",
    "privileged access approval review": "lexical",
    "What happens during a periodic access review?": "mixed",
    "How are critical security incidents escalated?": "semantic",
    "incident severity escalation notification": "lexical",
    "When must an incident be reported to security leadership?": "mixed",
    "Who approves employee expenses above the normal limit?": "semantic",
    "expense reimbursement receipt threshold": "lexical",
    "What documentation is required for an expense claim?": "mixed",
    "Who approves an emergency production change?": "semantic",
    "emergency change rollback approval": "lexical",
    "What is required before a standard production change?": "mixed",
    "How should restricted company data be handled?": "semantic",
}


ENTERPRISE_POLICY_EMBEDDING_IDENTITY = EmbeddingIdentity(
    requested_provider="test-provider",
    requested_model="enterprise-policy-benchmark",
    resolved_provider="test-provider",
    resolved_model="enterprise-policy-benchmark",
    dimension=24,
)


@dataclass(frozen=True)
class EnterprisePolicyBenchmarkItem:
    chunk: DocumentChunk
    embedding: tuple[float, ...]


def _chunk(chunk_id: str, content: str) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id="enterprise-policy-knowledge",
        content=content,
    )


# Concept dimensions:
# 0 approval
# 1 threshold
# 2 vendor
# 3 payment
# 4 invoice
# 5 purchase order
# 6 onboarding
# 7 access
# 8 privileged access
# 9 review
# 10 segregation of duties
# 11 incident
# 12 severity
# 13 escalation
# 14 notification
# 15 expense
# 16 receipt
# 17 reimbursement
# 18 change
# 19 emergency change
# 20 rollback
# 21 classification
# 22 restricted data
# 23 retention


ENTERPRISE_POLICY_BENCHMARK_ITEMS = (
    EnterprisePolicyBenchmarkItem(
        _chunk(
            "policy-vendor-onboarding",
            "New suppliers must complete vendor onboarding before they can be used for company purchases.",
        ),
        (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ),
    EnterprisePolicyBenchmarkItem(
        _chunk(
            "policy-vendor-documentation",
            "Vendor onboarding requires approved supplier information, tax documentation, banking details, and required compliance records.",
        ),
        (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ),
    EnterprisePolicyBenchmarkItem(
        _chunk(
            "policy-vendor-payment-approval",
            "Supplier payments require approval from an authorized business owner before payment is released.",
        ),
        (1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ),
    EnterprisePolicyBenchmarkItem(
        _chunk(
            "policy-vendor-payment-threshold",
            "Payments above the delegated approval threshold require additional approval from the designated financial authority.",
        ),
        (1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ),
    EnterprisePolicyBenchmarkItem(
        _chunk(
            "policy-invoice-validation",
            "Invoices must be validated against supplier records, purchase orders, and supporting documentation before payment.",
        ),
        (0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ),
    EnterprisePolicyBenchmarkItem(
        _chunk(
            "policy-purchase-order",
            "Purchases requiring a purchase order must have an approved purchase order before the supplier commitment is made.",
        ),
        (1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ),
    EnterprisePolicyBenchmarkItem(
        _chunk(
            "policy-access-request",
            "Employees request access through the approved access-management process and identify the business resources required.",
        ),
        (1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ),
    EnterprisePolicyBenchmarkItem(
        _chunk(
            "policy-privileged-access",
            "Privileged access requires explicit authorization from the designated resource owner and security authority.",
        ),
        (1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ),
    EnterprisePolicyBenchmarkItem(
        _chunk(
            "policy-access-review",
            "Access reviews periodically confirm that assigned permissions remain appropriate for the employee's current responsibilities.",
        ),
        (0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ),
    EnterprisePolicyBenchmarkItem(
        _chunk(
            "policy-segregation-of-duties",
            "Segregation of duties prevents a single person from requesting, approving, and executing the same sensitive transaction.",
        ),
        (1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ),
    EnterprisePolicyBenchmarkItem(
        _chunk(
            "policy-incident-classification",
            "Security incidents are classified according to business impact, affected systems, and severity.",
        ),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ),
    EnterprisePolicyBenchmarkItem(
        _chunk(
            "policy-incident-escalation",
            "Critical incidents must be escalated promptly to the incident manager and appropriate security leadership.",
        ),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ),
    EnterprisePolicyBenchmarkItem(
        _chunk(
            "policy-incident-notification",
            "Required incident notifications must be sent to designated stakeholders according to the incident severity and communication procedure.",
        ),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ),
    EnterprisePolicyBenchmarkItem(
        _chunk(
            "policy-expense-approval",
            "Employee expenses require approval by the employee's designated manager before reimbursement.",
        ),
        (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0),
    ),
    EnterprisePolicyBenchmarkItem(
        _chunk(
            "policy-expense-threshold",
            "Expenses above the delegated approval limit require additional approval from the designated financial authority.",
        ),
        (1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0),
    ),
    EnterprisePolicyBenchmarkItem(
        _chunk(
            "policy-expense-receipts",
            "Expense claims must include receipts or other required supporting evidence before reimbursement is processed.",
        ),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0),
    ),
    EnterprisePolicyBenchmarkItem(
        _chunk(
            "policy-standard-change",
            "Standard production changes require documented implementation steps, testing evidence, and approval before deployment.",
        ),
        (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
    ),
    EnterprisePolicyBenchmarkItem(
        _chunk(
            "policy-emergency-change",
            "Emergency production changes may bypass normal scheduling but require documented justification and expedited authorization.",
        ),
        (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0),
    ),
    EnterprisePolicyBenchmarkItem(
        _chunk(
            "policy-change-rollback",
            "Production changes must include a tested rollback procedure that can restore the previous stable state.",
        ),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0),
    ),
    EnterprisePolicyBenchmarkItem(
        _chunk(
            "policy-data-classification",
            "Company information must be classified according to its sensitivity and business impact before it is stored or shared.",
        ),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
    ),
    EnterprisePolicyBenchmarkItem(
        _chunk(
            "policy-restricted-data",
            "Restricted data may only be accessed or shared by authorized personnel using approved enterprise controls.",
        ),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0),
    ),
    EnterprisePolicyBenchmarkItem(
        _chunk(
            "policy-data-retention",
            "Enterprise records must be retained for the applicable retention period and securely disposed of when that period expires.",
        ),
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
    ),
)


def enterprise_policy_benchmark_embedding(text: str) -> EmbeddingResult:
    normalized = text.lower()
    vector = [0.0] * 24

    concept_terms = {
        0: ("approv", "authorize"),
        1: ("threshold", "limit"),
        2: ("vendor", "supplier"),
        3: ("payment", "pay"),
        4: ("invoice",),
        5: ("purchase order", "po"),
        6: ("onboard", "supplier setup"),
        7: ("access", "permission"),
        8: ("privileged",),
        9: ("review",),
        10: ("segregation", "duties"),
        11: ("incident",),
        12: ("severity", "critical"),
        13: ("escalat",),
        14: ("notification", "notify", "stakeholder"),
        15: ("expense",),
        16: ("receipt", "supporting evidence"),
        17: ("reimbursement",),
        18: ("change", "deployment"),
        19: ("emergency",),
        20: ("rollback", "restore"),
        21: ("classification", "classified"),
        22: ("restricted",),
        23: ("retention", "retained", "dispose"),
    }

    for index, terms in concept_terms.items():
        if any(term in normalized for term in terms):
            vector[index] = 1.0

    return EmbeddingResult(
        vector=tuple(vector),
        identity=ENTERPRISE_POLICY_EMBEDDING_IDENTITY,
    )


class EnterprisePolicyBenchmarkEmbeddingService:
    async def embed(self, text: str) -> tuple[float, ...]:
        return enterprise_policy_benchmark_embedding(text).vector

    async def embed_with_metadata(self, text: str) -> EmbeddingResult:
        return enterprise_policy_benchmark_embedding(text)


def enterprise_policy_benchmark_chunks() -> tuple[EmbeddedChunk, ...]:
    return tuple(
        EmbeddedChunk(
            chunk=item.chunk,
            embedding=item.embedding,
            embedding_identity=ENTERPRISE_POLICY_EMBEDDING_IDENTITY,
        )
        for item in ENTERPRISE_POLICY_BENCHMARK_ITEMS
    )


ENTERPRISE_POLICY_EVALUATION_CASES = (
    RetrievalEvaluationCase(
        query="Who can approve a high-value supplier payment?",
        relevant_chunk_ids=(
            "policy-vendor-payment-approval",
            "policy-vendor-payment-threshold",
        ),
        relevance_grades={
            "policy-vendor-payment-approval": 3.0,
            "policy-vendor-payment-threshold": 2.0,
        },
    ),
    RetrievalEvaluationCase(
        query="vendor payment approval threshold",
        relevant_chunk_ids=(
            "policy-vendor-payment-threshold",
            "policy-vendor-payment-approval",
        ),
        relevance_grades={
            "policy-vendor-payment-threshold": 3.0,
            "policy-vendor-payment-approval": 2.0,
        },
    ),
    RetrievalEvaluationCase(
        query="What approval is required before paying a new supplier?",
        relevant_chunk_ids=(
            "policy-vendor-payment-approval",
            "policy-vendor-payment-threshold",
        ),
        relevance_grades={
            "policy-vendor-payment-approval": 3.0,
            "policy-vendor-payment-threshold": 2.0,
        },
    ),
    RetrievalEvaluationCase(
        query="What must be checked before an invoice is paid?",
        relevant_chunk_ids=(
            "policy-invoice-validation",
            "policy-purchase-order",
        ),
        relevance_grades={
            "policy-invoice-validation": 3.0,
            "policy-purchase-order": 2.0,
        },
    ),
    RetrievalEvaluationCase(
        query="invoice validation purchase order matching",
        relevant_chunk_ids=(
            "policy-invoice-validation",
            "policy-purchase-order",
        ),
        relevance_grades={
            "policy-invoice-validation": 3.0,
            "policy-purchase-order": 2.0,
        },
    ),
    RetrievalEvaluationCase(
        query="How is a new supplier formally established?",
        relevant_chunk_ids=(
            "policy-vendor-onboarding",
            "policy-vendor-documentation",
        ),
        relevance_grades={
            "policy-vendor-onboarding": 3.0,
            "policy-vendor-documentation": 2.0,
        },
    ),
    RetrievalEvaluationCase(
        query="vendor onboarding required documentation",
        relevant_chunk_ids=(
            "policy-vendor-documentation",
            "policy-vendor-onboarding",
        ),
        relevance_grades={
            "policy-vendor-documentation": 3.0,
            "policy-vendor-onboarding": 2.0,
        },
    ),
    RetrievalEvaluationCase(
        query="What controls prevent one person from requesting and approving access?",
        relevant_chunk_ids=(
            "policy-segregation-of-duties",
            "policy-access-request",
            "policy-privileged-access",
        ),
        relevance_grades={
            "policy-segregation-of-duties": 3.0,
            "policy-access-request": 2.0,
            "policy-privileged-access": 1.0,
        },
    ),
    RetrievalEvaluationCase(
        query="privileged access approval review",
        relevant_chunk_ids=(
            "policy-privileged-access",
            "policy-access-review",
            "policy-access-request",
        ),
        relevance_grades={
            "policy-privileged-access": 3.0,
            "policy-access-review": 2.0,
            "policy-access-request": 1.0,
        },
    ),
    RetrievalEvaluationCase(
        query="What happens during a periodic access review?",
        relevant_chunk_ids=(
            "policy-access-review",
            "policy-privileged-access",
        ),
        relevance_grades={
            "policy-access-review": 3.0,
            "policy-privileged-access": 1.0,
        },
    ),
    RetrievalEvaluationCase(
        query="How are critical security incidents escalated?",
        relevant_chunk_ids=(
            "policy-incident-escalation",
            "policy-incident-classification",
            "policy-incident-notification",
        ),
        relevance_grades={
            "policy-incident-escalation": 3.0,
            "policy-incident-classification": 2.0,
            "policy-incident-notification": 1.0,
        },
    ),
    RetrievalEvaluationCase(
        query="incident severity escalation notification",
        relevant_chunk_ids=(
            "policy-incident-notification",
            "policy-incident-escalation",
            "policy-incident-classification",
        ),
        relevance_grades={
            "policy-incident-notification": 3.0,
            "policy-incident-escalation": 2.0,
            "policy-incident-classification": 1.0,
        },
    ),
    RetrievalEvaluationCase(
        query="When must an incident be reported to security leadership?",
        relevant_chunk_ids=(
            "policy-incident-notification",
            "policy-incident-escalation",
        ),
        relevance_grades={
            "policy-incident-notification": 3.0,
            "policy-incident-escalation": 2.0,
        },
    ),
    RetrievalEvaluationCase(
        query="Who approves employee expenses above the normal limit?",
        relevant_chunk_ids=(
            "policy-expense-threshold",
            "policy-expense-approval",
        ),
        relevance_grades={
            "policy-expense-threshold": 3.0,
            "policy-expense-approval": 2.0,
        },
    ),
    RetrievalEvaluationCase(
        query="expense reimbursement receipt threshold",
        relevant_chunk_ids=(
            "policy-expense-threshold",
            "policy-expense-receipts",
            "policy-expense-approval",
        ),
        relevance_grades={
            "policy-expense-threshold": 3.0,
            "policy-expense-receipts": 2.0,
            "policy-expense-approval": 1.0,
        },
    ),
    RetrievalEvaluationCase(
        query="What documentation is required for an expense claim?",
        relevant_chunk_ids=(
            "policy-expense-receipts",
            "policy-expense-approval",
        ),
        relevance_grades={
            "policy-expense-receipts": 3.0,
            "policy-expense-approval": 1.0,
        },
    ),
    RetrievalEvaluationCase(
        query="Who approves an emergency production change?",
        relevant_chunk_ids=(
            "policy-emergency-change",
            "policy-standard-change",
        ),
        relevance_grades={
            "policy-emergency-change": 3.0,
            "policy-standard-change": 1.0,
        },
    ),
    RetrievalEvaluationCase(
        query="emergency change rollback approval",
        relevant_chunk_ids=(
            "policy-emergency-change",
            "policy-change-rollback",
        ),
        relevance_grades={
            "policy-emergency-change": 3.0,
            "policy-change-rollback": 2.0,
        },
    ),
    RetrievalEvaluationCase(
        query="What is required before a standard production change?",
        relevant_chunk_ids=(
            "policy-standard-change",
            "policy-change-rollback",
            "policy-emergency-change",
        ),
        relevance_grades={
            "policy-standard-change": 3.0,
            "policy-change-rollback": 2.0,
            "policy-emergency-change": 1.0,
        },
    ),
    RetrievalEvaluationCase(
        query="How should restricted company data be handled?",
        relevant_chunk_ids=(
            "policy-restricted-data",
            "policy-data-classification",
        ),
        relevance_grades={
            "policy-restricted-data": 3.0,
            "policy-data-classification": 2.0,
        },
    ),
)


def enterprise_policy_evaluation_cases() -> tuple[RetrievalEvaluationCase, ...]:
    return ENTERPRISE_POLICY_EVALUATION_CASES
