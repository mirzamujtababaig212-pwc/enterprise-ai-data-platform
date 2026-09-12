from __future__ import annotations

import asyncio

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.control_plane.persistence.models import (
    Base,
    RetrievalEvaluationReleaseDecisionRecord,
)
from rag.evaluation.composite_release import (
    CompositeEvaluationReleaseDecision,
)
from rag.evaluation.external.release import (
    ExternalEvaluationReleaseDecision,
    ExternalEvaluationReleasePolicy,
)
from rag.evaluation.release import RetrievalEvaluationReleaseDecision
from rag.evaluation.stores.release_decision import (
    PostgreSQLRetrievalEvaluationReleaseDecisionStore,
)


def _decision(
    *,
    run_id: str = "run-1",
    passed: bool = True,
) -> CompositeEvaluationReleaseDecision:
    native = RetrievalEvaluationReleaseDecision(
        run_id=run_id,
        passed=True,
        errors=(),
    )

    external = ExternalEvaluationReleaseDecision(
        passed=passed,
        errors=() if passed else ("external quality gate failed",),
        policy=ExternalEvaluationReleasePolicy(
            name="application-external-evaluation-release",
            required=True,
        ),
    )

    return CompositeEvaluationReleaseDecision(
        run_id=run_id,
        native=native,
        external=external,
        passed=passed,
        errors=() if passed else ("external: external quality gate failed",),
    )


def _repository():
    engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(engine)

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    return (
        PostgreSQLRetrievalEvaluationReleaseDecisionStore(session_factory()),
        engine,
    )


def test_schema_contains_release_decision_table() -> None:
    repository, engine = _repository()

    try:
        assert inspect(engine).has_table("retrieval_evaluation_release_decisions")
    finally:
        repository._session.close()
        engine.dispose()


def test_save_and_get_round_trip_preserves_decision() -> None:
    repository, engine = _repository()

    try:
        decision = _decision()

        # The release-decision table has a foreign key to evaluation runs.
        # SQLite does not enforce foreign keys by default, so this isolated
        # store test can exercise the persistence boundary independently.
        asyncio.run(repository.save(decision))

        restored = asyncio.run(repository.get(decision.run_id))

        assert restored is not None
        assert restored.run_id == decision.run_id
        assert restored.passed is True
        assert restored.errors == ()

        assert restored.native.run_id == decision.native.run_id
        assert restored.native.passed is True
        assert restored.native.errors == ()

        assert restored.external.passed is True
        assert restored.external.errors == ()
        assert restored.external.policy == decision.external.policy
    finally:
        repository._session.close()
        engine.dispose()


def test_save_and_get_round_trip_preserves_failed_decision() -> None:
    repository, engine = _repository()

    try:
        decision = _decision(passed=False)

        asyncio.run(repository.save(decision))
        restored = asyncio.run(repository.get(decision.run_id))

        assert restored is not None
        assert restored.passed is False
        assert restored.errors == ("external: external quality gate failed",)
        assert restored.native.passed is True
        assert restored.external.passed is False
        assert restored.external.errors == ("external quality gate failed",)
        assert restored.external.policy.required is True
    finally:
        repository._session.close()
        engine.dispose()


def test_get_missing_decision_returns_none() -> None:
    repository, engine = _repository()

    try:
        assert asyncio.run(repository.get("missing")) is None
    finally:
        repository._session.close()
        engine.dispose()


def test_save_replaces_existing_decision() -> None:
    repository, engine = _repository()

    try:
        first = _decision(passed=False)
        second = _decision(passed=True)

        asyncio.run(repository.save(first))
        asyncio.run(repository.save(second))

        restored = asyncio.run(repository.get(first.run_id))

        assert restored is not None
        assert restored.passed is True
        assert restored.errors == ()
    finally:
        repository._session.close()
        engine.dispose()


def test_persisted_decision_has_no_raw_evaluation_payload() -> None:
    repository, engine = _repository()

    try:
        decision = _decision()

        asyncio.run(repository.save(decision))

        record = repository._session.get(
            RetrievalEvaluationReleaseDecisionRecord,
            decision.run_id,
        )

        assert record is not None
        assert record.errors == []
        assert record.native == {
            "run_id": "run-1",
            "passed": True,
            "errors": [],
        }
        assert record.external == {
            "passed": True,
            "errors": [],
            "policy": {
                "name": "application-external-evaluation-release",
                "required": True,
            },
        }

        # The composite release artifact must remain aggregate-only.
        assert "query" not in record.native
        assert "query" not in record.external
        assert "retrieved_contexts" not in record.native
        assert "retrieved_contexts" not in record.external
    finally:
        repository._session.close()
        engine.dispose()
