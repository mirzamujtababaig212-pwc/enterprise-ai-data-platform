import pytest

from common.builders.dlq_builder import DLQBuilder
from common.dlq.delta_dlq import DeltaDLQ
from common.dlq.noop_dlq import NoOpDLQ


def test_build_delta_dlq():

    dlq = DLQBuilder.build(
        DeltaDLQ,
        {}
    )

    assert isinstance(
        dlq,
        DeltaDLQ
    )


def test_build_noop_dlq():

    dlq = DLQBuilder.build(
        NoOpDLQ,
        {}
    )

    assert isinstance(
        dlq,
        NoOpDLQ
    )


def test_invalid_dlq():

    with pytest.raises(ValueError):
        DLQBuilder.build(None, {})
