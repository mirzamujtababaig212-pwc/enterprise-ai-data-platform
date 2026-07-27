import pytest

from common.dlq.delta_dlq import DeltaDLQ
from common.dlq.noop_dlq import NoOpDLQ
from common.factories.dlq_factory import DLQFactory


def test_create_delta_dlq():

    config = {
        "dlq": {
            "type": "delta"
        }
    }

    dlq = DLQFactory.create(config)

    assert isinstance(
        dlq,
        DeltaDLQ
    )


def test_create_noop_dlq():

    config = {
        "dlq": {
            "type": "noop"
        }
    }

    dlq = DLQFactory.create(config)

    assert isinstance(
        dlq,
        NoOpDLQ
    )


def test_invalid_dlq():

    config = {
        "dlq": {
            "type": "dummy"
        }
    }

    with pytest.raises(ValueError):
        DLQFactory.create(config)
