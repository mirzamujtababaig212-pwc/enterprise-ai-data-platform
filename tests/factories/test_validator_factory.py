import pytest

from common.factories.validator_factory import ValidatorFactory
from common.validation.composite_validator import CompositeValidator
from common.validation.noop_validator import NoOpValidator


def test_create_composite():

    config = {"validator": {"type": "composite"}}

    validator = ValidatorFactory.create(config)

    assert isinstance(validator, CompositeValidator)


def test_create_noop():

    config = {"validator": {"type": "noop"}}

    validator = ValidatorFactory.create(config)

    assert isinstance(validator, NoOpValidator)


def test_invalid_validator():

    config = {"validator": {"type": "dummy"}}

    with pytest.raises(ValueError):
        ValidatorFactory.create(config)
