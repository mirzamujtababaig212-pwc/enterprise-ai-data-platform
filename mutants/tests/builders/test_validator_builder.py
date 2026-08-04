import pytest

from common.builders.validator_builder import ValidatorBuilder
from common.validation.composite_validator import CompositeValidator
from common.validation.noop_validator import NoOpValidator


def test_build_composite():

    validator = ValidatorBuilder.build(CompositeValidator, {})

    assert isinstance(validator, CompositeValidator)


def test_build_noop():

    validator = ValidatorBuilder.build(NoOpValidator, {})

    assert isinstance(validator, NoOpValidator)


def test_invalid_validator():

    with pytest.raises(ValueError):
        ValidatorBuilder.build(None, {})
