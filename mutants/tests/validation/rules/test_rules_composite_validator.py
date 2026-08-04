from unittest.mock import MagicMock, patch

from common.validation.rules.composite_validator import CompositeValidator


def test_composite_validator_create():
    validator = CompositeValidator()
    assert len(validator.rules) == 3


@patch("common.validation.rules.composite_validator.NotNullRule")
@patch("common.validation.rules.composite_validator.RegexRule")
@patch("common.validation.rules.composite_validator.DuplicateRule")
def test_validate(
    mock_duplicate,
    mock_regex,
    mock_not_null,
):
    df = MagicMock()
    valid = MagicMock()
    rejected = MagicMock()
    rule = MagicMock()
    rule.validate.return_value = (
        valid,
        rejected,
    )
    validator = CompositeValidator()
    validator.rules = [
        rule,
        rule,
        rule,
    ]
    result_valid, result_invalid = validator.validate(df)
    assert result_valid is valid
    assert result_invalid is not None
