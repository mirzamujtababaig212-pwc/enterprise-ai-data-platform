from common.validation.composite_validator import CompositeValidator
from common.validation.noop_validator import NoOpValidator

VALIDATOR_REGISTRY = {
    "composite": CompositeValidator,
    "noop": NoOpValidator,
}
