class ValidatorBuilder:
    @staticmethod
    def build(validator_cls, config):

        if validator_cls is None:
            raise ValueError("Invalid Validator")

        return validator_cls()
