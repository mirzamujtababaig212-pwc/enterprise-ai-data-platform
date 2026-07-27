class CompositeValidator:
    def __init__(self):
        self.rules = [
            NotNullRule(),
            RegexRule(),
            DuplicateRule(),
        ]
    def validate(self, df):
        valid = df
        invalid = None
        for rule in self.rules:
            valid, rejected = rule.validate(valid)
            if rejected is not None:
                invalid = rejected if invalid is None else invalid.union(rejected)
        return valid, invalid
