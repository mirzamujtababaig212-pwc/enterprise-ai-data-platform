class SchemaRule:

    def __init__(self, expected_schema):
        self.expected_schema = expected_schema

    def validate(self, df):
        return df.schema == self.expected_schema
