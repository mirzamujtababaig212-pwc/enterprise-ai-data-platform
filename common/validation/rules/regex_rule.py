from pyspark.sql.functions import col


class RegexRule:

    def __init__(
        self,
        column,
        pattern
    ):
        self.column = column
        self.pattern = pattern

    def validate(self, df):

        condition = col(self.column).rlike(self.pattern)

        valid_df = df.filter(condition)

        invalid_df = df.filter(~condition)

        return valid_df, invalid_df
