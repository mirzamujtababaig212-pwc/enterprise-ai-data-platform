from pyspark.sql.functions import col


class RangeRule:

    def __init__(self, column, minimum, maximum):
        self.column = column
        self.minimum = minimum
        self.maximum = maximum

    def validate(self, df):

        condition = (col(self.column) >= self.minimum) & (
            col(self.column) <= self.maximum
        )

        valid_df = df.filter(condition)

        invalid_df = df.filter(~condition)

        return valid_df, invalid_df
