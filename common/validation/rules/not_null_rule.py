from pyspark.sql.functions import col


class NotNullRule:
    def __init__(self, columns):
        self.columns = columns
    def validate(self, df):
        condition = None
        for c in self.columns:
            expr = col(c).isNotNull()
            condition = expr if condition is None else condition & expr
        valid_df = df.filter(condition)
        invalid_df = df.filter(~condition)
        return valid_df, invalid_df
