class SparkSessionBuilder:

    @staticmethod
    def build(app_name):

        return (
            SparkSession.builder
            .appName(app_name)
            .getOrCreate()
        )
