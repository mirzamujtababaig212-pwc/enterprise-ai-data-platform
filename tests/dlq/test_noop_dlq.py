from common.dlq.noop_dlq import NoOpDLQ


def test_create():
    dlq = NoOpDLQ()
    assert dlq is not None


def test_write(spark):
    dlq = NoOpDLQ()
    df = spark.createDataFrame([(1, "Alice")], ["id", "name"])
    dlq.write(df)


def test_empty_dataframe(spark):
    dlq = NoOpDLQ()
    empty = spark.createDataFrame(
        [],
        "id INT, name STRING",
    )
    dlq.write(empty)


def test_large_dataset(spark):
    dlq = NoOpDLQ()
    rows = [(i, f"name{i}") for i in range(5000)]
    df = spark.createDataFrame(
        rows,
        ["id", "name"],
    )
    dlq.write(df)
