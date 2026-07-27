class DLQBuilder:

    @staticmethod
    def build(dlq_cls, config):

        if dlq_cls is None:
            raise ValueError("Invalid DLQ class")

        # DeltaDLQ requires a table argument
        if dlq_cls.__name__ == "DeltaDLQ":
            table = (
                config.get("table")
                or config.get("writer", {}).get("table")
                or "test_dlq"
            )
            return dlq_cls(table)

        # NoOpDLQ has no constructor arguments
        return dlq_cls()
