from common.validation.composite_validator import (
    CompositeValidator,
)
from common.validation.duplicate_validator import (
    DuplicateValidator,
)
from common.validation.noop_validator import (
    NoOpValidator,
)
from common.validation.null_validator import (
    NullValidator,
)
from common.validation.schema_validator import (
    SchemaValidator,
)


class ValidatorFactory:

    @staticmethod
    def create(config):

        validator_cfg = config.get("validator", {})
        validator_type = validator_cfg.get("type", "default")

        pipeline = config.get("pipeline", {}).get("class")

        # ---------------------------------------------------------
        # Explicit validator types
        # ---------------------------------------------------------

        if validator_type == "noop":
            return NoOpValidator()

        if validator_type == "composite":
            validators = []

            schema = validator_cfg.get("schema")
            required = validator_cfg.get("required")
            duplicate_keys = validator_cfg.get("duplicate_keys")

            if schema:
                validators.append(SchemaValidator(schema))

            if required:
                validators.append(NullValidator(required))

            if duplicate_keys:
                validators.append(DuplicateValidator(duplicate_keys))

            return CompositeValidator(validators)

        if validator_type != "default":
            raise ValueError(f"Unknown validator type: {validator_type}")

        # ---------------------------------------------------------
        # Pipeline-specific default validators
        # ---------------------------------------------------------

        if pipeline == "bronze":

            return CompositeValidator(
                [
                    SchemaValidator(
                        [
                            "kafka_key",
                            "kafka_topic",
                            "kafka_partition",
                            "kafka_offset",
                            "kafka_timestamp",
                            "raw_value",
                            "vehicle_id",
                            "event_time",
                            "latitude",
                            "longitude",
                            "speed",
                            "rpm",
                            "fuel_level",
                            "battery",
                            "engine_temperature",
                            "gear",
                            "ingestion_time",
                        ]
                    ),
                    NullValidator(
                        [
                            "vehicle_id",
                            "event_time",
                        ]
                    ),
                    DuplicateValidator(
                        [
                            "kafka_topic",
                            "kafka_partition",
                            "kafka_offset",
                        ],
                        order_columns=[
                            "kafka_timestamp",
                        ],
                    ),
                ]
            )

        if pipeline == "silver":
            return CompositeValidator(
                [
                    SchemaValidator(
                        [
                            "vehicle_id",
                            "event_time",
                            "latitude",
                            "longitude",
                            "speed",
                            "rpm",
                            "fuel_level",
                            "battery",
                            "engine_temperature",
                            "gear",
                            "kafka_key",
                            "kafka_topic",
                            "kafka_partition",
                            "kafka_offset",
                            "kafka_timestamp",
                            "raw_value",
                            "ingestion_time",
                            "speed_category",
                            "fuel_status",
                            "battery_status",
                            "vehicle_status",
                        ]
                    ),
                    NullValidator(
                        [
                            "vehicle_id",
                            "event_time",
                        ]
                    ),
                    DuplicateValidator(
                        [
                            "vehicle_id",
                            "event_time",
                        ],
                        order_columns=[
                            "kafka_partition",
                            "kafka_offset",
                        ],
                    ),
                ]
            )

        if pipeline == "gold":
            return NoOpValidator()

        raise ValueError(f"Unknown pipeline class: {pipeline}")
