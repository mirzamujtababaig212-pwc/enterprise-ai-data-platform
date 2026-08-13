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
            return CompositeValidator()

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
                            "topic",
                            "partition",
                            "offset",
                            "timestamp",
                            "ingestion_timestamp",
                            "ingestion_time",
                        ]
                    ),
                    NullValidator(["vehicle_id"]),
                    DuplicateValidator(["vehicle_id"]),
                ]
            )

        if pipeline == "silver":
            return CompositeValidator(
                [
                    SchemaValidator(
                        [
                            "vehicle_id",
                            "status",
                            "event_time",
                            "speed",
                            "fuel_level",
                            "battery",
                            "engine_temperature",
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
                        ]
                    ),
                ]
            )

        if pipeline == "gold":
            return NoOpValidator()

        raise ValueError(f"Unknown pipeline class: {pipeline}")
