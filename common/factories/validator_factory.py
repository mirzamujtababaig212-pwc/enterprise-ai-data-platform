from common.registry.validator_registry import VALIDATOR_REGISTRY
from common.validation.composite_validator import CompositeValidator
from common.validation.duplicate_validator import DuplicateValidator
from common.validation.noop_validator import NoOpValidator
from common.validation.null_validator import NullValidator
from common.validation.schema_validator import SchemaValidator


class ValidatorFactory:

    @staticmethod
    def create(config):

        validator_type = config["validator"]["type"]

        if validator_type not in VALIDATOR_REGISTRY:
            raise ValueError(f"Unknown validator type: {validator_type}")

        pipeline = config.get("pipeline", {}).get("class")

        if validator_type == "noop":
            return NoOpValidator()

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

        elif pipeline == "silver":
            return CompositeValidator(
                [
                    SchemaValidator(
                        [
                            "vehicle_id",
                            "status",
                            "event_timestamp",
                        ]
                    ),
                    NullValidator(["vehicle_id"]),
                    DuplicateValidator(["vehicle_id"]),
                ]
            )
        else:
            return CompositeValidator()
