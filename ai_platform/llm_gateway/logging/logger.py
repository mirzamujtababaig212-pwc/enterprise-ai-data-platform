import json
import logging
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    def format(self, record):
        # Base log structure
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # Optional structured fields
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id

        if hasattr(record, "endpoint"):
            log_record["endpoint"] = record.endpoint

        if hasattr(record, "method"):
            log_record["method"] = record.method

        if hasattr(record, "provider"):
            log_record["provider"] = record.provider

        if hasattr(record, "model"):
            log_record["model"] = record.model

        if hasattr(record, "latency_ms"):
            log_record["latency_ms"] = record.latency_ms

        if hasattr(record, "status_code"):
            log_record["status_code"] = record.status_code

        if hasattr(record, "client_ip"):
            log_record["client_ip"] = record.client_ip

        if hasattr(record, "user_agent"):
            log_record["user_agent"] = record.user_agent

        if hasattr(record, "tokens_in"):
            log_record["tokens_in"] = record.tokens_in

        if hasattr(record, "tokens_out"):
            log_record["tokens_out"] = record.tokens_out

        if hasattr(record, "estimated_cost"):
            log_record["estimated_cost"] = record.estimated_cost

        # Startup metadata
        if hasattr(record, "component"):
            log_record["component"] = record.component

        if hasattr(record, "version"):
            log_record["version"] = record.version

        if hasattr(record, "exception_type"):
            log_record["exception_type"] = record.exception_type

        return json.dumps(log_record)


# --------------------------------------------------
# Shared Logger Configuration
# --------------------------------------------------

logger = logging.getLogger("llm_gateway")

logger.setLevel(logging.INFO)

handler = logging.StreamHandler()

handler.setFormatter(JSONFormatter())

logger.handlers.clear()

logger.addHandler(handler)

logger.propagate = False


def get_logger():
    return logger
