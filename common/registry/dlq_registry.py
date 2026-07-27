from common.dlq.delta_dlq import DeltaDLQ
from common.dlq.noop_dlq import NoOpDLQ

DLQ_REGISTRY = {
    "delta": DeltaDLQ,
    "noop": NoOpDLQ,
}
