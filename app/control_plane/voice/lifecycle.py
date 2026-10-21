from app.control_plane.voice.models import VoiceCallStatus, VoiceEventType


_TERMINAL_STATUSES = frozenset(
    {
        VoiceCallStatus.COMPLETED,
        VoiceCallStatus.FAILED,
    }
)


def next_call_status(
    current: VoiceCallStatus | None,
    event_type: VoiceEventType,
) -> VoiceCallStatus:
    """Return the canonical call state without regressing terminal states.

    A missing current state means this is the first observed event. The
    function intentionally has no persistence or provider SDK dependency.
    Event deduplication and atomic persistence belong to the repository layer.
    """
    if current in _TERMINAL_STATUSES:
        return current

    if event_type is VoiceEventType.CALL_FAILED:
        return VoiceCallStatus.FAILED

    if event_type is VoiceEventType.CALL_COMPLETED:
        return VoiceCallStatus.COMPLETED

    if event_type is VoiceEventType.CALL_ANSWERED:
        return VoiceCallStatus.IN_PROGRESS

    if event_type is VoiceEventType.CALL_RECEIVED:
        return current or VoiceCallStatus.RECEIVED

    raise ValueError(f"Unsupported voice event type: {event_type}")
