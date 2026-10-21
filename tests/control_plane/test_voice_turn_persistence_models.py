from app.control_plane.persistence.models import Base, VoiceCallTurnRecord


def test_voice_call_turn_is_registered_in_shared_metadata():
    table = Base.metadata.tables["voice_call_turns"]

    assert table is VoiceCallTurnRecord.__table__
    assert {
        "turn_id",
        "tenant_id",
        "call_id",
        "turn_number",
        "callback_token",
        "status",
        "request_fingerprint",
        "agent_run_id",
        "response_text",
        "failure_code",
        "started_at",
        "completed_at",
        "created_at",
        "updated_at",
    } <= set(table.columns.keys())


def test_voice_call_turn_has_required_foreign_key_and_uniqueness():
    table = VoiceCallTurnRecord.__table__
    constraints = {constraint.name: constraint for constraint in table.constraints}

    assert any(
        foreign_key.target_fullname == "voice_calls.call_id"
        for foreign_key in table.c.call_id.foreign_keys
    )

    assert {
        column.name for column in constraints["uq_voice_call_turns_tenant_call_turn"].columns
    } == {"tenant_id", "call_id", "turn_number"}

    assert {
        column.name for column in constraints["uq_voice_call_turns_callback_token"].columns
    } == {"callback_token"}


def test_voice_call_turn_has_status_and_size_constraints():
    constraints = {constraint.name for constraint in VoiceCallTurnRecord.__table__.constraints}

    assert "ck_voice_call_turns_positive_turn_number" in constraints
    assert "ck_voice_call_turns_status" in constraints
    assert "ck_voice_call_turns_fingerprint_length" in constraints
    assert "ck_voice_call_turns_response_length" in constraints

    assert VoiceCallTurnRecord.__table__.c.status.default is None
    assert VoiceCallTurnRecord.__table__.c.status.server_default.arg == "awaiting_input"


def test_voice_turn_does_not_duplicate_transcripts_or_raw_payloads():
    columns = set(VoiceCallTurnRecord.__table__.columns.keys())

    assert "transcript" not in columns
    assert "raw_payload" not in columns
    assert "request_fingerprint" in columns
