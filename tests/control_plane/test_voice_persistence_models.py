from app.control_plane.persistence.models import Base, VoiceLineConfigRecord


def test_voice_line_config_is_registered_in_shared_metadata():
    table = Base.metadata.tables["voice_line_configs"]
    assert table is VoiceLineConfigRecord.__table__
    assert {
        "tenant_id",
        "configured_phone_number",
        "inbound_opening_greeting",
        "inbound_system_prompt_override",
        "outbound_opening_message",
        "outbound_system_prompt_override",
        "provider_key",
        "agent_uuid",
        "provider_sync_status",
        "voice_synthesizer",
        "spoken_language",
        "conversational_pause_pacing",
        "barge_in_sensitivity",
        "llm_reasoning_model",
        "max_call_duration_minutes",
    } <= set(table.columns.keys())


def test_voice_line_config_has_tenant_scoped_phone_uniqueness():
    table = VoiceLineConfigRecord.__table__
    constraints = {c.name: c for c in table.constraints}
    unique = constraints["uq_voice_line_configs_configured_phone_number"]
    assert {column.name for column in unique.columns} == {"configured_phone_number"}


def test_voice_line_config_defaults_to_paused_and_unconfigured():
    table = VoiceLineConfigRecord.__table__
    assert table.c.line_status.default.arg == "paused"
    assert table.c.provider_sync_status.default.arg == "not_configured"


def test_voice_line_config_has_database_check_constraints():
    names = {c.name for c in VoiceLineConfigRecord.__table__.constraints}
    assert "ck_voice_line_configs_line_status" in names
    assert "ck_voice_line_configs_provider_sync_status" in names
    assert "ck_voice_line_configs_pause_pacing" in names
    assert "ck_voice_line_configs_barge_in_sensitivity" in names
    assert "ck_voice_line_configs_max_call_duration" in names


def test_voice_call_tables_are_registered_with_deduplication_constraints():
    from app.control_plane.persistence.models import (
        VoiceCallRecord,
        VoiceProviderEventRecord,
    )

    calls = Base.metadata.tables["voice_calls"]
    events = Base.metadata.tables["voice_provider_events"]

    assert calls is VoiceCallRecord.__table__
    assert events is VoiceProviderEventRecord.__table__

    call_constraints = {c.name: c for c in calls.constraints}
    assert {column.name for column in call_constraints["uq_voice_calls_provider_call"].columns} == {
        "provider_key",
        "provider_call_id",
    }

    event_constraints = {c.name: c for c in events.constraints}
    assert {
        column.name
        for column in event_constraints["uq_voice_provider_events_provider_event"].columns
    } == {"provider_key", "provider_event_id"}

    assert events.c.call_id.foreign_keys
    assert any(fk.target_fullname == "voice_calls.call_id" for fk in events.c.call_id.foreign_keys)
    assert any(
        fk.target_fullname == "voice_line_configs.id" for fk in calls.c.voice_line_id.foreign_keys
    )

    assert "ck_voice_calls_status" in call_constraints
    assert "raw_payload" not in events.c
    assert "transcript" not in calls.c
