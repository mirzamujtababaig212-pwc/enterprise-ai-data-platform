from datetime import UTC, datetime, timedelta

import pytest

from app.control_plane.agent_runs.lease import create_lease


def test_create_lease_returns_id_and_future_expiry() -> None:
    before = datetime.now(UTC)

    lease_id, lease_expires_at = create_lease(60)

    after = datetime.now(UTC)

    assert lease_id
    assert lease_expires_at.tzinfo == UTC
    assert lease_expires_at > before
    assert lease_expires_at <= after + timedelta(seconds=60)


@pytest.mark.parametrize("lease_seconds", [0, -1])
def test_create_lease_rejects_non_positive_duration(
    lease_seconds: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="lease_seconds must be greater than zero",
    ):
        create_lease(lease_seconds)
