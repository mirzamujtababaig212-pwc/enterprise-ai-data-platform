from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4


def create_lease(lease_seconds: int) -> tuple[str, datetime]:
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be greater than zero.")

    lease_id = str(uuid4())
    lease_expires_at = datetime.now(UTC) + timedelta(seconds=lease_seconds)

    return lease_id, lease_expires_at
