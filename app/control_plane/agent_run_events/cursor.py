from __future__ import annotations

import base64
import json
from datetime import datetime


def encode_cursor(
    *,
    created_at: datetime,
    event_id: int,
) -> str:
    payload = {
        "created_at": created_at.isoformat(),
        "id": event_id,
    }

    raw = json.dumps(payload).encode()

    return base64.urlsafe_b64encode(raw).decode()


def decode_cursor(cursor: str) -> tuple[datetime, int]:
    raw = base64.urlsafe_b64decode(cursor.encode())

    payload = json.loads(raw)

    return (
        datetime.fromisoformat(payload["created_at"]),
        int(payload["id"]),
    )
