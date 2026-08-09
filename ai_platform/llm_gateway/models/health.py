from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    providers: dict[str, dict[str, Any]]
