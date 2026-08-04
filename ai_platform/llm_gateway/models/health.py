from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    providers: list[str]
