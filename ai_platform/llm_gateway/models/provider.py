from pydantic import BaseModel


class ProviderInfo(BaseModel):
    name: str
    healthy: bool
    available_models: list[str]
