from pydantic import BaseModel


class ServiceInfo(BaseModel):
    name: str
    version: str
    environment: str
    status: str = "healthy"
