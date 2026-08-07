from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: int
    message: str


class ErrorResponse(BaseModel):
    status: str = "error"
    error: ErrorDetail
    request_id: str | None = None
