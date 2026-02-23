from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class OCRResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    model: str
    markdown: str
    processing_ms: int


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    error_code: str
    message: str

