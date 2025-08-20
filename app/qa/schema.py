# app/qa/schema.py
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field, validator

class Citation(BaseModel):
    chunk_id: str
    source: Optional[str] = None
    path: Optional[str] = None
    section: Optional[str] = None

class Safety(BaseModel):
    blocked: bool = False
    reason: Optional[str] = None

class AnswerPayload(BaseModel):
    answer: str = Field(..., min_length=1)
    citations: List[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    safety: Safety = Field(default_factory=Safety)

    @validator("citations", pre=True)
    def _make_list(cls, v):
        return v or []
