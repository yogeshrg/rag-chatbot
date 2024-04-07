"""Pydantic models for API requests and responses."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str = Field(pattern="^(user|assistant|system)$")
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    citations: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    """Request for chat completion."""

    message: str = Field(..., min_length=1, max_length=10000)
    session_id: UUID | None = None
    stream: bool = True
    top_k: int | None = Field(default=None, ge=1, le=20)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)


class ChatResponse(BaseModel):
    """Response from chat completion."""

    message: ChatMessage
    session_id: UUID
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentUploadResponse(BaseModel):
    """Response for document upload."""

    document_id: UUID
    filename: str
    chunks_created: int
    status: str = "indexed"


class DocumentInfo(BaseModel):
    """Document metadata."""

    document_id: UUID
    filename: str
    file_size: int
    chunks: int
    created_at: datetime
    status: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    ollama_connected: bool
    vector_store_ready: bool
    document_count: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
