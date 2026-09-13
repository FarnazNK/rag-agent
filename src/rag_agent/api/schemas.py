from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"


class CreateOrganizationRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)


class CreateWorkspaceRequest(BaseModel):
    organization_id: str
    name: str = Field(min_length=2, max_length=120)


class OrganizationOut(BaseModel):
    id: str
    name: str


class WorkspaceOut(BaseModel):
    id: str
    organization_id: str
    name: str


class QueryRequest(BaseModel):
    workspace_id: str
    query: str = Field(min_length=1, max_length=2000)


class ChunkOut(BaseModel):
    chunk_id: str
    source: str
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    chunks: list[ChunkOut]
    latency_ms: float
    request_id: str


class IngestionResponse(BaseModel):
    ingestion_id: str
    document_id: str
    status: str
    progress: int
    error: str | None = None


class DocumentOut(BaseModel):
    id: str
    workspace_id: str
    filename: str
    status: str
    created_at: datetime


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
