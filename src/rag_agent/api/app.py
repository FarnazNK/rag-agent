from __future__ import annotations

import asyncio
import hashlib
import time
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_agent.api.metrics import (
    CACHE_HITS,
    ERRORS,
    INGESTION_FAILURES,
    LLM_LATENCY,
    REQUEST_LATENCY,
    REQUESTS,
    RETRIEVAL_LATENCY,
    TOKEN_USAGE,
    render_metrics,
)
from rag_agent.api.schemas import (
    ChunkOut,
    CreateOrganizationRequest,
    CreateWorkspaceRequest,
    DocumentOut,
    HealthResponse,
    IngestionResponse,
    LoginRequest,
    OrganizationOut,
    QueryRequest,
    QueryResponse,
    RegisterRequest,
    TokenResponse,
    WorkspaceOut,
)
from rag_agent.auth import create_access_token, decode_access_token, hash_password, verify_password
from rag_agent.config import get_settings
from rag_agent.db import get_db
from rag_agent.guardrails import PIILeakDetector, apply_guardrails
from rag_agent.models import (
    Document,
    IngestionJob,
    IngestionStatus,
    Membership,
    Organization,
    User,
    Workspace,
)
from rag_agent.services import cache, create_ingestion_job, retrieve_chunks, run_ingestion

ALLOWED_MIME = {"text/plain", "text/markdown"}


def _extract_bearer(auth_header: str | None) -> str:
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    return auth_header.split(" ", 1)[1]


async def _current_user(
    authorization: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> User:
    token = _extract_bearer(authorization)
    try:
        user_id = decode_access_token(token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="invalid token") from exc
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="user not found")
    return user


async def _assert_workspace_access(
    db: AsyncSession, *, user_id: str, workspace_id: str
) -> Workspace:
    workspace = await db.get(Workspace, workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    membership = (
        await db.execute(
            select(Membership).where(
                Membership.user_id == user_id,
                Membership.organization_id == workspace.organization_id,
            )
        )
    ).scalar_one_or_none()
    if membership is None:
        raise HTTPException(status_code=403, detail="forbidden")
    return workspace


def _build_answer(query: str, chunks: list[ChunkOut]) -> str:
    if not chunks:
        return "No relevant context found for this workspace."
    context = "\n".join(f"- {c.snippet}" for c in chunks[:3])
    citations = " ".join(f"[source: {c.source}]" for c in chunks[:3])
    return f"Answer: {query}\n\nContext:\n{context}\n\nCitations: {citations}"


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="RAG Agent API",
        version="1.0.0",
        description="Multi-tenant RAG API with persistent ingestion and tenant-scoped retrieval.",
    )
    rate_state: dict[str, tuple[int, float]] = {}
    rate_lock = asyncio.Lock()

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        now = time.time()
        client_key = request.client.host if request.client else "unknown"
        auth_header = request.headers.get("authorization")
        if auth_header:
            client_key = f"auth:{hashlib.sha256(auth_header.encode('utf-8')).hexdigest()}"
        async with rate_lock:
            count, window = rate_state.get(client_key, (0, now))
            if now - window > 60:
                count, window = 0, now
            count += 1
            rate_state[client_key] = (count, window)
            if count > 120:
                return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})

        req_id = request.headers.get("x-request-id", str(uuid4()))
        request.state.request_id = req_id
        response = await call_next(request)
        response.headers["x-request-id"] = req_id
        return response

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", service="rag-agent")

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(content=render_metrics(), media_type="text/plain; version=0.0.4")

    @app.post("/auth/register", response_model=TokenResponse)
    async def register(
        payload: RegisterRequest, db: AsyncSession = Depends(get_db)
    ) -> TokenResponse:
        existing = (
            await db.execute(select(User).where(User.email == payload.email))
        ).scalar_one_or_none()
        if existing is not None:
            raise HTTPException(status_code=409, detail="email already exists")
        user = User(email=payload.email, password_hash=hash_password(payload.password))
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return TokenResponse(access_token=create_access_token(user.id))

    @app.post("/auth/login", response_model=TokenResponse)
    async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)) -> TokenResponse:
        user = (
            await db.execute(select(User).where(User.email == payload.email))
        ).scalar_one_or_none()
        if user is None or not verify_password(payload.password, user.password_hash):
            raise HTTPException(status_code=401, detail="invalid credentials")
        return TokenResponse(access_token=create_access_token(user.id))

    @app.post("/orgs", response_model=OrganizationOut)
    async def create_org(
        payload: CreateOrganizationRequest,
        user: User = Depends(_current_user),
        db: AsyncSession = Depends(get_db),
    ) -> OrganizationOut:
        org = Organization(name=payload.name)
        db.add(org)
        await db.flush()
        db.add(Membership(user_id=user.id, organization_id=org.id, role="owner"))
        await db.commit()
        return OrganizationOut(id=org.id, name=org.name)

    @app.post("/workspaces", response_model=WorkspaceOut)
    async def create_workspace(
        payload: CreateWorkspaceRequest,
        user: User = Depends(_current_user),
        db: AsyncSession = Depends(get_db),
    ) -> WorkspaceOut:
        membership = (
            await db.execute(
                select(Membership).where(
                    Membership.user_id == user.id,
                    Membership.organization_id == payload.organization_id,
                )
            )
        ).scalar_one_or_none()
        if membership is None:
            raise HTTPException(status_code=403, detail="forbidden")
        workspace = Workspace(organization_id=payload.organization_id, name=payload.name)
        db.add(workspace)
        await db.commit()
        await db.refresh(workspace)
        return WorkspaceOut(
            id=workspace.id, organization_id=workspace.organization_id, name=workspace.name
        )

    @app.post("/workspaces/{workspace_id}/documents", response_model=IngestionResponse)
    async def upload_document(
        workspace_id: str,
        file: UploadFile = File(...),
        user: User = Depends(_current_user),
        db: AsyncSession = Depends(get_db),
    ) -> IngestionResponse:
        start = time.perf_counter()
        endpoint = "upload_document"
        await _assert_workspace_access(db, user_id=user.id, workspace_id=workspace_id)
        if file.content_type not in ALLOWED_MIME:
            raise HTTPException(status_code=415, detail="unsupported mime type")
        raw = await file.read(settings.max_upload_size_bytes + 1)
        if len(raw) > settings.max_upload_size_bytes:
            raise HTTPException(status_code=413, detail="file too large")
        content = raw.decode("utf-8", errors="ignore")
        if not content.strip():
            raise HTTPException(status_code=400, detail="empty file")

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        existing = (
            await db.execute(
                select(Document).where(
                    Document.workspace_id == workspace_id,
                    Document.content_hash == content_hash,
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            raise HTTPException(status_code=409, detail="duplicate ingestion request")

        try:
            document, job = await create_ingestion_job(
                db,
                workspace_id=workspace_id,
                user_id=user.id,
                filename=file.filename or "uploaded.txt",
                mime_type=file.content_type or "text/plain",
                content_hash=content_hash,
                raw_content=content,
            )
        except IntegrityError as exc:
            await db.rollback()
            raise HTTPException(status_code=409, detail="duplicate ingestion request") from exc
        job = await run_ingestion(db, document=document, job=job, content=content)
        if job.status == IngestionStatus.failed:
            INGESTION_FAILURES.inc()
            ERRORS.labels(endpoint=endpoint).inc()
        REQUESTS.labels(endpoint=endpoint, status=job.status.value).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)
        return IngestionResponse(
            ingestion_id=job.id,
            document_id=document.id,
            status=job.status.value,
            progress=job.progress,
            error=job.error,
        )

    @app.get("/ingestions/{ingestion_id}", response_model=IngestionResponse)
    async def ingestion_status(
        ingestion_id: str,
        user: User = Depends(_current_user),
        db: AsyncSession = Depends(get_db),
    ) -> IngestionResponse:
        job = await db.get(IngestionJob, ingestion_id)
        if job is None:
            raise HTTPException(status_code=404, detail="ingestion job not found")
        await _assert_workspace_access(db, user_id=user.id, workspace_id=job.workspace_id)
        return IngestionResponse(
            ingestion_id=job.id,
            document_id=job.document_id,
            status=job.status.value,
            progress=job.progress,
            error=job.error,
        )

    @app.post("/documents/{document_id}/reindex", response_model=IngestionResponse)
    async def reindex_document(
        document_id: str,
        user: User = Depends(_current_user),
        db: AsyncSession = Depends(get_db),
    ) -> IngestionResponse:
        doc = await db.get(Document, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="document not found")
        await _assert_workspace_access(db, user_id=user.id, workspace_id=doc.workspace_id)
        if not doc.raw_content.strip():
            raise HTTPException(status_code=400, detail="canonical document content missing")
        job = IngestionJob(document_id=doc.id, workspace_id=doc.workspace_id, user_id=user.id)
        db.add(job)
        await db.flush()
        job = await run_ingestion(db, document=doc, job=job, content=doc.raw_content)
        return IngestionResponse(
            ingestion_id=job.id,
            document_id=doc.id,
            status=job.status.value,
            progress=job.progress,
            error=job.error,
        )

    @app.delete("/documents/{document_id}", response_model=DocumentOut)
    async def delete_document(
        document_id: str,
        user: User = Depends(_current_user),
        db: AsyncSession = Depends(get_db),
    ) -> DocumentOut:
        doc = await db.get(Document, document_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="document not found")
        await _assert_workspace_access(db, user_id=user.id, workspace_id=doc.workspace_id)
        out = DocumentOut(
            id=doc.id,
            workspace_id=doc.workspace_id,
            filename=doc.filename,
            status=doc.status.value,
            created_at=doc.created_at,
        )
        await db.delete(doc)
        await db.commit()
        cache.invalidate_workspace(doc.workspace_id)
        return out

    @app.post("/query", response_model=QueryResponse)
    async def query(
        payload: QueryRequest,
        request: Request,
        user: User = Depends(_current_user),
        db: AsyncSession = Depends(get_db),
    ) -> QueryResponse:
        start = time.perf_counter()
        if len(payload.query.split()) > 512:
            raise HTTPException(status_code=413, detail="query exceeds token limit")
        await _assert_workspace_access(db, user_id=user.id, workspace_id=payload.workspace_id)
        endpoint = "query"
        cache_before = cache.get(payload.workspace_id, payload.query)
        if cache_before is not None:
            CACHE_HITS.inc()
        try:
            retrieval_start = time.perf_counter()
            chunks = await retrieve_chunks(
                db, workspace_id=payload.workspace_id, query=payload.query
            )
            RETRIEVAL_LATENCY.observe(time.perf_counter() - retrieval_start)
            llm_start = time.perf_counter()
            chunk_view = [
                ChunkOut(
                    chunk_id=c.chunk_id,
                    source=c.source,
                    score=c.score,
                    snippet=c.content[:200],
                )
                for c in chunks
            ]
            answer = _build_answer(payload.query, chunk_view)
            answer, _ = apply_guardrails(answer, [PIILeakDetector()], raise_on_block=False)
            LLM_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - llm_start)
            TOKEN_USAGE.labels(kind="input").inc(max(1, len(payload.query.split())))
            TOKEN_USAGE.labels(kind="output").inc(max(1, len(answer.split())))
        except OperationalError as exc:
            ERRORS.labels(endpoint=endpoint).inc()
            raise HTTPException(status_code=503, detail="database temporarily unavailable") from exc
        except RuntimeError as exc:
            ERRORS.labels(endpoint=endpoint).inc()
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        REQUESTS.labels(endpoint=endpoint, status="ok").inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)
        return QueryResponse(
            answer=answer,
            chunks=chunk_view,
            latency_ms=(time.perf_counter() - start) * 1000,
            request_id=request.state.request_id,
        )

    @app.exception_handler(HTTPException)
    async def _http_exc(_: Request, exc: HTTPException):
        if exc.status_code >= 500:
            return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return app
