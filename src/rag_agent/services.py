from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from dataclasses import dataclass
from typing import cast

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rag_agent.models import Chunk, Document, IngestionJob, IngestionStatus
from rag_agent.schemas import RetrievedChunk

TOKEN_RE = re.compile(r"\w+")


def chunk_text(text: str, *, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    if not text.strip():
        return []
    chunks: list[str] = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + chunk_size].strip())
        i += max(1, chunk_size - overlap)
    return [c for c in chunks if c]


@retry(
    retry=retry_if_exception_type(RuntimeError),
    wait=wait_exponential(multiplier=0.2, min=0.2, max=2),
    stop=stop_after_attempt(3),
    reraise=True,
)
def embed_text(text: str, *, dim: int = 1536) -> list[float]:
    if "EMBED_FAIL" in text:
        raise RuntimeError("embedding provider failure")
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    data = (digest * (dim // len(digest) + 1))[:dim]
    return [(b / 255.0) for b in data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def bm25ish_score(query: str, text: str) -> float:
    query_tokens = set(TOKEN_RE.findall(query.lower()))
    text_tokens = set(TOKEN_RE.findall(text.lower()))
    if not query_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


@dataclass(slots=True)
class QueryCandidate:
    chunk: Chunk
    dense: float
    sparse: float

    @property
    def fused(self) -> float:
        return (self.dense * 0.7) + (self.sparse * 0.3)


class RetrievalCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], tuple[float, list[RetrievedChunk]]] = {}
        self.ttl_seconds = 60.0
        self._lock = threading.Lock()

    def get(self, workspace_id: str, query: str) -> list[RetrievedChunk] | None:
        key = (workspace_id, query.strip().lower())
        with self._lock:
            value = self._cache.get(key)
            if value is None:
                return None
            ts, payload = value
            if time.time() - ts > self.ttl_seconds:
                self._cache.pop(key, None)
                return None
            return payload

    def set(self, workspace_id: str, query: str, chunks: list[RetrievedChunk]) -> None:
        with self._lock:
            self._cache[(workspace_id, query.strip().lower())] = (time.time(), chunks)

    def invalidate_workspace(self, workspace_id: str) -> None:
        with self._lock:
            keys = [k for k in self._cache if k[0] == workspace_id]
            for key in keys:
                self._cache.pop(key, None)


cache = RetrievalCache()


async def create_ingestion_job(
    session: AsyncSession,
    *,
    workspace_id: str,
    user_id: str,
    filename: str,
    mime_type: str,
    content_hash: str,
    metadata_json: dict | None = None,
    raw_content: str = "",
) -> tuple[Document, IngestionJob]:
    doc = Document(
        workspace_id=workspace_id,
        owner_user_id=user_id,
        filename=filename,
        mime_type=mime_type,
        metadata_json=metadata_json or {},
        content_hash=content_hash,
        raw_content=raw_content,
        status=IngestionStatus.pending,
    )
    job = IngestionJob(
        document=doc,
        workspace_id=workspace_id,
        user_id=user_id,
        status=IngestionStatus.pending,
        progress=0,
    )
    session.add_all([doc, job])
    await session.flush()
    return doc, job


async def run_ingestion(
    session: AsyncSession,
    *,
    document: Document,
    job: IngestionJob,
    content: str,
) -> IngestionJob:
    try:
        job.status = IngestionStatus.processing
        job.progress = 5
        document.status = IngestionStatus.processing
        chunks = chunk_text(content)
        if not chunks:
            raise ValueError("invalid document content")
        document.raw_content = content

        await session.execute(delete(Chunk).where(Chunk.document_id == document.id))
        for idx, chunk in enumerate(chunks):
            vector = embed_text(chunk)
            session.add(
                Chunk(
                    document_id=document.id,
                    workspace_id=document.workspace_id,
                    position=idx,
                    content=chunk,
                    embedding=cast(list[float], vector),
                    metadata_json={"source": document.filename, "position": idx},
                )
            )
            job.progress = int(((idx + 1) / len(chunks)) * 95)
        document.status = IngestionStatus.completed
        job.status = IngestionStatus.completed
        job.progress = 100
        job.error = None
    except Exception as exc:
        document.status = IngestionStatus.failed
        job.status = IngestionStatus.failed
        job.error = str(exc)
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    cache.invalidate_workspace(document.workspace_id)
    await session.refresh(job)
    return job


async def retrieve_chunks(
    session: AsyncSession,
    *,
    workspace_id: str,
    query: str,
    limit: int = 5,
) -> list[RetrievedChunk]:
    cached = cache.get(workspace_id, query)
    if cached is not None:
        return cached

    query_embedding = embed_text(query)
    if session.bind and session.bind.dialect.name == "postgresql":
        nearest_rows = (
            await session.execute(
                text(
                    "SELECT id FROM chunks "
                    "WHERE workspace_id = :workspace_id "
                    "ORDER BY embedding <=> CAST(:embedding AS vector) "
                    "LIMIT :limit"
                ),
                {
                    "workspace_id": workspace_id,
                    "embedding": str(query_embedding),
                    "limit": max(limit * 20, 50),
                },
            )
        ).all()
        candidate_ids = [row[0] for row in nearest_rows]
        if not candidate_ids:
            return []
        rows = (
            (
                await session.execute(
                    select(Chunk).where(
                        Chunk.id.in_(candidate_ids), Chunk.workspace_id == workspace_id
                    )
                )
            )
            .scalars()
            .all()
        )
    else:
        rows = (
            (await session.execute(select(Chunk).where(Chunk.workspace_id == workspace_id)))
            .scalars()
            .all()
        )
    scored = [
        QueryCandidate(
            chunk=chunk,
            dense=cosine_similarity(query_embedding, cast(list[float], chunk.embedding)),
            sparse=bm25ish_score(query, chunk.content),
        )
        for chunk in rows
    ]
    scored.sort(key=lambda c: c.fused, reverse=True)
    output = [
        RetrievedChunk(
            chunk_id=c.chunk.id,
            content=c.chunk.content,
            source=str(c.chunk.metadata_json.get("source", "unknown")),
            score=max(0.0, min(1.0, c.fused)),
            metadata={"position": c.chunk.position},
        )
        for c in scored[:limit]
    ]
    cache.set(workspace_id, query, output)
    return output
