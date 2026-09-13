from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document

from rag_agent.schemas import RetrievedChunk


class VectorStore(Protocol):
    def add_documents(self, docs: list[Document]) -> list[str]: ...
    def similarity_search(self, query: str, k: int) -> list[RetrievedChunk]: ...
    def count(self) -> int: ...


def _embed(text: str, dim: int = 1536) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    data = (digest * (dim // len(digest) + 1))[:dim]
    return [b / 255.0 for b in data]


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class InMemoryStore:
    _docs: list[Document] = field(default_factory=list)
    _ids: list[str] = field(default_factory=list)
    _vecs: list[list[float]] = field(default_factory=list)

    def __init__(self, persist_directory=None, collection_name=None, embeddings=None) -> None:
        self._docs = []
        self._ids = []
        self._vecs = []
        self._persist_path: Path | None = None
        if persist_directory is not None:
            directory = Path(persist_directory)
            directory.mkdir(parents=True, exist_ok=True)
            name = collection_name or "rag_agent_docs"
            self._persist_path = directory / f"{name}.json"
            if self._persist_path.exists():
                raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
                self._ids = [item["id"] for item in raw]
                self._docs = [
                    Document(page_content=item["content"], metadata=item.get("metadata", {}))
                    for item in raw
                ]
                self._vecs = [_embed(doc.page_content) for doc in self._docs]

    def add_documents(self, docs: list[Document]) -> list[str]:
        ids: list[str] = []
        start = len(self._docs)
        for i, doc in enumerate(docs):
            idx = str(start + i)
            ids.append(idx)
            self._docs.append(doc)
            self._ids.append(idx)
            self._vecs.append(_embed(doc.page_content))
        self._persist()
        return ids

    def similarity_search(self, query: str, k: int) -> list[RetrievedChunk]:
        qv = _embed(query)
        ranked = sorted(
            range(len(self._docs)),
            key=lambda i: _cos(qv, self._vecs[i]),
            reverse=True,
        )[:k]
        return [
            RetrievedChunk(
                chunk_id=self._docs[i].metadata.get("chunk_id", self._ids[i]),
                content=self._docs[i].page_content,
                source=self._docs[i].metadata.get("source", "unknown"),
                score=max(0.0, min(1.0, _cos(qv, self._vecs[i]))),
                metadata=self._docs[i].metadata,
            )
            for i in ranked
        ]

    def count(self) -> int:
        return len(self._docs)

    def _persist(self) -> None:
        if self._persist_path is None:
            return
        payload = [
            {
                "id": self._ids[i],
                "content": self._docs[i].page_content,
                "metadata": self._docs[i].metadata,
            }
            for i in range(len(self._docs))
        ]
        self._persist_path.write_text(json.dumps(payload), encoding="utf-8")


class ChromaStore(InMemoryStore):
    """Compatibility wrapper for callers expecting `ChromaStore`."""
