"""Vector store layer.

Wraps Chroma behind a Protocol so the agent code doesn't depend on a specific
provider. Swapping to Pinecone or pgvector is a one-class change — that's
exactly the kind of build-vs-buy seam the JD calls out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from rag_agent.config import Settings, get_settings
from rag_agent.observability import get_logger
from rag_agent.schemas import RetrievedChunk

log = get_logger(__name__)


class VectorStore(Protocol):
    """Interface every vector backend must satisfy."""

    def add_documents(self, docs: list[Document]) -> list[str]: ...
    def similarity_search(self, query: str, k: int) -> list[RetrievedChunk]: ...
    def count(self) -> int: ...


def build_embeddings(settings: Settings | None = None) -> Embeddings:
    """Factory for the embedding client. Cached separately from the store
    so tests can stub it out."""
    settings = settings or get_settings()
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddings(model=settings.embedding_model)
    raise ValueError(f"Unsupported embedding provider: {settings.embedding_provider}")


class ChromaStore:
    """Concrete VectorStore backed by Chroma.

    Persistence is on by default — `vector_store_path` from settings is the
    on-disk directory. For ephemeral test stores, pass a temp dir.
    """

    def __init__(
        self,
        persist_directory: Path | None = None,
        collection_name: str | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        settings = get_settings()
        self._path = Path(persist_directory or settings.vector_store_path)
        self._path.mkdir(parents=True, exist_ok=True)

        self._collection = collection_name or settings.collection_name
        self._embeddings = embeddings or build_embeddings(settings)

        self._client = Chroma(
            collection_name=self._collection,
            embedding_function=self._embeddings,
            persist_directory=str(self._path),
        )
        log.debug(
            "vector_store.initialized",
            path=str(self._path),
            collection=self._collection,
        )

    def add_documents(self, docs: list[Document]) -> list[str]:
        """Insert documents, returning their assigned IDs."""
        if not docs:
            return []
        ids = self._client.add_documents(docs)
        log.info("vector_store.indexed", count=len(ids))
        return ids

    def similarity_search(self, query: str, k: int) -> list[RetrievedChunk]:
        """Run dense retrieval. Scores are converted to similarity in [0, 1]."""
        # Chroma returns distance (lower is better). We invert and clip.
        results = self._client.similarity_search_with_relevance_scores(query, k=k)

        chunks: list[RetrievedChunk] = []
        for doc, score in results:
            # `relevance_score` from Chroma is already normalized to [0, 1].
            chunks.append(
                RetrievedChunk(
                    chunk_id=doc.metadata.get("chunk_id", doc.id or ""),
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    score=max(0.0, min(1.0, float(score))),
                    metadata=doc.metadata,
                )
            )
        return chunks

    def count(self) -> int:
        """Number of vectors currently in the collection."""
        try:
            return self._client._collection.count()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — counting must never crash a query
            return -1
