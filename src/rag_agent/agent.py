"""Public Agent entrypoint.

This is the only class downstream code should import. Everything else is
implementation detail. Keeping the surface small is deliberate — it lets us
refactor the internals (graph topology, retrievers, providers) without
breaking callers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from rag_agent.config import get_settings
from rag_agent.graph import build_graph
from rag_agent.observability import (
    configure_logging,
    get_logger,
    maybe_enable_langsmith,
)
from rag_agent.retrieval import HybridRetriever
from rag_agent.schemas import AgentState
from rag_agent.vectorstore import ChromaStore, VectorStore

log = get_logger(__name__)


class Agent:
    """A LangGraph-orchestrated RAG agent.

    Typical usage:
        agent = Agent.from_corpus(docs)
        reply = agent.ask("What is our PTO policy?")
    """

    def __init__(self, retriever: HybridRetriever) -> None:
        configure_logging()
        self._tracing_enabled = maybe_enable_langsmith()
        self._retriever = retriever
        self._graph = build_graph(retriever)
        log.info(
            "agent.ready",
            tracing=self._tracing_enabled,
            corpus_size=len(retriever.corpus),
            vector_count=retriever.store.count(),
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_corpus(
        cls,
        documents: list[Document],
        *,
        store: VectorStore | None = None,
        persist_directory: Path | None = None,
    ) -> Agent:
        """Build an agent from a list of LangChain documents.

        Indexes them into the vector store (if `store` is None, a Chroma
        store is created at the configured path).
        """
        store = store or ChromaStore(persist_directory=persist_directory)
        if store.count() <= 0 and documents:
            store.add_documents(documents)
        retriever = HybridRetriever(store=store, corpus=documents)
        return cls(retriever)

    @classmethod
    def from_existing_store(
        cls,
        corpus: list[Document],
        store: VectorStore,
    ) -> Agent:
        """Build an agent against an already-populated store. The corpus is
        still required for BM25 (which has no persistent format)."""
        retriever = HybridRetriever(store=store, corpus=corpus)
        return cls(retriever)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def ask(self, query: str, *, history: list[Any] | None = None) -> str:
        """Synchronous one-shot question. Returns the final answer string."""
        result = self.run(query, history=history)
        return result.final_answer or ""

    def run(self, query: str, *, history: list[Any] | None = None) -> AgentState:
        """Full run, returning the final state for inspection / eval hooks."""
        initial = AgentState(
            query=query,
            messages=[*(history or []), HumanMessage(content=query)],
        )
        # LangGraph returns a dict; revalidate as AgentState for type safety.
        final_dict = self._graph.invoke(initial)
        return AgentState.model_validate(final_dict)

    # ------------------------------------------------------------------
    # Introspection — useful for evals and debugging
    # ------------------------------------------------------------------

    @property
    def graph(self) -> Any:
        """Expose the compiled graph for visualization / tracing."""
        return self._graph
