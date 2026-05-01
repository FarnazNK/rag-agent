"""Prompt templates.

Centralized so prompt iteration doesn't require digging through node code.
Each prompt is a function returning a list of (role, content) tuples — easy
to feed into any chat model.
"""

from __future__ import annotations

from rag_agent.schemas import RetrievedChunk

ROUTER_SYSTEM = """You are a router for a retrieval-augmented assistant.
Decide whether the user's message requires retrieving documents from the
knowledge base, can be answered directly from general knowledge, or should
be refused (off-topic, harmful, or asking for private data you don't have).

Respond with exactly one word: retrieve, answer_directly, or refuse."""


QUERY_REWRITE_SYSTEM = """You rewrite user questions to maximize retrieval recall.
Rules:
- Expand acronyms and ambiguous pronouns using the conversation context.
- Keep the rewrite under 30 words.
- Preserve the user's intent exactly. Do not add new questions.
- Output ONLY the rewritten query, no preamble."""


GRADER_SYSTEM = """You grade whether retrieved context is sufficient to answer
the user's question. Reply in JSON with three fields:
- is_relevant: bool
- confidence: float between 0 and 1
- rationale: one short sentence

Be strict: if the context is tangentially related but doesn't actually
contain the answer, return is_relevant=false."""


ANSWER_SYSTEM = """You are a helpful assistant answering questions using ONLY
the provided context. Rules:
- If the context is insufficient, say so clearly. Do not guess.
- Cite sources inline using [source: filename] format.
- Be concise. No preamble like "Based on the context...".
"""


def build_router_prompt(query: str) -> list[tuple[str, str]]:
    return [("system", ROUTER_SYSTEM), ("user", query)]


def build_rewrite_prompt(query: str, history: str = "") -> list[tuple[str, str]]:
    user = f"Conversation so far:\n{history}\n\nLatest question:\n{query}" if history else query
    return [("system", QUERY_REWRITE_SYSTEM), ("user", user)]


def build_grader_prompt(query: str, chunks: list[RetrievedChunk]) -> list[tuple[str, str]]:
    context = "\n\n---\n\n".join(c.as_context_block() for c in chunks)
    user = f"Question:\n{query}\n\nRetrieved context:\n{context}"
    return [("system", GRADER_SYSTEM), ("user", user)]


def build_answer_prompt(query: str, chunks: list[RetrievedChunk]) -> list[tuple[str, str]]:
    context = "\n\n---\n\n".join(c.as_context_block() for c in chunks) or "(no context retrieved)"
    user = f"Context:\n{context}\n\nQuestion:\n{query}"
    return [("system", ANSWER_SYSTEM), ("user", user)]
