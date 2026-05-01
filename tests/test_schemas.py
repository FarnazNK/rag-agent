"""Tests for the Pydantic schemas."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from rag_agent.schemas import AgentState, GradingResult, RetrievedChunk


class TestRetrievedChunk:
    def test_as_context_block_includes_source_and_score(self):
        chunk = RetrievedChunk(
            chunk_id="c1",
            content="hello world",
            source="policy.md",
            score=0.87,
        )
        block = chunk.as_context_block()
        assert "policy.md" in block
        assert "0.87" in block
        assert "hello world" in block


class TestAgentState:
    def test_defaults_are_safe(self):
        state = AgentState(query="anything")
        assert state.messages == []
        assert state.chunks == []
        assert state.iterations == 0
        assert state.final_answer is None
        assert state.run_id  # UUID populated

    def test_messages_accept_langchain_types(self):
        state = AgentState(
            query="q",
            messages=[HumanMessage(content="hi"), AIMessage(content="hello")],
        )
        assert len(state.messages) == 2


class TestGradingResult:
    def test_confidence_must_be_in_unit_interval(self):
        # Valid
        GradingResult(is_relevant=True, rationale="ok", confidence=0.5)
        # Invalid — pydantic should raise
        import pytest

        with pytest.raises(ValueError):
            GradingResult(is_relevant=True, rationale="bad", confidence=1.5)
