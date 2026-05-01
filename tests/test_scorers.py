"""Tests for evaluation scorers."""

from __future__ import annotations

from rag_agent.evals.dataset import EvalCase
from rag_agent.evals.scorers import (
    ContainsScorer,
    ExactMatchScorer,
    RetrievalRecallScorer,
)
from rag_agent.schemas import AgentState, RetrievedChunk


def _state(answer: str = "", sources: list[str] | None = None, **kwargs) -> AgentState:
    chunks = [
        RetrievedChunk(chunk_id=f"c{i}", content="", source=src, score=1.0)
        for i, src in enumerate(sources or [])
    ]
    return AgentState(query="q", final_answer=answer, chunks=chunks, **kwargs)


class TestContainsScorer:
    def test_skipped_when_no_expected(self):
        result = ContainsScorer()(
            EvalCase(case_id="c1", query="q"),
            _state(answer="anything"),
        )
        assert result.passed
        assert "skipped" in result.rationale

    def test_all_substrings_present_passes(self):
        case = EvalCase(case_id="c1", query="q", expected_contains=["15", "PTO"])
        result = ContainsScorer()(case, _state(answer="New hires get 15 PTO days."))
        assert result.passed
        assert result.score == 1.0

    def test_partial_match_gives_partial_score(self):
        case = EvalCase(case_id="c1", query="q", expected_contains=["15", "PTO", "vacation"])
        result = ContainsScorer()(case, _state(answer="New hires get 15 PTO days."))
        assert not result.passed
        assert 0.6 < result.score < 0.7  # 2/3

    def test_case_insensitive(self):
        case = EvalCase(case_id="c1", query="q", expected_contains=["pto"])
        result = ContainsScorer()(case, _state(answer="Your PTO balance is updated monthly."))
        assert result.passed


class TestRetrievalRecallScorer:
    def test_all_expected_sources_retrieved(self):
        case = EvalCase(case_id="c1", query="q", expected_sources=["a.md", "b.md"])
        result = RetrievalRecallScorer()(case, _state(sources=["a.md", "b.md", "c.md"]))
        assert result.passed
        assert result.score == 1.0

    def test_missing_source_fails(self):
        case = EvalCase(case_id="c1", query="q", expected_sources=["a.md", "b.md"])
        result = RetrievalRecallScorer()(case, _state(sources=["a.md"]))
        assert not result.passed
        assert result.score == 0.5


class TestExactMatchScorer:
    def test_refuse_checks_route_not_answer(self):
        case = EvalCase(case_id="c1", query="q", should_refuse=True, expected_exact="refused")
        # Correct refusal
        result = ExactMatchScorer()(case, _state(answer="something", route="refuse"))
        assert result.passed
        # Failed refusal — agent answered instead
        result = ExactMatchScorer()(case, _state(answer="here is the info", route="retrieve"))
        assert not result.passed
