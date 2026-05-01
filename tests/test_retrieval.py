"""Tests for retrieval fusion logic.

These exercise the deterministic core of retrieval — no LLM, no network.
"""

from __future__ import annotations

from langchain_core.documents import Document

from rag_agent.retrieval import HybridRetriever, _reciprocal_rank_fusion
from rag_agent.schemas import RetrievedChunk


def _chunk(cid: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        content=f"content for {cid}",
        source="test",
        score=score,
    )


class TestReciprocalRankFusion:
    def test_empty_inputs_return_empty(self):
        assert _reciprocal_rank_fusion([]) == []
        assert _reciprocal_rank_fusion([[], []]) == []

    def test_single_list_preserves_order(self):
        chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
        fused = _reciprocal_rank_fusion([chunks])
        assert [c.chunk_id for c in fused] == ["a", "b", "c"]

    def test_overlap_between_lists_boosts_shared_items(self):
        # 'a' appears top in both lists -> should rank #1 after fusion.
        list1 = [_chunk("a"), _chunk("b"), _chunk("c")]
        list2 = [_chunk("a"), _chunk("d"), _chunk("e")]
        fused = _reciprocal_rank_fusion([list1, list2])
        assert fused[0].chunk_id == "a"

    def test_scores_are_normalized_to_unit_interval(self):
        list1 = [_chunk("a"), _chunk("b")]
        list2 = [_chunk("c"), _chunk("d")]
        fused = _reciprocal_rank_fusion([list1, list2])
        for c in fused:
            assert 0.0 <= c.score <= 1.0
        # Top item should have the maximum normalized score.
        assert fused[0].score == 1.0


class TestHybridRetriever:
    def test_bm25_search_returns_empty_for_empty_corpus(self):
        class FakeStore:
            def add_documents(self, docs):
                return []

            def similarity_search(self, query, k):
                return []

            def count(self):
                return 0

        retriever = HybridRetriever(store=FakeStore(), corpus=[])
        # Direct BM25 path
        assert retriever._bm25_search("anything", k=5) == []

    def test_bm25_search_ranks_exact_keyword_matches_first(self):
        class FakeStore:
            def add_documents(self, docs):
                return []

            def similarity_search(self, query, k):
                return []

            def count(self):
                return 0

        # BM25Okapi does exact-token matching (no stemming), so queries must
        # share tokens with the matching docs. This is the well-known
        # tradeoff vs. dense retrieval — we keep BM25 for exact matches and
        # rely on embeddings for morphological / semantic similarity.
        corpus = [
            Document(page_content="the cat sat on the mat", metadata={"source": "a.md"}),
            Document(page_content="quantum mechanics is hard", metadata={"source": "b.md"}),
            Document(page_content="python programming is fun", metadata={"source": "c.md"}),
        ]
        retriever = HybridRetriever(store=FakeStore(), corpus=corpus)
        results = retriever._bm25_search("cat mat", k=1)

        assert len(results) == 1
        assert results[0].source == "a.md"
        # Normalized score should be at the top of the [0,1] range.
        assert results[0].score == 1.0
