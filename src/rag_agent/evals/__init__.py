"""Evaluation harness.

A small, honest eval framework. Not a full LangSmith replacement — but enough
to (a) catch regressions before shipping, (b) surface which cases fail and why,
and (c) show the seams where a larger eval platform would plug in.

The three pieces:
    - `dataset`: loading YAML test cases into typed `EvalCase` objects
    - `scorers`: pluggable scoring functions (exact match, contains, LLM-judge)
    - `runner`: executes cases against an Agent, aggregates, prints a report
"""

from rag_agent.evals.dataset import EvalCase, EvalDataset, load_dataset
from rag_agent.evals.runner import EvalReport, EvalResult, run_evaluation
from rag_agent.evals.scorers import (
    ContainsScorer,
    ExactMatchScorer,
    LLMJudgeScorer,
    RetrievalRecallScorer,
    Scorer,
    ScoreResult,
)

__all__ = [
    "EvalCase",
    "EvalDataset",
    "EvalReport",
    "EvalResult",
    "ContainsScorer",
    "ExactMatchScorer",
    "LLMJudgeScorer",
    "RetrievalRecallScorer",
    "Scorer",
    "ScoreResult",
    "load_dataset",
    "run_evaluation",
]
