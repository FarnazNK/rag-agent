"""Eval dataset loading.

Cases live in YAML so non-engineers can contribute. Each case is validated
through Pydantic — schema mistakes fail loudly at load time rather than
three minutes into a run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """A single test case."""

    case_id: str
    query: str
    # What the answer MUST contain (case-insensitive substring). Optional.
    expected_contains: list[str] = Field(default_factory=list)
    # Exact string the answer should equal (rarely useful for generative; kept
    # for router tests where the output is a fixed token).
    expected_exact: str | None = None
    # Sources that retrieval should have surfaced. Used by RetrievalRecallScorer.
    expected_sources: list[str] = Field(default_factory=list)
    # Free-form criteria for the LLM judge. E.g. "cites the PTO policy doc".
    judge_criteria: str | None = None
    # Tags for filtering (e.g. "smoke", "regression", "safety").
    tags: list[str] = Field(default_factory=list)
    # Whether the agent should refuse. Tested by checking route=refuse.
    should_refuse: bool = False

    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalDataset(BaseModel):
    """A named collection of cases."""

    name: str
    description: str = ""
    cases: list[EvalCase]

    def filter_by_tag(self, tag: str) -> EvalDataset:
        return EvalDataset(
            name=f"{self.name}[{tag}]",
            description=self.description,
            cases=[c for c in self.cases if tag in c.tags],
        )


def load_dataset(path: Path | str) -> EvalDataset:
    """Load a YAML dataset file. Raises on schema errors."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return EvalDataset.model_validate(raw)
