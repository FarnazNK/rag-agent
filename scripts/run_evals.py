"""Run the eval suite against the sample corpus.

Usage:
    python scripts/run_evals.py                    # run full suite
    python scripts/run_evals.py --tag smoke        # filter by tag
    python scripts/run_evals.py --out results.json # save machine-readable
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console

# Make the package importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.documents import Document  # noqa: E402

from rag_agent.agent import Agent  # noqa: E402
from rag_agent.evals import (  # noqa: E402
    ContainsScorer,
    ExactMatchScorer,
    LLMJudgeScorer,
    RetrievalRecallScorer,
    load_dataset,
    run_evaluation,
)

app = typer.Typer(add_completion=False)
console = Console()


def _load_corpus(corpus_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for i, path in enumerate(sorted(corpus_dir.glob("*.md"))):
        docs.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": path.name, "chunk_id": f"doc-{i}"},
            )
        )
    return docs


@app.command()
def main(
    dataset: Path = typer.Option(
        Path("data/eval_datasets/hr_smoke.yaml"),
        help="Path to the eval dataset YAML.",
    ),
    corpus: Path = typer.Option(
        Path("data/sample_corpus"),
        help="Corpus directory to index for retrieval.",
    ),
    tag: str = typer.Option("", help="Only run cases with this tag."),
    out: Path = typer.Option(None, help="Write full results to this JSON file."),
    no_judge: bool = typer.Option(False, help="Skip the (expensive) LLM judge scorer."),
    fail_under: float = typer.Option(
        1.0,
        help="Exit non-zero if pass_rate is below this threshold. Default 1.0.",
    ),
) -> None:
    ds = load_dataset(dataset)
    if tag:
        ds = ds.filter_by_tag(tag)
    if not ds.cases:
        console.print(f"[red]No cases to run for tag={tag!r}[/red]")
        raise typer.Exit(code=1)

    docs = _load_corpus(corpus)
    if not docs:
        console.print(f"[red]No corpus found in {corpus}[/red]")
        raise typer.Exit(code=1)

    agent = Agent.from_corpus(docs)

    scorers = [
        ExactMatchScorer(),
        ContainsScorer(),
        RetrievalRecallScorer(),
    ]
    if not no_judge:
        scorers.append(LLMJudgeScorer())

    report = run_evaluation(agent, ds, scorers, console=console)

    if out:
        report.write_json(out)
        console.print(f"[dim]Wrote results to {out}[/dim]")

    if report.pass_rate < fail_under:
        console.print(
            f"[red]Pass rate {report.pass_rate:.0%} < threshold {fail_under:.0%} — "
            f"failing the run.[/red]"
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
