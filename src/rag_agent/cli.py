"""Command-line interface.

A thin Typer app so reviewers can `rag-agent ask "question"` and see it
work end to end. All business logic stays in the library.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from langchain_core.documents import Document
from rich.console import Console
from rich.panel import Panel

from rag_agent.agent import Agent

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _load_sample_corpus(data_dir: Path) -> list[Document]:
    """Load text files from `data_dir` as a corpus. One file = one doc."""
    if not data_dir.exists():
        return []
    docs: list[Document] = []
    for i, path in enumerate(sorted(data_dir.glob("*.md"))):
        docs.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": path.name, "chunk_id": f"doc-{i}"},
            )
        )
    return docs


@app.command()
def ingest(
    data_dir: Path = typer.Option(
        Path("data/sample_corpus"),
        help="Directory containing .md files to ingest.",
    ),
) -> None:
    """Index the sample corpus into the vector store."""
    docs = _load_sample_corpus(data_dir)
    if not docs:
        console.print(f"[red]No .md files found in {data_dir}[/red]")
        raise typer.Exit(code=1)
    Agent.from_corpus(docs)
    console.print(f"[green]Indexed {len(docs)} documents.[/green]")


@app.command()
def ask(
    query: str = typer.Argument(..., help="Question to ask the agent."),
    data_dir: Path = typer.Option(
        Path("data/sample_corpus"),
        help="Corpus directory (must match the one used at ingest time).",
    ),
    show_trace: bool = typer.Option(False, "--trace", help="Print intermediate state."),
) -> None:
    """Ask the agent a single question."""
    docs = _load_sample_corpus(data_dir)
    if not docs:
        console.print(f"[red]No corpus found in {data_dir}. Run `ingest` first.[/red]")
        raise typer.Exit(code=1)

    agent = Agent.from_corpus(docs)
    state = agent.run(query)

    if show_trace:
        console.print(Panel.fit(f"route: {state.route}", title="router"))
        if state.rewritten_query:
            console.print(Panel.fit(state.rewritten_query, title="rewritten query"))
        for c in state.chunks:
            console.print(
                Panel.fit(
                    c.content[:200] + ("..." if len(c.content) > 200 else ""),
                    title=f"{c.source} (score={c.score:.2f})",
                )
            )
        if state.grading:
            console.print(
                Panel.fit(
                    f"relevant={state.grading.is_relevant}\n"
                    f"confidence={state.grading.confidence}\n"
                    f"rationale={state.grading.rationale}",
                    title="grader",
                )
            )

    console.print(Panel(state.final_answer or "(no answer)", title="answer", border_style="green"))


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
