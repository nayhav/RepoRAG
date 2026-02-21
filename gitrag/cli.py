"""CLI interface for GitRAG using Click and Rich."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from gitrag.config import load_config
from gitrag.core.pipeline import RAGPipeline
from gitrag.core.types import GeneratedAnswer

console = Console()
error_console = Console(stderr=True)


def _load_pipeline(repo_path: str, config: str | None) -> RAGPipeline:
    """Create a pipeline instance from CLI arguments."""
    repo = Path(repo_path)
    if not repo.exists():
        error_console.print(f"[red]Error:[/red] Repository path does not exist: {repo}")
        raise SystemExit(1)
    cfg = load_config(config)
    return RAGPipeline(repo, cfg)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def main() -> None:
    """GitRAG — Chat with any local GitHub repository."""


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

@main.command()
@click.argument("repo_path")
@click.option("--config", default=None, help="Path to config.yaml")
@click.option("--force", is_flag=True, help="Clear existing index before re-indexing")
def index(repo_path: str, config: str | None, force: bool) -> None:
    """Index a local repository."""
    pipeline = _load_pipeline(repo_path, config)

    with console.status("[bold green]Indexing repository…", spinner="dots"):
        stats = pipeline.index(force=force)

    table = Table(title="Indexing Complete", show_header=False, border_style="green")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Files", str(stats.total_files))
    table.add_row("Chunks", str(stats.total_chunks))
    table.add_row("Languages", ", ".join(stats.languages))
    table.add_row("Duration", f"{stats.duration_seconds:.2f}s")
    console.print(table)


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

@main.command()
@click.argument("repo_path")
@click.argument("question")
@click.option("--config", default=None, help="Path to config.yaml")
def query(repo_path: str, question: str, config: str | None) -> None:
    """Single-shot query against an indexed repository."""
    pipeline = _load_pipeline(repo_path, config)

    with console.status("[bold cyan]Thinking…", spinner="dots"):
        answer, _conv = pipeline.query(question)

    _print_answer(answer)


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

@main.command()
@click.argument("repo_path")
@click.option("--config", default=None, help="Path to config.yaml")
def chat(repo_path: str, config: str | None) -> None:
    """Start an interactive chat session with an indexed repository."""
    pipeline = _load_pipeline(repo_path, config)

    status = pipeline.get_status()
    if not status.get("index_exists"):
        error_console.print(
            "[yellow]Warning:[/yellow] No index found. Run [bold]gitrag index[/bold] first."
        )
        raise SystemExit(1)

    console.print(Panel(
        f"[bold]GitRAG Chat[/bold]\n"
        f"Repository: [cyan]{repo_path}[/cyan]\n"
        f"Chunks indexed: [green]{status.get('chunks_count', '?')}[/green]\n\n"
        f"Type your question, or use [bold]/quit[/bold], [bold]/clear[/bold], [bold]/stats[/bold].",
        title="Welcome",
        border_style="blue",
    ))

    conversation_id: str | None = None

    while True:
        try:
            user_input = console.input("\n[bold blue]You>[/bold blue] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # Slash commands
        if user_input.lower() == "/quit":
            console.print("[dim]Goodbye![/dim]")
            break

        if user_input.lower() == "/clear":
            if conversation_id:
                pipeline.clear_conversation(conversation_id)
            conversation_id = None
            console.print("[dim]Conversation cleared.[/dim]")
            continue

        if user_input.lower() == "/stats":
            _print_status(pipeline.get_status())
            continue

        # Regular question
        with console.status("[bold cyan]Thinking…", spinner="dots"):
            answer, conv = pipeline.query(user_input, conversation_id=conversation_id)
        conversation_id = conv.conversation_id

        _print_answer(answer)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@main.command()
@click.argument("repo_path")
@click.option("--config", default=None, help="Path to config.yaml")
def status(repo_path: str, config: str | None) -> None:
    """Show index status for a repository."""
    pipeline = _load_pipeline(repo_path, config)
    _print_status(pipeline.get_status())


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_answer(answer: GeneratedAnswer) -> None:
    """Render a GeneratedAnswer with markdown and citations."""
    console.print(Panel(Markdown(answer.content), title="Answer", border_style="green"))

    if answer.citations:
        table = Table(title="Citations", border_style="cyan")
        table.add_column("File", style="cyan")
        table.add_column("Lines", style="yellow")
        table.add_column("Symbol", style="green")
        for cite in answer.citations:
            lines = f"{cite.start_line}-{cite.end_line}" if cite.end_line != cite.start_line else str(cite.start_line)
            table.add_row(cite.file_path, lines, cite.symbol_name)
        console.print(table)


def _print_status(info: dict) -> None:
    """Render index status as a table."""
    table = Table(title="Index Status", show_header=False, border_style="blue")
    table.add_column("Key", style="bold")
    table.add_column("Value")
    for key, value in info.items():
        table.add_row(key, str(value))
    console.print(table)


if __name__ == "__main__":
    main()
