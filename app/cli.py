"""CLI for RAG Chatbot."""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from app.core.config import settings
from app.services.rag_service import rag_service

app = typer.Typer(help="RAG Chatbot CLI")
console = Console()


@app.command()
def chat(
    message: str = typer.Argument(None, help="Message to send (omit for interactive mode)"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of documents to retrieve"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream response"),
):
    """Chat with the RAG system."""
    asyncio.run(_chat_async(message, top_k, stream))


async def _chat_async(message: str | None, top_k: int, stream: bool):
    console.print(Panel.fit(f"[bold blue]{settings.app_name}[/bold blue] v{settings.app_version}"))
    console.print(f"Using model: [cyan]{settings.ollama_llm_model}[/cyan]")
    console.print(f"Embeddings: [cyan]{settings.ollama_embed_model}[/cyan]")
    console.print()

    await rag_service.initialize()

    if not rag_service.is_ready():
        console.print("[red]RAG service not ready. Check Ollama connection.[/red]")
        return

    if message:
        await _send_message(message, top_k, stream)
    else:
        console.print("[dim]Enter 'quit' or 'exit' to quit. Enter 'clear' to clear screen.[/dim]")
        console.print()
        while True:
            try:
                user_input = Prompt.ask("[bold green]You[/bold green]")
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                if user_input.lower() == "clear":
                    console.clear()
                    continue
                if not user_input.strip():
                    continue
                await _send_message(user_input, top_k, stream)
            except (KeyboardInterrupt, EOFError):
                break


async def _send_message(message: str, top_k: int, stream: bool):
    console.print()
    with console.status("[bold yellow]Thinking...[/bold yellow]", spinner="dots"):
        response = await rag_service.query(message, top_k=top_k, stream=stream)

    if stream:
        console.print("[bold blue]Assistant[/bold blue]: ", end="")
        full_response = ""
        async for token in response.async_response_gen():
            console.print(token, end="")
            full_response += token
        console.print()
    else:
        full_response = str(response)
        console.print(Markdown(f"**Assistant:** {full_response}"))

    # Show citations
    if hasattr(response, "source_nodes") and response.source_nodes:
        console.print("\n[dim]Sources:[/dim]")
        for i, node in enumerate(response.source_nodes[:3], 1):
            filename = node.metadata.get("filename", "unknown")
            score = node.score if node.score else 0
            console.print(f"  [dim]{i}.[/dim] {filename} (score: {score:.3f})")
    console.print()


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="File or directory to ingest"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively ingest directory"),
):
    """Ingest documents into the vector store."""
    asyncio.run(_ingest_async(path, recursive))


async def _ingest_async(path: Path, recursive: bool):
    console.print(Panel.fit(f"[bold blue]{settings.app_name}[/bold blue] - Document Ingestion"))
    await rag_service.initialize()

    if path.is_file():
        files = [path]
    elif path.is_dir():
        if recursive:
            files = list(path.rglob("*"))
        else:
            files = list(path.glob("*"))
        files = [
            f for f in files if f.is_file() and f.suffix.lower() in settings.supported_extensions
        ]
    else:
        console.print(f"[red]Path not found: {path}[/red]")
        return

    if not files:
        console.print("[yellow]No supported files found[/yellow]")
        return

    console.print(f"Found [cyan]{len(files)}[/cyan] files to ingest")
    with console.status("[bold yellow]Indexing...[/bold yellow]", spinner="dots"):
        result = await rag_service.add_documents(files)

    console.print(
        f"[green]Success![/green] Added {result['chunks']} chunks from {result['added']} documents"
    )


@app.command()
def list_docs():
    """List all indexed documents."""
    asyncio.run(_list_docs_async())


async def _list_docs_async():
    console.print(Panel.fit(f"[bold blue]{settings.app_name}[/bold blue] - Indexed Documents"))
    await rag_service.initialize()

    docs = await rag_service.list_documents()
    if not docs:
        console.print("[yellow]No documents indexed[/yellow]")
        return

    for doc in docs:
        console.print(
            f"  [cyan]{doc['document_id']}[/cyan]  {doc['filename']}  ({doc['chunks']} chunks)"
        )


@app.command()
def health():
    """Check system health."""
    asyncio.run(_health_async())


async def _health_async():
    console.print(Panel.fit(f"[bold blue]{settings.app_name}[/bold blue] - Health Check"))
    ollama = await rag_service.check_ollama_connection()
    ready = rag_service.is_ready()
    docs = await rag_service.list_documents() if ready else []

    console.print(
        f"  Ollama: {'[green]Connected[/green]' if ollama else '[red]Disconnected[/red]'}"
    )
    console.print(f"  Vector Store: {'[green]Ready[/green]' if ready else '[red]Not Ready[/red]'}")
    console.print(f"  Documents: [cyan]{len(docs)}[/cyan]")


if __name__ == "__main__":
    app()
