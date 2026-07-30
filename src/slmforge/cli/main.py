"""SLMForge CLI entrypoint."""

from __future__ import annotations

import typer

from slmforge.data.prefetch import prefetch

app = typer.Typer(
    no_args_is_help=True,
    help="SLMForge -- plug-and-play SLM builder.",
)

data_app = typer.Typer(help="Dataset operations.")
app.add_typer(data_app, name="data")


@app.command()
def init() -> None:
    """Initialise a .slmforge/ config directory in the current folder."""
    typer.echo("init: not yet implemented (M7)")


@app.command()
def build(
    auto: bool = typer.Option(False, "--auto", help="Skip all confirmation prompts."),
    recipe: str | None = typer.Option(None, "--recipe", help="Run a bundled recipe by name."),
) -> None:
    """Discover data in cwd, detect task, fine-tune, eval, and print usage doc."""
    typer.echo(f"build: not yet implemented (M7). auto={auto} recipe={recipe}")


@app.command()
def eval(build_id: str) -> None:
    """Re-run eval on an existing build."""
    typer.echo(f"eval: not yet implemented (M7). build_id={build_id}")


@app.command()
def serve(build_id: str, port: int = 8000) -> None:
    """Start a local vLLM endpoint for a build."""
    typer.echo(f"serve: not yet implemented (M7). build_id={build_id} port={port}")


@app.command(name="list")
def list_cmd() -> None:
    """List all builds in this folder."""
    typer.echo("list: not yet implemented (M7)")


@app.command()
def usage(build_id: str) -> None:
    """Print the USAGE.md for a build."""
    typer.echo(f"usage: not yet implemented (M7). build_id={build_id}")


@app.command()
def ui() -> None:
    """Launch the localhost web UI."""
    typer.echo("ui: not yet implemented (M7)")


@data_app.command(name="prefetch")
def prefetch_dataset(dataset_id: str) -> None:
    """Download and cache a public dataset."""
    path = prefetch(dataset_id)
    typer.echo(f"Dataset cached at: {path}")


if __name__ == "__main__":
    app()
