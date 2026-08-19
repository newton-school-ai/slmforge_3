"""SLMForge CLI entrypoint."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from sqlalchemy.orm import Session

from slmforge.api.db import SessionLocal, engine
from slmforge.data.prefetch import prefetch
from slmforge.engine.state import Base
from slmforge.finetune.checkpoint import (
    build_dir,
    get_build_from_db,
    new_build_id,
    resume_build,
    run_build_epochs,
)

app = typer.Typer(
    no_args_is_help=True,
    help="SLMForge -- plug-and-play SLM builder.",
)

data_app = typer.Typer(help="Dataset operations.")
app.add_typer(data_app, name="data")

console = Console()

SANITY_EPOCHS = 3
SANITY_SEED = 42


def _ensure_db() -> None:
    Base.metadata.create_all(bind=engine)


def _db_session() -> Session:
    _ensure_db()
    return SessionLocal()


@app.command()
def init() -> None:
    """Initialise a .slmforge/ config directory in the current folder."""
    typer.echo("init: not yet implemented (M7)")


@app.command()
def build(
    auto: bool = typer.Option(
        False,
        "--auto",
        help="Skip all confirmation prompts.",
    ),
    recipe: str | None = typer.Option(
        None,
        "--recipe",
        help="Run a bundled recipe by name.",
    ),
    task: str | None = typer.Option(
        None,
        "--task",
        help="Override the automatically detected task type.",
    ),
    base: str | None = typer.Option(
        None,
        "--base",
        help="Override the automatically selected base model.",
    ),
    template: str | None = typer.Option(
        None,
        "--template",
        help="Override the automatically selected prompt template.",
    ),
    resume: str | None = typer.Option(
        None,
        "--resume",
        help="Resume a pre-empted build from its last checkpoint.",
    ),
    kill_after_epoch: int | None = typer.Option(
        None,
        "--kill-after-epoch",
        help="Stop after N epochs (testing helper for checkpoint/resume).",
    ),
    seed: int = typer.Option(
        SANITY_SEED,
        "--seed",
        help="Random seed for deterministic training.",
    ),
) -> None:
    """Discover data in cwd, detect task, fine-tune, eval, and print usage doc."""
    del auto, task, base, template

    session = _db_session()
    cwd = Path.cwd()

    try:
        if resume is not None:
            _run_resume(resume, session=session, cwd=cwd)
            return

        if recipe != "sanity":
            typer.echo(
                "build: full pipeline not yet implemented (M7).\n"
                f"recipe={recipe}\n"
                "Use --recipe sanity for checkpoint/resume smoke testing."
            )
            raise typer.Exit(code=1)

        build_id = new_build_id()
        build_dir(build_id, cwd).mkdir(parents=True, exist_ok=True)

        _, metrics = run_build_epochs(
            build_id,
            total_epochs=SANITY_EPOCHS,
            seed=seed,
            recipe=recipe,
            kill_after_epoch=kill_after_epoch,
            cwd=cwd,
            session=session,
        )

        if kill_after_epoch is not None and metrics is None:
            row = get_build_from_db(session, build_id)
            checkpoint_path = row.checkpoint_path if row else "unknown"
            console.print(
                Panel(
                    f"Build interrupted after epoch {kill_after_epoch}.\n"
                    f"build_id: {build_id}\n"
                    f"checkpoint: {checkpoint_path}\n"
                    f"Resume with: slmforge build --resume {build_id}",
                    title="Checkpoint saved",
                )
            )
            raise typer.Exit(code=0)

        assert metrics is not None
        console.print(
            Panel(
                f"build_id: {build_id}\n"
                f"final eval: {metrics}\n"
                f"Resume with: slmforge build --resume {build_id}",
                title="Build complete",
            )
        )
    finally:
        session.close()


def _run_resume(build_id: str, *, session: Session, cwd: Path) -> None:
    checkpoint, metrics = resume_build(build_id, cwd=cwd, session=session)
    console.print(
        Panel(
            f"Resumed build {build_id}\n"
            f"epochs completed: {checkpoint.epoch}/{checkpoint.total_epochs}\n"
            f"checkpoint: {checkpoint.checkpoint_path}\n"
            f"final eval: {metrics}",
            title="Build resumed",
        )
    )


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
