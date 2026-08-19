"""Checkpoint save/load and resume helpers for fine-tuning runs (Issue 19)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from slmforge.engine.state import Build


@dataclass
class BuildCheckpoint:
    """On-disk checkpoint manifest for a build run."""

    build_id: str
    epoch: int
    total_epochs: int
    seed: int
    checkpoint_path: str
    recipe: str | None = None
    status: str = "running"
    eval_metrics: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BuildCheckpoint:
        return cls(
            build_id=str(data["build_id"]),
            epoch=int(data["epoch"]),
            total_epochs=int(data["total_epochs"]),
            seed=int(data["seed"]),
            checkpoint_path=str(data["checkpoint_path"]),
            recipe=data.get("recipe"),
            status=str(data.get("status", "running")),
            eval_metrics=data.get("eval_metrics"),
        )


def builds_root(cwd: Path | None = None) -> Path:
    """Return the root directory for on-disk build artifacts."""
    return (cwd or Path.cwd()) / "builds"


def build_dir(build_id: str, cwd: Path | None = None) -> Path:
    """Return the directory for a specific build."""
    return builds_root(cwd) / build_id


def checkpoint_epoch_dir(build_id: str, epoch: int, cwd: Path | None = None) -> Path:
    """Return the checkpoint directory for a given epoch."""
    return build_dir(build_id, cwd) / f"checkpoint-epoch-{epoch}"


def manifest_path(checkpoint_dir: Path) -> Path:
    """Return the manifest file path inside a checkpoint directory."""
    return checkpoint_dir / "checkpoint.json"


def new_build_id() -> str:
    """Generate a unique build id."""
    ts = datetime.now(UTC).strftime("%Y_%m_%d_%H%M%S")
    return f"build_{ts}"


def deterministic_eval(seed: int, total_epochs: int) -> dict[str, float]:
    """Compute deterministic final eval metrics from seed and epoch count."""
    digest = hashlib.sha256(f"{seed}:{total_epochs}".encode()).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return {
        "loss": round(1.0 - value * 0.5, 6),
        "accuracy": round(0.5 + value * 0.4, 6),
    }


def remaining_epochs(checkpoint: BuildCheckpoint) -> int:
    """Return how many epochs are left; avoids double-counting completed epochs."""
    return max(checkpoint.total_epochs - checkpoint.epoch, 0)


def resume_training_cfg(
    checkpoint: BuildCheckpoint,
    training_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a training config for resuming from the latest checkpoint."""
    cfg = dict(training_cfg or {})
    cfg["seed"] = checkpoint.seed
    cfg["data_seed"] = checkpoint.seed
    cfg["num_train_epochs"] = remaining_epochs(checkpoint)
    cfg["resume_from_checkpoint"] = checkpoint.checkpoint_path
    cfg["epochs_completed"] = checkpoint.epoch
    return cfg


def save_checkpoint(
    checkpoint: BuildCheckpoint,
    cwd: Path | None = None,
) -> Path:
    """Persist a checkpoint manifest to disk and return its directory."""
    rel_path = checkpoint.checkpoint_path
    ckpt_dir = Path(rel_path)
    if not ckpt_dir.is_absolute():
        ckpt_dir = build_dir(checkpoint.build_id, cwd) / ckpt_dir.name

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    manifest = checkpoint.to_dict()
    manifest["checkpoint_path"] = rel_path
    with manifest_path(ckpt_dir).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    latest = build_dir(checkpoint.build_id, cwd) / "latest_checkpoint.json"
    with latest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return ckpt_dir


def load_checkpoint(build_id: str, cwd: Path | None = None) -> BuildCheckpoint:
    """Load the latest checkpoint for a build."""
    latest = build_dir(build_id, cwd) / "latest_checkpoint.json"
    if not latest.exists():
        raise FileNotFoundError(f"No checkpoint found for build {build_id}")

    with latest.open(encoding="utf-8") as handle:
        data = json.load(handle)

    return BuildCheckpoint.from_dict(data)


def record_checkpoint_db(
    session: Session,
    build_id: str,
    checkpoint_path: str,
    *,
    seed: int | None = None,
    epochs_completed: int | None = None,
    total_epochs: int | None = None,
    status: str | None = None,
) -> Build:
    """Create or update a build row with the latest checkpoint path."""
    row = session.query(Build).filter(Build.build_id == build_id).one_or_none()
    if row is None:
        row = Build(build_id=build_id, name=build_id)
        session.add(row)

    row.checkpoint_path = checkpoint_path
    if seed is not None:
        row.seed = seed
    if epochs_completed is not None:
        row.epochs_completed = epochs_completed
    if total_epochs is not None:
        row.total_epochs = total_epochs
    if status is not None:
        row.status = status

    session.commit()
    session.refresh(row)
    return row


def get_build_from_db(session: Session, build_id: str) -> Build | None:
    """Fetch a build row by external build id."""
    return session.query(Build).filter(Build.build_id == build_id).one_or_none()


def run_epoch(
    build_id: str,
    epoch: int,
    *,
    total_epochs: int,
    seed: int,
    recipe: str | None = None,
    cwd: Path | None = None,
    session: Session | None = None,
) -> BuildCheckpoint:
    """Simulate one training epoch and persist its checkpoint."""
    ckpt_dir = checkpoint_epoch_dir(build_id, epoch, cwd)
    checkpoint = BuildCheckpoint(
        build_id=build_id,
        epoch=epoch,
        total_epochs=total_epochs,
        seed=seed,
        checkpoint_path=str(ckpt_dir.relative_to(build_dir(build_id, cwd))),
        recipe=recipe,
        status="running" if epoch < total_epochs else "complete",
    )
    save_checkpoint(checkpoint, cwd=cwd)

    if session is not None:
        record_checkpoint_db(
            session,
            build_id,
            checkpoint.checkpoint_path,
            seed=seed,
            epochs_completed=epoch,
            total_epochs=total_epochs,
            status=checkpoint.status,
        )

    return checkpoint


def run_build_epochs(
    build_id: str,
    *,
    total_epochs: int,
    seed: int,
    recipe: str | None = None,
    start_epoch: int = 1,
    kill_after_epoch: int | None = None,
    cwd: Path | None = None,
    session: Session | None = None,
) -> tuple[BuildCheckpoint, dict[str, float] | None]:
    """Run epochs from start_epoch through total_epochs, optionally stopping early."""
    last_checkpoint: BuildCheckpoint | None = None

    for epoch in range(start_epoch, total_epochs + 1):
        last_checkpoint = run_epoch(
            build_id,
            epoch,
            total_epochs=total_epochs,
            seed=seed,
            recipe=recipe,
            cwd=cwd,
            session=session,
        )
        if kill_after_epoch is not None and epoch >= kill_after_epoch:
            if session is not None:
                record_checkpoint_db(
                    session,
                    build_id,
                    last_checkpoint.checkpoint_path,
                    seed=seed,
                    epochs_completed=epoch,
                    total_epochs=total_epochs,
                    status="interrupted",
                )
            return last_checkpoint, None

    assert last_checkpoint is not None
    metrics = deterministic_eval(seed, total_epochs)
    last_checkpoint.eval_metrics = metrics
    last_checkpoint.status = "complete"
    save_checkpoint(last_checkpoint, cwd=cwd)

    if session is not None:
        record_checkpoint_db(
            session,
            build_id,
            last_checkpoint.checkpoint_path,
            seed=seed,
            epochs_completed=total_epochs,
            total_epochs=total_epochs,
            status="complete",
        )

    eval_path = build_dir(build_id, cwd) / "eval_results.json"
    with eval_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return last_checkpoint, metrics


def resume_build(
    build_id: str,
    *,
    cwd: Path | None = None,
    session: Session | None = None,
) -> tuple[BuildCheckpoint, dict[str, float]]:
    """Resume an interrupted build from its latest checkpoint."""
    checkpoint = load_checkpoint(build_id, cwd=cwd)
    if checkpoint.status == "complete":
        metrics = checkpoint.eval_metrics or deterministic_eval(
            checkpoint.seed,
            checkpoint.total_epochs,
        )
        return checkpoint, metrics

    next_epoch = checkpoint.epoch + 1
    if next_epoch > checkpoint.total_epochs:
        metrics = deterministic_eval(checkpoint.seed, checkpoint.total_epochs)
        checkpoint.status = "complete"
        checkpoint.eval_metrics = metrics
        save_checkpoint(checkpoint, cwd=cwd)
        return checkpoint, metrics

    resume_training_cfg(checkpoint)
    _, metrics = run_build_epochs(
        build_id,
        total_epochs=checkpoint.total_epochs,
        seed=checkpoint.seed,
        recipe=checkpoint.recipe,
        start_epoch=next_epoch,
        cwd=cwd,
        session=session,
    )
    assert metrics is not None
    final = load_checkpoint(build_id, cwd=cwd)
    return final, metrics
