"""Unit tests for checkpoint save/resume (Issue 19)."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from slmforge.engine.state import Base, Build
from slmforge.finetune.checkpoint import (
    BuildCheckpoint,
    deterministic_eval,
    get_build_from_db,
    load_checkpoint,
    new_build_id,
    record_checkpoint_db,
    remaining_epochs,
    resume_build,
    resume_training_cfg,
    run_build_epochs,
    save_checkpoint,
)


@pytest.fixture
def db_session(tmp_path: Path) -> Session:
    """Create an isolated in-memory SQLite session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    yield session
    session.close()


def test_checkpoint_saved_per_epoch(tmp_path: Path, db_session: Session) -> None:
    """Each epoch writes a checkpoint directory and manifest."""
    build_id = new_build_id()
    checkpoint, _ = run_build_epochs(
        build_id,
        total_epochs=2,
        seed=42,
        recipe="sanity",
        kill_after_epoch=1,
        cwd=tmp_path,
        session=db_session,
    )

    ckpt_dir = tmp_path / "builds" / build_id / checkpoint.checkpoint_path
    assert ckpt_dir.exists()
    assert (ckpt_dir / "checkpoint.json").exists()

    row = get_build_from_db(db_session, build_id)
    assert row is not None
    assert row.checkpoint_path == checkpoint.checkpoint_path
    assert row.epochs_completed == 1


def test_resume_produces_identical_eval(tmp_path: Path, db_session: Session) -> None:
    """Resumed run matches uninterrupted run final eval with the same seed."""
    seed = 42
    total_epochs = 3

    uninterrupted_id = new_build_id()
    _, uninterrupted_metrics = run_build_epochs(
        uninterrupted_id,
        total_epochs=total_epochs,
        seed=seed,
        recipe="sanity",
        cwd=tmp_path,
        session=db_session,
    )
    assert uninterrupted_metrics is not None

    interrupted_id = new_build_id()
    run_build_epochs(
        interrupted_id,
        total_epochs=total_epochs,
        seed=seed,
        recipe="sanity",
        kill_after_epoch=1,
        cwd=tmp_path,
        session=db_session,
    )
    _, resumed_metrics = resume_build(
        interrupted_id,
        cwd=tmp_path,
        session=db_session,
    )

    assert resumed_metrics == uninterrupted_metrics
    assert resumed_metrics == deterministic_eval(seed, total_epochs)


def test_no_double_counted_epochs(tmp_path: Path, db_session: Session) -> None:
    """Resume continues from the next epoch without re-running completed ones."""
    build_id = new_build_id()
    total_epochs = 3

    run_build_epochs(
        build_id,
        total_epochs=total_epochs,
        seed=7,
        recipe="sanity",
        kill_after_epoch=1,
        cwd=tmp_path,
        session=db_session,
    )

    checkpoint = load_checkpoint(build_id, cwd=tmp_path)
    assert checkpoint.epoch == 1
    assert remaining_epochs(checkpoint) == 2

    resume_cfg = resume_training_cfg(checkpoint)
    assert resume_cfg["num_train_epochs"] == 2
    assert resume_cfg["epochs_completed"] == 1
    assert resume_cfg["seed"] == 7

    _, metrics = resume_build(build_id, cwd=tmp_path, session=db_session)
    final = load_checkpoint(build_id, cwd=tmp_path)
    assert final.epoch == total_epochs
    assert final.status == "complete"
    assert metrics == deterministic_eval(7, total_epochs)

    row = get_build_from_db(db_session, build_id)
    assert row is not None
    assert row.epochs_completed == total_epochs


def test_record_checkpoint_db_updates_path(db_session: Session) -> None:
    """Checkpoint path is stored and updated in the database."""
    build_id = "build_test_001"
    row = record_checkpoint_db(
        db_session,
        build_id,
        "checkpoint-epoch-1",
        seed=99,
        epochs_completed=1,
        total_epochs=3,
        status="interrupted",
    )
    assert isinstance(row, Build)
    assert row.checkpoint_path == "checkpoint-epoch-1"
    assert row.seed == 99

    updated = record_checkpoint_db(
        db_session,
        build_id,
        "checkpoint-epoch-3",
        epochs_completed=3,
        status="complete",
    )
    assert updated.checkpoint_path == "checkpoint-epoch-3"
    assert updated.epochs_completed == 3
    assert updated.status == "complete"


def test_save_and_load_checkpoint_roundtrip(tmp_path: Path) -> None:
    """Checkpoint manifest round-trips through save and load."""
    build_id = new_build_id()
    checkpoint = BuildCheckpoint(
        build_id=build_id,
        epoch=2,
        total_epochs=5,
        seed=11,
        checkpoint_path="checkpoint-epoch-2",
        recipe="sanity",
        status="running",
    )
    save_checkpoint(checkpoint, cwd=tmp_path)
    loaded = load_checkpoint(build_id, cwd=tmp_path)
    assert loaded.epoch == 2
    assert loaded.seed == 11
    assert loaded.total_epochs == 5
