import pytest

from slmforge.data.source_guard import validate


def test_rejects_unknown_source() -> None:
    with pytest.raises(ValueError, match="Unknown source type"):
        validate("unknown", "data/file.json")


def test_rejects_internal_path() -> None:
    with pytest.raises(ValueError, match="Internal data paths"):
        validate("public", "_internal/data.json")


def test_accepts_valid_source() -> None:
    validate("public", "data/file.json")
