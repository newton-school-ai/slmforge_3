import pytest
from slmforge.data.source_guard import validate


def test_validate_accepts_valid_source():
    """Valid source configurations should not raise errors."""
    valid_configs = [
        {"type": "public", "id": "dataset-1"},
        {"type": "synthetic", "path": "local/data.json"},
        {"type": "local", "path": "path/to/data.csv"},
    ]
    for config in valid_configs:
        validate(config)  # Should not raise


def test_validate_rejects_unregistered_type():
    """Unregistered types should raise a ValueError."""
    config = {"type": "unknown-type", "id": "dataset-1"}
    with pytest.raises(ValueError, match="Unregistered source type: unknown-type"):
        validate(config)


def test_validate_rejects_internal_path():
    """Configurations with paths containing _internal/ should raise a ValueError."""
    configs = [
        {"type": "local", "path": "data/_internal/secret.csv"},
        {"type": "public", "id": "my_org/_internal/data"},
    ]
    for config in configs:
        with pytest.raises(ValueError, match="Internal paths are forbidden"):
            validate(config)
