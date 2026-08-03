from pathlib import Path

from slmforge.data.sources.registry import get_source_adapter


def validate(source_type: str, source_path: str | Path) -> None:
    """Validate source type and ensure internal paths are not allowed."""
    # Raises ValueError if the source type is not registered.
    get_source_adapter(source_type)

    path = Path(source_path)

    if "_internal" in path.parts:
        msg = "Internal data paths are not allowed."
        raise ValueError(msg)
