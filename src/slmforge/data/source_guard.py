"""
src/slmforge/data/source_guard.py
===================================
Validates data source configurations to prevent internal data leaks.
"""

from typing import Any, Dict
from slmforge.data.sources.registry import _REGISTRY


def validate(source_meta: Dict[str, Any]) -> None:
    """Validate a source configuration dictionary.

    Raises
    ------
    ValueError
        If the source type is not registered or if the path indicates internal data.
    """
    source_type = source_meta.get("type")
    if source_type not in _REGISTRY:
        raise ValueError(f"Unregistered source type: {source_type}")

    path = source_meta.get("path") or source_meta.get("id")
    if path and "_internal/" in str(path):
        raise ValueError(f"Internal paths are forbidden in source config: {path}")


__all__ = ["validate"]
