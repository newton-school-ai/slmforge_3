"""
src/slmforge/data/preview.py
============================
Sample-preview helper for SLMForge.

Provides a lightweight way to peek at the first *N* records from any
supported data source without loading the entire dataset into memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from slmforge.data.ingest import load, detect_format

_DEFAULT_N = 5


def preview(
    path: str | Path,
    n: int = _DEFAULT_N,
    fmt: str | None = None,
) -> List[Dict[str, Any]]:
    """Return the first *n* records from *path*.

    Parameters
    ----------
    path:
        File or directory to preview.
    n:
        Maximum number of records to return.  Defaults to ``5``.
    fmt:
        Optional explicit format string.  When omitted, auto-detected.

    Returns
    -------
    List[Dict[str, Any]]
        Up to *n* records from the beginning of the dataset.

    Examples
    --------
    >>> from slmforge.data.preview import preview
    >>> records = preview("my_data.jsonl")          # first 5 records
    >>> records = preview("my_data.csv", n=10)      # first 10 records
    >>> records = preview("/data/txts/", n=3)       # first 3 txt files
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    records: List[Dict[str, Any]] = []
    for record in load(path, fmt=fmt):
        records.append(record)
        if len(records) >= n:
            break

    return records


def preview_info(path: str | Path, n: int = _DEFAULT_N) -> Dict[str, Any]:
    """Return a summary dict containing format, record count, and sample rows.

    Useful for UI display / API responses.

    Parameters
    ----------
    path:
        File or directory to inspect.
    n:
        Number of sample records to include in the response.

    Returns
    -------
    Dict[str, Any]
        ``{"path": ..., "format": ..., "n_returned": ..., "records": [...]}``
    """
    p = Path(path)
    fmt = detect_format(p)
    records = preview(p, n=n, fmt=fmt)

    return {
        "path": str(p),
        "format": fmt,
        "n_returned": len(records),
        "records": records,
    }


__all__ = ["preview", "preview_info"]
