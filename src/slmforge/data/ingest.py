"""src/slmforge/data/ingest.py.
===========================
Multi-format ingestion layer for SLMForge.

Supported formats
-----------------
- JSONL  (.jsonl)
- CSV    (.csv)
- Parquet (.parquet)
- TXT folder - a directory whose files are all .txt

Format detection
----------------
1. Try ``python-magic`` (MIME sniffing on the raw bytes).
2. Fall back to file-extension when libmagic is not available or the
   path is a directory.

Each public reader returns an ``Iterator[Dict[str, Any]]`` so callers
always receive a normalised record stream regardless of format.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional python-magic import (graceful fallback)
# ---------------------------------------------------------------------------
try:
    import magic as _magic  # python-magic

    _MAGIC_AVAILABLE = True
except (ImportError, OSError):  # libmagic shared library not installed
    _MAGIC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------
FORMAT_JSONL = "jsonl"
FORMAT_CSV = "csv"
FORMAT_PARQUET = "parquet"
FORMAT_TXT_FOLDER = "txt-folder"

# MIME to format mapping (python-magic output)
_MIME_TO_FORMAT: dict[str, str] = {
    "application/json": FORMAT_JSONL,  # single-object JSON
    "application/x-ndjson": FORMAT_JSONL,  # libmagic: "New Line Delimited JSON" = JSONL
    "text/csv": FORMAT_CSV,
    "application/csv": FORMAT_CSV,
    "application/octet-stream": FORMAT_PARQUET,  # parquet has no standard MIME
    "application/vnd.apache.parquet": FORMAT_PARQUET,
}

# Extension to format mapping (fallback / directory check)
_EXT_TO_FORMAT: dict[str, str] = {
    ".jsonl": FORMAT_JSONL,
    ".csv": FORMAT_CSV,
    ".parquet": FORMAT_PARQUET,
}


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def detect_format(path: str | Path) -> str:
    """Detect the data format of *path*.

    Strategy
    --------
    1. If *path* is a directory, return ``txt-folder``.
    2. If ``python-magic`` is available, sniff the MIME type and map to a
       known format.
    3. Fall back to the file extension.
    4. Last resort: peek at the first 4 bytes for the Parquet magic ``PAR1``.

    Parameters
    ----------
    path:
        File or directory path.

    Returns
    -------
    str
        One of ``"jsonl"``, ``"csv"``, ``"parquet"``, ``"txt-folder"``.

    Raises
    ------
    ValueError
        When the format cannot be determined.

    """
    p = Path(path)

    # 1 - directory
    if p.is_dir():
        return FORMAT_TXT_FOLDER

    # 2 - python-magic MIME sniffing
    if _MAGIC_AVAILABLE:
        try:
            mime = _magic.from_file(str(p), mime=True)
            # "text/plain" is ambiguous - refine with extension
            if mime == "text/plain":
                ext = p.suffix.lower()
                if ext == ".jsonl":
                    return FORMAT_JSONL
                if ext == ".csv":
                    return FORMAT_CSV
                # Unknown extension + text/plain: fall through; don't assume JSONL
            else:
                fmt = _MIME_TO_FORMAT.get(mime)
                if fmt:
                    return fmt
        except (OSError, ValueError) as exc:
            logger.debug("python-magic MIME sniff failed, falling back to extension: %s", exc)

    # 3 - extension fallback
    ext = p.suffix.lower()
    if ext in _EXT_TO_FORMAT:
        return _EXT_TO_FORMAT[ext]

    # 4 - last-resort: peek at file bytes to detect parquet magic bytes PAR1
    try:
        with open(p, "rb") as fh:
            header = fh.read(4)
        if header == b"PAR1":
            return FORMAT_PARQUET
    except OSError:
        pass

    msg = (
        f"Cannot determine format for '{path}'. "
        "Provide a file with a recognised extension (.jsonl, .csv, .parquet) "
        "or a directory of .txt files."
    )
    raise ValueError(
        msg,
    )


# ---------------------------------------------------------------------------
# Per-format readers
# ---------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one record per line from a JSONL file.

    Blank lines are skipped; malformed lines raise ``json.JSONDecodeError``.
    """
    p = Path(path)
    with p.open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                yield json.loads(raw)


def read_csv(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one dict-per-row from a CSV file (header row required)."""
    p = Path(path)
    with p.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield dict(row)


def read_parquet(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield one dict-per-row from a Parquet file using pyarrow."""
    import pyarrow.parquet as pq  # lazy import - not required for other formats

    table = pq.read_table(str(path))
    for batch in table.to_batches():
        batch_dict = batch.to_pydict()
        keys = list(batch_dict.keys())
        n_rows = len(batch_dict[keys[0]]) if keys else 0
        for i in range(n_rows):
            yield {k: batch_dict[k][i] for k in keys}


def read_txt_folder(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield ``{"text": <file_content>}`` for every ``.txt`` file in *path*.

    Files are processed in sorted order for reproducibility.
    Non-``.txt`` files are silently skipped.
    """
    p = Path(path)
    if not p.is_dir():
        msg = f"Expected a directory, got: {path}"
        raise ValueError(msg)

    txt_files = sorted(p.glob("*.txt"))
    for txt_file in txt_files:
        content = txt_file.read_text(encoding="utf-8")
        yield {"text": content, "source_file": txt_file.name}


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------

_FORMAT_READERS = {
    FORMAT_JSONL: read_jsonl,
    FORMAT_CSV: read_csv,
    FORMAT_PARQUET: read_parquet,
    FORMAT_TXT_FOLDER: read_txt_folder,
}


def load(path: str | Path, fmt: str | None = None) -> Iterator[dict[str, Any]]:
    """Load *path* and return a normalised record stream.

    Parameters
    ----------
    path:
        File or directory to load.
    fmt:
        Optional explicit format string (``"jsonl"``, ``"csv"``,
        ``"parquet"``, ``"txt-folder"``).  When omitted, the format is
        auto-detected via :func:`detect_format`.

    Yields
    ------
    Dict[str, Any]
        One record per iteration.

    Raises
    ------
    ValueError
        On unknown path or format.

    """
    if fmt is None:
        fmt = detect_format(path)

    reader = _FORMAT_READERS.get(fmt)
    if reader is None:
        msg = f"Unsupported format '{fmt}'. Choose from: {list(_FORMAT_READERS)}"
        raise ValueError(msg)

    yield from reader(path)


__all__ = [
    "FORMAT_CSV",
    "FORMAT_JSONL",
    "FORMAT_PARQUET",
    "FORMAT_TXT_FOLDER",
    "detect_format",
    "load",
    "read_csv",
    "read_jsonl",
    "read_parquet",
    "read_txt_folder",
]
