"""slmforge.data - data loading, ingestion, and preview utilities."""

from slmforge.data.ingest import (
    FORMAT_CSV,
    FORMAT_JSONL,
    FORMAT_PARQUET,
    FORMAT_TXT_FOLDER,
    detect_format,
    load,
)
from slmforge.data.preview import preview, preview_info

__all__ = [
    "FORMAT_JSONL",
    "FORMAT_CSV",
    "FORMAT_PARQUET",
    "FORMAT_TXT_FOLDER",
    "detect_format",
    "load",
    "preview",
    "preview_info",
]
