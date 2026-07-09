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
from slmforge.data.builder import DatasetBuilder
from slmforge.data.card import generate_card

__all__ = [
    "FORMAT_JSONL",
    "FORMAT_CSV",
    "FORMAT_PARQUET",
    "FORMAT_TXT_FOLDER",
    "detect_format",
    "load",
    "preview",
    "preview_info",
    "DatasetBuilder",
    "generate_card",
]
