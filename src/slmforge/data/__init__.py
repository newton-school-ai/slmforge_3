"""slmforge.data - data loading, ingestion, preview, building, and card utilities."""

from slmforge.data.builder import DatasetBuilder, DEFAULT_SEED
from slmforge.data.card import generate_card
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
    "DatasetBuilder",
    "DEFAULT_SEED",
    "generate_card",
    "FORMAT_JSONL",
    "FORMAT_CSV",
    "FORMAT_PARQUET",
    "FORMAT_TXT_FOLDER",
    "detect_format",
    "load",
    "preview",
    "preview_info",
]
