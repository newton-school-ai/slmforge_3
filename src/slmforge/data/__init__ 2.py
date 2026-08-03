"""slmforge.data - data loading, ingestion, preview, building, and card utilities."""

from slmforge.data.builder import DEFAULT_SEED, DatasetBuilder
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
    "DEFAULT_SEED",
    "FORMAT_CSV",
    "FORMAT_JSONL",
    "FORMAT_PARQUET",
    "FORMAT_TXT_FOLDER",
    "DatasetBuilder",
    "detect_format",
    "generate_card",
    "load",
    "preview",
    "preview_info",
]
