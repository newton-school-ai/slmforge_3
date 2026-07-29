from __future__ import annotations

import logging
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")

DATASET_ALIASES = {
    "samsum": "knkarthick/samsum",
}


def prefetch(dataset_id: str) -> Path:
    """Download and cache a public dataset locally."""

    dataset_path = CACHE_DIR / dataset_id
    card_path = dataset_path / "dataset_card.md"

    # Subsequent prefetches are no-ops only if cache is complete
    if dataset_path.exists() and any(dataset_path.iterdir()):
        return dataset_path

    dataset_path.mkdir(parents=True, exist_ok=True)

    hf_dataset_id = DATASET_ALIASES.get(dataset_id, dataset_id)

    dataset = load_dataset(hf_dataset_id)

    # Save DatasetDict / Dataset locally
    dataset.save_to_disk(str(dataset_path))

    # Extract license information from loaded dataset metadata
    license_info = "Unknown"

    try:
        license_info = getattr(dataset.info, "license", "Unknown")
    except AttributeError:
        logger.debug("dataset.info has no 'license' attribute; defaulting to 'Unknown'")

    # Write dataset attribution card
    card_path.write_text(
        "# Dataset Card\n\n"
        f"Dataset: {dataset_id}\n\n"
        f"HuggingFace ID: {hf_dataset_id}\n\n"
        f"License: {license_info}\n",
        encoding="utf-8",
    )

    return dataset_path
