"""
src/slmforge/data/card.py
==========================
Auto-generate a Markdown dataset card for a built ``DatasetDict``.

The card documents every source, split sizes, and the seed used so that
anyone inspecting the dataset can reproduce it exactly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import datasets

from slmforge.data.sources.base import Source


def generate_card(
    sources: List[Source],
    dataset_dict: datasets.DatasetDict,
    seed: int,
) -> str:
    """Return a Markdown dataset card as a string.

    Parameters
    ----------
    sources:
        The ``Source`` instances that were used to build the dataset.
    dataset_dict:
        The resulting ``DatasetDict`` (must contain ``train``, ``val``, ``eval``).
    seed:
        The seed that was used for splitting.

    Returns
    -------
    str
        A complete Markdown document.
    """
    total = sum(len(dataset_dict[split]) for split in dataset_dict)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = []

    # ---- Header ----------------------------------------------------------
    lines.append("# Dataset Card")
    lines.append("")
    lines.append(f"**Generated:** {timestamp}")
    lines.append(f"**Seed:** {seed}")
    lines.append(f"**Total records:** {total}")
    lines.append("")

    # ---- Split summary ---------------------------------------------------
    lines.append("## Splits")
    lines.append("")
    lines.append("| Split | Records | Percentage |")
    lines.append("|-------|---------|------------|")

    for split_name in ("train", "val", "eval"):
        if split_name in dataset_dict:
            n = len(dataset_dict[split_name])
            pct = (n / total * 100) if total else 0.0
            lines.append(f"| {split_name} | {n} | {pct:.1f}% |")

    lines.append("")

    # ---- Source inventory -------------------------------------------------
    lines.append("## Sources")
    lines.append("")
    lines.append("| # | Type | ID / Path | Records | Licence |")
    lines.append("|---|------|-----------|---------|---------|")

    for idx, src in enumerate(sources, start=1):
        meta: Dict[str, Any] = src.metadata()
        src_type = meta.get("type", "unknown")
        src_id = meta.get("id", meta.get("path", "n/a"))
        src_size = meta.get("size", "n/a")
        src_licence = meta.get("licence", meta.get("license", "unspecified"))
        lines.append(f"| {idx} | {src_type} | {src_id} | {src_size} | {src_licence} |")

    lines.append("")

    # ---- Reproducibility notes -------------------------------------------
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(
        f"This dataset was split deterministically using seed **{seed}**.  "
        "Re-running `DatasetBuilder.build()` with the same sources and seed "
        "will produce identical train / val / eval splits."
    )
    lines.append("")

    return "\n".join(lines)


__all__ = ["generate_card"]
