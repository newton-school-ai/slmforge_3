"""src/slmforge/data/builder.py.
=============================
Dataset builder with deterministic seeded splits.

Merges records from one or more ``Source`` adapters into a HuggingFace
``datasets.DatasetDict`` with train / val / eval splits (80 / 10 / 10).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import datasets

if TYPE_CHECKING:
    from slmforge.data.sources.base import Source

# Default random seed for reproducible splits.
DEFAULT_SEED: int = 42

# Split ratios
_TRAIN_RATIO = 0.80  # 80 %
_VAL_RATIO = 0.10  # 10 %
_EVAL_RATIO = 0.10  # 10 %


class DatasetBuilder:
    """Build a ``DatasetDict`` with deterministic train/val/eval splits.

    Usage
    -----
    >>> dd = DatasetBuilder.build(sources, seed=42)
    >>> dd.keys()  # dict_keys(['train', 'val', 'eval'])
    """

    @staticmethod
    def build(
        sources: list[Source],
        seed: int = DEFAULT_SEED,
    ) -> datasets.DatasetDict:
        """Merge *sources* and split into train / val / eval.

        Parameters
        ----------
        sources:
            One or more ``Source`` instances whose ``iter_records()`` will be
            consumed.
        seed:
            Random seed for the split.  Using the same seed guarantees
            identical splits across runs (reproducibility).

        Returns
        -------
        datasets.DatasetDict
            Keys ``"train"`` (~80 %), ``"val"`` (~10 %), ``"eval"`` (~10 %).

        Raises
        ------
        ValueError
            When *sources* is empty or yields zero records.

        """
        if not sources:
            msg = "At least one Source must be provided."
            raise ValueError(msg)

        # ------------------------------------------------------------------
        # 1. Collect all records from every source
        # ------------------------------------------------------------------
        all_records: list[dict] = []
        for src in sources:
            all_records.extend(src.iter_records())

        if not all_records:
            msg = "Sources yielded zero records.  Cannot build dataset from empty data."
            raise ValueError(msg)

        # ------------------------------------------------------------------
        # 2. Create a HuggingFace Dataset from the flat list of dicts
        # ------------------------------------------------------------------
        full_dataset = datasets.Dataset.from_list(all_records)

        # ------------------------------------------------------------------
        # 3. Two-stage split:  full -> train + _rest  ->  _rest -> val + eval
        #
        # Stage 1:  80 % train, 20 % remainder
        # Stage 2:  remainder 50/50 -> val (10 %) + eval (10 %)
        # ------------------------------------------------------------------
        rest_fraction = _VAL_RATIO + _EVAL_RATIO  # 0.20

        split_1 = full_dataset.train_test_split(
            test_size=rest_fraction,
            seed=seed,
        )

        train_ds = split_1["train"]
        rest_ds = split_1["test"]

        split_2 = rest_ds.train_test_split(
            test_size=0.5,
            seed=seed,
        )

        val_ds = split_2["train"]
        eval_ds = split_2["test"]

        return datasets.DatasetDict(
            {
                "train": train_ds,
                "val": val_ds,
                "eval": eval_ds,
            },
        )


__all__ = ["DEFAULT_SEED", "DatasetBuilder"]
