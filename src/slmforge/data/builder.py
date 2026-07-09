import random
from typing import List, Optional
import datasets
from slmforge.data.sources.base import Source

class DatasetBuilder:
    @staticmethod
    def build(sources: List[Source], seed: int = 42) -> datasets.DatasetDict:
        """Aggregate records from all sources and split them into train/val/eval splits."""
        if not sources:
            raise ValueError("No sources provided.")

        records = []
        for src in sources:
            try:
                records.extend(list(src.iter_records()))
            except NotImplementedError:
                # If a source is not implemented/supported, we can skip it or let it raise.
                # Since InternalSource raises NotImplementedError, let's let it raise or handle it.
                # Actually, the unit tests check that InternalSource raises NotImplementedError.
                # If the builder is called with InternalSource, raising NotImplementedError is appropriate.
                raise

        if not records:
            raise ValueError("No records found in any source.")

        if len(records) < 3:
            raise ValueError("At least 3 records are required to split into train, val, and eval sets.")

        # Create a Hugging Face Dataset from the aggregated list
        dataset = datasets.Dataset.from_list(records)

        # Generate deterministic splits using python's random generator with the seed
        indices = list(range(len(dataset)))
        rng = random.Random(seed)
        rng.shuffle(indices)

        # Calculate split sizes for 80/10/10 split
        n = len(dataset)
        n_val = max(1, int(round(n * 0.1)))
        n_eval = max(1, int(round(n * 0.1)))
        n_train = n - n_val - n_eval

        # Re-adjust in case n is small to ensure at least 1 record per split
        if n_train <= 0:
            n_train = 1
            n_val = 1
            n_eval = n - 2

        train_indices = indices[:n_train]
        val_indices = indices[n_train : n_train + n_val]
        eval_indices = indices[n_train + n_val :]

        train_ds = dataset.select(train_indices)
        val_ds = dataset.select(val_indices)
        eval_ds = dataset.select(eval_indices)

        return datasets.DatasetDict({
            "train": train_ds,
            "val": val_ds,
            "eval": eval_ds,
        })
