import pytest
from typing import Iterator, Dict, Any
from slmforge.data.sources.base import Source
from slmforge.data.sources import LocalSource, PublicHFSource, SyntheticSource
from slmforge.data import DatasetBuilder, generate_card
import datasets

class MockSource(Source):
    def __init__(self, name: str, records: list):
        self.name = name
        self.records = records

    def iter_records(self) -> Iterator[Dict[str, Any]]:
        yield from self.records

    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "mock",
            "id": self.name,
            "size": len(self.records),
            "licence": "MIT"
        }

def test_builder_basic_split():
    # Create 100 unique records
    records = [{"id": i, "text": f"text_{i}"} for i in range(100)]
    source = MockSource("test_source", records)

    # Build dataset dict
    dataset_dict = DatasetBuilder.build([source], seed=42)

    assert isinstance(dataset_dict, datasets.DatasetDict)
    assert "train" in dataset_dict
    assert "val" in dataset_dict
    assert "eval" in dataset_dict

    # Check 80/10/10 split sizes
    assert len(dataset_dict["train"]) == 80
    assert len(dataset_dict["val"]) == 10
    assert len(dataset_dict["eval"]) == 10

def test_builder_determinism():
    records = [{"id": i, "text": f"text_{i}"} for i in range(100)]
    source = MockSource("test_source", records)

    # Two runs with the same seed
    ds_dict_1 = DatasetBuilder.build([source], seed=42)
    ds_dict_2 = DatasetBuilder.build([source], seed=42)

    assert ds_dict_1["train"]["text"] == ds_dict_2["train"]["text"]
    assert ds_dict_1["val"]["text"] == ds_dict_2["val"]["text"]
    assert ds_dict_1["eval"]["text"] == ds_dict_2["eval"]["text"]

    # One run with a different seed
    ds_dict_3 = DatasetBuilder.build([source], seed=100)

    # Should be different
    assert ds_dict_1["train"]["text"] != ds_dict_3["train"]["text"]

def test_builder_no_leak():
    records = [{"id": i, "text": f"text_{i}"} for i in range(100)]
    source = MockSource("test_source", records)

    dataset_dict = DatasetBuilder.build([source], seed=42)

    train_texts = set(dataset_dict["train"]["text"])
    val_texts = set(dataset_dict["val"]["text"])
    eval_texts = set(dataset_dict["eval"]["text"])

    # Disjoint check
    assert not train_texts.intersection(val_texts)
    assert not train_texts.intersection(eval_texts)
    assert not val_texts.intersection(eval_texts)

def test_builder_empty_or_too_small():
    source_empty = MockSource("empty", [])
    with pytest.raises(ValueError, match="No records found in any source."):
        DatasetBuilder.build([source_empty])

    records_too_small = [{"id": 1, "text": "only 1"}]
    source_small = MockSource("small", records_too_small)
    with pytest.raises(ValueError, match="At least 3 records are required"):
        DatasetBuilder.build([source_small])

def test_card_generation():
    sources = [
        PublicHFSource(dataset_id="cnn_dailymail", size="100k", licence="Apache-2.0"),
        SyntheticSource(generator_id="feedback_summariser", size="20k", licence="Pod-authored"),
        LocalSource(path="/data/custom_dataset", size="10k", licence="Proprietary")
    ]

    card_md = generate_card(sources, build_id="build_12345")

    # Check build ID title
    assert "# Dataset Card -- build_12345" in card_md

    # Check table headers
    assert "| Type | Identifier | Size | Licence |" in card_md
    assert "|------|-----------|------|---------|" in card_md

    # Check each source row
    assert "| public | cnn_dailymail | 100k | Apache-2.0 |" in card_md
    assert "| synthetic | feedback_summariser | 20k | Pod-authored |" in card_md
    assert "| local | /data/custom_dataset | 10k | Proprietary |" in card_md

    # Check splits section
    assert "## Splits" in card_md
    assert "- Train: 80%" in card_md
    assert "- Val: 10%" in card_md
    assert "- Held-out eval: 10% (seeded, frozen)" in card_md
