"""
tests/unit/test_dataset_builder.py
===================================
Unit tests for DatasetBuilder (seeded 80/10/10 splits) and dataset card
generation.

Run with:
    pytest tests/unit/test_dataset_builder.py -v
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List

import datasets
import pytest

from slmforge.data.builder import DEFAULT_SEED, DatasetBuilder
from slmforge.data.card import generate_card
from slmforge.data.sources.base import Source


# ---------------------------------------------------------------------------
# Helpers -- stub sources for testing
# ---------------------------------------------------------------------------


class StubSource(Source):
    """A minimal Source that yields *n* records with predictable content."""

    def __init__(
        self,
        n: int = 100,
        *,
        source_type: str = "stub",
        source_id: str = "stub-001",
        licence: str = "MIT",
    ) -> None:
        self._n = n
        self._type = source_type
        self._id = source_id
        self._licence = licence

    def iter_records(self) -> Iterator[Dict[str, Any]]:
        for i in range(self._n):
            yield {"text": f"{self._id}-record-{i}", "idx": i}

    def metadata(self) -> Dict[str, Any]:
        return {
            "type": self._type,
            "id": self._id,
            "size": self._n,
            "licence": self._licence,
        }


class EmptySource(Source):
    """A source that yields zero records."""

    def iter_records(self) -> Iterator[Dict[str, Any]]:
        return iter([])

    def metadata(self) -> Dict[str, Any]:
        return {"type": "empty"}


# ---------------------------------------------------------------------------
# DatasetBuilder.build() tests
# ---------------------------------------------------------------------------


class TestDatasetBuilderBuild:
    """Tests for DatasetBuilder.build()."""

    def test_build_returns_dataset_dict(self) -> None:
        """Return type is DatasetDict with keys train, val, eval."""
        dd = DatasetBuilder.build([StubSource(100)])
        assert isinstance(dd, datasets.DatasetDict)
        assert set(dd.keys()) == {"train", "val", "eval"}

    def test_default_seed_is_42(self) -> None:
        """DEFAULT_SEED should be 42 as specified in the issue."""
        assert DEFAULT_SEED == 42

    def test_same_seed_same_splits(self) -> None:
        """Two calls with the same seed must produce identical splits."""
        sources: List[Source] = [StubSource(200)]
        dd1 = DatasetBuilder.build(sources, seed=42)
        dd2 = DatasetBuilder.build(sources, seed=42)

        for split in ("train", "val", "eval"):
            assert list(dd1[split]["text"]) == list(dd2[split]["text"])

    def test_different_seed_different_splits(self) -> None:
        """Two calls with different seeds should (almost certainly) differ."""
        sources: List[Source] = [StubSource(200)]
        dd1 = DatasetBuilder.build(sources, seed=42)
        dd2 = DatasetBuilder.build(sources, seed=99)

        # At least one split should differ
        any_diff = False
        for split in ("train", "val", "eval"):
            if list(dd1[split]["text"]) != list(dd2[split]["text"]):
                any_diff = True
                break
        assert any_diff, "Different seeds produced identical splits."

    def test_split_ratios(self) -> None:
        """Train ~80 %, val ~10 %, eval ~10 % of total records."""
        n = 1000
        dd = DatasetBuilder.build([StubSource(n)])
        total = sum(len(dd[s]) for s in dd)
        assert total == n

        train_pct = len(dd["train"]) / total
        val_pct = len(dd["val"]) / total
        eval_pct = len(dd["eval"]) / total

        # Allow +/-2 % tolerance for rounding
        assert 0.78 <= train_pct <= 0.82, f"train ratio {train_pct:.2%}"
        assert 0.08 <= val_pct <= 0.12, f"val ratio {val_pct:.2%}"
        assert 0.08 <= eval_pct <= 0.12, f"eval ratio {eval_pct:.2%}"

    def test_no_eval_leak(self) -> None:
        """No record should appear in more than one split (no eval leak)."""
        dd = DatasetBuilder.build([StubSource(500)])

        train_texts = set(dd["train"]["text"])
        val_texts = set(dd["val"]["text"])
        eval_texts = set(dd["eval"]["text"])

        assert train_texts.isdisjoint(val_texts), "Train and val overlap!"
        assert train_texts.isdisjoint(eval_texts), "Train and eval overlap!"
        assert val_texts.isdisjoint(eval_texts), "Val and eval overlap!"

        # Also verify union equals original set
        union = train_texts | val_texts | eval_texts
        assert len(union) == 500

    def test_no_eval_leak_multiple_sources(self) -> None:
        """No eval leak when merging multiple sources."""
        sources: List[Source] = [
            StubSource(200, source_type="a", source_id="a-1"),
            StubSource(300, source_type="b", source_id="b-1"),
        ]
        dd = DatasetBuilder.build(sources)
        total = sum(len(dd[s]) for s in dd)
        assert total == 500

        train_texts = set(dd["train"]["text"])
        val_texts = set(dd["val"]["text"])
        eval_texts = set(dd["eval"]["text"])

        assert train_texts.isdisjoint(val_texts)
        assert train_texts.isdisjoint(eval_texts)
        assert val_texts.isdisjoint(eval_texts)

    def test_empty_sources_raises(self) -> None:
        """Passing an empty sources list raises ValueError."""
        with pytest.raises(ValueError, match="At least one Source"):
            DatasetBuilder.build([])

    def test_sources_with_zero_records_raises(self) -> None:
        """Sources that yield zero records should raise ValueError."""
        with pytest.raises(ValueError, match="zero records"):
            DatasetBuilder.build([EmptySource()])

    def test_all_records_preserved(self) -> None:
        """Every record from every source appears exactly once across splits."""
        n = 300
        dd = DatasetBuilder.build([StubSource(n)])

        all_texts: list[str] = []
        for split in ("train", "val", "eval"):
            all_texts.extend(dd[split]["text"])

        assert sorted(all_texts) == sorted(f"stub-001-record-{i}" for i in range(n))


# ---------------------------------------------------------------------------
# Dataset card generation tests
# ---------------------------------------------------------------------------


class TestGenerateCard:
    """Tests for the generate_card() function."""

    @pytest.fixture()
    def card_fixture(self) -> tuple[str, list[Source]]:
        """Build a dataset and generate the card for reuse."""
        sources: List[Source] = [
            StubSource(80, source_type="public", source_id="squad", licence="CC-BY-4.0"),
            StubSource(20, source_type="synthetic", source_id="synth-v1", licence="Apache-2.0"),
        ]
        dd = DatasetBuilder.build(sources, seed=42)
        card = generate_card(sources, dd, seed=42)
        return card, sources

    def test_card_is_string(self, card_fixture: tuple[str, list[Source]]) -> None:
        card, _ = card_fixture
        assert isinstance(card, str)

    def test_card_contains_seed(self, card_fixture: tuple[str, list[Source]]) -> None:
        card, _ = card_fixture
        assert "42" in card

    def test_card_lists_sources(self, card_fixture: tuple[str, list[Source]]) -> None:
        """Card markdown contains each source's type, id, size, licence."""
        card, sources = card_fixture
        for src in sources:
            meta = src.metadata()
            assert meta["type"] in card
            assert str(meta["id"]) in card
            assert str(meta["size"]) in card
            assert meta["licence"] in card

    def test_card_contains_split_table(self, card_fixture: tuple[str, list[Source]]) -> None:
        card, _ = card_fixture
        assert "train" in card
        assert "val" in card
        assert "eval" in card
        # Table header markers
        assert "Split" in card
        assert "Records" in card

    def test_card_contains_total(self, card_fixture: tuple[str, list[Source]]) -> None:
        card, _ = card_fixture
        assert "100" in card  # total = 80 + 20

    def test_card_contains_reproducibility_note(
        self, card_fixture: tuple[str, list[Source]]
    ) -> None:
        card, _ = card_fixture
        assert "deterministic" in card.lower() or "reproducib" in card.lower()
