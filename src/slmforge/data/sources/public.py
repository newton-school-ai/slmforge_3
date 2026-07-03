from typing import Iterator, Dict, Any

from .base import BaseSource


class PublicHFSource(BaseSource):
    def __init__(
        self,
        dataset_id: str,
        split: str = "train",
        text_field: str = "text",
    ):
        self._dataset_id = dataset_id
        self._split = split
        self._text_field = text_field
        self._ds = None  # lazy load

    def _load(self):
        if self._ds is None:
            from datasets import load_dataset
            self._ds = load_dataset(
                self._dataset_id,
                split=self._split,
            )

    def iter_records(self) -> Iterator[Dict[str, Any]]:
        self._load()
        for row in self._ds:
            yield {
                "text": row[self._text_field],
                "meta": {"source": self._dataset_id},
            }

    def metadata(self) -> Dict[str, Any]:
        self._load()
        return {
            "type": "public_hf",
            "id": self._dataset_id,
            "split": self._split,
            "size": len(self._ds),
            "licence": self._ds.info.license or "unknown",
        }