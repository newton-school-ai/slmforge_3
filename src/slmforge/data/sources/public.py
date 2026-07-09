from typing import Iterator, Dict, Any
from .base import Source

class PublicHFSource(Source):
    def __init__(self, dataset_id: str = "default_id", size: str = "unknown", licence: str = "unknown"):
        self.dataset_id = dataset_id
        self.size = size
        self.licence = licence

    def iter_records(self) -> Iterator[Dict[str, Any]]:
        yield {"text": "public hf data sample"}

    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "public",
            "id": self.dataset_id,
            "size": self.size,
            "licence": self.licence,
        }
