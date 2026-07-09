from typing import Iterator, Dict, Any
from .base import Source

class SyntheticSource(Source):
    def __init__(self, generator_id: str = "default_synthetic", size: str = "unknown", licence: str = "unknown"):
        self.generator_id = generator_id
        self.size = size
        self.licence = licence

    def iter_records(self) -> Iterator[Dict[str, Any]]:
        yield {"text": "synthetic data sample"}

    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "synthetic",
            "id": self.generator_id,
            "size": self.size,
            "licence": self.licence,
        }
