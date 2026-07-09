from typing import Iterator, Dict, Any
from .base import Source

class LocalSource(Source):
    def __init__(self, path: str = "default_path", size: str = "unknown", licence: str = "unknown"):
        self.path = path
        self.size = size
        self.licence = licence

    def iter_records(self) -> Iterator[Dict[str, Any]]:
        yield {"text": "local data sample"}

    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "local",
            "path": self.path,
            "size": self.size,
            "licence": self.licence,
        }
