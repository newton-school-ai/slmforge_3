from typing import Iterator, Dict, Any
from .base import Source

class LocalSource(Source):
    def iter_records(self) -> Iterator[Dict[str, Any]]:
        yield {"text": "local data sample"}

    def metadata(self) -> Dict[str, Any]:
        return {"type": "local"}
