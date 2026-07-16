from typing import Iterator, Dict, Any
from .base import Source


class PublicHFSource(Source):
    def iter_records(self) -> Iterator[Dict[str, Any]]:
        yield {"text": "public hf data sample"}

    def metadata(self) -> Dict[str, Any]:
        return {"type": "public"}
