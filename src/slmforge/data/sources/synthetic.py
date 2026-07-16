from typing import Iterator, Dict, Any
from .base import Source


class SyntheticSource(Source):
    def iter_records(self) -> Iterator[Dict[str, Any]]:
        yield {"text": "synthetic data sample"}

    def metadata(self) -> Dict[str, Any]:
        return {"type": "synthetic"}
