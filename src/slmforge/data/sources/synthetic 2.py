from collections.abc import Iterator
from typing import Any

from .base import Source


class SyntheticSource(Source):
    def iter_records(self) -> Iterator[dict[str, Any]]:
        yield {"text": "synthetic data sample"}

    def metadata(self) -> dict[str, Any]:
        return {"type": "synthetic"}
