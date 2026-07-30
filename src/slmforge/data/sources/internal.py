from collections.abc import Iterator
from typing import Any

from .base import Source


class InternalSource(Source):
    def iter_records(self) -> Iterator[dict[str, Any]]:
        msg = "Internal sources are not yet supported"
        raise NotImplementedError(msg)

    def metadata(self) -> dict[str, Any]:
        msg = "Internal sources are not yet supported"
        raise NotImplementedError(msg)
