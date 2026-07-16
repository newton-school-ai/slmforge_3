from typing import Iterator, Dict, Any
from .base import Source


class InternalSource(Source):
    def iter_records(self) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError("Internal sources are not yet supported")

    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError("Internal sources are not yet supported")
