from typing import Iterator, Dict, Any

from .base import BaseSource


class InternalSource(BaseSource):
    """Stub - not yet implemented."""

    def iter_records(self) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError(
            "InternalSource is not yet available. "
            "This adapter will be implemented when NST "
            "internal data pipelines are ready."
        )

    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError(
            "InternalSource metadata not available."
        )
    