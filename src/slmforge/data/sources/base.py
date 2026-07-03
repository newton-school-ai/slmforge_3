from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any


class BaseSource(ABC):
    """Abstract base for all data sources."""

    @abstractmethod
    def iter_records(self) -> Iterator[Dict[str, Any]]:
        """Yield normalised records: {"text": ..., "meta": ...}"""
        ...

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Return source metadata: type, id/path, size, licence."""
        ...