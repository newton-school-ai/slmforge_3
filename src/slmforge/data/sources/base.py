from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


class Source(ABC):
    @abstractmethod
    def iter_records(self) -> Iterator[dict[str, Any]]:
        pass

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        pass
