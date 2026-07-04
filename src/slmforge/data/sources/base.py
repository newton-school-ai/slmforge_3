from abc import ABC, abstractmethod
from typing import Iterator, Dict, Any

class Source(ABC):
    @abstractmethod
    def iter_records(self) -> Iterator[Dict[str, Any]]:
        pass

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        pass
