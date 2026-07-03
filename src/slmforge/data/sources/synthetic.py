from typing import Iterator, Dict, Any, Callable

from .base import BaseSource


class SyntheticSource(BaseSource):
    def __init__(
        self,
        generator_fn: Callable,
        count: int = 1000,
        name: str = "synthetic",
    ):
        self._gen = generator_fn
        self._count = count
        self._name = name

    def iter_records(self) -> Iterator[Dict[str, Any]]:
        for i in range(self._count):
            yield self._gen(i)

    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "synthetic",
            "name": self._name,
            "size": self._count,
            "licence": "generated",
        }