from pathlib import Path
from typing import Iterator, Dict, Any

from .base import BaseSource


class LocalSource(BaseSource):
    def __init__(self, path: str, text_field: str = "text"):
        self._path = Path(path)
        self._text_field = text_field

    def iter_records(self) -> Iterator[Dict[str, Any]]:
        from slmforge.data.ingest import load_file

        for record in load_file(self._path):
            yield {
                "text": record[self._text_field],
                "meta": {"path": str(self._path)},
            }

    def metadata(self) -> Dict[str, Any]:
        return {
            "type": "local",
            "path": str(self._path),
            "size": self._path.stat().st_size,
            "licence": "local",
        }