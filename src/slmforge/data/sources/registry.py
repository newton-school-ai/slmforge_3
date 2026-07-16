from typing import Type
from .base import Source
from .synthetic import SyntheticSource
from .public import PublicHFSource
from .local import LocalSource
from .internal import InternalSource

_REGISTRY = {
    "synthetic": SyntheticSource,
    "public": PublicHFSource,
    "local": LocalSource,
    "internal": InternalSource,
}


def get_source_adapter(source_type: str) -> Type[Source]:
    if source_type not in _REGISTRY:
        raise ValueError(f"Unknown source type: {source_type}")
    return _REGISTRY[source_type]
