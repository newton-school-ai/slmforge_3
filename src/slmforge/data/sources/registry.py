from typing import Dict, Type

from .base import BaseSource
from .synthetic import SyntheticSource
from .public import PublicHFSource
from .local import LocalSource
from .internal import InternalSource

_REGISTRY: Dict[str, Type[BaseSource]] = {
    "synthetic": SyntheticSource,
    "public_hf": PublicHFSource,
    "local": LocalSource,
    "internal": InternalSource,
}


def get_source(source_type: str, **kwargs) -> BaseSource:
    """Factory: get a source adapter by type name."""
    if source_type not in _REGISTRY:
        raise ValueError(
            f"Unknown source type: {source_type}. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[source_type](**kwargs)


def list_source_types() -> list:
    return list(_REGISTRY.keys())