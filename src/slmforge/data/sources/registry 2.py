from .base import Source
from .internal import InternalSource
from .local import LocalSource
from .public import PublicHFSource
from .synthetic import SyntheticSource

_REGISTRY = {
    "synthetic": SyntheticSource,
    "public": PublicHFSource,
    "local": LocalSource,
    "internal": InternalSource,
}


def get_source_adapter(source_type: str) -> type[Source]:
    if source_type not in _REGISTRY:
        msg = f"Unknown source type: {source_type}"
        raise ValueError(msg)
    return _REGISTRY[source_type]
