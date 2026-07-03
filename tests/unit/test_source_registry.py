import pytest
from slmforge.data.sources import (
    get_source_adapter,
    SyntheticSource,
    PublicHFSource,
    LocalSource,
    InternalSource
)

def test_registry_valid_types():
    assert get_source_adapter("synthetic") == SyntheticSource
    assert get_source_adapter("public") == PublicHFSource
    assert get_source_adapter("local") == LocalSource
    assert get_source_adapter("internal") == InternalSource

def test_registry_rejects_unknown_types():
    with pytest.raises(ValueError, match="Unknown source type: unknown"):
        get_source_adapter("unknown")

def test_internal_raises_not_implemented():
    source = InternalSource()
    with pytest.raises(NotImplementedError, match="Internal sources are not yet supported"):
        source.metadata()
    
    with pytest.raises(NotImplementedError, match="Internal sources are not yet supported"):
        # We need to call next() to start the generator and trigger the error
        next(source.iter_records())

def test_adapters_yield_same_shape():
    sources = [
        SyntheticSource(),
        PublicHFSource(),
        LocalSource()
    ]
    
    for source in sources:
        records = list(source.iter_records())
        assert len(records) > 0
        for record in records:
            assert isinstance(record, dict)
            assert "text" in record
