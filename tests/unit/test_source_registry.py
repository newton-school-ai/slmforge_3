import pytest

from slmforge.data.sources.registry import (
    get_source,
    list_source_types,
)
from slmforge.data.sources.synthetic import SyntheticSource
from slmforge.data.sources.local import LocalSource
from slmforge.data.sources.public import PublicHFSource
from slmforge.data.sources.internal import InternalSource


def dummy_generator(i):
    return {
        "text": f"sample {i}",
        "meta": {"id": i},
    }


def test_get_synthetic_source():
    source = get_source(
        "synthetic",
        generator_fn=dummy_generator,
    )
    assert isinstance(source, SyntheticSource)


def test_get_local_source():
    source = get_source(
        "local",
        path="dummy.txt",
    )
    assert isinstance(source, LocalSource)


def test_get_public_source():
    source = get_source(
        "public_hf",
        dataset_id="dummy",
    )
    assert isinstance(source, PublicHFSource)


def test_get_internal_source():
    source = get_source("internal")
    assert isinstance(source, InternalSource)


def test_unknown_source_type():
    with pytest.raises(ValueError):
        get_source("unknown")


def test_list_source_types():
    assert set(list_source_types()) == {
        "synthetic",
        "public_hf",
        "local",
        "internal",
    }
