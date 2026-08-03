from slmforge.engine.state import (
    Build,
    Dataset,
    Eval,
    Run,
    Serve,
    Source,
)


def test_build_model() -> None:
    obj = Build(name="test_build")
    assert obj.name == "test_build"


def test_run_model() -> None:
    obj = Run(name="test_run")
    assert obj.name == "test_run"


def test_source_model() -> None:
    obj = Source(name="test_source")
    assert obj.name == "test_source"


def test_dataset_model() -> None:
    obj = Dataset(name="test_dataset")
    assert obj.name == "test_dataset"


def test_eval_model() -> None:
    obj = Eval(name="test_eval")
    assert obj.name == "test_eval"


def test_serve_model() -> None:
    obj = Serve(name="test_serve")
    assert obj.name == "test_serve"
