from slmforge.engine.state import (
    Build,
    Run,
    Source,
    Dataset,
    Eval,
    Serve,
)


def test_build_model():
    obj = Build(name="test_build")
    assert obj.name == "test_build"


def test_run_model():
    obj = Run(name="test_run")
    assert obj.name == "test_run"


def test_source_model():
    obj = Source(name="test_source")
    assert obj.name == "test_source"


def test_dataset_model():
    obj = Dataset(name="test_dataset")
    assert obj.name == "test_dataset"


def test_eval_model():
    obj = Eval(name="test_eval")
    assert obj.name == "test_eval"


def test_serve_model():
    obj = Serve(name="test_serve")
    assert obj.name == "test_serve"
