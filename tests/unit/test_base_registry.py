import pytest

from slmforge.finetune.registry import (
    get_base,
    list_bases,
    validate_hf_id,
)


def test_registry_contains_four_models() -> None:
    models = list_bases()

    assert len(models) == 4
    assert "phi-3-mini" in models
    assert "llama-3.1-8b-instruct" in models
    assert "qwen-2.5-7b-instruct" in models
    assert "deepseek-v3-distill" in models


def test_get_base_contains_metadata() -> None:
    model = get_base("phi-3-mini")

    assert model["hf_id"]
    assert model["lora_recipe"]
    assert model["vram"]
    assert model["license"]
    assert len(model["recommended_tasks"]) > 0


def test_unknown_base_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown base model"):
        get_base("bert-base")


def test_validate_hf_id() -> None:
    assert validate_hf_id("microsoft/Phi-3-mini-4k-instruct")
    assert not validate_hf_id("random/model")
