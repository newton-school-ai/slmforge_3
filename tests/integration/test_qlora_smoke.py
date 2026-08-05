"""tests/integration/test_qlora_smoke.py.
=========================================
Integration smoke tests for QLoRA (4-bit NF4) training harness.
"""

from __future__ import annotations

from pathlib import Path

import datasets
import pytest
import torch

from slmforge.finetune.qlora import get_qlora_config, is_8b_class, train_qlora


@pytest.fixture
def tiny_dataset() -> datasets.DatasetDict:
    """Create a 50-record synthetic dataset for smoke testing."""
    train_records = [
        {"text": f"Instruction: Solve problem #{i}.\nOutput: Solution #{i} is valid."}
        for i in range(40)
    ]
    eval_records = [
        {"text": f"Instruction: Solve problem #{i}.\nOutput: Solution #{i} is valid."}
        for i in range(40, 50)
    ]
    return datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_list(train_records),
            "eval": datasets.Dataset.from_list(eval_records),
        }
    )


@pytest.mark.gpu
def test_qlora_auto_selected_for_8b_class() -> None:
    """Test that QLoRA auto-selection logic returns True for registered 8B-class models."""
    assert is_8b_class("llama-3.1-8b-instruct")
    assert is_8b_class("qwen-2.5-7b-instruct")
    assert is_8b_class("deepseek-v3-distill")
    assert is_8b_class("meta-llama/Llama-3.1-8B-Instruct")
    assert is_8b_class("Qwen/Qwen2.5-7B-Instruct")

    # Small models should return False for QLoRA auto-selection
    assert not is_8b_class("phi-3-mini")
    assert not is_8b_class("sshleifer/tiny-gpt2")


@pytest.mark.gpu
def test_get_qlora_config_parameters() -> None:
    """Test building standard 4-bit NF4 BitsAndBytesConfig."""
    config = get_qlora_config()
    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_use_double_quant is True


@pytest.mark.gpu
def test_qlora_training_smoke_and_memory_limit(
    tiny_dataset: datasets.DatasetDict, tmp_path: Path
) -> None:
    """Test QLoRA training harness end-to-end and verify memory footprint < 16GB."""
    base_model = "sshleifer/tiny-gpt2"
    out_dir = tmp_path / "qlora_adapter"

    lora_cfg = {"r": 4, "lora_alpha": 8, "target_modules": ["c_attn"]}
    training_cfg = {
        "output_dir": str(out_dir),
        "num_train_epochs": 1.0,
        "per_device_train_batch_size": 4,
        "seed": 42,
    }

    adapter_path = train_qlora(
        dataset=tiny_dataset,
        base=base_model,
        lora_cfg=lora_cfg,
        training_cfg=training_cfg,
    )

    assert Path(adapter_path).exists()
    assert (Path(adapter_path) / "adapter_config.json").exists()

    # Memory check (< 16 GB)
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024**3)
        assert max_mem_gb < 16.0, f"Memory footprint exceeded 16GB: {max_mem_gb:.2f}GB"
