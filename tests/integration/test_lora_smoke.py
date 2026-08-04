"""tests/integration/test_lora_smoke.py.
========================================
Integration smoke tests for LoRA training harness.
"""

from __future__ import annotations

import json
from pathlib import Path

import datasets
import pytest
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from slmforge.finetune.lora import train_lora


@pytest.fixture
def tiny_dataset() -> datasets.DatasetDict:
    """Create a 50-record synthetic dataset (40 train, 10 eval)."""
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
def test_train_tiny_adapter_end_to_end(tiny_dataset: datasets.DatasetDict, tmp_path: Path) -> None:
    """Test training a tiny adapter on a 50-record dataset end-to-end."""
    base_model = "sshleifer/tiny-gpt2"
    out_dir = tmp_path / "adapter_end_to_end"

    lora_cfg = {"r": 4, "lora_alpha": 8, "target_modules": ["c_attn"]}
    training_cfg = {
        "output_dir": str(out_dir),
        "num_train_epochs": 1.0,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "logging_steps": 1,
        "seed": 42,
    }

    adapter_path = train_lora(
        dataset=tiny_dataset,
        base=base_model,
        lora_cfg=lora_cfg,
        training_cfg=training_cfg,
    )

    assert Path(adapter_path).exists()
    assert (Path(adapter_path) / "adapter_config.json").exists()
    adapter_weights_exist = (Path(adapter_path) / "adapter_model.safetensors").exists() or (
        Path(adapter_path) / "adapter_model.bin"
    ).exists()
    assert adapter_weights_exist, "Adapter weights file not found!"


@pytest.mark.gpu
def test_adapter_loads_and_generates_token(
    tiny_dataset: datasets.DatasetDict, tmp_path: Path
) -> None:
    """Test that saved adapter loads onto base model and generates tokens."""
    base_model = "sshleifer/tiny-gpt2"
    out_dir = tmp_path / "adapter_gen"

    lora_cfg = {"r": 4, "lora_alpha": 8, "target_modules": ["c_attn"]}
    training_cfg = {
        "output_dir": str(out_dir),
        "num_train_epochs": 1.0,
        "per_device_train_batch_size": 4,
        "seed": 42,
    }

    adapter_path = train_lora(
        dataset=tiny_dataset,
        base=base_model,
        lora_cfg=lora_cfg,
        training_cfg=training_cfg,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    base = AutoModelForCausalLM.from_pretrained(base_model)
    model = PeftModel.from_pretrained(base, adapter_path)

    inputs = tokenizer("Instruction: Solve problem #1.\nOutput:", return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assert len(generated_text) > 0
    assert outputs.shape[1] > inputs["input_ids"].shape[1]


@pytest.mark.gpu
def test_reproducibility_same_seed(tiny_dataset: datasets.DatasetDict, tmp_path: Path) -> None:
    """Test that training twice with the same seed produces identical eval scores."""
    base_model = "sshleifer/tiny-gpt2"
    lora_cfg = {"r": 4, "lora_alpha": 8, "target_modules": ["c_attn"]}

    out_dir_1 = tmp_path / "run_1"
    training_cfg_1 = {
        "output_dir": str(out_dir_1),
        "num_train_epochs": 1.0,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "seed": 42,
    }

    adapter_path_1 = train_lora(
        dataset=tiny_dataset,
        base=base_model,
        lora_cfg=lora_cfg,
        training_cfg=training_cfg_1,
    )

    out_dir_2 = tmp_path / "run_2"
    training_cfg_2 = {
        "output_dir": str(out_dir_2),
        "num_train_epochs": 1.0,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "seed": 42,
    }

    adapter_path_2 = train_lora(
        dataset=tiny_dataset,
        base=base_model,
        lora_cfg=lora_cfg,
        training_cfg=training_cfg_2,
    )

    metrics_1_path = Path(adapter_path_1) / "eval_results.json"
    metrics_2_path = Path(adapter_path_2) / "eval_results.json"

    assert metrics_1_path.exists()
    assert metrics_2_path.exists()

    with open(metrics_1_path, encoding="utf-8") as f1, open(metrics_2_path, encoding="utf-8") as f2:
        eval1 = json.load(f1)
        eval2 = json.load(f2)

    eval_loss_1 = eval1.get("eval_loss")
    eval_loss_2 = eval2.get("eval_loss")

    assert eval_loss_1 is not None and eval_loss_2 is not None
    assert pytest.approx(eval_loss_1, abs=1e-4) == eval_loss_2
