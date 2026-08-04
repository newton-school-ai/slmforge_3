"""src/slmforge/finetune/lora.py.
============================
LoRA training harness using HuggingFace Trainer and PEFT.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import datasets
import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from slmforge.finetune.registry import _REGISTRY, get_base


def _resolve_base_hf_id(base: str | dict[str, Any]) -> str:
    """Resolve base model input (registry key, dict, or HF path) to HF repo ID."""
    if isinstance(base, dict):
        if "hf_id" in base:
            return str(base["hf_id"])
        raise ValueError("Base model dict must contain 'hf_id'")

    if isinstance(base, str):
        if base in _REGISTRY:
            return get_base(base)["hf_id"]
        return base

    raise TypeError(f"Invalid type for base: {type(base)}")


def _format_record_to_text(example: dict[str, Any]) -> str:
    """Format a dataset record into a plain text string for language modeling."""
    if "text" in example and example["text"]:
        return str(example["text"])

    if "prompt" in example and "target" in example:
        return f"{example['prompt']}{example['target']}"

    if "prompt" in example and "response" in example:
        return f"{example['prompt']}{example['response']}"

    if "instruction" in example:
        inst = example["instruction"]
        inp = example.get("input", "")
        out = example.get("output", "")
        if inp:
            return f"Instruction: {inst}\nInput: {inp}\nOutput: {out}"
        return f"Instruction: {inst}\nOutput: {out}"

    # Fallback: combine primitive values
    return " ".join(
        str(v)
        for k, v in example.items()
        if isinstance(v, (str, int, float)) and not k.startswith("_")
    )


def train_lora(
    dataset: Any,
    base: str | dict[str, Any],
    lora_cfg: dict[str, Any] | LoraConfig | None = None,
    training_cfg: dict[str, Any] | TrainingArguments | None = None,
) -> str:
    """Train a LoRA adapter on a dataset using HF Trainer and PEFT.

    Parameters
    ----------
    dataset : Any
        HF DatasetDict, Dataset, list of records, or path to dataset.
    base : str | dict[str, Any]
        Base model name from registry, HF repo ID, or base dict.
    lora_cfg : dict[str, Any] | LoraConfig | None
        LoRA configuration dictionary or LoraConfig instance.
    training_cfg : dict[str, Any] | TrainingArguments | None
        Training configuration dictionary or TrainingArguments instance.

    Returns
    -------
    str
        Path to the saved trained adapter output directory.
    """
    hf_id = _resolve_base_hf_id(base)

    # Parse training_cfg
    if isinstance(training_cfg, TrainingArguments):
        training_dict = training_cfg.to_dict()
        output_dir = training_cfg.output_dir
        seed = training_cfg.seed
    else:
        training_dict = dict(training_cfg) if training_cfg else {}
        output_dir = training_dict.get("output_dir", "./outputs/adapter")
        seed = training_dict.get("seed", 42)

    # Set deterministic seed
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Process dataset into train and eval split
    eval_ds = None
    if isinstance(dataset, datasets.DatasetDict):
        train_ds = dataset["train"]
        eval_ds = dataset.get("eval") or dataset.get("val") or dataset.get("validation")
    elif isinstance(dataset, datasets.Dataset):
        train_ds = dataset
    elif isinstance(dataset, list):
        train_ds = datasets.Dataset.from_list(dataset)
    elif isinstance(dataset, (str, Path)):
        ds_path = Path(dataset)
        if ds_path.exists():
            loaded = datasets.load_from_disk(str(ds_path))
            if isinstance(loaded, datasets.DatasetDict):
                train_ds = loaded["train"]
                eval_ds = loaded.get("eval") or loaded.get("val") or loaded.get("validation")
            else:
                train_ds = loaded
        else:
            loaded = datasets.load_dataset(str(dataset))
            if isinstance(loaded, datasets.DatasetDict):
                train_ds = loaded["train"]
                eval_ds = loaded.get("eval") or loaded.get("val") or loaded.get("validation")
            else:
                train_ds = loaded
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")

    # Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        trust_remote_code=True,
    )

    max_seq_len = training_dict.get("max_seq_length", 512)

    def tokenize_fn(examples: dict[str, Any]) -> dict[str, Any]:
        if "input_ids" in examples:
            return examples

        if "text" in examples:
            texts = examples["text"]
        else:
            keys = list(examples.keys())
            num_items = len(examples[keys[0]])
            texts = []
            for i in range(num_items):
                item = {k: examples[k][i] for k in keys}
                texts.append(_format_record_to_text(item))

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        tokenized["labels"] = [list(ids) for ids in tokenized["input_ids"]]
        return tokenized

    tokenized_train_ds = train_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in train_ds.column_names if c not in ("input_ids", "labels", "attention_mask")],
    )

    tokenized_eval_ds = None
    if eval_ds is not None:
        tokenized_eval_ds = eval_ds.map(
            tokenize_fn,
            batched=True,
            remove_columns=[c for c in eval_ds.column_names if c not in ("input_ids", "labels", "attention_mask")],
        )

    # Configure PEFT LoRA
    if isinstance(lora_cfg, LoraConfig):
        peft_config = lora_cfg
    else:
        lora_dict = dict(lora_cfg) if lora_cfg else {}
        r = lora_dict.get("r", 8)
        lora_alpha = lora_dict.get("lora_alpha", 16)
        target_modules = lora_dict.get("target_modules", None)
        lora_dropout = lora_dict.get("lora_dropout", 0.05)
        bias = lora_dict.get("bias", "none")
        task_type = lora_dict.get("task_type", TaskType.CAUSAL_LM)

        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=task_type,
        )

    peft_model = get_peft_model(model, peft_config)

    # Set up TrainingArguments
    has_eval = tokenized_eval_ds is not None
    eval_strat = "epoch" if has_eval else "no"

    targs_dict = {
        "output_dir": output_dir,
        "num_train_epochs": training_dict.get("num_train_epochs", 1.0),
        "per_device_train_batch_size": training_dict.get("per_device_train_batch_size", 2),
        "per_device_eval_batch_size": training_dict.get("per_device_eval_batch_size", 2),
        "gradient_accumulation_steps": training_dict.get("gradient_accumulation_steps", 1),
        "learning_rate": training_dict.get("learning_rate", 2e-4),
        "weight_decay": training_dict.get("weight_decay", 0.01),
        "warmup_ratio": training_dict.get("warmup_ratio", 0.03),
        "logging_steps": training_dict.get("logging_steps", 1),
        "save_strategy": training_dict.get("save_strategy", "epoch"),
        "save_total_limit": training_dict.get("save_total_limit", 2),
        "seed": seed,
        "data_seed": seed,
        "report_to": training_dict.get("report_to", "none"),
        "fp16": training_dict.get("fp16", torch.cuda.is_available()),
        "use_cpu": training_dict.get(
            "use_cpu",
            not torch.cuda.is_available()
            and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        ),
    }

    if hasattr(TrainingArguments, "eval_strategy"):
        targs_dict["eval_strategy"] = eval_strat
    else:
        targs_dict["evaluation_strategy"] = eval_strat

    if isinstance(training_cfg, TrainingArguments):
        training_args = training_cfg
    else:
        training_args = TrainingArguments(**targs_dict)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_eval_ds if has_eval else None,
        data_collator=data_collator,
    )

    trainer.train()

    eval_metrics: dict[str, Any] = {}
    if has_eval:
        eval_metrics = trainer.evaluate()

    # Save adapter only (saves adapter config + adapter weights, NOT base model)
    adapter_path = Path(output_dir)
    adapter_path.mkdir(parents=True, exist_ok=True)

    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    # Store evaluation metrics for inspection / testing determinism
    metrics_file = adapter_path / "eval_results.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    return str(adapter_path)
