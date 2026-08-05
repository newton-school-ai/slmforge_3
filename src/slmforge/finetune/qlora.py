"""src/slmforge/finetune/qlora.py.
=============================
QLoRA (4-bit NF4) quantisation path for 8B-class models.
"""

from __future__ import annotations

from typing import Any

import torch
from peft import prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

from slmforge.finetune.lora import train_lora
from slmforge.finetune.registry import _REGISTRY, get_base

# Registered 8B-class base models or models with qlora recipe
_8B_CLASS_KEYWORDS = ("8b", "7b", "llama-3.1-8b", "qwen-2.5-7b", "deepseek-v3")


def is_8b_class(base: str | dict[str, Any]) -> bool:
    """Check whether a base model belongs to the 8B-class requiring QLoRA quantisation.

    Parameters
    ----------
    base : str | dict[str, Any]
        Model key in registry, HF repo ID, or base metadata dict.

    Returns
    -------
    bool
        True if model is an 8B-class model (vram >= 16GB or 8b/7b keywords), False otherwise.
    """
    if isinstance(base, dict):
        vram = base.get("vram", "")
        if "16" in vram or "24" in vram or "32" in vram:
            return True
        hf_id = base.get("hf_id", "").lower()
        return any(kw in hf_id for kw in _8B_CLASS_KEYWORDS)

    if isinstance(base, str):
        if base in _REGISTRY:
            model_info = get_base(base)
            vram = model_info.get("vram", "")
            if "16" in vram or "24" in vram or "32" in vram:
                return True
            hf_id = model_info.get("hf_id", "").lower()
            return any(kw in hf_id for kw in _8B_CLASS_KEYWORDS)

        base_lower = base.lower()
        return any(kw in base_lower for kw in _8B_CLASS_KEYWORDS)

    return False


def get_qlora_config(
    quant_type: str = "nf4",
    double_quant: bool = True,
    compute_dtype: torch.dtype | None = None,
) -> BitsAndBytesConfig:
    """Build a 4-bit NF4 BitsAndBytesConfig for QLoRA.

    Parameters
    ----------
    quant_type : str
        Quantisation data type, default "nf4".
    double_quant : bool
        Whether to use nested/double quantisation, default True.
    compute_dtype : torch.dtype | None
        Compute dtype for 4-bit base weights during forward pass.
        Defaults to bfloat16 if GPU supports it, else float16.

    Returns
    -------
    BitsAndBytesConfig
        4-bit quantisation configuration.
    """
    if compute_dtype is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def prepare_qlora_model(model: Any) -> Any:
    """Prepare a quantised 4-bit model for k-bit training with PEFT."""
    return prepare_model_for_kbit_training(model)


def train_qlora(
    dataset: Any,
    base: str | dict[str, Any],
    lora_cfg: dict[str, Any] | None = None,
    training_cfg: dict[str, Any] | None = None,
) -> str:
    """Train a QLoRA 4-bit quantised adapter on an 8B-class base model.

    Parameters
    ----------
    dataset : Any
        HF DatasetDict, Dataset, list of records, or path to dataset.
    base : str | dict[str, Any]
        Base model name from registry, HF repo ID, or base dict.
    lora_cfg : dict[str, Any] | None
        LoRA configuration parameters.
    training_cfg : dict[str, Any] | None
        Training configuration parameters.

    Returns
    -------
    str
        Path to the saved trained adapter output directory.
    """
    t_cfg = dict(training_cfg) if training_cfg else {}
    t_cfg["use_qlora"] = True

    return train_lora(
        dataset=dataset,
        base=base,
        lora_cfg=lora_cfg,
        training_cfg=t_cfg,
    )
