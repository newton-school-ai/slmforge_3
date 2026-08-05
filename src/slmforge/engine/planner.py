"""src/slmforge/engine/planner.py.
=================================
VRAM and walltime estimator and run planner for SLM fine-tuning.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from slmforge.finetune.registry import _REGISTRY, get_base

# Default shared GPU VRAM limit in GB
DEFAULT_SHARED_BUDGET_GB: float = 16.0

# Parameter counts (in billions) for known bases or estimation
MODEL_PARAM_MAP: dict[str, float] = {
    "phi-3-mini": 3.8,
    "llama-3.1-8b-instruct": 8.0,
    "qwen-2.5-7b-instruct": 7.6,
    "deepseek-v3-distill": 7.0,
    "sshleifer/tiny-gpt2": 0.005,
    "hf-internal-testing/tiny-random-gpt2": 0.001,
}


@dataclass
class RunPlan:
    """Estimated resource allocation and walltime plan for a fine-tuning run."""

    vram_gb: float
    est_minutes: float
    over_budget: bool
    shared_budget_gb: float
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert RunPlan to dictionary."""
        return asdict(self)

    def to_event(self) -> dict[str, Any]:
        """Emit plan as a streaming event dict before training starts."""
        return {
            "event": "plan",
            "data": {
                "vram_gb": round(self.vram_gb, 2),
                "est_minutes": round(self.est_minutes, 2),
                "over_budget": self.over_budget,
                "shared_budget_gb": self.shared_budget_gb,
                "warning": self.warning,
            },
        }


def _estimate_param_count_billions(base: str | dict[str, Any]) -> float:
    """Helper to infer or estimate base model parameter count in billions."""
    if isinstance(base, dict):
        base_name = base.get("hf_id", "") or base.get("name", "")
    else:
        base_name = str(base)

    if base_name in MODEL_PARAM_MAP:
        return MODEL_PARAM_MAP[base_name]

    if base_name in _REGISTRY:
        return MODEL_PARAM_MAP.get(base_name, 7.0)

    base_lower = base_name.lower()
    if "8b" in base_lower:
        return 8.0
    if "7b" in base_lower:
        return 7.0
    if "3b" in base_lower or "mini" in base_lower:
        return 3.8
    if "1b" in base_lower:
        return 1.0
    if "tiny" in base_lower or "micro" in base_lower:
        return 0.005

    return 7.0


def estimate_run(
    dataset_size: int,
    base: str | dict[str, Any],
    lora_cfg: dict[str, Any] | None = None,
    epochs: float = 1.0,
    shared_budget_gb: float = DEFAULT_SHARED_BUDGET_GB,
    batch_size: int = 2,
    seq_len: int = 512,
    use_qlora: bool | None = None,
) -> RunPlan:
    """Estimate VRAM usage and walltime duration for a training run.

    Parameters
    ----------
    dataset_size : int
        Total number of records/samples in dataset.
    base : str | dict[str, Any]
        Registered base model name, HF ID, or metadata dict.
    lora_cfg : dict[str, Any] | None
        LoRA configuration dict (e.g., {"r": 8, "lora_alpha": 16}).
    epochs : float
        Number of training epochs.
    shared_budget_gb : float
        Shared GPU VRAM memory limit in GB (default 16.0 GB).
    batch_size : int
        Per-device training batch size (default 2).
    seq_len : int
        Maximum sequence length (default 512).
    use_qlora : bool | None
        Whether QLoRA 4-bit quantisation is enabled. If None, auto-detected
        for 8B-class models or models with recipe="qlora".

    Returns
    -------
    RunPlan
        RunPlan dataclass containing vram_gb, est_minutes, over_budget status,
        and warning messages if applicable.
    """
    params_b = _estimate_param_count_billions(base)
    lora_dict = lora_cfg or {}
    r = lora_dict.get("r", 8)

    if use_qlora is None:
        if isinstance(base, dict):
            recipe = base.get("lora_recipe", "")
            use_qlora = recipe == "qlora" or params_b >= 7.0
        elif isinstance(base, str) and base in _REGISTRY:
            recipe = get_base(base).get("lora_recipe", "")
            use_qlora = recipe == "qlora" or params_b >= 7.0
        else:
            use_qlora = params_b >= 7.0

    # 1. Base Model Weights VRAM
    if use_qlora:
        weights_vram = (params_b * 0.5) + 0.5
    else:
        weights_vram = (params_b * 2.0) + 0.3

    # 2. Adapter & Optimizer VRAM
    adapter_vram = (params_b * 0.01) * (r / 8.0)

    # 3. Activation & KV Cache VRAM
    activation_vram = (batch_size * (seq_len / 512.0) * (params_b / 7.0)) * 0.4

    # 4. PyTorch / CUDA Overhead
    cuda_overhead = 1.0 if params_b > 1.0 else 0.2

    total_vram_gb = round(weights_vram + adapter_vram + activation_vram + cuda_overhead, 2)

    # 5. Walltime (minutes)
    steps_per_epoch = math.ceil(dataset_size / max(batch_size, 1))
    total_steps = steps_per_epoch * epochs

    if params_b < 0.1:
        sec_per_step = 0.05
    elif use_qlora:
        sec_per_step = 0.6 + (params_b * 0.05)
    else:
        sec_per_step = 0.4 + (params_b * 0.04)

    total_seconds = total_steps * sec_per_step + 10.0
    est_minutes = round(max(total_seconds / 60.0, 0.1), 2)

    over_budget = total_vram_gb > shared_budget_gb
    warning = None
    if over_budget:
        warning = (
            f"Estimated VRAM ({total_vram_gb:.1f} GB) exceeds shared GPU budget "
            f"({shared_budget_gb:.1f} GB). Consider using QLoRA 4-bit, reducing batch size, "
            f"or decreasing sequence length."
        )

    return RunPlan(
        vram_gb=total_vram_gb,
        est_minutes=est_minutes,
        over_budget=over_budget,
        shared_budget_gb=shared_budget_gb,
        warning=warning,
    )


def plan_event(plan: RunPlan) -> dict[str, Any]:
    """Helper to format a RunPlan into a streaming event payload."""
    return plan.to_event()
