from __future__ import annotations

from typing import TypedDict


class BaseModel(TypedDict):
    hf_id: str
    lora_recipe: str
    vram: str
    license: str
    recommended_tasks: list[str]


_REGISTRY: dict[str, BaseModel] = {
    "phi-3-mini": {
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "lora_recipe": "qlora",
        "vram": "8 GB",
        "license": "MIT",
        "recommended_tasks": [
            "chat",
            "classification",
            "instruction",
            "qa",
            "summarisation",
        ],
    },
    "llama-3.1-8b-instruct": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "lora_recipe": "qlora",
        "vram": "16 GB",
        "license": "Llama 3.1 Community",
        "recommended_tasks": [
            "chat",
            "instruction",
            "qa",
            "summarisation",
        ],
    },
    "qwen-2.5-7b-instruct": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "lora_recipe": "qlora",
        "vram": "16 GB",
        "license": "Apache-2.0",
        "recommended_tasks": [
            "chat",
            "classification",
            "instruction",
            "qa",
        ],
    },
    "deepseek-v3-distill": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "lora_recipe": "qlora",
        "vram": "16 GB",
        "license": "MIT",
        "recommended_tasks": [
            "chat",
            "instruction",
            "qa",
            "summarisation",
        ],
    },
}


def get_base(name: str) -> BaseModel:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown base model: {name}")
    return _REGISTRY[name]


def list_bases() -> list[str]:
    return list(_REGISTRY.keys())


def validate_hf_id(hf_id: str) -> bool:
    return any(model["hf_id"] == hf_id for model in _REGISTRY.values())
