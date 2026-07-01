from typing import Literal

from pydantic import BaseModel


class Source(BaseModel):
    type: Literal["local", "public", "synthetic", "internal"]
    path: str | None = None
    id: str | None = None
    generator: str | None = None
    size: int | None = None
    seed: int | None = None


class LoRAConfig(BaseModel):
    r: int
    alpha: int
    dropout: float


class TrainingConfig(BaseModel):
    epochs: int
    batch_size: int
    lr: float


class EvalConfig(BaseModel):
    llm_judge: bool


class BuildRequest(BaseModel):
    sources: list[Source]
    task_type: Literal[
        "classification",
        "summarisation",
        "qa",
        "instruction",
        "chat",
        "auto",
    ]
    base_model: str
    lora: LoRAConfig
    training: TrainingConfig
    eval: EvalConfig


class BuildResponse(BaseModel):
    build_id: str
    status: str
