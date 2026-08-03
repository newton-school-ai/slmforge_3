from typing import Literal

from pydantic import BaseModel

TaskType = Literal[
    "classification",
    "chat",
    "instruction",
    "qa",
    "summarisation",
]

BaseModelType = Literal[
    "phi-3-mini",
    "llama3",
    "mistral",
]

TemplateType = Literal[
    "chat",
    "classification",
    "instruction",
    "qa",
    "summarisation",
]


class BuildRequest(BaseModel):
    name: str

    base_model: BaseModelType = "phi-3-mini"

    task_type: TaskType | None = None

    template: TemplateType | None = None

    epochs: int = 3
