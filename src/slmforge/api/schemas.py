from pydantic import BaseModel


class BuildRequest(BaseModel):
    name: str
    base_model: str = "phi-3-mini"
    task_type: str | None = None
    epochs: int = 3