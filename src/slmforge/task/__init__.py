from .detector import (
    detect_task,
    CLASSIFICATION,
    SUMMARISATION,
    QA,
    INSTRUCTION,
    CHAT,
    ALL_TASK_TYPES,
)
from .templates import (
    TaskTemplate,
    ClassificationTemplate,
    SummarisationTemplate,
    QATemplate,
    InstructionTemplate,
    ChatTemplate,
    get_template,
    render_record,
    strip_record,
)

__all__ = [
    "detect_task",
    "CLASSIFICATION",
    "SUMMARISATION",
    "QA",
    "INSTRUCTION",
    "CHAT",
    "ALL_TASK_TYPES",
    "TaskTemplate",
    "ClassificationTemplate",
    "SummarisationTemplate",
    "QATemplate",
    "InstructionTemplate",
    "ChatTemplate",
    "get_template",
    "render_record",
    "strip_record",
]
