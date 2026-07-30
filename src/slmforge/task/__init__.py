from .detector import (
    ALL_TASK_TYPES,
    CHAT,
    CLASSIFICATION,
    INSTRUCTION,
    QA,
    SUMMARISATION,
    detect_task,
)
from .templates import (
    ChatTemplate,
    ClassificationTemplate,
    InstructionTemplate,
    QATemplate,
    SummarisationTemplate,
    TaskTemplate,
    get_template,
    render_record,
    strip_record,
)

__all__ = [
    "ALL_TASK_TYPES",
    "CHAT",
    "CLASSIFICATION",
    "INSTRUCTION",
    "QA",
    "SUMMARISATION",
    "ChatTemplate",
    "ClassificationTemplate",
    "InstructionTemplate",
    "QATemplate",
    "SummarisationTemplate",
    "TaskTemplate",
    "detect_task",
    "get_template",
    "render_record",
    "strip_record",
]
