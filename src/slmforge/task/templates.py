"""
src/slmforge/task/templates.py
=============================
Canonical templates and formatting utilities for different task types.
"""

from __future__ import annotations

import re
from typing import Any

from slmforge.task.detector import (
    CLASSIFICATION,
    SUMMARISATION,
    QA,
    INSTRUCTION,
    CHAT,
)


def _find_key(
    keys: list[str], candidates: list[str], exclude: list[str] | None = None
) -> str | None:
    """Helper to find first matching key in record from candidates list, excluding specific keys."""
    exclude = exclude or []
    for cand in candidates:
        for k in keys:
            if k.lower() == cand.lower() and k not in exclude:
                return k
    return None


def render_messages(messages: list[dict[str, str]], format_type: str) -> tuple[str, str]:
    """Render a list of messages into prompt and target strings.

    Parameters
    ----------
    messages : list[dict[str, str]]
        A list of message dictionaries, each containing "role" and "content".
    format_type : str
        The chat format style, either "phi-3" or "llama-3.1".

    Returns
    -------
    tuple[str, str]
        A tuple of (prompt, target).
    """
    if not messages:
        return "", ""

    if messages[-1]["role"] != "assistant":
        raise ValueError("The last message must be from the assistant to generate a target.")

    prompt_msgs = messages[:-1]
    target_msg = messages[-1]

    prompt = ""
    if format_type == "phi-3":
        for msg in prompt_msgs:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|{role}|>\n{content}<|end|>\n"
        prompt += "<|assistant|>\n"
        target = f"{target_msg['content']}<|end|>"

    elif format_type == "llama-3.1":
        prompt = "<|begin_of_text|>"
        for msg in prompt_msgs:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        target = f"{target_msg['content']}<|eot_id|>"

    else:
        raise ValueError(f"Unsupported format_type: {format_type}")

    return prompt, target


def strip_messages(prompt: str, target: str, format_type: str) -> list[dict[str, str]]:
    """Parse a rendered prompt and target back into a list of messages.

    Parameters
    ----------
    prompt : str
        The rendered prompt string.
    target : str
        The rendered target string.
    format_type : str
        The chat format style, either "phi-3" or "llama-3.1".

    Returns
    -------
    list[dict[str, str]]
        A list of message dictionaries, each containing "role" and "content".
    """
    full_text = prompt + target

    if format_type == "phi-3":
        pattern = re.compile(r"<\|(system|user|assistant)\|>\n?(.*?)(?:<\|end\|>\n?|$)", re.DOTALL)
        matches = pattern.findall(full_text)
        return [{"role": r, "content": c} for r, c in matches]

    elif format_type == "llama-3.1":
        if full_text.startswith("<|begin_of_text|>"):
            full_text = full_text[len("<|begin_of_text|>") :]
        pattern = re.compile(
            r"<\|start_header_id\|>(system|user|assistant)<\|end_header_id\|>\n\n(.*?)(?:<\|eot_id\|>|$)",
            re.DOTALL,
        )
        matches = pattern.findall(full_text)
        return [{"role": r, "content": c} for r, c in matches]

    else:
        raise ValueError(f"Unsupported format_type: {format_type}")


class TaskTemplate:
    """Base class for all task templates."""

    def render(self, record: dict[str, Any], format_type: str) -> tuple[str, str]:
        """Render a record into (prompt, target) strings."""
        raise NotImplementedError

    def strip(self, prompt: str, target: str, format_type: str) -> dict[str, Any]:
        """Strip rendered prompt and target back into the original record."""
        raise NotImplementedError


class ClassificationTemplate(TaskTemplate):
    """Template for text classification tasks."""

    def __init__(
        self,
        text_key: str | None = None,
        label_key: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.text_key = text_key
        self.label_key = label_key
        self.system_prompt = system_prompt or "Classify the text into the correct category."

    def _bind_keys(self, record: dict[str, Any]) -> None:
        keys = list(record.keys())
        if self.label_key is None:
            self.label_key = (
                _find_key(
                    keys,
                    [
                        "label",
                        "class",
                        "category",
                        "sentiment",
                        "intent",
                        "target",
                        "classification",
                        "y",
                    ],
                )
                or "label"
            )
        if self.text_key is None:
            self.text_key = _find_key(
                keys, ["text", "input", "sentence", "document"], exclude=[self.label_key]
            )
            if self.text_key is None:
                non_label_keys = [k for k in keys if k != self.label_key]
                self.text_key = non_label_keys[0] if non_label_keys else "text"

    def render(self, record: dict[str, Any], format_type: str) -> tuple[str, str]:
        self._bind_keys(record)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": str(record.get(self.text_key, ""))},
            {"role": "assistant", "content": str(record.get(self.label_key, ""))},
        ]
        return render_messages(messages, format_type)

    def strip(self, prompt: str, target: str, format_type: str) -> dict[str, Any]:
        messages = strip_messages(prompt, target, format_type)
        user_content = ""
        assistant_content = ""
        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "assistant":
                assistant_content = msg["content"]

        text_key = self.text_key or "text"
        label_key = self.label_key or "label"
        return {
            text_key: user_content,
            label_key: assistant_content,
        }


class SummarisationTemplate(TaskTemplate):
    """Template for summarization tasks."""

    def __init__(
        self,
        document_key: str | None = None,
        summary_key: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.document_key = document_key
        self.summary_key = summary_key
        self.system_prompt = system_prompt or "Summarize the text concisely."

    def _bind_keys(self, record: dict[str, Any]) -> None:
        keys = list(record.keys())
        if self.summary_key is None:
            self.summary_key = (
                _find_key(
                    keys,
                    ["summary", "abstract", "headline", "title", "summarisation", "summarization"],
                )
                or "summary"
            )
        if self.document_key is None:
            self.document_key = _find_key(
                keys, ["document", "article", "text", "context", "body"], exclude=[self.summary_key]
            )
            if self.document_key is None:
                non_summary_keys = [k for k in keys if k != self.summary_key]
                self.document_key = non_summary_keys[0] if non_summary_keys else "document"

    def render(self, record: dict[str, Any], format_type: str) -> tuple[str, str]:
        self._bind_keys(record)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": str(record.get(self.document_key, ""))},
            {"role": "assistant", "content": str(record.get(self.summary_key, ""))},
        ]
        return render_messages(messages, format_type)

    def strip(self, prompt: str, target: str, format_type: str) -> dict[str, Any]:
        messages = strip_messages(prompt, target, format_type)
        user_content = ""
        assistant_content = ""
        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "assistant":
                assistant_content = msg["content"]

        doc_key = self.document_key or "document"
        sum_key = self.summary_key or "summary"
        return {
            doc_key: user_content,
            sum_key: assistant_content,
        }


class QATemplate(TaskTemplate):
    """Template for question answering tasks."""

    def __init__(
        self,
        question_key: str | None = None,
        answer_key: str | None = None,
        context_key: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.question_key = question_key
        self.answer_key = answer_key
        self.context_key = context_key
        self.system_prompt = system_prompt

    def _bind_keys(self, record: dict[str, Any]) -> None:
        keys = list(record.keys())
        if self.answer_key is None:
            self.answer_key = _find_key(keys, ["answer", "reply", "a", "response"]) or "answer"
        if self.question_key is None:
            self.question_key = _find_key(
                keys, ["question", "query", "q", "prompt"], exclude=[self.answer_key]
            )
            if self.question_key is None:
                non_answer_keys = [k for k in keys if k != self.answer_key]
                self.question_key = non_answer_keys[0] if non_answer_keys else "question"
        if self.context_key is None:
            self.context_key = _find_key(
                keys,
                ["context", "document", "passage"],
                exclude=[self.question_key, self.answer_key],
            )

    def render(self, record: dict[str, Any], format_type: str) -> tuple[str, str]:
        self._bind_keys(record)
        context_val = record.get(self.context_key) if self.context_key else None

        if context_val is not None:
            user_content = f"Context: {context_val}\nQuestion: {record.get(self.question_key, '')}"
            sys_prompt = self.system_prompt or "Answer the question based on the provided context."
        else:
            user_content = str(record.get(self.question_key, ""))
            sys_prompt = self.system_prompt or "Answer the question."

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": str(record.get(self.answer_key, ""))},
        ]
        return render_messages(messages, format_type)

    def strip(self, prompt: str, target: str, format_type: str) -> dict[str, Any]:
        messages = strip_messages(prompt, target, format_type)
        user_content = ""
        assistant_content = ""
        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "assistant":
                assistant_content = msg["content"]

        q_key = self.question_key or "question"
        a_key = self.answer_key or "answer"

        result = {
            a_key: assistant_content,
        }

        if user_content.startswith("Context:"):
            match = re.match(r"^Context:\s*(.*?)\nQuestion:\s*(.*)$", user_content, re.DOTALL)
            if match:
                c_key = self.context_key or "context"
                result[c_key] = match.group(1)
                result[q_key] = match.group(2)
            else:
                result[q_key] = user_content
        else:
            result[q_key] = user_content

        return result


class InstructionTemplate(TaskTemplate):
    """Template for instruction-following tasks."""

    def __init__(
        self,
        instruction_key: str | None = None,
        input_key: str | None = None,
        response_key: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.instruction_key = instruction_key
        self.input_key = input_key
        self.response_key = response_key
        self.system_prompt = system_prompt or "You are a helpful assistant."

    def _bind_keys(self, record: dict[str, Any]) -> None:
        keys = list(record.keys())
        if self.response_key is None:
            self.response_key = _find_key(keys, ["response", "output"]) or "response"
        if self.instruction_key is None:
            self.instruction_key = _find_key(
                keys, ["instruction", "prompt"], exclude=[self.response_key]
            )
            if self.instruction_key is None:
                non_response_keys = [k for k in keys if k != self.response_key]
                self.instruction_key = non_response_keys[0] if non_response_keys else "instruction"
        if self.input_key is None:
            self.input_key = _find_key(
                keys, ["input"], exclude=[self.instruction_key, self.response_key]
            )

    def render(self, record: dict[str, Any], format_type: str) -> tuple[str, str]:
        self._bind_keys(record)
        input_val = record.get(self.input_key) if self.input_key else None

        if input_val is not None and str(input_val).strip() != "":
            user_content = (
                f"Instruction: {record.get(self.instruction_key, '')}\nInput: {input_val}"
            )
        else:
            user_content = str(record.get(self.instruction_key, ""))

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": str(record.get(self.response_key, ""))},
        ]
        return render_messages(messages, format_type)

    def strip(self, prompt: str, target: str, format_type: str) -> dict[str, Any]:
        messages = strip_messages(prompt, target, format_type)
        user_content = ""
        assistant_content = ""
        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
            elif msg["role"] == "assistant":
                assistant_content = msg["content"]

        inst_key = self.instruction_key or "instruction"
        resp_key = self.response_key or "response"

        result = {
            resp_key: assistant_content,
        }

        if user_content.startswith("Instruction:"):
            match = re.match(r"^Instruction:\s*(.*?)\nInput:\s*(.*)$", user_content, re.DOTALL)
            if match:
                in_key = self.input_key or "input"
                result[inst_key] = match.group(1)
                result[in_key] = match.group(2)
            else:
                result[inst_key] = user_content
        else:
            result[inst_key] = user_content

        return result


class ChatTemplate(TaskTemplate):
    """Template for chat/dialogue tasks."""

    def __init__(
        self,
        messages_key: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.messages_key = messages_key
        self.system_prompt = system_prompt
        self._role_key = "role"
        self._content_key = "content"

    def _bind_keys(self, record: dict[str, Any]) -> None:
        keys = list(record.keys())
        if self.messages_key is None:
            self.messages_key = (
                _find_key(keys, ["messages", "conversations", "dialogue", "turns", "chat"])
                or "messages"
            )

    def render(self, record: dict[str, Any], format_type: str) -> tuple[str, str]:
        self._bind_keys(record)
        raw_msgs = record.get(self.messages_key, [])
        if not isinstance(raw_msgs, list):
            raw_msgs = []

        role_key = "role"
        content_key = "content"
        if raw_msgs and isinstance(raw_msgs[0], dict):
            keys = {k.lower() for k in raw_msgs[0].keys()}
            if "from" in keys and "value" in keys:
                role_key = "from"
                content_key = "value"
            elif "speaker" in keys and "text" in keys:
                role_key = "speaker"
                content_key = "text"
            else:
                role_key = (
                    _find_key(list(raw_msgs[0].keys()), ["role", "from", "speaker"]) or "role"
                )
                content_key = (
                    _find_key(list(raw_msgs[0].keys()), ["content", "value", "text"]) or "content"
                )

        self._role_key = role_key
        self._content_key = content_key

        messages = []
        for msg in raw_msgs:
            if not isinstance(msg, dict):
                continue
            r = str(msg.get(role_key, ""))
            c = str(msg.get(content_key, ""))

            if r.lower() in ("human", "user", "speaker"):
                role = "user"
            elif r.lower() in ("gpt", "assistant", "system_gpt"):
                role = "assistant"
            elif r.lower() == "system":
                role = "system"
            else:
                role = r.lower()

            messages.append({"role": role, "content": c})

        return render_messages(messages, format_type)

    def strip(self, prompt: str, target: str, format_type: str) -> dict[str, Any]:
        messages = strip_messages(prompt, target, format_type)

        reconstructed = []
        for msg in messages:
            r = msg["role"]
            c = msg["content"]

            orig_role = r
            if self._role_key.lower() == "from":
                if r == "user":
                    orig_role = "human"
                elif r == "assistant":
                    orig_role = "gpt"
                elif r == "system":
                    orig_role = "system"
            elif self._role_key.lower() == "speaker":
                if r == "user":
                    orig_role = "user"
                elif r == "assistant":
                    orig_role = "assistant"
                elif r == "system":
                    orig_role = "system"

            reconstructed.append(
                {
                    self._role_key: orig_role,
                    self._content_key: c,
                }
            )

        msg_key = self.messages_key or "messages"
        return {msg_key: reconstructed}


def get_template(task_type: str, **kwargs: Any) -> TaskTemplate:
    """Get a template instance for the specified task type.

    Parameters
    ----------
    task_type : str
        The task type.
    **kwargs : Any
        Optional arguments for initializing the template.

    Returns
    -------
    TaskTemplate
        A task template instance.
    """
    if task_type == CLASSIFICATION:
        return ClassificationTemplate(**kwargs)
    elif task_type == SUMMARISATION:
        return SummarisationTemplate(**kwargs)
    elif task_type == QA:
        return QATemplate(**kwargs)
    elif task_type == INSTRUCTION:
        return InstructionTemplate(**kwargs)
    elif task_type == CHAT:
        return ChatTemplate(**kwargs)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")


def render_record(
    record: dict[str, Any],
    task_type: str,
    format_type: str,
    **kwargs: Any,
) -> tuple[str, str]:
    """Render a dataset record into prompt and target strings based on task type.

    Parameters
    ----------
    record : dict[str, Any]
        The dataset record.
    task_type : str
        The task type.
    format_type : str
        The chat format style, either "phi-3" or "llama-3.1".
    **kwargs : Any
        Optional arguments for initializing the template.

    Returns
    -------
    tuple[str, str]
        A tuple of (prompt, target).
    """
    template = get_template(task_type, **kwargs)
    return template.render(record, format_type)


def strip_record(
    prompt: str,
    target: str,
    task_type: str,
    format_type: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Parse a rendered prompt and target back into the original record.

    Parameters
    ----------
    prompt : str
        The rendered prompt.
    target : str
        The rendered target.
    task_type : str
        The task type.
    format_type : str
        The chat format style.
    **kwargs : Any
        Optional arguments for initializing the template.

    Returns
    -------
    dict[str, Any]
        The reconstructed record.
    """
    template = get_template(task_type, **kwargs)
    return template.strip(prompt, target, format_type)
