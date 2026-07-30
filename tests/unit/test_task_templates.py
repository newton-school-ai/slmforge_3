"""
tests/unit/test_task_templates.py
==================================
Unit tests for task templates, prompt rendering, and roundtrip parsing.
"""

from __future__ import annotations

import pytest

from slmforge.task import (
    CLASSIFICATION,
    ClassificationTemplate,
    SummarisationTemplate,
    QATemplate,
    InstructionTemplate,
    ChatTemplate,
    render_record,
    strip_record,
)


@pytest.mark.parametrize("format_type", ["phi-3", "llama-3.1"])
def test_classification_template_roundtrip(format_type: str) -> None:
    # 1. Standard keys
    record = {
        "text": "The movie was absolutely amazing!",
        "label": "positive",
    }
    template = ClassificationTemplate()
    prompt, target = template.render(record, format_type)

    # Check format-specific headers
    if format_type == "phi-3":
        assert "<|system|>\n" in prompt
        assert "<|user|>\n" in prompt
        assert "<|assistant|>\n" in prompt
        assert target.endswith("<|end|>")
    else:
        assert "<|begin_of_text|>" in prompt
        assert "<|start_header_id|>system<|end_header_id|>" in prompt
        assert "<|start_header_id|>user<|end_header_id|>" in prompt
        assert "<|start_header_id|>assistant<|end_header_id|>" in prompt
        assert target.endswith("<|eot_id|>")

    reconstructed = template.strip(prompt, target, format_type)
    assert reconstructed == record

    # 2. Custom keys
    custom_record = {
        "review": "Not worth the price.",
        "sentiment": "negative",
    }
    custom_template = ClassificationTemplate(text_key="review", label_key="sentiment")
    prompt_c, target_c = custom_template.render(custom_record, format_type)
    reconstructed_c = custom_template.strip(prompt_c, target_c, format_type)
    assert reconstructed_c == custom_record

    # 3. Auto-detected custom keys
    auto_template = ClassificationTemplate()
    prompt_a, target_a = auto_template.render(custom_record, format_type)
    reconstructed_a = auto_template.strip(prompt_a, target_a, format_type)
    assert reconstructed_a == custom_record


@pytest.mark.parametrize("format_type", ["phi-3", "llama-3.1"])
def test_summarisation_template_roundtrip(format_type: str) -> None:
    record = {
        "document": "The quick brown fox jumps over the lazy dog. This is a very famous sentence.",
        "summary": "Fox jumps over dog.",
    }
    template = SummarisationTemplate()
    prompt, target = template.render(record, format_type)
    reconstructed = template.strip(prompt, target, format_type)
    assert reconstructed == record

    # Custom keys
    custom_record = {
        "article": "Some long news article goes here.",
        "headline": "Short title",
    }
    custom_template = SummarisationTemplate(document_key="article", summary_key="headline")
    prompt_c, target_c = custom_template.render(custom_record, format_type)
    reconstructed_c = custom_template.strip(prompt_c, target_c, format_type)
    assert reconstructed_c == custom_record


@pytest.mark.parametrize("format_type", ["phi-3", "llama-3.1"])
def test_qa_template_roundtrip(format_type: str) -> None:
    # 1. QA without context
    record_no_ctx = {
        "question": "What is the capital of Japan?",
        "answer": "Tokyo",
    }
    template_no_ctx = QATemplate()
    prompt_n, target_n = template_no_ctx.render(record_no_ctx, format_type)
    reconstructed_n = template_no_ctx.strip(prompt_n, target_n, format_type)
    assert reconstructed_n == record_no_ctx

    # 2. QA with context
    record_ctx = {
        "context": "Tokyo is the capital and largest city of Japan.",
        "question": "What is the capital of Japan?",
        "answer": "Tokyo",
    }
    template_ctx = QATemplate()
    prompt_c, target_c = template_ctx.render(record_ctx, format_type)
    reconstructed_c = template_ctx.strip(prompt_c, target_c, format_type)
    assert reconstructed_c == record_ctx

    # 3. Custom keys
    custom_record = {
        "passage": "Paris is the capital of France.",
        "query": "Where is Paris?",
        "reply": "France",
    }
    custom_template = QATemplate(context_key="passage", question_key="query", answer_key="reply")
    prompt_cust, target_cust = custom_template.render(custom_record, format_type)
    reconstructed_cust = custom_template.strip(prompt_cust, target_cust, format_type)
    assert reconstructed_cust == custom_record


@pytest.mark.parametrize("format_type", ["phi-3", "llama-3.1"])
def test_instruction_template_roundtrip(format_type: str) -> None:
    # 1. Instruction without input
    record_no_in = {
        "instruction": "Write a poem about rain.",
        "response": "Raindrops fall from the sky...",
    }
    template_no_in = InstructionTemplate()
    prompt_n, target_n = template_no_in.render(record_no_in, format_type)
    reconstructed_n = template_no_in.strip(prompt_n, target_n, format_type)
    assert reconstructed_n == record_no_in

    # 2. Instruction with input
    record_in = {
        "instruction": "Translate the sentence into Spanish.",
        "input": "Hello, how are you?",
        "response": "Hola, como estas?",
    }
    template_in = InstructionTemplate()
    prompt_i, target_i = template_in.render(record_in, format_type)
    reconstructed_i = template_in.strip(prompt_i, target_i, format_type)
    assert reconstructed_i == record_in

    # 3. Custom keys
    custom_record = {
        "prompt": "Calculate 123 + 456.",
        "output": "579",
    }
    custom_template = InstructionTemplate(instruction_key="prompt", response_key="output")
    prompt_c, target_c = custom_template.render(custom_record, format_type)
    reconstructed_c = custom_template.strip(prompt_c, target_c, format_type)
    assert reconstructed_c == custom_record


@pytest.mark.parametrize("format_type", ["phi-3", "llama-3.1"])
def test_chat_template_roundtrip(format_type: str) -> None:
    # 1. Standard OpenAI format (role/content)
    record_openai = {
        "messages": [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "What is prime number?"},
            {"role": "assistant", "content": "A number divisible only by 1 and itself."},
            {"role": "user", "content": "Is 2 prime?"},
            {"role": "assistant", "content": "Yes, 2 is the only even prime number."},
        ]
    }
    template = ChatTemplate()
    prompt, target = template.render(record_openai, format_type)
    reconstructed = template.strip(prompt, target, format_type)
    assert reconstructed == record_openai

    # 2. ShareGPT format (from/value)
    record_sharegpt = {
        "conversations": [
            {"from": "system", "value": "You are a translator."},
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Bonjour"},
        ]
    }
    template_sgpt = ChatTemplate()
    prompt_s, target_s = template_sgpt.render(record_sharegpt, format_type)
    reconstructed_s = template_sgpt.strip(prompt_s, target_s, format_type)
    assert reconstructed_s == record_sharegpt

    # 3. Custom speaker/text format
    record_speaker = {
        "turns": [
            {"speaker": "user", "text": "Hi"},
            {"speaker": "assistant", "text": "Hello, how can I help?"},
        ]
    }
    template_speaker = ChatTemplate()
    prompt_sp, target_sp = template_speaker.render(record_speaker, format_type)
    reconstructed_sp = template_speaker.strip(prompt_sp, target_sp, format_type)
    assert reconstructed_sp == record_speaker


@pytest.mark.parametrize("format_type", ["phi-3", "llama-3.1"])
def test_render_and_strip_record_helpers(format_type: str) -> None:
    record = {
        "text": "This is a classification test.",
        "label": "test",
    }
    prompt, target = render_record(record, CLASSIFICATION, format_type)
    reconstructed = strip_record(prompt, target, CLASSIFICATION, format_type)
    assert reconstructed == record
