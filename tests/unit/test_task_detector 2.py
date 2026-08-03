"""tests/unit/test_task_detector.py.

Unit tests for the Task Detector (`src/slmforge/task/detector.py`).
"""

from __future__ import annotations

from slmforge.task import (
    CHAT,
    CLASSIFICATION,
    INSTRUCTION,
    QA,
    SUMMARISATION,
    detect_task,
)


def test_detect_chat() -> None:
    # OpenAI role/content format
    chat_records_openai = [
        {
            "messages": [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris is the capital of France."},
            ],
        },
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        },
    ]
    res = detect_task(chat_records_openai)
    assert res["task_type"] == CHAT
    assert res["confidence"] == 1.0
    assert not res["fallback_prompt"]

    # ShareGPT from/value format
    chat_records_sharegpt = [
        {
            "conversations": [
                {"from": "human", "value": "What is the capital of France?"},
                {"from": "gpt", "value": "Paris."},
            ],
        },
    ]
    res2 = detect_task(chat_records_sharegpt)
    assert res2["task_type"] == CHAT
    assert res2["confidence"] == 1.0


def test_detect_qa() -> None:
    # Matching QA fields with question marks
    qa_records = [
        {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
        {"question": "How far is the moon?", "answer": "384,400 km"},
        {"question": "Why is the sky blue?", "answer": "Rayleigh scattering"},
        {"question": "What is water made of?", "answer": "Hydrogen and Oxygen"},
    ]
    res = detect_task(qa_records)
    assert res["task_type"] == QA
    assert res["confidence"] >= 0.7


def test_detect_summarisation() -> None:
    # Document/summary pairs with long document and short summary
    sum_records = [
        {
            "document": (
                "The Hubble Space Telescope has captured a stunning new image of a distant galaxy. "
                "The galaxy, located millions of light-years away, shows vibrant star-forming regions "
                "and intricate dust lanes. Scientists say this image provides crucial clues about "
                "the early universe and galactic evolution."
            ),
            "summary": "Hubble captures new galaxy image aiding star formation study.",
        },
        {
            "document": (
                "A new medical trial has shown promising results for an Alzheimer's drug. "
                "The drug target beta-amyloid plaques in the brain, slowing cognitive decline "
                "by nearly thirty percent in early-stage patients. Researchers call it a major "
                "breakthrough, though further trials are needed to confirm safety and long-term efficacy."
            ),
            "summary": "Alzheimer's drug slows cognitive decline in trial.",
        },
    ]
    res = detect_task(sum_records)
    assert res["task_type"] == SUMMARISATION
    assert res["confidence"] >= 0.7


def test_detect_classification() -> None:
    # Key label/class with discrete unique labels
    class_records = [
        {"text": "This product is absolutely amazing!", "label": "positive"},
        {"text": "Worst experience ever.", "label": "negative"},
        {"text": "It was okay, nothing special.", "label": "neutral"},
        {"text": "Highly recommend buying this.", "label": "positive"},
        {"text": "Terrible quality, broke immediately.", "label": "negative"},
    ]
    res = detect_task(class_records)
    assert res["task_type"] == CLASSIFICATION
    assert res["confidence"] >= 0.7


def test_detect_instruction() -> None:
    # Prompt/response pairs starting with instruction verbs
    inst_records = [
        {
            "instruction": "Write a python function to add two numbers.",
            "response": "def add(a, b):\n    return a + b",
        },
        {
            "instruction": "Explain the concept of quantum computing in simple terms.",
            "response": "Quantum computing is a type of computing...",
        },
        {
            "instruction": "List five fruits that are high in vitamin C.",
            "response": "1. Orange\n2. Kiwi\n3. Strawberry\n4. Guava\n5. Papaya",
        },
        {
            "instruction": "Create a CSS stylesheet for a dark-mode website layout.",
            "response": "body { background-color: #121212; color: #ffffff; }",
        },
    ]
    res = detect_task(inst_records)
    assert res["task_type"] == INSTRUCTION
    assert res["confidence"] >= 0.7


def test_empty_records() -> None:
    res = detect_task([])
    assert res["task_type"] == INSTRUCTION
    assert res["confidence"] == 0.0
    assert res["fallback_prompt"] is not None


def test_low_confidence_fallback() -> None:
    # Unclear columns, short texts, no matches
    ambiguous_records = [
        {"data_field_1": "hello world", "data_field_2": "foo bar"},
        {"data_field_1": "another data sample", "data_field_2": "some value"},
    ]
    res = detect_task(ambiguous_records)
    # The confidence should be low (< 0.5)
    assert res["confidence"] < 0.5
    assert res["fallback_prompt"] is not None
    # Check that alternatives are populated
    assert isinstance(res["alternatives"], list)
