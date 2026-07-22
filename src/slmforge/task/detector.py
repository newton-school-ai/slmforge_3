"""
Task Type Detector Module

This module provides heuristics for detecting the NLP task type based on schema shape and content features.
The detector uses heuristics to identify task types from sample data in order to support a "just point at data" UX.

Supported Task Types:
- classification
- summarisation
- qa
- instruction
- chat

Heuristics documented:
- Classification: Looks for 'label', 'target', 'class', or 'category' columns. Boosts confidence if the number of unique labels is small.
- Summarisation: Looks for 'summary' or 'abstract' columns alongside 'text', 'document', or 'article' columns. Boosts confidence if the source text is longer than the summary on average.
- QA: Looks for 'question' and 'answer' columns. Presence of 'context' boosts confidence.
- Instruction: Looks for 'instruction' and 'response' (or 'output') columns. Presence of 'input' boosts confidence.
- Chat: Looks for 'messages' or 'conversations' columns. Boosts confidence if the content is a list of dicts with 'role' and 'content' keys.

Returns:
A dictionary containing:
- 'task': Detected task type (str)
- 'confidence': Confidence score between 0.0 and 1.0 (float)
- 'alternatives': List of alternative tasks with their scores
- 'fallback_prompt': A string containing a fallback prompt if confidence is low.
"""

from typing import List, Dict, Any

TASK_TYPES = ["classification", "summarisation", "qa", "instruction", "chat"]


def analyze_schema_and_content(sample_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Scores different tasks based on sample data."""
    scores = {t: 0.0 for t in TASK_TYPES}

    if not sample_data:
        return scores

    keys = list(sample_data[0].keys())
    key_lower = [str(k).lower() for k in keys]

    # 1. Classification Heuristic
    if any(k in key_lower for k in ["label", "target", "class", "category"]):
        scores["classification"] += 0.6
        label_key = next(
            (k for k, kl in zip(keys, key_lower) if kl in ["label", "target", "class", "category"]),
            None,
        )
        if label_key:
            unique_labels = set(
                str(row.get(label_key)) for row in sample_data if row.get(label_key) is not None
            )
            if 0 < len(unique_labels) <= 20:
                scores["classification"] += 0.3

    # 2. QA Heuristic
    if "question" in key_lower and "answer" in key_lower:
        scores["qa"] += 0.7
        if "context" in key_lower:
            scores["qa"] += 0.2

    # 3. Summarisation Heuristic
    if any(k in key_lower for k in ["summary", "abstract"]) and any(
        k in key_lower for k in ["text", "document", "article"]
    ):
        scores["summarisation"] += 0.7

        summary_key = next(
            (k for k, kl in zip(keys, key_lower) if kl in ["summary", "abstract"]), None
        )
        text_key = next(
            (k for k, kl in zip(keys, key_lower) if kl in ["text", "document", "article"]), None
        )

        if summary_key and text_key:
            summary_len = sum(len(str(row.get(summary_key, ""))) for row in sample_data)
            text_len = sum(len(str(row.get(text_key, ""))) for row in sample_data)

            if text_len > summary_len and text_len > 0:
                scores["summarisation"] += 0.2

    # 4. Instruction Heuristic
    if "instruction" in key_lower and ("response" in key_lower or "output" in key_lower):
        scores["instruction"] += 0.7
        if "input" in key_lower:
            scores["instruction"] += 0.2

    # 5. Chat Heuristic
    if "messages" in key_lower or "conversations" in key_lower:
        scores["chat"] += 0.7

        msg_key = next(
            (k for k, kl in zip(keys, key_lower) if kl in ["messages", "conversations"]), None
        )
        if msg_key:
            sample_val = sample_data[0].get(msg_key)
            if isinstance(sample_val, list) and len(sample_val) > 0:
                if (
                    isinstance(sample_val[0], dict)
                    and "role" in sample_val[0]
                    and "content" in sample_val[0]
                ):
                    scores["chat"] += 0.2

    # Normalize scores between 0 and 1 (cap at 1.0) and round to 2 decimals
    for k in scores:
        scores[k] = round(min(1.0, scores[k]), 2)

    return scores


def detect_task_type(sample_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Detects the task type from a sample of data.

    Args:
        sample_data: A list of dictionaries representing the rows of the dataset.

    Returns:
        Dict containing task, confidence, alternatives, and fallback_prompt.
    """
    scores = analyze_schema_and_content(sample_data)

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_task, best_score = sorted_scores[0]

    alternatives = [{"task": t, "confidence": s} for t, s in sorted_scores[1:] if s > 0]

    fallback_prompt = None
    if best_score < 0.6:
        fallback_prompt = (
            "Could not confidently determine the task type from the provided data schema. "
            "Please explicitly specify the task type (e.g., classification, summarisation, qa, instruction, chat)."
        )

    if best_score == 0.0:
        best_task = "unknown"

    return {
        "task": best_task,
        "confidence": best_score,
        "alternatives": alternatives,
        "fallback_prompt": fallback_prompt,
    }
