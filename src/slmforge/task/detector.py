"""
src/slmforge/task/detector.py
=============================

Task detector v1 for automatically classifying datasets into canonical task types:
- ``chat``
- ``qa``
- ``summarisation``
- ``classification``
- ``instruction``

Heuristics Overview
------------------
1. **chat**:
   - Matches if a dataset column (e.g., `messages`, `conversations`, `turns`, `dialogue`) contains lists of dictionaries.
   - Checks if these dictionaries contain conversation role structures, such as:
     - OpenAI format: `role` (user/assistant/system) and `content`.
     - ShareGPT format: `from` (human/gpt) and `value`.
     - Custom speaker formats: `speaker` and `text`.
   - Returns a confidence score of 1.0 if such structured conversational history is verified.

2. **qa**:
   - Look for question/answer key pairs (e.g., `question` paired with `answer`/`reply`, or `q` paired with `a`).
   - Analyzes content features: checks what fraction of values in the question column end with a question mark `?` or begin with common interrogative words (who, what, where, when, why, how, can, is, do, etc.).
   - Scores higher if a separate `context` column is also present.

3. **summarisation**:
   - Looks for document/summary key pairs (e.g., `document`, `article`, `text`, or `context` paired with `summary`, `abstract`, or `headline`).
   - Analyzes text length ratios: computes the average length of the document values and summary values, checking if the document is on average significantly longer than the summary (ratio >= 2.5) and the document has a minimum substantial average length.

4. **classification**:
   - Identifies candidate target/label columns containing small categorical sets (e.g., `label`, `class`, `category`, `sentiment`, `intent`, `target`).
   - Evaluates cardinality: checks if the unique value count is small (e.g., <= 20 or <= 15% of samples) and consists of short strings (average length <= 35 characters) or integers/booleans.

5. **instruction**:
   - Looks for standard instruction keys (e.g., `instruction`, `prompt`, `input`, `output`, `response`).
   - Verifies that the output column is not categorical (high cardinality / distinct responses).
   - Scans prompt content for active verbs/imperatives at the start of sentences (e.g., write, explain, summarize, translate, generate, list, etc.).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Canonical task types
CLASSIFICATION = "classification"
SUMMARISATION = "summarisation"
QA = "qa"
INSTRUCTION = "instruction"
CHAT = "chat"

ALL_TASK_TYPES = [CLASSIFICATION, SUMMARISATION, QA, INSTRUCTION, CHAT]

# Verbs indicative of instructions
INSTRUCTION_VERBS = {
    "write",
    "explain",
    "summarize",
    "summarise",
    "translate",
    "generate",
    "list",
    "create",
    "parse",
    "format",
    "find",
    "determine",
    "analyze",
    "analyse",
    "calculate",
    "describe",
    "convert",
    "extract",
    "compare",
    "classify",
    "evaluate",
    "predict",
    "correct",
    "rewrite",
    "extract",
}

# Question words indicative of QA
QUESTION_WORDS = {
    "who",
    "what",
    "where",
    "when",
    "why",
    "how",
    "which",
    "whom",
    "whose",
    "is",
    "are",
    "can",
    "do",
    "does",
    "did",
    "was",
    "were",
    "could",
    "should",
    "would",
}


def _get_keys_and_samples(
    records: Any, max_samples: int = 100
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Helper to extract unique keys and a subset of records from records input."""
    samples: List[Dict[str, Any]] = []
    keys: set[str] = set()

    # If it is a Hugging Face Dataset or similar that provides column_names
    if hasattr(records, "column_names") and isinstance(records.column_names, list):
        keys = set(records.column_names)

    # Convert to indexable or iterate
    count = 0
    try:
        n = len(records)
        limit = min(n, max_samples)
        for i in range(limit):
            rec = records[i]
            if isinstance(rec, dict):
                samples.append(rec)
                if not keys:
                    keys.update(rec.keys())
    except Exception:
        # Fallback to iteration
        for rec in records:
            if isinstance(rec, dict):
                samples.append(rec)
                if not keys:
                    keys.update(rec.keys())
                count += 1
                if count >= max_samples:
                    break

    if not keys and samples:
        for s in samples:
            keys.update(s.keys())

    return list(keys), samples


def detect_task(records: List[Dict[str, Any]] | Any) -> Dict[str, Any]:
    """Automatically detect the task type of a dataset of records.

    Parameters
    ----------
    records : List[Dict[str, Any]] or datasets.Dataset
        The list of dataset records (or Hugging Face Dataset) to analyze.

    Returns
    -------
    Dict[str, Any]
        A dictionary with the following keys:
        - "task_type": The predicted task type (str).
        - "confidence": The confidence score (float between 0.0 and 1.0).
        - "alternatives": A list of sorted (task_type, confidence) tuples of alternatives.
        - "fallback_prompt": A string explaining recommendations if confidence is low, else None.
    """
    keys, samples = _get_keys_and_samples(records)

    if not samples:
        return {
            "task_type": INSTRUCTION,
            "confidence": 0.0,
            "alternatives": [],
            "fallback_prompt": "No records found to analyze. Defaulting to 'instruction'.",
        }

    # Initialize scores for all task types
    scores: Dict[str, float] = {
        CHAT: 0.0,
        QA: 0.0,
        SUMMARISATION: 0.0,
        CLASSIFICATION: 0.0,
        INSTRUCTION: 0.0,
    }

    # Lowercase keys for case-insensitive matching
    lower_keys = [k.lower() for k in keys]
    key_map = {k.lower(): k for k in keys}

    # 1. Chat Heuristic
    # Look for list-of-dict conversation structures
    has_chat_keys = any(
        w in lower_keys for w in ["messages", "conversations", "dialogue", "turns", "chat"]
    )
    is_chat_struct = False

    for sample in samples:
        for key in keys:
            val = sample.get(key)
            if isinstance(val, list) and len(val) > 0:
                elem = val[0]
                if isinstance(elem, dict):
                    elem_keys = {ek.lower() for ek in elem.keys()}
                    # OpenAI (role, content), ShareGPT (from, value), or speaker/text
                    if (
                        {"role", "content"}.issubset(elem_keys)
                        or {"from", "value"}.issubset(elem_keys)
                        or {"speaker", "text"}.issubset(elem_keys)
                    ):
                        is_chat_struct = True
                        break
        if is_chat_struct:
            break

    if is_chat_struct:
        scores[CHAT] = 1.0
    elif has_chat_keys:
        scores[CHAT] = 0.8
    elif any("chat" in k or "dialogue" in k or "conversation" in k for k in lower_keys):
        scores[CHAT] = 0.5

    # 2. QA Heuristic
    q_keys = [key_map[k] for k in lower_keys if any(w in k for w in ["question", "query", "q"])]
    a_keys = [
        key_map[k]
        for k in lower_keys
        if any(w in k for w in ["answer", "reply", "a"])
        and not any(w in k for w in ["class", "label"])
    ]
    ctx_keys = [
        key_map[k] for k in lower_keys if any(w in k for w in ["context", "document", "passage"])
    ]

    # If instruction and response are present but contain QA markers
    if not q_keys and "prompt" in lower_keys and "response" in lower_keys:
        q_keys.append(key_map["prompt"])
    if not a_keys and "prompt" in lower_keys and "response" in lower_keys:
        a_keys.append(key_map["response"])

    qa_base = 0.0
    if q_keys and a_keys:
        qa_base = 0.7
        if ctx_keys:
            qa_base = 0.8
    elif q_keys:
        qa_base = 0.3
    elif a_keys:
        qa_base = 0.2

    # Check content features for QA
    qa_content_bonus = 0.0
    if q_keys:
        q_key = q_keys[0]
        q_vals = [s.get(q_key, "") for s in samples if s.get(q_key) is not None]
        if q_vals:
            q_str_vals = [str(v).strip().lower() for v in q_vals]
            ends_with_q = sum(1 for v in q_str_vals if v.endswith("?"))
            starts_with_q_word = sum(
                1
                for v in q_str_vals
                if any(v.startswith(w + " ") or v.startswith(w + "'") for w in QUESTION_WORDS)
            )

            total_q_checks = len(q_vals)
            q_ratio = (ends_with_q + starts_with_q_word) / total_q_checks
            if q_ratio > 0.6:
                qa_content_bonus = 0.2
            elif q_ratio > 0.3:
                qa_content_bonus = 0.1

    scores[QA] = min(1.0, qa_base + qa_content_bonus)

    # 3. Summarisation Heuristic
    doc_keys = [
        key_map[k]
        for k in lower_keys
        if any(w in k for w in ["document", "article", "text", "context", "body"])
    ]
    sum_keys = [
        key_map[k]
        for k in lower_keys
        if any(
            w in k
            for w in ["summary", "abstract", "headline", "title", "summarisation", "summarization"]
        )
    ]

    sum_base = 0.0
    if doc_keys and sum_keys:
        sum_base = 0.7
    elif sum_keys:
        sum_base = 0.4

    sum_content_bonus = 0.0
    if doc_keys and sum_keys:
        doc_key = doc_keys[0]
        sum_key = sum_keys[0]
        doc_lens = []
        sum_lens = []
        for s in samples:
            doc_val = s.get(doc_key, "")
            sum_val = s.get(sum_key, "")
            if doc_val and sum_val:
                doc_lens.append(len(str(doc_val)))
                sum_lens.append(len(str(sum_val)))
        if doc_lens and sum_lens:
            avg_doc_len = sum(doc_lens) / len(doc_lens)
            avg_sum_len = sum(sum_lens) / len(sum_lens)
            if avg_sum_len > 0:
                ratio = avg_doc_len / avg_sum_len
                # Summaries are shorter than documents
                if ratio >= 2.5 and avg_doc_len > 120:
                    sum_content_bonus = min(0.3, (ratio - 2.5) * 0.05 + 0.1)

    scores[SUMMARISATION] = min(1.0, sum_base + sum_content_bonus)

    # 4. Classification Heuristic
    best_class_score = 0.0
    for key in keys:
        vals = [s.get(key) for s in samples if s.get(key) is not None]
        if not vals:
            continue
        all_ints = all(isinstance(v, (int, bool)) for v in vals)
        str_vals = [str(v) for v in vals]
        avg_len = sum(len(v) for v in str_vals) / len(str_vals) if str_vals else 0.0

        # Filter out non-hashable values (lists, dicts) before creating the set
        hashable_vals = [v for v in vals if isinstance(v, (str, int, float, bool)) or v is None]
        unique_vals = set(hashable_vals)
        cardinality = len(unique_vals)

        if cardinality == 0:
            continue
        if cardinality == 1 and len(samples) > 5:
            # Constant value column, not a useful class target
            continue

        is_label_key = any(
            w in key.lower()
            for w in [
                "label",
                "class",
                "category",
                "sentiment",
                "intent",
                "target",
                "classification",
                "y",
            ]
        )

        c_score = 0.0
        # Determine if output behaves like discrete labels
        if all_ints and cardinality <= 20:
            c_score = 0.7
            if is_label_key:
                c_score += 0.25
        elif cardinality <= 20 or (len(samples) >= 10 and cardinality / len(samples) <= 0.15):
            if avg_len <= 35:
                c_score = 0.6
                if is_label_key:
                    c_score += 0.35
                if cardinality == 2:
                    c_score += 0.05

        # Avoid classification false positives for very small sample sets without clear label keys
        if len(samples) < 5 and not is_label_key:
            c_score = min(c_score, 0.3)

        best_class_score = max(best_class_score, c_score)

    scores[CLASSIFICATION] = min(1.0, best_class_score)

    # 5. Instruction Heuristic
    inst_keys = [
        key_map[k] for k in lower_keys if any(w in k for w in ["instruction", "prompt", "input"])
    ]
    out_keys = [
        key_map[k]
        for k in lower_keys
        if any(w in k for w in ["output", "response"])
        and not any(w in k for w in ["class", "label"])
    ]

    inst_base = 0.0
    if inst_keys and out_keys:
        inst_base = 0.7
    elif inst_keys:
        inst_base = 0.4

    inst_content_bonus = 0.0
    if inst_keys:
        inst_key = inst_keys[0]
        inst_vals = [s.get(inst_key, "") for s in samples if s.get(inst_key) is not None]
        if inst_vals:
            inst_str_vals = [str(v).strip().lower() for v in inst_vals]
            starts_with_verb = sum(
                1 for v in inst_str_vals if any(v.startswith(w + " ") for w in INSTRUCTION_VERBS)
            )
            verb_ratio = starts_with_verb / len(inst_str_vals)
            if verb_ratio >= 0.2:
                inst_content_bonus = 0.2
            elif verb_ratio >= 0.05:
                inst_content_bonus = 0.1

    # Verify that the output keys are not categorical (ensure it's not classification)
    if out_keys:
        out_key = out_keys[0]
        out_vals = [s.get(out_key) for s in samples if s.get(out_key) is not None]
        if out_vals:
            unique_out = len(set(out_vals))
            out_cardinality_ratio = unique_out / len(out_vals) if out_vals else 0.0
            # If the output is highly repetitive and labels, penalize instruction
            if out_cardinality_ratio <= 0.15 and len(out_vals) >= 10:
                inst_base -= 0.4

    scores[INSTRUCTION] = max(0.0, min(1.0, inst_base + inst_content_bonus))

    # Determine predicted task and alternatives
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    predicted_task, confidence = sorted_scores[0]

    # If confidence is 0.0, fall back to instruction
    if confidence == 0.0:
        predicted_task = INSTRUCTION

    alternatives = [(t, float(s)) for t, s in sorted_scores if t != predicted_task and s > 0.0]

    fallback_prompt = None
    if confidence < 0.5:
        alt_str = ", ".join([f"'{t}' (conf: {s:.2f})" for t, s in alternatives])
        fallback_prompt = (
            f"Low confidence ({confidence:.2f}) in automatic task detection. "
            f"Predicted '{predicted_task}' as the best guess. "
            f"Alternative tasks found: [{alt_str}]. "
            "To override this automatic prediction, run your build with the '--task' option, "
            "e.g., `slmforge build --task <task_type>`."
        )

    return {
        "task_type": predicted_task,
        "confidence": float(confidence),
        "alternatives": alternatives,
        "fallback_prompt": fallback_prompt,
    }
