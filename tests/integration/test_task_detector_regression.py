"""
tests/integration/test_task_detector_regression.py
==================================================

Regression suite for Task Detector. Loads labeled samples from `tests/fixtures/task_detection/`
and asserts that the task detector achieves at least 85% accuracy on individual samples.
"""

from __future__ import annotations

import json
from pathlib import Path

from slmforge.task import (
    detect_task,
    CLASSIFICATION,
    SUMMARISATION,
    QA,
    INSTRUCTION,
    CHAT,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "task_detection"


def test_detector_regression_accuracy() -> None:
    """Load all labeled fixtures and assert >= 85% accuracy on individual samples."""
    task_types = [CLASSIFICATION, SUMMARISATION, QA, INSTRUCTION, CHAT]

    total_samples = 0
    correct_samples = 0

    for task_type in task_types:
        fixture_path = FIXTURES_DIR / f"{task_type}.jsonl"
        assert fixture_path.exists(), f"Missing fixture file: {fixture_path}"

        # Load records
        records = []
        with open(fixture_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        # 1. Verify detection of the entire dataset as a whole
        dataset_res = detect_task(records)
        assert dataset_res["task_type"] == task_type, (
            f"Failed to detect correct task type for entire {task_type} dataset. "
            f"Got: {dataset_res['task_type']}"
        )

        # 2. Evaluate accuracy on individual samples
        for record in records:
            total_samples += 1
            res = detect_task([record])
            if res["task_type"] == task_type:
                correct_samples += 1
            else:
                # Log incorrect prediction for debugging/transparency
                print(
                    f"[Mismatch] Expected {task_type}, predicted {res['task_type']} for record: {record}"
                )

    accuracy = correct_samples / total_samples
    print("\nTask Detector Regression Summary:")
    print(f"Total samples: {total_samples}")
    print(f"Correct: {correct_samples}")
    print(f"Accuracy: {accuracy:.2%}")

    assert total_samples >= 30, f"Expected at least 30 labeled samples, found {total_samples}"
    assert accuracy >= 0.85, f"Task detector accuracy fell below 85% threshold. Got: {accuracy:.2%}"
