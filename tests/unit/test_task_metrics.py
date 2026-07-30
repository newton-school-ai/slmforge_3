"""
tests/unit/test_task_metrics.py
================================
Unit tests for the task metric selector, metric definitions, and plug-in registry.
"""

from __future__ import annotations

import pytest

from slmforge.task import (
    CHAT,
    CLASSIFICATION,
    INSTRUCTION,
    QA,
    SUMMARISATION,
    get_metric_suite,
    register_metric,
    set_task_suite,
)


def test_metric_suites_non_empty() -> None:
    """Each canonical task type must return a non-empty metrics suite."""
    for task_type in [CLASSIFICATION, SUMMARISATION, QA, INSTRUCTION, CHAT]:
        suite = get_metric_suite(task_type)
        assert len(suite) > 0
        for name, fn in suite.items():
            assert callable(fn)
            assert isinstance(name, str)


def test_accuracy_metric() -> None:
    """Verify accuracy metric works with standard label formats."""
    suite = get_metric_suite(CLASSIFICATION)
    acc = suite["accuracy"]

    preds = ["positive", "negative", "neutral"]
    refs = ["positive", "positive", "neutral"]
    # 2 out of 3 match => 2/3 approx 0.6666...
    score = acc(preds, refs)
    assert abs(score - 2 / 3) < 1e-4

    # Empty inputs return 0.0
    assert acc([], []) == 0.0


def test_f1_metrics() -> None:
    """Verify macro and micro F1 scores."""
    suite = get_metric_suite(CLASSIFICATION)
    f1_macro = suite["f1_macro"]
    f1_micro = suite["f1_micro"]

    preds = ["a", "b", "c"]
    refs = ["a", "a", "c"]

    score_macro = f1_macro(preds, refs)
    score_micro = f1_micro(preds, refs)

    assert 0.0 <= score_macro <= 1.0
    assert 0.0 <= score_micro <= 1.0
    # Micro F1 for single-label multiclass equals accuracy
    assert abs(score_micro - 2 / 3) < 1e-4


def test_rouge_metrics() -> None:
    """Verify ROUGE scores return floats in standard range."""
    suite = get_metric_suite(SUMMARISATION)
    r1 = suite["rouge1"]
    r2 = suite["rouge2"]
    rl = suite["rougeL"]

    preds = ["The quick brown fox jumps over the lazy dog."]
    refs = ["A quick brown fox jumped over a lazy dog."]

    assert 0.0 <= r1(preds, refs) <= 1.0
    assert 0.0 <= r2(preds, refs) <= 1.0
    assert 0.0 <= rl(preds, refs) <= 1.0

    # Test exact match
    assert abs(rl(preds, preds) - 1.0) < 1e-4


def test_exact_match_metric() -> None:
    """Verify exact match strips whitespace and is case-insensitive."""
    suite = get_metric_suite(QA)
    em = suite["exact_match"]

    preds = ["  Tokyo  ", "London", "paris"]
    refs = ["tokyo", " London ", "Paris"]

    assert em(preds, refs) == 1.0
    assert em(["Tokyo"], ["Paris"]) == 0.0


def test_qa_f1_metric() -> None:
    """Verify SQuAD-style token-overlap F1 score."""
    suite = get_metric_suite(QA)
    f1 = suite["f1"]

    # 1. Exact match F1
    assert abs(f1(["hello world"], ["hello world"]) - 1.0) < 1e-4

    # 2. Token overlap F1
    # pred: "hello world" (2 tokens)
    # ref: "hello there world" (3 tokens)
    # common: {"hello", "world"} (2 tokens)
    # precision = 2/2 = 1.0
    # recall = 2/3
    # F1 = 2 * (1.0 * 2/3) / (1.0 + 2/3) = (4/3) / (5/3) = 0.8
    score = f1(["hello world"], ["hello there world"])
    assert abs(score - 0.8) < 1e-4

    # 3. No overlap
    assert f1(["hello"], ["world"]) == 0.0


def test_bleu_metric() -> None:
    """Verify sacrebleu corpus BLEU score runs and is normalized to [0.0, 1.0]."""
    suite = get_metric_suite(INSTRUCTION)
    bleu = suite["bleu"]

    preds = ["the cat is on the mat"]
    refs = ["the cat is on the mat"]

    score = bleu(preds, refs)
    # BLEU score for exact match should be 1.0 (sacrebleu score 100 normalized by /100)
    assert abs(score - 1.0) < 1e-2
    assert 0.0 <= score <= 1.0


def test_plugin_registration() -> None:
    """Verify that we can register custom metrics and map them to task types."""

    # 1. Register custom metric
    def custom_metric(preds: list[str], refs: list[str]) -> float:
        return 0.42

    register_metric("my_custom_metric", custom_metric)

    # 2. Set suite for a task
    set_task_suite(CLASSIFICATION, ["accuracy", "my_custom_metric"])

    suite = get_metric_suite(CLASSIFICATION)
    assert "accuracy" in suite
    assert "my_custom_metric" in suite
    assert "f1_macro" not in suite  # was removed by override
    assert suite["my_custom_metric"]([], []) == 0.42

    # 3. Setting unregistered metric should raise ValueError
    with pytest.raises(ValueError, match="is not registered"):
        set_task_suite(CLASSIFICATION, ["unregistered_metric"])

    # 4. Requesting unsupported task type should raise ValueError
    with pytest.raises(ValueError, match="Unsupported task type"):
        get_metric_suite("invalid_task_type")

    # Restore default classification suite for consistency in subsequent tests
    set_task_suite(CLASSIFICATION, ["accuracy", "f1_macro", "f1_micro"])
