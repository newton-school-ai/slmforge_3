"""
src/slmforge/task/metrics.py
===========================

Per-task metric selector and registry.
Maps canonical task types to default evaluation metric suites.

Default Mapping:
- **classification**: accuracy, f1_macro, f1_micro
- **summarisation**: rouge1, rouge2, rougeL
- **qa**: exact_match, f1 (token overlap F1)
- **instruction**: rougeL, bleu
- **chat**: rougeL, bleu

Plug-in Interface:
- Register custom metrics using ``register_metric(name, function)``.
- Override or customize task suites using ``set_task_suite(task_type, metric_names)``.
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.metrics import accuracy_score, f1_score

from slmforge.task.detector import (
    CHAT,
    CLASSIFICATION,
    INSTRUCTION,
    QA,
    SUMMARISATION,
)


def accuracy_metric(predictions: list[str], references: list[str]) -> float:
    """Compute the accuracy score of the predictions against the references.

    Parameters
    ----------
    predictions : list[str]
        List of predicted strings.
    references : list[str]
        List of reference strings.

    Returns
    -------
    float
        Accuracy score between 0.0 and 1.0.
    """
    if not predictions or not references:
        return 0.0
    return float(accuracy_score(references, predictions))


def f1_macro_metric(predictions: list[str], references: list[str]) -> float:
    """Compute the macro F1 score.

    Parameters
    ----------
    predictions : list[str]
        List of predicted strings.
    references : list[str]
        List of reference strings.

    Returns
    -------
    float
        Macro F1 score between 0.0 and 1.0.
    """
    if not predictions or not references:
        return 0.0
    return float(f1_score(references, predictions, average="macro", zero_division=0.0))


def f1_micro_metric(predictions: list[str], references: list[str]) -> float:
    """Compute the micro F1 score.

    Parameters
    ----------
    predictions : list[str]
        List of predicted strings.
    references : list[str]
        List of reference strings.

    Returns
    -------
    float
        Micro F1 score between 0.0 and 1.0.
    """
    if not predictions or not references:
        return 0.0
    return float(f1_score(references, predictions, average="micro", zero_division=0.0))


def _compute_rouge(predictions: list[str], references: list[str], rouge_type: str) -> float:
    """Helper to compute average ROUGE scores using rouge-score library."""
    if not predictions or not references:
        return 0.0
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores.append(score[rouge_type].fmeasure)
    return float(sum(scores) / len(scores)) if scores else 0.0


def rouge1_metric(predictions: list[str], references: list[str]) -> float:
    """Compute the average ROUGE-1 F1 score.

    Parameters
    ----------
    predictions : list[str]
        List of predicted strings.
    references : list[str]
        List of reference strings.

    Returns
    -------
    float
        Average ROUGE-1 F-measure between 0.0 and 1.0.
    """
    return _compute_rouge(predictions, references, "rouge1")


def rouge2_metric(predictions: list[str], references: list[str]) -> float:
    """Compute the average ROUGE-2 F1 score.

    Parameters
    ----------
    predictions : list[str]
        List of predicted strings.
    references : list[str]
        List of reference strings.

    Returns
    -------
    float
        Average ROUGE-2 F-measure between 0.0 and 1.0.
    """
    return _compute_rouge(predictions, references, "rouge2")


def rougeL_metric(predictions: list[str], references: list[str]) -> float:
    """Compute the average ROUGE-L F1 score.

    Parameters
    ----------
    predictions : list[str]
        List of predicted strings.
    references : list[str]
        List of reference strings.

    Returns
    -------
    float
        Average ROUGE-L F-measure between 0.0 and 1.0.
    """
    return _compute_rouge(predictions, references, "rougeL")


def exact_match_metric(predictions: list[str], references: list[str]) -> float:
    """Compute exact match accuracy, ignoring leading/trailing whitespaces and case.

    Parameters
    ----------
    predictions : list[str]
        List of predicted strings.
    references : list[str]
        List of reference strings.

    Returns
    -------
    float
        Exact match score between 0.0 and 1.0.
    """
    if not predictions or not references:
        return 0.0
    matches = [
        1.0 if p.strip().lower() == r.strip().lower() else 0.0
        for p, r in zip(predictions, references)
    ]
    return float(sum(matches) / len(matches)) if matches else 0.0


def qa_f1_metric(predictions: list[str], references: list[str]) -> float:
    """Compute the token-level overlap F1 score (SQuAD style).

    Parameters
    ----------
    predictions : list[str]
        List of predicted strings.
    references : list[str]
        List of reference strings.

    Returns
    -------
    float
        Token-level F1 score between 0.0 and 1.0.
    """
    if not predictions or not references:
        return 0.0

    def compute_f1(pred: str, ref: str) -> float:
        pred_tokens = pred.strip().lower().split()
        ref_tokens = ref.strip().lower().split()
        if not pred_tokens or not ref_tokens:
            return 1.0 if pred_tokens == ref_tokens else 0.0
        common = set(pred_tokens) & set(ref_tokens)
        num_same = len(common)
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        return 2 * (precision * recall) / (precision + recall)

    scores = [compute_f1(p, r) for p, r in zip(predictions, references)]
    return float(sum(scores) / len(scores)) if scores else 0.0


def bleu_metric(predictions: list[str], references: list[str]) -> float:
    """Compute the normalized corpus BLEU score using sacrebleu library.

    Parameters
    ----------
    predictions : list[str]
        List of predicted strings.
    references : list[str]
        List of reference strings.

    Returns
    -------
    float
        BLEU score normalized to range [0.0, 1.0].
    """
    if not predictions or not references:
        return 0.0
    import sacrebleu

    try:
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        score = float(bleu.score / 100.0)
        return min(1.0, max(0.0, score))
    except (ValueError, TypeError, ZeroDivisionError, AttributeError):
        return 0.0


# Plug-in registry for metric functions
_METRIC_REGISTRY: dict[str, Callable[[list[str], list[str]], float]] = {
    "accuracy": accuracy_metric,
    "f1_macro": f1_macro_metric,
    "f1_micro": f1_micro_metric,
    "rouge1": rouge1_metric,
    "rouge2": rouge2_metric,
    "rougeL": rougeL_metric,
    "exact_match": exact_match_metric,
    "f1": qa_f1_metric,
    "bleu": bleu_metric,
}

# Task to metric names mapping
_TASK_METRIC_MAP: dict[str, list[str]] = {
    CLASSIFICATION: ["accuracy", "f1_macro", "f1_micro"],
    SUMMARISATION: ["rouge1", "rouge2", "rougeL"],
    QA: ["exact_match", "f1"],
    INSTRUCTION: ["rougeL", "bleu"],
    CHAT: ["rougeL", "bleu"],
}


def register_metric(name: str, fn: Callable[[list[str], list[str]], float]) -> None:
    """Register a custom metric callable under the given name.

    Parameters
    ----------
    name : str
        The metric identifier name.
    fn : Callable[[list[str], list[str]], float]
        A callable taking predictions and references and returning a float.
    """
    _METRIC_REGISTRY[name] = fn


def set_task_suite(task_type: str, metric_names: list[str]) -> None:
    """Override or set the active metrics suite for a task type.

    Parameters
    ----------
    task_type : str
        The task type.
    metric_names : list[str]
        List of metric names to map to this task type.
    """
    for name in metric_names:
        if name not in _METRIC_REGISTRY:
            raise ValueError(f"Metric '{name}' is not registered in the metric registry.")
    _TASK_METRIC_MAP[task_type] = list(metric_names)


def get_metric_suite(task_type: str) -> dict[str, Callable[[list[str], list[str]], float]]:
    """Get the mapping of metric name to callable for the given task type.

    Parameters
    ----------
    task_type : str
        The task type.

    Returns
    -------
    dict[str, Callable[[list[str], list[str]], float]]
        A dictionary of registered metric functions.
    """
    if task_type not in _TASK_METRIC_MAP:
        raise ValueError(f"Unsupported task type: {task_type}")

    names = _TASK_METRIC_MAP[task_type]
    return {name: _METRIC_REGISTRY[name] for name in names}
