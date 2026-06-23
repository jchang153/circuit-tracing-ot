"""Decoded-text checkers for relaxed MCQA evaluation and filtering."""

from __future__ import annotations

from collections.abc import Sequence


def causalab_substring_checker(neural_output: object, causal_output: object) -> bool:
    """Match causal-abstractions-ot's relaxed decoded-text checker."""
    neural_text = str(neural_output)
    causal_text = str(causal_output)
    return causal_text in neural_text or neural_text in causal_text


def checker_accuracy(predicted_texts: Sequence[object], expected_texts: Sequence[object]) -> float:
    """Average the relaxed substring checker over decoded predictions."""
    total = len(expected_texts)
    if total == 0:
        return 0.0
    correct = sum(
        int(causalab_substring_checker(predicted, expected))
        for predicted, expected in zip(predicted_texts, expected_texts)
    )
    return float(correct) / float(total)


def selection_metric_from_metrics(metrics: dict[str, object]) -> tuple[str, float]:
    """Return the preferred scalar metric for calibration selection."""
    if "checker_acc" in metrics:
        return "checker_acc", float(metrics["checker_acc"])
    return "exact_acc", float(metrics.get("exact_acc", 0.0))
