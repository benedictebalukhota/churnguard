"""Unit tests for ChurnGuard evaluation metric calculations."""
import numpy as np

from evaluate_model import classification_metrics


def test_confusion_matrix_and_rates_are_consistent():
    result = classification_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.10, 0.80, 0.70, 0.20]),
        0.50,
    )
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (1, 1, 1, 1)
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["flagged_accounts"] == 2


def test_threshold_includes_equal_scores():
    result = classification_metrics(
        np.array([0, 1]),
        np.array([0.275, 0.275]),
        0.275,
    )
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (0, 1, 0, 1)
