"""Tests for chronological evaluation metric calculations."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_temporal import metrics

def test_metrics_use_expected_confusion_matrix_order():
    result = metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.7, 0.2]), 0.5)
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (1, 1, 1, 1)
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["flagged_accounts"] == 2

def test_threshold_is_inclusive():
    result = metrics(np.array([0, 1]), np.array([0.3, 0.3]), 0.3)
    assert (result["tn"], result["fp"], result["fn"], result["tp"]) == (0, 1, 0, 1)