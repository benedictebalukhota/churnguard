"""Tests for the temporal snapshot data validator."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate_temporal_data import validate


def test_missing_temporal_columns_blocks_validation(tmp_path):
    source = tmp_path / "incomplete.csv"
    pd.DataFrame({"AccountID": [1], "Churn": [0]}).to_csv(source, index=False)

    result = validate(source, horizon_days=30, provenance_path=None)

    assert result["status"] == "BLOCKED"
    assert any("Missing required columns" in error for error in result["errors"])


def test_nonpositive_horizon_is_rejected(tmp_path):
    source = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(source, index=False)

    try:
        validate(source, horizon_days=0, provenance_path=None)
    except ValueError as exc:
        assert "positive integer" in str(exc)
    else:
        raise AssertionError("A zero-day horizon must be rejected.")
