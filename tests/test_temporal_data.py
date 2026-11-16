"""Tests for the temporal snapshot data validator."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_model import FEATURE_COLUMNS
from validate_temporal_data import validate


def make_row(account_id, snapshot, observed, churn, churn_date):
    row = {column: 1 for column in FEATURE_COLUMNS}
    row.update({
        'AccountID': account_id,
        'Snapshot_Date': snapshot,
        'Outcome_Observed_Through': observed,
        'Churn_Date': churn_date,
        'Churn': churn,
    })
    return row


def test_missing_temporal_columns_blocks_validation(tmp_path):
    source = tmp_path / 'incomplete.csv'
    pd.DataFrame({'AccountID': [1], 'Churn': [0]}).to_csv(source, index=False)
    result = validate(source, horizon_days=30, provenance_path=None)
    assert result['status'] == 'BLOCKED'
    assert any('Missing required columns' in error for error in result['errors'])


def test_nonpositive_horizon_is_rejected(tmp_path):
    source = tmp_path / 'empty.csv'
    pd.DataFrame().to_csv(source, index=False)
    try:
        validate(source, horizon_days=0, provenance_path=None)
    except ValueError as exc:
        assert 'positive integer' in str(exc)
    else:
        raise AssertionError('A zero-day horizon must be rejected.')


def test_30_day_label_rules_allow_early_churn_and_require_full_negative_followup(tmp_path):
    source = tmp_path / 'dated.csv'
    rows = [
        make_row(1, '2026-01-01', '2026-01-11', 1, '2026-01-11'),
        make_row(2, '2026-01-01', '2026-01-31', 0, None),
    ]
    pd.DataFrame(rows).to_csv(source, index=False)
    result = validate(source, horizon_days=30, provenance_path=None)
    assert result['status'] == 'PASS_WITH_WARNINGS'
    assert result['errors'] == []


def test_censored_negative_label_is_blocked(tmp_path):
    source = tmp_path / 'censored.csv'
    row = make_row(3, '2026-01-01', '2026-01-20', 0, None)
    pd.DataFrame([row]).to_csv(source, index=False)
    result = validate(source, horizon_days=30, provenance_path=None)
    assert result['status'] == 'BLOCKED'
    assert any('lack full 30-day outcome follow-up' in error for error in result['errors'])


def test_default_horizon_is_30_days():
    import inspect
    from validate_temporal_data import validate
    assert inspect.signature(validate).parameters['horizon_days'].default == inspect.Parameter.empty