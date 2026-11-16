"""Validate a dated ChurnGuard snapshot dataset before temporal evaluation.

The validator does not train a model. It checks that required dates exist,
labels have complete follow-up, snapshots are not duplicated, and a separate
feature-provenance CSV has no feature timestamps after each snapshot.

Example:
  python validate_temporal_data.py --data snapshots.csv
  python validate_temporal_data.py --data snapshots.xlsx --provenance feature_provenance.csv

Expected columns:
  AccountID, Snapshot_Date, Outcome_Observed_Through, Churn_Date, Churn,
  plus the 17 ChurnGuard model features.

Provenance CSV columns:
  AccountID, Snapshot_Date, Feature, Feature_Observed_At, Source_System,
  Transformation_Version
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from train_model import FEATURE_COLUMNS, ID_COL, TARGET


DATE_COLUMNS = ["Snapshot_Date", "Outcome_Observed_Through", "Churn_Date"]
REQUIRED = [ID_COL, *DATE_COLUMNS, TARGET, *FEATURE_COLUMNS]


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input must be a CSV or Excel workbook.")


def validate(data_path: Path, horizon_days: int, provenance_path: Path | None):
    if horizon_days <= 0:
        raise ValueError("--horizon-days must be a positive integer.")
    df = load_table(data_path)
    errors: list[str] = []
    warnings: list[str] = []

    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        errors.append(f"Missing required columns: {missing}")
    if missing:
        return {"status": "BLOCKED", "rows": int(len(df)), "errors": errors,
                "warnings": warnings, "checks": {}}

    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    if df[ID_COL].isna().any():
        errors.append("AccountID contains missing values.")
    if df["Snapshot_Date"].isna().any():
        errors.append("Snapshot_Date contains missing or invalid timestamps.")
    if df["Outcome_Observed_Through"].isna().any():
        errors.append("Outcome_Observed_Through contains missing or invalid timestamps.")
    if df[TARGET].isna().any() or not set(df[TARGET].dropna().unique()).issubset({0, 1}):
        errors.append("Churn must contain only binary 0/1 values without missing labels.")

    valid_dates = df["Snapshot_Date"].notna() & df["Outcome_Observed_Through"].notna()
    expected_end = df["Snapshot_Date"] + pd.to_timedelta(horizon_days, unit="D")
    churn_date = df["Churn_Date"]
    churn_label = pd.to_numeric(df[TARGET], errors="coerce")
    # Negative labels require full horizon follow-up. Positive labels only require
    # observation through the actual churn date, within the prediction horizon.
    incomplete = (
        valid_dates & (churn_label == 0) &
        (df["Outcome_Observed_Through"] < expected_end)
    )
    if incomplete.any():
        errors.append(
            f"{int(incomplete.sum())} rows lack full {horizon_days}-day outcome follow-up; "
            "mark them censored or exclude them rather than labelling them non-churn."
        )

    future_churn = df["Churn_Date"].notna() & df["Snapshot_Date"].notna() & (
        df["Churn_Date"] <= df["Snapshot_Date"]
    )
    if future_churn.any():
        errors.append(
            f"{int(future_churn.sum())} rows have Churn_Date on or before Snapshot_Date."
        )

    horizon_end = expected_end
    positive_invalid = (churn_label == 1) & (
        churn_date.isna() | (churn_date > horizon_end) |
        (churn_date <= df["Snapshot_Date"])
    )
    negative_invalid = (churn_label == 0) & (
        churn_date.notna() & (churn_date <= horizon_end)
    )
    positive_not_observed = (
        (churn_label == 1) & churn_date.notna() &
        (df["Outcome_Observed_Through"] < churn_date)
    )
    positive_invalid = positive_invalid | positive_not_observed
    if positive_invalid.any():
        errors.append(
            f"{int(positive_invalid.sum())} positive labels do not have a churn date "
            "inside the declared post-snapshot horizon with follow-up through that date."
        )
    if negative_invalid.any():
        errors.append(
            f"{int(negative_invalid.sum())} negative labels have a churn date "
            "inside the declared horizon."
        )

    duplicate_snapshots = int(df.duplicated([ID_COL, "Snapshot_Date"]).sum())
    if duplicate_snapshots:
        errors.append(
            f"{duplicate_snapshots} duplicate AccountID + Snapshot_Date rows found."
        )

    if (df["Snapshot_Date"].dropna() > pd.Timestamp.now(tz="UTC")).any():
        warnings.append("Some Snapshot_Date values are in the future.")

    provenance_summary = {"provided": provenance_path is not None}
    if provenance_path is not None:
        if not provenance_path.exists():
            errors.append(f"Provenance file not found: {provenance_path}")
        else:
            provenance = load_table(provenance_path)
            required_prov = [
                ID_COL, "Snapshot_Date", "Feature", "Feature_Observed_At",
                "Source_System", "Transformation_Version",
            ]
            missing_prov = sorted(set(required_prov) - set(provenance.columns))
            if missing_prov:
                errors.append(f"Provenance file missing columns: {missing_prov}")
            else:
                provenance["Snapshot_Date"] = pd.to_datetime(
                    provenance["Snapshot_Date"], errors="coerce", utc=True
                )
                provenance["Feature_Observed_At"] = pd.to_datetime(
                    provenance["Feature_Observed_At"], errors="coerce", utc=True
                )
                invalid_times = (
                    provenance["Snapshot_Date"].isna() |
                    provenance["Feature_Observed_At"].isna()
                )
                future_features = (
                    provenance["Feature_Observed_At"] > provenance["Snapshot_Date"]
                )
                invalid_names = ~provenance["Feature"].isin(FEATURE_COLUMNS)
                if invalid_times.any():
                    errors.append(
                        f"{int(invalid_times.sum())} provenance rows have invalid timestamps."
                    )
                if future_features.any():
                    errors.append(
                        f"{int(future_features.sum())} feature provenance rows use data "
                        "observed after the prediction snapshot."
                    )
                if invalid_names.any():
                    errors.append(
                        f"{int(invalid_names.sum())} provenance rows reference unknown features."
                    )
                expected = set(FEATURE_COLUMNS)
                covered = set(provenance["Feature"].dropna().unique())
                absent = sorted(expected - covered)
                if absent:
                    errors.append(f"No provenance entries supplied for features: {absent}")
                snapshot_keys = set(
                    zip(df[ID_COL].astype(str), df["Snapshot_Date"].astype(str))
                )
                provenance_keys = set(
                    zip(provenance[ID_COL].astype(str), provenance["Snapshot_Date"].astype(str))
                )
                missing_snapshot_keys = snapshot_keys - provenance_keys
                if missing_snapshot_keys:
                    errors.append(
                        f"Provenance is missing for {len(missing_snapshot_keys)} account snapshots."
                    )
                per_snapshot_coverage = provenance.groupby(
                    [provenance[ID_COL].astype(str), provenance["Snapshot_Date"].astype(str)]
                )["Feature"].nunique()
                incomplete_snapshots = int((per_snapshot_coverage < len(expected)).sum())
                if incomplete_snapshots:
                    errors.append(
                        f"{incomplete_snapshots} account snapshots do not have provenance "
                        "for all 17 model features."
                    )
                provenance_summary.update({
                    "rows": int(len(provenance)),
                    "features_covered": len(covered),
                    "features_expected": len(expected),
                    "future_feature_timestamp_rows": int(future_features.sum()),
                    "account_snapshots_with_incomplete_feature_lineage": incomplete_snapshots,
                    "account_snapshots_missing_lineage": len(missing_snapshot_keys),
                })
    else:
        warnings.append(
            "No feature-provenance file supplied. Snapshot dates alone cannot prove "
            "that every feature was available at prediction time."
        )

    summary = {
        "status": "PASS_WITH_WARNINGS" if not errors and warnings else (
            "PASS" if not errors else "BLOCKED"
        ),
        "rows": int(len(df)),
        "accounts": int(df[ID_COL].nunique(dropna=True)),
        "horizon_days": int(horizon_days),
        "snapshot_min": (
            df["Snapshot_Date"].min().isoformat()
            if df["Snapshot_Date"].notna().any() else None
        ),
        "snapshot_max": (
            df["Snapshot_Date"].max().isoformat()
            if df["Snapshot_Date"].notna().any() else None
        ),
        "churn_rate": (
            float(churn_label.mean()) if churn_label.notna().any() else None
        ),
        "duplicate_account_snapshot_rows": duplicate_snapshots,
        "provenance": provenance_summary,
        "errors": errors,
        "warnings": warnings,
        "note": (
            "Passing structural checks does not establish that the business churn "
            "definition is correct. A data owner must approve label semantics and "
            "feature lineage before temporal model evaluation."
        ),
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Dated CSV or Excel snapshot table")
    parser.add_argument(
        "--horizon-days", type=int, default=30,
        help="Prediction horizon in days (default: proposed 30-day horizon)",
    )
    parser.add_argument("--provenance", default=None, help="Optional feature-provenance CSV")
    parser.add_argument("--output", default="artifacts/temporal_data_validation.json")
    args = parser.parse_args()

    result = validate(
        Path(args.data), args.horizon_days,
        Path(args.provenance) if args.provenance else None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved validation report: {output}")
    raise SystemExit(0 if result["status"] != "BLOCKED" else 1)
