"""Run a chronological 30-day churn evaluation on validated snapshots."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss, confusion_matrix
from train_model import (FEATURE_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, RENAME_MAP, SENTINELS,
                         build_pipeline, DEFAULT_THRESHOLD)
from validate_temporal_data import validate

def prepare_features(frame):
    data = frame.rename(columns=RENAME_MAP).copy()
    for col in data.select_dtypes(include=["object", "string"]).columns:
        data[col] = data[col].astype("string").str.strip()
    for col in NUMERIC_COLUMNS:
        values = data[col].astype("string").str.strip()
        data[col] = pd.to_numeric(values.mask(values.isin(SENTINELS)), errors="coerce")
    data["Login_device"] = data["Login_device"].astype("string").str.strip()
    data["Login_device"] = data["Login_device"].mask(data["Login_device"].eq("&&&&"), pd.NA)
    data["Gender"] = data["Gender"].replace({"M": "Male", "F": "Female"})
    data["account_segment"] = data["account_segment"].replace({"Regular +": "Regular Plus", "Super +": "Super Plus"})
    X = data[FEATURE_COLUMNS].copy()
    for col in X.select_dtypes(include=["object", "string"]).columns:
        X[col] = X[col].map(lambda value: np.nan if pd.isna(value) else str(value)).astype(object)
    return X

def metrics(y, scores, threshold):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"average_precision": float(average_precision_score(y, scores)),
            "roc_auc": float(roc_auc_score(y, scores)),
            "brier_score": float(brier_score_loss(y, scores)),
            "log_loss": float(log_loss(y, scores, labels=[0, 1])),
            "threshold": float(threshold), "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
            "precision": float(tp / (tp + fp)) if tp + fp else None,
            "recall": float(tp / (tp + fn)) if tp + fn else None,
            "flagged_accounts": int(pred.sum())}

def evaluate(args):
    path = Path(args.data)
    raw = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    validation = validate(path, args.horizon_days, Path(args.provenance) if args.provenance else None)
    if validation["status"] == "BLOCKED":
        raise ValueError("Temporal data validation is BLOCKED: " + "; ".join(validation["errors"]))
    required_dates = ["Snapshot_Date", "Outcome_Observed_Through"]
    for col in required_dates:
        raw[col] = pd.to_datetime(raw[col], errors="coerce", utc=True)
    train_cutoff = pd.to_datetime(args.train_cutoff, utc=True)
    test_start = pd.to_datetime(args.test_start, utc=True)
    test_end = pd.to_datetime(args.test_end, utc=True)
    evaluation_as_of = pd.to_datetime(args.evaluation_as_of, utc=True)
    if not train_cutoff < test_start <= test_end <= evaluation_as_of:
        raise ValueError("Dates must satisfy train-cutoff < test-start <= test-end <= evaluation-as-of.")
    train = raw[(raw["Snapshot_Date"] <= train_cutoff) & (raw["Outcome_Observed_Through"] <= train_cutoff)].copy()
    test = raw[(raw["Snapshot_Date"] >= test_start) & (raw["Snapshot_Date"] <= test_end) &
               (raw["Outcome_Observed_Through"] <= evaluation_as_of)].copy()
    train = train.sort_values("Snapshot_Date").drop_duplicates("AccountID", keep="last")
    test = test.sort_values("Snapshot_Date").drop_duplicates("AccountID", keep="first")
    overlap = set(train["AccountID"].astype(str)) & set(test["AccountID"].astype(str))
    if args.account_scope == "unseen":
        test = test[~test["AccountID"].astype(str).isin(overlap)].copy()
    else:
        test = test[test["AccountID"].astype(str).isin(overlap)].copy()
    if train.empty or test.empty:
        raise ValueError("The selected date windows and account scope produced an empty train or test set.")
    if train["Churn"].nunique() < 2 or test["Churn"].nunique() < 2:
        raise ValueError("Both training and test partitions must contain churn and non-churn examples.")
    X_train, X_test = prepare_features(train), prepare_features(test)
    y_train, y_test = train["Churn"].astype(int), test["Churn"].astype(int)
    numeric = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_COLUMNS]
    model = build_pipeline(numeric, CATEGORICAL_COLUMNS.copy())
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    report = {"evaluation_type": "chronological_account_level", "horizon_days": args.horizon_days,
              "train_cutoff": train_cutoff.isoformat(), "test_start": test_start.isoformat(),
              "test_end": test_end.isoformat(), "evaluation_as_of": evaluation_as_of.isoformat(),
              "account_scope": args.account_scope, "threshold": DEFAULT_THRESHOLD,
              "train_rows_after_one_snapshot_per_account": int(len(train)),
              "test_rows_after_one_snapshot_per_account": int(len(test)),
              "train_accounts": int(train["AccountID"].nunique()),
              "test_accounts": int(test["AccountID"].nunique()),
              "account_ids_overlapping_before_scope_filter": int(len(overlap)),
              "train_churn_rate": float(y_train.mean()), "test_churn_rate": float(y_test.mean()),
              "metrics": metrics(y_test, scores, DEFAULT_THRESHOLD),
              "data_validation_status": validation["status"],
              "warning": "Threshold 0.275 is held fixed for comparability; select operational thresholds using training/validation data and business costs, never this test partition.",
              "limitations": ["This is only as reliable as the supplied churn labels and feature timestamps.",
                             "Account-scope mode evaluates either unseen or returning accounts, not both.",
                             "The report contains aggregate metrics only; raw account records are not written."]}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)); print("Saved:", out)
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--train-cutoff", required=True, help="UTC date or ISO timestamp")
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--evaluation-as-of", required=True, help="Latest date through which outcomes are observed")
    parser.add_argument("--horizon-days", type=int, default=30)
    parser.add_argument("--account-scope", choices=["unseen", "returning"], default="unseen")
    parser.add_argument("--provenance", default=None)
    parser.add_argument("--output", default="artifacts/temporal_evaluation_report.json")
    evaluate(parser.parse_args())