"""Reproduce ChurnGuard holdout, calibration, threshold and subgroup audits.

Usage:
    python evaluate_model.py --data "Customer Churn Data.xlsx"

This script retrains the documented pipeline on the same stratified split as
train_model.py. It does not change the deployed model or upload source data.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

from train_model import (
    DEFAULT_THRESHOLD, FEATURE_COLUMNS, ID_COL, NUMERIC_COLUMNS,
    TARGET, build_pipeline, clean_source,
)


def classification_metrics(y_true, scores, threshold):
    y_true = np.asarray(y_true, dtype=int)
    predicted = np.asarray(scores) >= float(threshold)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "churn_accounts": int(y_true.sum()),
        "churn_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "flagged_accounts": int(predicted.sum()),
        "flagged_rate": float(predicted.mean()),
    }


def evaluate(data_path: Path, output_path: Path):
    raw = pd.read_excel(data_path, sheet_name="Data for DSBA")
    clean = clean_source(raw)
    X = clean[FEATURE_COLUMNS].copy()
    for column in X.select_dtypes(include=["object", "string"]).columns:
        X[column] = X[column].map(
            lambda value: np.nan if pd.isna(value) else str(value)
        ).astype(object)
    y = clean[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    numeric = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    model = build_pipeline(numeric, CATEGORICAL_COLUMNS.copy())
    model.fit(X_train, y_train)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = model.predict_proba(X_test)[:, 1]

    prevalence = float(y_test.mean())
    report = {
        "audit_version": "1.0.0",
        "data": {
            "rows": int(len(clean)),
            "test_rows": int(len(y_test)),
            "churn_total": int(y.sum()),
            "nonchurn_total": int((1 - y).sum()),
            "split": "Stratified 80/20, random_state=42",
        },
        "overall": {
            "average_precision": float(average_precision_score(y_test, scores)),
            "roc_auc": float(roc_auc_score(y_test, scores)),
            "brier_score": float(brier_score_loss(y_test, scores)),
            "prevalence_baseline_brier": float(
                brier_score_loss(y_test, np.full(len(y_test), prevalence))
            ),
            "log_loss": float(log_loss(y_test, scores)),
            "test_churn_rate": prevalence,
            "threshold_0_275": classification_metrics(y_test, scores, 0.275),
            "threshold_0_5": classification_metrics(y_test, scores, 0.5),
        },
        "threshold_sweep": {},
        "calibration": {},
        "subgroup_metrics": {},
        "sensitive_feature_ablation": {},
        "limitations": [
            "Random holdout is not a future-period temporal validation.",
            "The source has no extraction timestamp, churn horizon, or feature availability timestamps.",
            "Scores are not certified as calibrated individual probabilities.",
            "Subgroup comparisons are descriptive and do not establish fairness or causality.",
            "No business intervention costs or outcomes are available.",
        ],
    }

    for threshold in [0.10, 0.15, 0.20, 0.25, 0.275, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80]:
        report["threshold_sweep"][str(threshold)] = classification_metrics(
            y_test, scores, threshold
        )

    observed, predicted = calibration_curve(
        y_test, scores, n_bins=10, strategy="quantile"
    )
    report["calibration"] = {
        "strategy": "10 quantile bins",
        "mean_predicted_score": predicted.tolist(),
        "observed_churn_rate": observed.tolist(),
        "interpretation": (
            "Descriptive reliability curve only. Quantile bins can have unequal "
            "widths; assess calibration on future data before interpreting scores "
            "as individual probabilities."
        ),
    }

    for column in ["Gender", "Marital_Status"]:
        report["subgroup_metrics"][column] = {}
        groups = X_test[column].astype("object").where(
            X_test[column].notna(), "Missing"
        )
        for value, indices in groups.groupby(groups).groups.items():
            positions = np.array([X_test.index.get_loc(index) for index in indices])
            group_y = y_test.loc[indices]
            group_scores = scores[positions]
            metrics = classification_metrics(group_y, group_scores, DEFAULT_THRESHOLD)
            metrics["minimum_group_size_met"] = metrics["n"] >= 20
            report["subgroup_metrics"][column][str(value)] = metrics

    # Retrain without the two demographic features to quantify their effect
    # on this split; this is a diagnostic, not a fairness certification.
    reduced_features = [
        c for c in FEATURE_COLUMNS if c not in {"Gender", "Marital_Status"}
    ]
    reduced_categoricals = [
        c for c in CATEGORICAL_COLUMNS if c not in {"Gender", "Marital_Status"}
    ]
    reduced_numeric = [c for c in reduced_features if c not in reduced_categoricals]
    reduced_model = build_pipeline(reduced_numeric, reduced_categoricals)
    reduced_model.fit(X_train[reduced_features], y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reduced_scores = reduced_model.predict_proba(X_test[reduced_features])[:, 1]
    report["sensitive_feature_ablation"] = {
        "excluded_features": ["Gender", "Marital_Status"],
        "average_precision": float(average_precision_score(y_test, reduced_scores)),
        "roc_auc": float(roc_auc_score(y_test, reduced_scores)),
        "threshold_0_275": classification_metrics(y_test, reduced_scores, 0.275),
        "interpretation": (
            "Single-split diagnostic only. A performance decrease does not by "
            "itself justify using demographic features."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved evaluation report: {output_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Customer Churn Data.xlsx")
    parser.add_argument("--output", default="artifacts/evaluation_report.json")
    args = parser.parse_args()
    evaluate(Path(args.data), Path(args.output))
