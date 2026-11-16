"""Audit duplicate-feature overlap and AccountID-order sensitivity.

Usage:
    python audit_data.py --data "Customer Churn Data.xlsx"

AccountID is not a timestamp. The ordered splits below are stress tests only,
not temporal validation. The source workbook is never written to the report.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from train_model import (
    CATEGORICAL_COLUMNS, DEFAULT_THRESHOLD, FEATURE_COLUMNS, ID_COL,
    NUMERIC_COLUMNS, TARGET, LIGHTGBM_PARAMS, build_pipeline, clean_source,
)


def score_metrics(y_true, scores):
    y_true = np.asarray(y_true, dtype=int)
    predicted = np.asarray(scores) >= DEFAULT_THRESHOLD
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "churn_rate": float(y_true.mean()),
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "threshold": DEFAULT_THRESHOLD,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
    }


def fit_and_score(X, y, train_idx, test_idx):
    numeric = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    model = build_pipeline(numeric, CATEGORICAL_COLUMNS.copy())
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = model.predict_proba(X.iloc[test_idx])[:, 1]
    return score_metrics(y.iloc[test_idx], scores)


def audit(data_path: Path, output_path: Path):
    raw = pd.read_excel(data_path, sheet_name="Data for DSBA")
    clean = clean_source(raw)
    X = clean[FEATURE_COLUMNS].copy()
    for column in X.select_dtypes(include=["object", "string"]).columns:
        X[column] = X[column].map(
            lambda value: np.nan if pd.isna(value) else str(value)
        ).astype(object)
    y = clean[TARGET].astype(int)

    # Hash feature values only (not AccountID or target) so exact duplicate
    # model inputs are grouped even if they belong to different account IDs.
    feature_hash = pd.util.hash_pandas_object(
        X.astype("string").fillna("<MISSING>"), index=False
    ).astype("uint64")
    group_sizes = feature_hash.value_counts()
    duplicate_groups = group_sizes[group_sizes > 1]
    conflicting_groups = (
        pd.DataFrame({"group": feature_hash, "target": y})
        .groupby("group")["target"].nunique()
    )
    conflicting_groups = int((conflicting_groups > 1).sum())

    # Reproduce the standard random split to quantify duplicate overlap.
    from sklearn.model_selection import train_test_split
    all_idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=0.20, stratify=y, random_state=42
    )
    train_hashes = set(feature_hash.iloc[train_idx])
    test_hashes = set(feature_hash.iloc[test_idx])
    overlap_hashes = train_hashes.intersection(test_hashes)
    test_rows_with_train_duplicate = int(
        feature_hash.iloc[test_idx].isin(train_hashes).sum()
    )

    # Group-aware split prevents identical feature vectors crossing partitions.
    grouped_split = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=42
    )
    grouped_train, grouped_test = next(
        grouped_split.split(X, y, groups=feature_hash)
    )
    grouped_metrics = fit_and_score(X, y, grouped_train, grouped_test)

    # AccountID is monotonic in the workbook, but is not a date. These
    # contiguous splits are only sensitivity checks for ordering effects.
    order = np.argsort(clean[ID_COL].to_numpy())
    cut = int(0.80 * len(order))
    id_train, id_test = order[:cut], order[cut:]
    id_forward = fit_and_score(X, y, id_train, id_test)
    id_reverse = fit_and_score(X, y, id_test, id_train)

    numeric_corr = clean[NUMERIC_COLUMNS + [TARGET]].corr(
        numeric_only=True
    )[TARGET].drop(TARGET)
    numeric_corr = numeric_corr.reindex(
        numeric_corr.abs().sort_values(ascending=False).index
    )

    id_bins = pd.qcut(clean[ID_COL], 10, duplicates="drop")
    id_bin_rates = (
        pd.DataFrame({"id_bin": id_bins.astype(str), "churn": y})
        .groupby("id_bin", sort=False)["churn"]
        .agg(count="count", churn_rate="mean")
        .reset_index()
        .to_dict(orient="records")
    )

    report = {
        "audit_version": "1.0.0",
        "source": {
            "rows": int(len(clean)),
            "account_id_unique": bool(clean[ID_COL].is_unique),
            "account_id_monotonic_increasing": bool(
                clean[ID_COL].is_monotonic_increasing
            ),
            "account_id_target_pearson": float(
                clean[ID_COL].corr(clean[TARGET])
            ),
            "account_id_target_spearman": float(
                clean[ID_COL].corr(clean[TARGET], method="spearman")
            ),
            "account_id_quantile_churn_rates": id_bin_rates,
        },
        "duplicate_feature_audit": {
            "duplicate_feature_groups": int(len(duplicate_groups)),
            "rows_in_duplicate_groups": int(duplicate_groups.sum()),
            "largest_group_size": int(group_sizes.max()),
            "conflicting_target_groups": conflicting_groups,
            "random_split_duplicate_hashes_crossing_partitions": int(
                len(overlap_hashes)
            ),
            "random_split_test_rows_with_training_duplicate": (
                test_rows_with_train_duplicate
            ),
            "interpretation": (
                "Exact repeated feature vectors can make a random holdout "
                "optimistic. Group-aware evaluation keeps them in one partition."
            ),
        },
        "random_stratified_holdout": fit_and_score(X, y, train_idx, test_idx),
        "group_aware_holdout": grouped_metrics,
        "accountid_order_sensitivity_only": {
            "first_80_percent_ids_train_last_20_percent_ids_test": id_forward,
            "last_20_percent_ids_train_first_80_percent_ids_test": id_reverse,
            "warning": (
                "AccountID is not a timestamp. These results are ordering "
                "stress tests, not temporal validation or evidence of drift."
            ),
        },
        "numeric_feature_target_pearson": {
            str(k): float(v) for k, v in numeric_corr.items()
        },
        "limitations": [
            "No extraction date, observation timestamp, or churn prediction horizon is available.",
            "Feature association does not establish target leakage or causality.",
            "The dataset's provenance and feature-generation process require confirmation.",
            "Group-aware holdout addresses exact duplicate feature vectors only, not near-duplicates or temporal leakage.",
            "A future-dated, point-in-time dataset is required for temporal validation.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved data audit: {output_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Customer Churn Data.xlsx")
    parser.add_argument("--output", default="artifacts/data_audit_report.json")
    args = parser.parse_args()
    audit(Path(args.data), Path(args.output))
