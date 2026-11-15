"""Train the ChurnGuard LightGBM pipeline using the Milestone 2 design.

Usage:
    python train_model.py --data "Customer Churn Data.xlsx"

The source workbook is deliberately not included in this repository. Supply it
locally; the script writes artifacts/churnguard_pipeline.joblib and metadata.
"""
from __future__ import annotations

import argparse
import warnings
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import lightgbm
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

SEED = 42
TARGET = "Churn"
ID_COL = "AccountID"

RENAME_MAP = {
    "CC_Contacted_LY": "CC_Contacted_L12m",
    "Complain_ly": "Complain_l12m",
    "coupon_used_for_payment": "coupon_used_l12m",
    "cashback": "cashback_l12m",
}
NUMERIC_COLUMNS = [
    "Tenure", "City_Tier", "CC_Contacted_L12m", "Service_Score",
    "Account_user_count", "CC_Agent_Score", "rev_per_month",
    "Complain_l12m", "rev_growth_yoy", "coupon_used_l12m",
    "Day_Since_CC_connect", "cashback_l12m",
]
SENTINELS = {"#", "@", "+", "$", "*", "&&&&"}
FEATURE_COLUMNS = [
    "Tenure", "City_Tier", "CC_Contacted_L12m", "Payment", "Gender",
    "Service_Score", "Account_user_count", "account_segment",
    "CC_Agent_Score", "Marital_Status", "rev_per_month", "Complain_l12m",
    "rev_growth_yoy", "coupon_used_l12m", "Day_Since_CC_connect",
    "cashback_l12m", "Login_device",
]
CATEGORICAL_COLUMNS = [
    "Payment", "Gender", "account_segment", "Marital_Status", "Login_device"
]

# These are the winning parameters recorded in the submitted Milestone 2 notebook.
LIGHTGBM_PARAMS = {
    "objective": "binary",
    "class_weight": "balanced",
    "random_state": SEED,
    "verbosity": -1,
    "n_jobs": 2,
    "deterministic": True,
    "force_col_wise": True,
    "reg_lambda": 1,
    "num_leaves": 47,
    "n_estimators": 500,
    "min_child_samples": 70,
    "max_depth": -1,
    "learning_rate": 0.08,
    "colsample_bytree": 1.0,
}

# Milestone 2 selected this training-out-of-fold F2 threshold to favour recall.
DEFAULT_THRESHOLD = 0.275


def clean_source(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented Milestone 1 / 2 cleaning rules."""
    clean = raw.rename(columns=RENAME_MAP).copy()

    for col in clean.select_dtypes(include=["object", "string"]).columns:
        clean[col] = clean[col].astype("string").str.strip()

    for col in NUMERIC_COLUMNS:
        values = clean[col].astype("string").str.strip()
        clean[col] = pd.to_numeric(values.mask(values.isin(SENTINELS)), errors="coerce")

    clean["Login_device"] = clean["Login_device"].astype("string").str.strip()
    clean["Login_device"] = clean["Login_device"].mask(
        clean["Login_device"].eq("&&&&"), pd.NA
    )
    clean["Gender"] = clean["Gender"].replace({"M": "Male", "F": "Female"})
    clean["account_segment"] = clean["account_segment"].replace(
        {"Regular +": "Regular Plus", "Super +": "Super Plus"}
    )

    if clean[TARGET].isna().any() or not set(clean[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Churn must be a complete binary target containing only 0 and 1.")
    if clean[ID_COL].isna().any() or not clean[ID_COL].is_unique:
        raise ValueError("AccountID must be present and unique for this account-level dataset.")
    missing_columns = set(FEATURE_COLUMNS + [TARGET]) - set(clean.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    return clean


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", RobustScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ], remainder="drop")
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", LGBMClassifier(**LIGHTGBM_PARAMS)),
    ])


def train(data_path: Path, output_dir: Path) -> dict:
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find source workbook: {data_path}")

    raw = pd.read_excel(data_path, sheet_name="Data for DSBA")
    clean = clean_source(raw)
    X = clean[FEATURE_COLUMNS].copy()
    # Convert pandas StringDtype missing sentinels to NumPy NaN for sklearn imputers.
    for col in X.select_dtypes(include=["object", "string"]).columns:
        X[col] = X[col].map(lambda value: np.nan if pd.isna(value) else str(value)).astype(object)
    y = clean[TARGET].astype(int).copy()

    # Match the stratified split used in Milestone 2 so the saved model aligns
    # with the reported holdout evaluation. The 20% test partition stays unseen.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )

    numeric_cols = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
    categorical_cols = CATEGORICAL_COLUMNS.copy()
    pipeline = build_pipeline(numeric_cols, categorical_cols)
    pipeline.fit(X_train, y_train)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            category=UserWarning,
        )
        test_scores = pipeline.predict_proba(X_test)[:, 1]
    from sklearn.metrics import average_precision_score, roc_auc_score
    metrics = {
        "test_average_precision": float(average_precision_score(y_test, test_scores)),
        "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
        "train_accounts": int(len(X_train)),
        "held_out_test_accounts": int(len(X_test)),
        "test_churn_rate": float(y_test.mean()),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "churnguard_pipeline.joblib"
    joblib.dump(pipeline, model_path, compress=3)

    metadata = {
        "model_name": "ChurnGuard LightGBM",
        "model_version": "1.0.0",
        "business_domain": "E-commerce account churn",
        "prediction_unit": "account",
        "target": {"field": "Churn", "positive_class": 1, "negative_class": 0},
        "risk_score": "predict_proba[:, 1], bounded from 0.0 to 1.0; treat as a ranking score until separately calibrated",
        "default_threshold": DEFAULT_THRESHOLD,
        "threshold_source": "Milestone 2 training out-of-fold F2 optimization",
        "feature_columns": FEATURE_COLUMNS,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "training_parameters": LIGHTGBM_PARAMS,
        "training_split": "Stratified 80/20; random_state=42; model fitted on training partition only",
        "milestone_2_reference_metrics": {
            "cross_validation_average_precision_mean": 0.9718,
            "cross_validation_average_precision_std": 0.0091,
            "holdout_average_precision": 0.9869,
            "holdout_roc_auc": 0.9970,
        },
        "reproduction_metrics": metrics,
        "versions": {
            "python": __import__("sys").version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "joblib": joblib.__version__,
        },
        "limitations": [
            "The workbook has no extraction timestamp or explicit churn prediction horizon.",
            "Point-in-time leakage has not been ruled out; temporal validation is required before production.",
            "The tenure unit and the currency/units of revenue and cashback are undocumented.",
            "The score is not certified as a calibrated probability.",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model: {model_path}")
    print(json.dumps(metrics, indent=2))
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Customer Churn Data.xlsx", help="Path to source workbook")
    parser.add_argument("--output", default="artifacts", help="Output folder")
    args = parser.parse_args()
    train(Path(args.data), Path(args.output))
