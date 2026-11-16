---
title: ChurnGuard – Account Churn Risk
emoji: 🛡️
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: 6.5.1
python_version: "3.13"
app_file: app.py
pinned: false
---

# ChurnGuard — Account Churn Risk Intelligence

ChurnGuard is a Gradio application that scores an e-commerce account for churn risk using the tuned LightGBM pipeline selected in the project's Milestone 2 analysis.

**Input:** 17 account-level usage, service, profile and commercial features.  
**Output:** a model risk score between `0.0` and `1.0`, plus an alert based on a configurable threshold.  
**Default alert threshold:** `0.275`, selected by maximizing F2 on training out-of-fold predictions in Milestone 2.

> **Deployment status:** the application is deployed on Hugging Face Spaces; GitHub Actions checks have passed and live predictions have been verified.

> **Important interpretation:** the displayed score is the model's positive-class score and is useful for ranking accounts. It is not guaranteed to be a calibrated individual churn probability. The source dataset has no extraction timestamp or explicit prediction horizon; point-in-time leakage has not been ruled out. Treat this as a prototype / controlled-pilot tool, not a production decision system.

## App preview

The interface lets a user enter account attributes, select an alert threshold, and receive:

- Churn risk score (`0.000` to `1.000`)
- Risk band: lower score, watchlist, or high model score
- Threshold-based retention flag
- Suggested operational next step based on recorded support / onboarding signals
- Structured JSON response for integrations

The operating threshold controls the alert decision; it does not change the underlying score.

## Run locally

Python 3.13 is recommended to match the model artifact build environment. After adding the fitted model file to `artifacts/`, clone the repository and run:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

Open the local Gradio URL printed in the terminal.

## Deploy to Hugging Face Spaces

The recommended deployment keeps GitHub as the source of truth and uses the GitHub Actions workflow at `.github/workflows/sync-to-huggingface.yml`.

Before enabling the workflow:

1. Create a Gradio Space.
2. Add the GitHub Actions repository secret `HF_TOKEN` with a Hugging Face write token scoped to that Space.
3. Confirm the GitHub Actions repository secret `HF_TOKEN` contains a Hugging Face write token for the target Space.
4. The current workflow targets `benedictebalukhota/churnguard` directly in `.github/workflows/sync-to-huggingface.yml`. Update that value if your Space has a different ID.
5. Confirm `artifacts/churnguard_pipeline.joblib` is present in this repository.
6. Push to `main`; GitHub Actions will mirror the files to the Space and Hugging Face will rebuild it.

The current workflow attempts to sync on every push to `main` and on manual dispatch. See `DEPLOYMENT.md` for the full steps.

For a one-time/manual deployment, create a Gradio Space and push or upload the complete repository contents after the model file has been added. The Space reads its SDK, Python and app entry-point settings from the YAML block at the top of this README.

## Repository layout

```text
churnguard/
├── .github/workflows/
│   ├── sync-to-huggingface.yml
│   └── tests.yml
├── artifacts/
│   ├── README.md
│   ├── metadata.json
│   └── churnguard_pipeline.joblib   # fitted model artifact
├── assets/
│   └── README.md
├── examples/
│   ├── README.md
│   └── sample_account.json
├── tests/
│   ├── README.md
│   └── test_app.py
├── app.py
├── train_model.py
├── evaluate_model.py
├── client_example.py
├── requirements.txt
├── requirements-train.txt
├── README.md
├── DEPLOYMENT.md
├── MODEL_CARD.md
└── LICENSE
```

## Retrain the model

The original customer workbook is intentionally **not** committed to GitHub. To reproduce the saved artifact locally, obtain the authorized `Customer Churn Data.xlsx` file, place it beside `train_model.py`, and run:

```bash
pip install -r requirements-train.txt
python train_model.py --data "Customer Churn Data.xlsx" --output artifacts
```

Review the new metadata and evaluation results before replacing the saved model. The script writes `artifacts/churnguard_pipeline.joblib` and `artifacts/metadata.json`. Upload only the fitted model artifact and metadata; do not upload the raw Excel workbook or customer-level records.

## Reproduce the evaluation

With the authorized source workbook available locally, run:

```bash
pip install -r requirements-train.txt
python evaluate_model.py --data "Customer Churn Data.xlsx" --output artifacts/evaluation_report.json
```

The script recreates the fixed stratified holdout, threshold sweep, calibration bins, subgroup metrics, and sensitive-feature ablation. It writes only aggregated evaluation results; never commit the source workbook or account-level records.

The reproduced holdout at threshold 0.275 yielded precision 89.18%, recall 97.89%, and F1 93.33% (TN=1,828, FP=45, FN=8, TP=371). This is a random holdout, not future-period validation. Gender and Marital_Status subgroup results are documented in MODEL_CARD.md and should be treated as descriptive diagnostics, not proof of fairness. The score remains a ranking signal, not a calibrated individual probability.

## Proposed 30-day temporal validation

ChurnGuard's proposed prediction horizon for the next validation design is **30 days**. This is a target definition for future evaluation, not evidence that the current deployed model predicts 30-day churn.

For each account snapshot, a positive label means churn occurred after the snapshot and within 30 days. A negative label requires complete observation through the full 30-day window without churn. Positive examples must be observed through their churn date; incomplete negative follow-up must be treated as censored/excluded.

Read `docs/TEMPORAL_DATA_CONTRACT.md` before preparing data. The validator expects `AccountID`, `Snapshot_Date`, `Outcome_Observed_Through`, `Churn_Date`, `Churn`, all 17 existing model features, and (ideally) a feature-provenance CSV documenting when each feature was observed.

Run the structural checks locally:

```bash
python validate_temporal_data.py --data snapshots.csv --provenance feature_provenance.csv
```

The validator defaults to 30 days. Override with `--horizon-days` only if the business later approves a different horizon. It does not train a model or prove that churn definitions and source timestamps are correct.

See `docs/PRODUCTION_MONITORING_PLAN.md` for the controlled-pilot gates. The current source workbook still cannot support temporal validation because it has no snapshot dates or feature observation timestamps.

## Model summary

| Item | Value |
|---|---:|
| Accounts in source | 11,260 |
| Observed churn prevalence | 16.84% |
| Selected algorithm | LightGBM classifier |
| Mean 5-fold CV Average Precision | 0.9718 ± 0.0091 |
| Holdout Average Precision | 0.9869 |
| Holdout ROC-AUC | 0.9970 |
| Default alert threshold | 0.275 |

These values are from the Milestone 2 random holdout design. They should not be represented as proof of future-period production performance.

## Data definitions and known limitations

- `Tenure` unit is not defined by the source.
- Revenue and cashback currency / units are not documented.
- The score direction for service and customer-care scores is not fully described.
- The churn prediction horizon, feature availability timestamps and data extraction date are absent.
- Probability calibration and performance on a future-dated dataset remain to be assessed.
- Account risk and account business value are separate. A high-risk account is not automatically the highest-value account to retain.

## Business use and guardrails

Use a risk flag to prioritize a human review, not to automatically deny service or issue a discount. A complaint should route to service recovery; early-tenure accounts may need onboarding support; other flagged accounts should receive a contextual review. Financial incentives require contribution-margin information and controlled campaign tests. Review false-positive / false-negative rates across relevant groups and protect customer data.

## API

The Gradio app exposes two named endpoints:

- `predict_churn`: structured account inputs and a threshold, returning score, band, alert decision and action prompt.
- `predict_json`: JSON payload and threshold, returning a JSON response.

Each public Gradio Space exposes its available API methods through the **Use via API** link in the interface footer.

## References

- ChurnGuard Capstone Milestone 1 Business Report (22 October 2026)
- ChurnGuard Capstone Milestone 2 Business Report (8 November 2026)
- Hugging Face Spaces Gradio SDK documentation: https://huggingface.co/docs/hub/spaces-sdks-gradio
- Hugging Face Spaces configuration: https://huggingface.co/docs/hub/main/en/spaces-config-reference


## Prediction explanations

The Gradio interface includes local SHAP explanations for each account prediction. ChurnGuard uses the fitted LightGBM model's TreeSHAP contributions and groups one-hot encoded categories back to their original input fields.

- Positive SHAP contributions push the model's raw output toward the positive class (Churn = 1).
- Negative contributions push it away from the positive class.
- Contributions are in raw-margin (log-odds) units, not percentage points.
- SHAP values describe model behaviour for this input; they do not establish causality.
