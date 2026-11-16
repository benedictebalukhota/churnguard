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

> **Repository setup status:** the source code, fitted model artifact, metadata, tests, examples and deployment workflow are present. GitHub Actions application checks have passed. Hugging Face deployment is still being verified.

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
3. Add the GitHub Actions repository variable `HF_REPO_ID` with the exact Hugging Face repository ID, for example `your-hf-username/churnguard`.
4. Upload the fitted `artifacts/churnguard_pipeline.joblib` file to this GitHub repository.
5. Push to `main`; GitHub Actions will mirror the files to the Space and Hugging Face will rebuild it.

The workflow intentionally skips deployment while the fitted model file is absent. See `DEPLOYMENT.md` for the full steps.

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
