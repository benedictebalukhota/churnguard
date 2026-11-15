# ChurnGuard deployment guide

This repository uses GitHub as the source of truth and Hugging Face Spaces to host the Gradio application.

- **GitHub** stores source code, version history, tests and release automation.
- **Hugging Face Spaces** builds and runs the Gradio app and provides its app URL and API.

## Current setup status

The repository contains the app code, training script, model metadata, sample input, tests and GitHub Actions workflows.

**The fitted model binary is still missing:** `artifacts/churnguard_pipeline.joblib`. The Gradio app loads this file at startup, so the app cannot run locally or on Hugging Face until the matching Milestone 2 artifact is uploaded to this path.

The raw `Customer Churn Data.xlsx` workbook is intentionally excluded. Do not commit raw customer-level data to a public repository.

## 1. Complete the model artifact

1. Locate the fitted `churnguard_pipeline.joblib` generated for Milestone 2.
2. Confirm it is the pipeline matching `artifacts/metadata.json` and the current `app.py`.
3. Upload it to the GitHub path `artifacts/churnguard_pipeline.joblib`.
4. Do not upload the source Excel workbook or customer-level rows.

If the artifact is no longer available, obtain the authorized workbook outside GitHub, retrain with `train_model.py`, review the evaluation outputs, and then upload only the approved model artifact and metadata.

## 2. Configure GitHub → Hugging Face automatic sync

The repository workflow is:

`.github/workflows/sync-to-huggingface.yml`

It runs on pushes to `main` and can also be started manually from the GitHub Actions tab. It uses the official `huggingface/hub-sync` action to mirror the GitHub repository to a Hugging Face Space.

### Create the Hugging Face Space

1. Open https://huggingface.co/new-space.
2. Create a Space named `churnguard` (or use another name and put its exact ID into the GitHub variable).
3. Choose **Gradio** as the SDK.
4. Keep the Space private while testing if that is appropriate for your use case.

### Create a Hugging Face write token

1. Open https://huggingface.co/settings/tokens.
2. Create a fine-grained token with write access only to the target Space, where available.
3. Copy the token once. Do not paste it into code, documentation, issues or commits.

### Add the token as a GitHub Actions secret

In GitHub, open:

`Settings → Secrets and variables → Actions → Secrets → New repository secret`

Create:

| Name | Value |
|---|---|
| `HF_TOKEN` | Your Hugging Face write token |

### Add the Space ID as a GitHub Actions variable

Open:

`Settings → Secrets and variables → Actions → Variables → New repository variable`

Create:

| Name | Value |
|---|---|
| `HF_REPO_ID` | Your exact Space ID, for example `your-hf-username/churnguard` |

The workflow uses `vars.HF_REPO_ID` and `secrets.HF_TOKEN`. It does not require editing the YAML to hard-code a username or token.

### Run and verify the deployment

1. Make sure `artifacts/churnguard_pipeline.joblib` exists in GitHub.
2. Commit and push the completed files to `main`.
3. Open the repository's **Actions** tab.
4. Open the **Sync ChurnGuard to Hugging Face** run and inspect the logs.
5. Open the Space's **Build logs** and then its **App** tab.
6. Test an example account and check **Use via API**.

The deployment workflow intentionally skips the sync if the model artifact is absent, so an incomplete Space is not published as a successful app.

## 3. What happens after setup

After the initial connection:

1. Edit code in GitHub (or locally and push it to GitHub).
2. Commit the change to `main`.
3. GitHub Actions mirrors the files to the configured Hugging Face Space.
4. Hugging Face rebuilds the Space.

GitHub remains the source of truth. The sync is one-way: edits made directly in Hugging Face can be overwritten by the next GitHub sync. The action also mirrors file deletions, so be careful when removing repository files.

The Hugging Face sync action excludes GitHub's `.github/` folder from the Space upload. GitHub Actions workflows remain in GitHub and are not copied into the Space.

## 4. Run and test locally

After the model artifact is in place:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

Open the local Gradio URL printed in the terminal.

Run tests:

```bash
python -m pip install pytest
python -m pytest -q
```

## 5. Retrain only when authorized data is available

Get the authorized Excel workbook from the data owner and place it beside `train_model.py`. Then run:

```bash
pip install -r requirements-train.txt
python train_model.py --data "Customer Churn Data.xlsx" --output artifacts
```

Review model metrics and metadata before replacing the deployed artifact. Keep the raw workbook out of GitHub and public Hugging Face repositories.

## 6. Production-readiness checklist

- Confirm the churn definition, observation date and prediction horizon.
- Verify that every model input is available at the time the score is calculated; run a temporal validation and leakage audit.
- Confirm the meaning and unit of tenure, revenue, cashback and satisfaction scores.
- Conduct probability calibration if the score will be described as an actual probability.
- Choose the alert threshold using contact capacity, intervention cost and account contribution margin.
- Test campaign lift against a holdout group before claiming saved revenue or reduced churn.
- Review fairness, privacy, subgroup errors, monitoring, retention policy and access controls.
- Avoid entering identifiable or confidential customer information into a public demo.
