# ChurnGuard deployment guide

This repository can be hosted in two places:

- **GitHub** stores the source code, version history, issues and collaboration.
- **Hugging Face Spaces** runs the Gradio application and gives it a public or private app URL.

Do not upload `Customer Churn Data.xlsx` or any customer-level source extract to a public repository. The training workbook is intentionally excluded from this package. Start with a private GitHub repository unless you have reviewed the data and model sharing permissions.

## A. Put the code on GitHub

### Option 1 — GitHub website

1. Download and unzip `ChurnGuard_GitHub_Repository.zip`.
2. Sign in to GitHub and create a new repository named `churnguard`.
3. Prefer **Private** while you are still checking data rights and deployment settings.
4. Create it without an automatically generated README or license, because this project folder already contains both.
5. Open the empty repository, choose **Add file → Upload files**, and upload the contents of the extracted `churnguard-github-repo` folder (not the ZIP file itself).
6. Commit the files to the `main` branch.

### Option 2 — Git command line

From the extracted project folder:

```bash
git init
git branch -M main
git add .
git commit -m "Build ChurnGuard churn risk scoring app"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/churnguard.git
git push -u origin main
```

Authenticate through GitHub's normal sign-in / credential manager or SSH setup. Do not paste access tokens into source files, README files, or notebooks.

## B. Publish the app on Hugging Face Spaces

1. Create a new Space and select **Gradio** as its SDK.
2. Give it a name such as `churnguard` and choose Public or Private deliberately.
3. Upload the same repository contents to the Space, or use the Git remote method below.
4. Wait for the build to complete. Open the App tab and try the included sample account examples.
5. Use **Use via API** in the app footer to inspect the API contract and example calls.

The `README.md` YAML block configures the Space SDK, Gradio version, Python version and entry-point file. `requirements.txt` declares the application dependencies.

### Push the same local folder to both GitHub and Hugging Face

Create an empty GitHub repository and a Gradio Space first. Then, inside this folder:

```bash
git init
git branch -M main
git add .
git commit -m "Build ChurnGuard churn risk scoring app"
git remote add github https://github.com/YOUR_GITHUB_USERNAME/churnguard.git
git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/churnguard.git
git push -u github main
git push -u space main
```

The Hugging Face push triggers a Space rebuild. If a remote repository was initialized with its own README or license, reconcile the histories before pushing rather than force-pushing blindly.

## C. Test the local app

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

Run the tests:

```bash
python -m pip install pytest
python -m pytest -q
```

## D. Retrain when the project data is updated

Get the authorized Excel workbook from its owner, place it beside `train_model.py`, then:

```bash
pip install -r requirements-train.txt
python train_model.py --data "Customer Churn Data.xlsx" --output artifacts
```

Review new metadata and evaluation results before replacing the saved model. Keep the raw workbook out of GitHub and Hugging Face public repos.

## E. Production-readiness checklist

- Confirm the churn definition, observation date and prediction horizon.
- Verify that every model input is available at the time the score is calculated; run a temporal validation and leakage audit.
- Confirm the meaning and unit of tenure, revenue, cashback and satisfaction scores.
- Conduct probability calibration if the score will be described as an actual probability.
- Choose the alert threshold using contact capacity, intervention cost and account contribution margin.
- Test campaign lift against a holdout group before claiming saved revenue or reduced churn.
- Review fairness, privacy, subgroup errors, monitoring, retention policy and access controls.
- Avoid entering identifiable or confidential customer information into a public demo.
