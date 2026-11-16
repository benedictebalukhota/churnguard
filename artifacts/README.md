
# Model artifacts

This folder contains the fitted machine-learning pipeline and its metadata.

## Files

- `churnguard_pipeline.joblib` — fitted preprocessing and LightGBM model used by `app.py`.
- `metadata.json` — model version, expected input fields, training configuration, evaluation notes and limitations.

## Model status

The fitted model artifact has been uploaded to this repository at:

`artifacts/churnguard_pipeline.joblib`

The application loads this file during startup. Keep the model and metadata consistent when updating or retraining the model.

## Important

- Do not upload the original customer dataset or confidential customer-level records.
- Do not replace the model artifact without reviewing the new training and evaluation results.
- Keep the model input schema consistent with `app.py`.
