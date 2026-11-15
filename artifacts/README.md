# Model artifacts

This folder stores the fitted preprocessing + LightGBM pipeline and its metadata.

Expected runtime artifact:

- `churnguard_pipeline.joblib` — fitted model used by `app.py`.
- `metadata.json` — model version, expected input fields, operating threshold and evaluation notes.

**Important:** the fitted `.joblib` binary is not yet present in this GitHub repository. The application cannot start until the matching file is added at exactly:

`artifacts/churnguard_pipeline.joblib`

Use the artifact generated with the supplied Milestone 2 training process. Do not substitute a newly trained model without reviewing its evaluation metrics. Do not upload the source Excel workbook or customer-level training rows to a public repository.
