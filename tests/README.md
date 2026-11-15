# Tests

The application tests cover the expected 17-field request schema, score bounds, and the fact that changing the alert threshold does not change the model score.

The prediction tests require `artifacts/churnguard_pipeline.joblib`. The GitHub Actions workflow reports a warning and skips model-dependent tests until that artifact is added.
