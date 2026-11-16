# ChurnGuard LightGBM model card

## Summary

ChurnGuard is an account-level binary classifier intended to help an e-commerce retention team prioritize accounts for human review. The model returns the score for the positive class (`Churn = 1`) on a 0.0–1.0 scale.

## Training data

- Source: the supplied Customer Churn Data workbook (Data for DSBA sheet)
- Records: 11,260 accounts
- Target prevalence: 16.84% churned accounts
- Features: 17 account-level fields covering tenure, usage, service/contact history, payment preference, account segment, revenue and engagement
- The workbook itself is not included in this repository.

## Model and evaluation

- Algorithm: LightGBM classifier
- Selected hyperparameters: the RandomizedSearchCV winner documented in Milestone 2
- Split: stratified 80/20 with a fixed seed of 42
- Cross-validation: five-fold stratified CV on the training partition
- Primary metric: Average Precision, because churn is the minority class
- Milestone 2 CV Average Precision: 0.9718 ± 0.0091
- Milestone 2 holdout Average Precision: 0.9869
- Milestone 2 holdout ROC-AUC: 0.9970

## Intended use

- Demonstration and academic capstone evaluation
- Prioritizing accounts for a human-led retention review
- Testing a controlled pilot after business and data validation

## Out-of-scope use

- Automatically cancelling, denying, or restricting customer services
- Automatically issuing discounts or financial offers
- Treating the raw score as a guaranteed or calibrated individual probability
- Deploying a production intervention without future-period validation and business approval

## Known limitations

1. There is no extraction date, churn horizon, or field-availability timestamp in the source. Temporal leakage has not been ruled out.
2. The unit for `Tenure` and the currency / unit for revenue and cashback are undocumented.
3. Score direction and the semantic definitions of some source variables require confirmation.
4. The saved score has not been calibrated as a literal probability.
5. The test design is a random holdout, not a future-dated temporal test.
6. Feature importance is descriptive of model splits and is not causal explanation.
7. Retention economics, campaign costs and incremental treatment effects were not included in the dataset.

## Recommended controls

- Confirm the point-in-time feature availability and prediction horizon.
- Run temporal validation on a dated sample before production use.
- Calibrate probabilities if users need to interpret scores as probabilities.
- Choose thresholds using business capacity, costs and contribution margin.
- Compare retention outcomes against a holdout group.
- Review subgroup error rates, fairness, privacy, monitoring and access controls.


## Explainability

The application displays local SHAP (TreeSHAP) contributions for individual predictions. Contributions from one-hot encoded categorical fields and numeric preprocessing fields are grouped to their original source features. Positive contributions move the model's raw margin toward Churn = 1; negative contributions move it away. Values are expressed in raw-margin (log-odds) units, not percentage points, and must not be interpreted as causal effects.
