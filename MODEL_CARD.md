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
- Milestone 2 / reproduced random-holdout Average Precision: 0.9869
- Reproduced random-holdout ROC-AUC: 0.9970
- Reproduced Brier score: 0.0140 (constant-prevalence baseline: 0.1400)
- Reproduced log loss: 0.0519
- At the provisional 0.275 threshold: precision 89.18%, recall 97.89%, F1 93.33%; confusion matrix TN=1,828, FP=45, FN=8, TP=371 (2,252 test accounts)

The metrics above are from a reproducible stratified random 80/20 split (seed 42), not a temporal or external validation. The 0.275 threshold is provisional and should be selected against intervention capacity and business costs before operational use.

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
8. The holdout metrics are unusually strong and should be independently checked for leakage, duplicates across time, target-derived fields and source-generation artefacts before any production claim.
9. Subgroup metrics differ across observed Gender and Marital_Status groups. The current single-split audit is descriptive, lacks confidence intervals, and is not a fairness certification.

## Recommended controls

- Confirm the point-in-time feature availability and prediction horizon.
- Run temporal validation on a dated sample before production use.
- Calibrate probabilities if users need to interpret scores as probabilities.
- Choose thresholds using business capacity, costs and contribution margin.
- Compare retention outcomes against a holdout group.
- Review subgroup error rates, fairness, privacy, monitoring and access controls.


## Explainability

The application displays local SHAP (TreeSHAP) contributions for individual predictions. Contributions from one-hot encoded categorical fields and numeric preprocessing fields are grouped to their original source features. Positive contributions move the model's raw margin toward Churn = 1; negative contributions move it away. Values are expressed in raw-margin (log-odds) units, not percentage points, and must not be interpreted as causal effects.

## Reproducible evaluation and subgroup review

Run the evaluation locally with the original authorized workbook:

```bash
pip install -r requirements-train.txt
python evaluate_model.py --data "Customer Churn Data.xlsx" --output artifacts/evaluation_report.json
```

The evaluation script retrains the documented pipeline using the fixed stratified split and writes aggregated metrics only. It reports threshold trade-offs, quantile calibration bins, subgroup error rates for Gender and Marital_Status, and a diagnostic ablation excluding those two features. The raw workbook and account-level rows must remain local and must not be committed.

At threshold 0.275, subgroup recall on this holdout was 95.80% for Female accounts and 99.13% for Male accounts. Marital_Status subgroup recall ranged from 91.30% to 99.49%. Small groups and differing observed churn rates require careful interpretation. The ablation excluding Gender and Marital_Status reduced Average Precision to 0.9763 and ROC-AUC to 0.9946 on this same split; this performance change alone does not justify retaining demographic attributes. Review necessity, lawfulness, fairness, and alternatives with appropriate stakeholders.

The Brier score is better than a constant-prevalence baseline on this random holdout, but this does not certify probability calibration. Continue to present the output as a risk score. Temporal validation remains blocked until dated, point-in-time data and an explicit prediction horizon are available.

## Stage 4C: duplicate and ordering sensitivity audit

A supplementary audit was run against the supplied workbook. It found 264 groups of repeated feature vectors (528 rows in total); each repeated feature group had the same observed target. In the original random split, 94 test rows had an identical feature vector represented in the training partition.

A five-fold stratified group split, grouping identical feature vectors so they cannot cross partitions, produced Average Precision 0.9795, ROC-AUC 0.9934 and Brier score 0.0183. At threshold 0.275, precision was 87.35% and recall was 94.72% (TN=1,821, FP=52, FN=20, TP=359). This is a sensitivity check for exact duplicate feature rows, not a substitute for external or temporal validation.

AccountID is unique and monotonically increasing in the workbook, but it is not documented as a timestamp. Its Pearson correlation with the target is -0.0095. An ID-ordered split (first 80% of IDs for training, final 20% for testing) produced Average Precision 0.9568 and ROC-AUC 0.9908. Reversing train/test direction produced Average Precision 0.8122 and ROC-AUC 0.9384. These are ordering stress tests only: they must not be interpreted as temporal validation, evidence of drift, or proof that AccountID encodes time.

No definitive target-leakage feature was established by this audit. Feature-target association and high predictive performance are not, by themselves, proof of leakage. The source's provenance, field definitions, observation windows, and target construction still require verification. In particular, confirm that complaint, contact, tenure, revenue, coupon and cashback fields were all measured strictly before the churn outcome and within the intended prediction window.

Run the additional audit locally:

```bash
python audit_data.py --data "Customer Churn Data.xlsx" --output artifacts/data_audit_report.json
```

The script writes aggregated statistics only. It does not write account-level data or the source workbook.
