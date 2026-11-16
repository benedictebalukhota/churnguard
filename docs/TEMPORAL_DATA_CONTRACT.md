# ChurnGuard temporal-data contract

## Proposed prediction horizon

**The proposed ChurnGuard prediction horizon is 30 days.** This is a validation-design choice, not a claim that the current model has been trained or validated for 30-day churn. Confirm the operational definition of churn with the data and business owners before labels are generated.

For a snapshot at time T, the label is 1 when churn occurs strictly after T and on or before T + 30 days. A label of 0 is valid only when the account is observed through at least T + 30 days with no churn in that interval. A positive row needs outcome observation through its actual churn date. Rows without sufficient follow-up are censored/excluded, never silently labelled 0.

## Required account-snapshot table

One row represents one account at one prediction snapshot. Required columns:

| Column | Requirement |
|---|---|
| AccountID | Stable account identifier; not a model feature |
| Snapshot_Date | Timestamp when the prediction would have been made |
| Outcome_Observed_Through | For churned rows, at least Churn_Date; for non-churn rows, at least Snapshot_Date + 30 days |
| Churn_Date | Actual churn date; blank for accounts with no churn in the fully observed 30-day horizon |
| Churn | Binary target derived using the 30-day rule above |
| All 17 model features | Values available at Snapshot_Date, never values recorded afterwards |

## Define churn operationally

Before building labels, document what counts as churn: cancellation, inactivity rules (if any), reactivation handling, exclusions, and the authoritative source system. Do not assume a cancellation date and an inactivity-based churn date mean the same thing. Resolve timezone and date-boundary conventions.

## Feature provenance table

Maintain a separate long-format file with one row per account snapshot and feature:

| Column | Meaning |
|---|---|
| AccountID | Account identifier matching the snapshot table |
| Snapshot_Date | Snapshot identifier |
| Feature | Exact model feature name |
| Feature_Observed_At | Latest source timestamp used to calculate this feature |
| Source_System | Originating system/table |
| Transformation_Version | Query, code, or business-rule version |

Every feature's Feature_Observed_At must be less than or equal to Snapshot_Date. Aggregated features must record the end of their observation window, not merely the time the aggregate was computed. For rolling-window fields, document the exact start and end dates.

## Existing model feature list

- Tenure
- City_Tier
- CC_Contacted_L12m
- Payment
- Gender
- Service_Score
- Account_user_count
- account_segment
- CC_Agent_Score
- Marital_Status
- rev_per_month
- Complain_l12m
- rev_growth_yoy
- coupon_used_l12m
- Day_Since_CC_connect
- cashback_l12m
- Login_device

Confirm each field's definition, unit, observation window, missing-value meaning, and whether it can change after churn. Demographic fields require separate necessity, legal, privacy, and fairness review.

## Evaluation protocol

1. Sort snapshots by Snapshot_Date.
2. Choose and freeze a chronological training cutoff before examining test performance.
3. Train only on snapshots whose 30-day outcome windows are fully known by the training cutoff.
4. Test on later snapshots whose outcome windows have matured by the evaluation date.
5. Prevent the same AccountID crossing train/test for an unseen-account evaluation. If evaluating future snapshots of known accounts, report this separately as a returning-account evaluation.
6. Fit imputers, encoders, feature selection, calibration, and threshold selection on training/validation data only.
7. Report Average Precision, ROC-AUC, Brier score, log loss, confusion matrix, precision, recall, and subgroup metrics with uncertainty intervals.
8. Compare against simple baselines and existing random/group-aware holdouts.
9. Preserve a frozen future test period for final reporting.

The implementation is `evaluate_temporal.py`. It takes `--train-cutoff`, `--test-start`, `--test-end`, and `--evaluation-as-of` dates. It selects the latest matured training snapshot and earliest matured test snapshot per account. `--account-scope unseen` excludes accounts present in training; `--account-scope returning` evaluates only accounts represented in both periods. The two modes answer different questions and should be reported separately.

## Privacy and handling

Keep raw account-level data in an approved restricted environment. Do not commit raw datasets, identifiers, or feature-level provenance records to GitHub. Only publish aggregated metrics after privacy review.