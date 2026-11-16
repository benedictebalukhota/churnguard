# ChurnGuard temporal-data contract

## Purpose

This contract defines the minimum evidence required before a future-period evaluation. The current workbook has no snapshot date, feature observation timestamps, or explicit prediction horizon, so it cannot support temporal validation.

## Required account-snapshot table

One row represents one account at one prediction snapshot. Required columns:

| Column | Requirement |
|---|---|
| AccountID | Stable account identifier; not a timestamp or model feature |
| Snapshot_Date | Timestamp when the prediction would have been made |
| Outcome_Observed_Through | Date through which churn status is known; must be at least Snapshot_Date + horizon |
| Churn_Date | Actual churn date, blank only when no churn occurred in the fully observed horizon |
| Churn | Binary target derived using the agreed horizon definition |
| All 17 model features | Values available at Snapshot_Date, never values recorded afterwards |

## Define the prediction horizon before collecting labels

Choose and document a business-approved horizon (for example, 30, 60, or 90 days). Do not treat this example as a decision already made. Define churn operationally, including cancellations, inactivity rules, reactivations, and exclusions. For a horizon H, a positive label means churn occurred after the snapshot and on or before Snapshot_Date + H. A negative label requires evidence that the account was observed through the full horizon without churn. Incomplete follow-up must be marked censored or excluded, not silently labelled 0.

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

Every feature's Feature_Observed_At must be less than or equal to Snapshot_Date. Aggregated features must record the end of their observation window, not merely the time the aggregate was computed.

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
2. Choose a cutoff date before looking at test performance.
3. Train only on snapshots whose full outcome horizon is known by the training cutoff.
4. Test on later snapshots whose full outcome horizon is known.
5. Prevent the same AccountID from crossing train and test if the intended question is performance on unseen accounts. If evaluating repeat snapshots for existing accounts, define that as a separate returning-account test.
6. Fit imputers, encoders, feature selection, calibration, and threshold selection on training data only.
7. Report Average Precision, ROC-AUC, Brier score, log loss, confusion matrix, precision, recall, and subgroup metrics with uncertainty intervals.
8. Compare against a simple baseline and the existing random/group-aware holdouts.
9. Preserve a frozen test period for final reporting.

## Privacy and handling

Keep raw account-level data in an approved restricted environment. Do not commit raw datasets, identifiers, or feature-level provenance records to GitHub. Only publish aggregated metrics after privacy review.