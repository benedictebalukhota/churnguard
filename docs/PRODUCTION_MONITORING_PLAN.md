# ChurnGuard controlled-pilot monitoring plan

## Gate 0: before a pilot

Do not describe the model as temporally validated until all of the following are complete:

- [ ] Business owner approves the operational churn definition and prediction horizon.
- [ ] Data owner provides dated account snapshots and outcome follow-up through the full horizon.
- [ ] Data owner documents all 17 feature definitions, units, source systems, aggregation windows, and latest observation timestamps.
- [ ] Every feature is proven to have been available on or before its prediction snapshot.
- [ ] Duplicate and repeated-account handling is explicitly defined.
- [ ] Temporal train/validation/test cutoffs are frozen before model selection.
- [ ] Group-aware evaluation is included where the intended use is unseen accounts.
- [ ] Legal, privacy, security, and fairness reviews approve the proposed use of customer attributes.
- [ ] A human review process and escalation path are documented.
- [ ] A baseline and a no-model comparison are specified.

## Gate 1: offline temporal evaluation

Report at minimum:

- Average Precision and ROC-AUC.
- Brier score and log loss, with calibration analysis.
- Confusion matrix, precision, recall, specificity, and alert volume at the proposed threshold.
- Confidence intervals, ideally from account-level bootstrap resampling.
- Performance by time period and relevant customer groups, with sample sizes.
- Comparison with the existing random and group-aware evaluation.
- A clear account of censored/incomplete outcome windows and excluded records.

Choose the alert threshold using retention-team capacity, intervention costs, and the relative cost of missed churn versus unnecessary outreach. Do not choose a threshold solely to maximize a statistical metric.

## Gate 2: shadow mode

Run predictions without exposing them to agents or triggering interventions. Store a model version, snapshot timestamp, score, threshold, feature-schema version, and prediction ID in an access-controlled system. When outcomes mature, compare predictions with observed outcomes.

Review at a fixed cadence (for example, weekly operational checks and monthly model review) agreed with the business owner. This cadence is a proposed starting point, not a measured requirement.

## Monitoring signals

| Area | Monitor | Response |
|---|---|---|
| Data quality | Missing fields, invalid categories, range violations, duplicate snapshots | Pause scoring for affected records; fix source mapping |
| Feature drift | Distribution shifts and new/unseen categories | Investigate source changes; do not auto-retrain |
| Score drift | Score distribution and alert volume | Check data pipeline and customer mix |
| Outcome performance | Average Precision, recall, precision, Brier score after labels mature | Reassess threshold and model fitness |
| Calibration | Predicted score bands versus observed churn rates | Recalibrate only using a separate training/validation period |
| Fairness | Group-level error rates, alert rates, sample sizes and uncertainty | Investigate disparities before expanding use |
| Operations | Human-review completion, contact outcomes, complaints, opt-outs | Adjust workflow and safeguards |
| Security | Access logs, retention, exports, secrets | Restrict access and follow incident process |

## Pause / rollback triggers

Pause use and investigate when:

- Feature lineage or snapshot timing cannot be verified.
- Required inputs are missing or schema changes unexpectedly.
- A scoring release changes output behaviour without review.
- Matured outcome performance falls below a pre-agreed business or statistical floor.
- A material subgroup disparity appears and cannot be explained or mitigated.
- Customer data is exposed or used outside approved purposes.
- Agents begin treating scores as certain predictions or automating adverse decisions.

Do not automatically retrain on a drift alert. Confirm the cause, rebuild a dated evaluation, compare candidate models against a frozen test period, obtain approval, and preserve rollback to the previous model.

## Pilot use restrictions

- Scores prioritize human review; they are not decisions.
- Do not automatically deny, restrict, or cancel customer services.
- Do not automatically issue discounts or financial incentives based on score alone.
- Do not interpret SHAP contributions as causes.
- Do not represent the raw model score as a calibrated probability.
- Keep account-level data and logs in an approved restricted environment.
- Define retention periods and delete data when no longer needed.

## Current status

The current workbook cannot pass the temporal-validation gate because it has no snapshot dates, feature observation timestamps, or explicit prediction horizon. The deployed model remains a prototype pending those inputs and the reviews above.
