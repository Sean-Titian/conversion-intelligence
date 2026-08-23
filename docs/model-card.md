# Model card

## Intended use

This model is an educational decision-support prototype for ranking e-commerce sessions by
conversion likelihood. It demonstrates prediction-time contracts, imbalanced-class evaluation,
cost-sensitive threshold selection, and negative controls.

## Out-of-scope use

- Automated denial, pricing, credit, employment, or other high-impact decisions
- Targeting based solely on age, country, or another sensitive/proxy attribute
- Claims that a high-importance feature causes conversion
- Deployment without current, representative validation data

## Model variants

- `acquisition`: uses only features nominally available at session start and is a candidate for
  prospective validation.
- `in_session`: uses the source's **final** page total. Despite the historical code key, it is a
  retrospective upper bound, not a valid real-time model from this dataset.

## Evaluation

The training command writes discrimination, calibration, capacity, and operating-point metrics to
`reports/generated/metrics.json`. Average precision is primary because the positive class is rare;
ROC-AUC is supplementary. Precision and recall are evaluated at a threshold selected on validation
data using explicit costs. `conversion-audit` adds five split seeds, a row-bootstrap interval, exact
profile holdout, label permutation, and input corruption. The bootstrap is not dependence-aware.

## Known limitations

- The source dataset has no timestamp, so temporal drift cannot be measured.
- The source has no stable user/session identifier, so entity leakage and clustered uncertainty
  cannot be measured.
- Final page count is not timestamped at a prospective scoring point.
- The label records same-session conversion and does not measure retention or long-term value.
- Demographic and geographic features require fairness review before operational use.
- Aggregate educational data cannot substitute for a production data contract, monitoring, and incident response.
