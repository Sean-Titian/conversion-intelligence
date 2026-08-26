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

`conversion-decide` performs the policy analysis separately from model benchmarking. It fits on
train partitions, searches every distinct validation-score boundary, freezes the selected cutoff,
and compares it with both a fixed 0.50 convention and the theoretical calibrated-probability cost
threshold `C_FP / (C_FP + C_FN)` on the held-out test partition. It repeats both
acquisition-profile-grouped and row-random splits over five seeds, sweeps illustrative FN:FP loss
ratios, and tests validation selection ceilings while keeping equal-score rows together.

The primary 4:1 sensitivity scenario gives the retrospective full-session logistic model a mean
validation-selected threshold of 0.208 and a 17.3% reduction in modeled test loss versus 0.50 across
five grouped holdouts. Against the theoretical 0.20 reference, however, mean modeled loss is 0.001%
higher; validation search adds no material benefit over applying the stated cost matrix correctly.
These are unitless classification-loss weights, not measured dollars, ROI, or causal treatment
effects. Even switching the acquisition-time analysis from 0.50 to a cost-aware rule produces only
about a 0.8% mean reduction, so it does not support a strong deployment claim.

## Known limitations

- The source dataset has no timestamp, so temporal drift cannot be measured.
- The source has no stable user/session identifier, so entity leakage and clustered uncertainty
  cannot be measured.
- Final page count is not timestamped at a prospective scoring point.
- The label records same-session conversion and does not measure retention or long-term value.
- The policy treats false positives and false negatives as classification decisions. Without
  randomized intervention data, it cannot identify who would convert because of an action.
- The 4:1 loss ratio and 5%/10% validation selection ceilings are explicit sensitivity assumptions,
  not measured business inputs or hard operational quotas.
- Demographic and geographic features require fairness review before operational use.
- Aggregate educational data cannot substitute for a production data contract, monitoring, and incident response.
