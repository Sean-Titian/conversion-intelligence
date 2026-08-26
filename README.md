# Conversion Intelligence

Decision-aware conversion modeling with prediction-time contracts, leakage checks, calibrated
evaluation, and an experimentation handoff.

This is the conversion-modeling track of my **Growth Analytics Portfolio**. It connects the
Applied / Product Data Science story on my resume to a public, reproducible implementation without
redistributing the restricted educational source data.

[![CI](https://github.com/Sean-Titian/conversion-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/Sean-Titian/conversion-intelligence/actions/workflows/ci.yml)

The central finding is deliberately uncomfortable: **the score that looks best is the least ready
to deploy.** Final pages visited makes same-session conversion easy to classify, but the source does
not show how many pages were known at a prospective intervention time.

## Start with the decision time

| Scenario | Features | Valid interpretation | Status |
| --- | --- | --- | --- |
| Acquisition | country, age, user status, source | Coarse risk at session start | Candidate for prospective validation |
| Retrospective full session | acquisition features + final total pages | Descriptive upper bound after observing the session | **Not a real-time deployment result** |

The code retains `in_session` as the second scenario key for compatibility, but its contract says
`session_end_in_source_data`. A genuine in-session model would require timestamped page events and a
fixed scoring cutoff, such as features known after page three or two minutes.

## Reality-audited results

The source case contains 316,200 rows and a 3.23% conversion rate. Validation removes two impossible
ages. Results below use logistic regression, train-only preprocessing, a 60/20/20
train/validation/test split, and five split seeds. Average precision (AP) is primary because only
3.23% of rows are positive; ROC-AUC is supplementary.

| Model / control | AP, mean (seed range) | Brier ↓ | Lift @ top 5% | Recall @ top 5% | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training-prevalence constant | 0.032 | 0.0312 | —¹ | —¹ | 0.500 |
| **Acquisition-time model** | **0.133 (0.127–0.137)** | 0.0293 | 5.25× | 0.262 | 0.820 |
| Final page count only² | 0.747 | 0.0135 | 16.18× | 0.809 | 0.976 |
| Retrospective full-session model² | 0.841 (0.832–0.850) | 0.0109 | 17.32× | 0.866 | 0.986 |

¹ A constant score ties every row, so a top-5% ranking is undefined; AP and ROC-AUC are the
meaningful baseline values. ² Single-seed result for pages-only; the full model is
averaged over five seeds. Neither is a deployable same-session claim with this dataset. Top-5%
metrics use deterministic stable-input-order tie breaking and record the boundary-tie size, so
their row-order sensitivity is explicit.

### Resume metric crosswalk

The resume compresses the original single-holdout reproduction as ROC-AUC / PR-AUC
**0.986 / 0.844 → 0.820 / 0.129** after removing final page views. The rebuilt repository uses a
60/20/20 train/validation/test design and five seeds, so its canonical estimates are
**0.986 / 0.841 → 0.820 / 0.133**. The small numerical differences come from the evaluation
protocol; both tell the same substantive story: the highest score depends on information that is
not available at acquisition time. The reviewed aggregates and protocol labels are recorded in the
[machine-readable benchmark](reports/source-benchmark.json).

The original resume's 0.50-to-0.20 example was only a precision/recall trade-off. The rebuilt
decision analysis no longer chooses 0.20 by hand: it searches every distinct validation-score
boundary under an explicit loss function, freezes the selected threshold, and evaluates it on a
held-out test partition within that split. The result remains a retrospective full-session
exercise, not a recommended production cutoff.

### What the stress tests say

- On the seed-42 test set, an iid row bootstrap puts acquisition AP at **0.127 [0.118, 0.135]**
  and retrospective AP at **0.839 [0.827, 0.852]**. These intervals are intentionally labelled
  naive: without user or time IDs, dependence-aware uncertainty is not identifiable.
- A label-permutation negative control returns AP **0.0317** and ROC-AUC **0.492**, close to chance.
  That makes an accidental evaluation path that reads the true label less likely.
- The validated data has only **14,270 full-session feature profiles**, and 299,903 rows repeat an
  earlier row across every recorded column, including the label. Repeated low-cardinality profiles
  are not automatically copied users, because there is no session or user key. As a sensitivity
  test, forbidding exact feature
  profiles from crossing splits gives AP **0.117** for acquisition and **0.839** for the
  retrospective model. The grouped acquisition test also has a lower 2.80% prevalence, so its AP
  is not directly comparable; the latter result barely changes.
- Adding synthetic page-count error with standard deviation two lowers retrospective AP from 0.839
  to about **0.76**. Masking 5% of cells lowers it to about **0.80**. These are input-corruption
  checks, not substitutes for a real out-of-time evaluation.

The high retrospective score is therefore not mainly explained by memorizing exact feature
profiles. It is mostly the very strong relationship between the final page count and conversion:
page count alone reaches AP 0.747. That relationship can be genuine descriptively while still being
unusable at the claimed decision time and non-causal.

## Cost-sensitive decision policy

The policy minimizes the validation objective
`(false positives + r × false negatives) / rows × 1,000`, where `r` is an explicit
false-negative-to-false-positive loss ratio. The primary illustrative scenario uses `r = 4`.
For calibrated probabilities, that ratio has a theoretical cost threshold of
`1 / (1 + 4) = 0.20`. The code selects an empirical threshold independently on each validation
split, but evaluates it against both the fixed 0.50 convention and this much stronger 0.20
cost-aware reference.

The main stress test keeps identical acquisition-feature profiles out of different partitions and
repeats a nominal 60/20/20 train/validation/test protocol over five seeds. Group constraints make
the realized row shares approximate rather than exact.

| Retrospective full-session result | Mean (five grouped holdouts) | Seed range |
| --- | ---: | ---: |
| Validation-selected threshold | **0.208** | 0.198–0.216 |
| Precision | 0.665 | 0.638–0.690 |
| Recall | 0.814 | 0.799–0.836 |
| Modeled loss / 1,000, fixed 0.50 | 44.124 | 39.493–51.581 |
| Modeled loss / 1,000, theoretical 0.20 | 36.471 | 33.054–41.589 |
| Modeled loss / 1,000, validation-selected | **36.467** | 33.084–41.541 |
| Selected vs fixed 0.50 | **17.286% lower** | 14.955%–19.466% lower |
| Selected vs theoretical 0.20 | **0.001% higher** | 0.556% higher–0.336% lower |

Loss rows average the five split-level losses; relative rows average five within-split ratios. The
ratio of the displayed mean losses is therefore not the displayed mean relative change. This is why
36.467 is slightly below 36.471 while the mean within-split comparison is 0.001% higher.

The row-random sensitivity also selects a threshold near 0.20. The central lesson is not that a
complex threshold search beats a sound cost rule: it does not. Nearly all of the 17.3% reduction
relative to 0.50 comes from representing the 4:1 loss assumption, while validation selection is
effectively tied with the theoretical 0.20 reference. A 5% validation selection ceiling chooses a
mean threshold of 0.213 and flags 3.9% of test sessions; it is a sensitivity check, not a hard batch
quota. The selected policy stays below 5% on all five test splits, while fixed 0.20 slightly exceeds
it in one split under score shift.
The reported seed ranges measure split sensitivity, not confidence intervals; test rows can overlap
across seeds, and acquisition profiles are only a proxy because the source has no stable user ID.

This is an honest audit of an **offline classification decision rule**. The cost units are scenario
weights—not dollars, ROI, or causal conversion lift. Even switching the acquisition-time model from
0.50 to a cost-aware rule reduces modeled loss by only about 0.8%, so the data does not support a
strong deployable pre-session policy. A production decision would require measured action costs,
incremental treatment response, a separate top-K rule for hard capacity, and prospective or
out-of-time validation.

## Product recommendation

- Use the acquisition score only as a low-stakes prioritization signal after prospective validation;
  do not automate consequential decisions from demographic or geographic attributes.
- Do not present the 0.986 ROC-AUC as the headline deployment result. Use it as evidence that final
  engagement is associated with conversion and as motivation to collect timestamped behavioral data.
- Test a specific intervention—such as a recommendation module—with random assignment. At a 3.23%
  baseline, 10% relative lift, 80% power, and 5% attrition, the planning approximation is **51,893
  users per arm**.
- Monitor calibration, performance by segment, latency, refunds, and opt-outs before expanding use.

## What was improved

- Replaced notebook-global one-hot encoding with a train-only `Pipeline` and `ColumnTransformer`.
- Encoded decision time and deployment status as tested prediction contracts.
- Added AP, Brier score, calibration error, lift, recall at capacity, and explicit threshold costs.
- Replaced a hand-picked cutoff example with exact validation-only threshold selection, explicit
  fixed-0.50 and theoretical cost-threshold references, validation selection ceilings, tie handling,
  and cost-ratio sensitivity.
- Added five-seed evaluation, iid bootstrap intervals, exact-profile holdout, label permutation,
  and missing/unseen/noisy-input checks.
- Added training–scoring schema parity tests, including reordered fields, irrelevant fields, missing
  values, and unseen categories.
- Added a pre-registered-style [experiment handoff](docs/experiment-design.md) instead of turning
  feature importance into a causal claim.
- Excluded raw educational data and binary model artifacts from version control.

## Reproduce it

Python 3.11+ is required. The reviewed source benchmark records Python 3.12.13 and the exact
algorithm-sensitive package versions in `requirements-benchmark.txt`; the synthetic workflow is
also tested on Python 3.11–3.13 in CI.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the workflow without the original dataset:

```bash
conversion-synthesize --output data/synthetic/conversion.csv --rows 50000
conversion-train --data data/synthetic/conversion.csv --output reports/synthetic
pytest
```

To reproduce the reviewed source benchmark, use Python 3.12.13 and install the exact lock before
the package. Then provide an authorized local copy of the source data:

```bash
python -m pip install -r requirements-benchmark.txt
python -m pip install -e ".[dev]"
conversion-train --data data/raw/conversion_project.csv --output reports/generated
conversion-audit --data data/raw/conversion_project.csv --output reports/robustness/realism_audit.json
conversion-decide --data data/raw/conversion_project.csv \
  --output reports/generated/decision-optimization.json \
  --benchmark-output reports/generated/decision-optimization-benchmark.json
python -m conversion_intelligence.experiment --baseline 0.0323 --relative-lift 0.10
```

The audit report is rebuilt locally and ignored by Git. See the [data contract](data/README.md) and
[model card](docs/model-card.md) for interpretation boundaries.

## Repository map

```text
.
├── src/conversion_intelligence/   # validation, models, audit, power, CLI
├── tests/                         # contracts, parity, metrics, and split tests
├── docs/                          # experiment design and model card
├── data/                          # contract and ignored local inputs
├── reports/                       # reproducible, ignored generated outputs
└── .github/workflows/ci.yml       # lint and test gate
```

## What the data cannot establish

- No timestamps: no out-of-time test, delayed-label audit, or drift estimate.
- No stable user/session identifier: no user-level leakage test or clustered uncertainty.
- No prospective page snapshot: no evidence that final page count was available before conversion.
- No intervention assignment: prediction and association do not estimate incremental lift.
- Undocumented redistribution terms: the original data is not included.

The next credible increment is better data, not a more complex classifier: log user/session IDs,
feature event times, score time, intervention exposure, delayed outcomes, and guardrails. Only then
does it make sense to add production serving, monitoring, or model-registry machinery.

## Portfolio context and license

This is an educational portfolio project, not employment experience. The synthetic generator keeps
the pipeline runnable without redistributing the source dataset. Original code is available under
the [MIT License](LICENSE); [third-party notices](THIRD_PARTY_NOTICES.md) define the source boundary.
