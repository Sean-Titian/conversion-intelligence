# Experiment handoff: from prediction to causality

The observational model can rank likely converters; it cannot establish that making users visit more pages will increase conversion. The recommended next step is a randomized test of a specific product intervention.

## Decision

Test whether a lightweight recommendation module increases completed purchases for eligible sessions without harming latency or user trust.

## Design

- **Unit of randomization:** authenticated user, persisted across sessions to avoid cross-arm exposure.
- **Population:** eligible users before the recommendation module is rendered.
- **Control:** existing experience.
- **Treatment:** recommendation module with the same checkout flow and pricing.
- **Primary metric:** users completing at least one purchase within seven days of first exposure.
- **Secondary metrics:** product-detail views, add-to-cart rate, and revenue per eligible user.
- **Guardrails:** page latency, bounce rate, error rate, refund rate, and opt-out/contact rate.
- **Analysis:** intention-to-treat difference in proportions with a 95% confidence interval.
- **Heterogeneity:** pre-registered cuts for new versus returning users and acquisition channel; interaction estimates are exploratory unless powered in advance.

## Integrity checks

1. Log assignment before rendering the experience.
2. Verify sample ratio and invariant pre-treatment features daily.
3. Do not stop early on a nominal p-value; use the pre-registered horizon or a sequential design.
4. Treat page views as a mediator/diagnostic metric, not proof of causality.
5. Report absolute lift and uncertainty, not only statistical significance.

Use the repository's power calculation to turn a baseline rate and minimum detectable effect into a sample-size requirement:

```bash
python -m conversion_intelligence.experiment --baseline 0.0323 --relative-lift 0.10
```
