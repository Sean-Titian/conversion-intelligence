# Experiment handoff: from prediction to causality

## Status and decision

This is a **design draft, not rollout approval**. The observational model ranks likely converters;
it cannot establish that increasing page views or selecting a high-score user changes conversion.
The proposed intervention is a lightweight recommendation module, with unchanged checkout and
pricing. Its hypothesis is a higher seven-day purchase probability without unacceptable customer
harm. No experiment has been run here, and no real user-level seven-day baseline is available.

The source's 3.23% session conversion rate cannot be relabeled as the user-level baseline below.
Numeric guardrail margins, acceptable costs, eligibility consent, observation completeness, and
the enrollment/calendar horizon still need owner approval before this design is launchable.

## Scoring, admission, and assignment are separate

| Boundary | Unit / timing | Required decision |
| --- | --- | --- |
| Model scoring | Request at the declared prediction time | Only already-available features; no final pages or outcomes |
| Optional capacity admission | Pre-assignment eligible population | Freeze score/floor/budget and a distinct-user rule, symmetrically for both arms |
| Experiment assignment | One eligible authenticated user, once | Separate random assignment before rendering; persist the arm across sessions |
| Primary analysis | All assigned users | Fixed outcome window from assignment; ITT by original arm |

The request-level `select_top_k` helper is not a user allocator or randomizer. If a future design
uses it before enrollment, it must first resolve one eligible request per distinct user and manage
cross-batch capacity separately. Its deterministic hash tie rule is not experiment randomization.
The same admission rule must apply before arm assignment; never select treatment users afterward
because they rendered, clicked, or acquired a high post-treatment score.

An experiment inside a pre-selected cohort would estimate the module effect **within that cohort**,
not the incremental value of the ranking policy. Testing ranking/routing itself needs a separately
specified policy experiment with a budget-matched comparison. Neither system is implemented here.

## Population, clock, and estimand

- **Population:** consented, eligible authenticated users under a frozen pre-assignment eligibility
  snapshot, with stable identity and one assignment per user. Repeated visits do not re-enroll.
- **Allocation:** control is the existing experience; treatment is the recommendation module.
  Assignment precedes the first possible render and remains stable across later sessions.
- **Clock:** `assignment_at`, not first exposure. For user i the primary binary outcome is at least
  one completed purchase in `[assignment_at_i, assignment_at_i + 7 days)`. A purchase exactly at
  the right boundary is outside this pre-specified window. Returning or rendering later does not
  reset the clock.
- **Denominator:** every originally assigned user, analyzed in the assigned arm, whether the module
  rendered, the user engaged, or the experience crossed over. Rendering/exposure is a diagnostic,
  not a post-assignment eligibility filter. Non-rendering alone does not make an outcome missing.
- **Estimand:** treatment-minus-control difference in these user purchase risks for the enrolled
  population. Report absolute and relative effects, denominators, and a two-sided 95% interval.
  No numerical treatment effect or uncertainty interval is claimed by this planning document.

Wait until the final assignment's full window plus a pre-declared ingestion lag has elapsed.
Independently verify source coverage or complete endpoint snapshots for all assigned users.
An absent purchase event is zero only with demonstrated outcome coverage; an immature or
unobserved outcome is not zero. Missingness must not silently remove users from ITT or be fixed
by increasing planned enrollment. Unresolved coverage, differential missingness, assignment
contamination, or identity problems keep the decision on hold.

Triggered/exposed-only analyses are not interchangeable with this ITT estimand. Any future
triggered analysis requires a separately justified, symmetric counterfactual trigger and logging
checks; it must not replace the primary population simply because it gives a better result.

## Metrics, integrity, and decision gates

- Primary: the assignment-origin seven-day user purchase risk above.
- Secondary: product views, add-to-cart, and revenue per assigned user; engagement is a mediator
  or diagnostic, not proof of why conversion changed.
- Guardrails: latency, errors, bounce, refunds, and opt-out/contact risk. Freeze each denominator,
  collection method, maturity window, harm direction, and acceptable margin before launch.
  Refunds may need longer observation than purchases; purchase maturity alone cannot certify them.
- Check assignment uniqueness, SRM, pre-treatment balance/placebos, event/availability timing,
  source coverage, and cross-arm exposure. SRM or a negative-control failure is an investigation
  gate, not an invitation to keep favorable metrics.
- Freeze one primary comparison, calendar/enrollment horizon, attrition/missingness handling, and
  multiplicity strategy. Do not stop early on a nominal p-value. Powered interaction tests need
  their own design; otherwise subgroup cuts are exploratory.
- Statistical primary evidence alone does not authorize launch: guardrail non-inferiority,
  observability, integrity, and measured business costs must also support a pre-specified rule.
  Those margins and a complete decision rule have not been supplied, so readiness remains on hold.

## Reproduce the illustrative planning calculation

```bash
python -m conversion_intelligence.experiment --baseline 0.0323 --relative-lift 0.10
```

This command returns **51,893 independent units per arm (103,786 total)** under the displayed
assumptions, including its default alpha, target power, and attrition inflation. These rates are
caller-supplied assumptions, not measured inputs for the proposed user experiment. For a real
plan, estimate the untreated rate with the **same unit, population/admission rule, outcome, and
assignment-origin window**. Do not use a session denominator or infer a seven-day baseline from
the restricted source table without the missing identities/timestamps.

The function uses equal allocation and independent Bernoulli units, with pooled variance under
the null and separate variances under the alternative. It inverts a normal approximation while
neglecting the distant rejection tail, then divides by `(1 - attrition)` and rounds up enrollment.
It is not an exact test, a design-effect/cluster adjustment, a guardrail or multiplicity calculation,
or an empirical operating-characteristics study. Extreme numerically unsupported inputs are
rejected. The attrition parameter is only a hypothetical enrollment buffer, not permission to
discard assigned users or a correction for missing-outcome bias. Validate the final design's full
decision rule by an appropriate prospective rehearsal or simulation before asserting attained power.

The [statsmodels planning-method documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.samplesize_proportions_2indep_onetail.html)
describes the corresponding one-tail inversion approximation for a two-sided test and its ignored
far tail. Microsoft's [pre-experiment patterns](https://www.microsoft.com/en-us/research/articles/patterns-of-trustworthy-experimentation-pre-experiment-stage/)
discuss randomization-unit and counterfactual-logging requirements; its
[post-experiment patterns](https://www.microsoft.com/en-us/research/articles/patterns-of-trustworthy-experimentation-post-experiment-stage/)
explain why triggered and overall-population analyses require different interpretation. The
specific seven-day assignment-origin estimand here is this project's design choice, not an
implementation or a claim that those sources validate this population or launch plan.
