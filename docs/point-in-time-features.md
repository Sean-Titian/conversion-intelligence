# Point-in-time feature contract

## Why this extension exists

The source conversion case has one especially important limitation: `total_pages_visited` is the
final session total. It is strongly associated with conversion, but it is not a valid feature for a
score made earlier in the session. A high retrospective score therefore does not establish that the
same model could support a live product decision.

This extension demonstrates the data contract that would be required before an in-session feature
could be evaluated honestly. It builds a point-in-time feature table from synthetic event records
with explicit event and availability timestamps. The purpose is to prove feature-timing semantics,
key integrity, and batch/reference parity—not to produce a larger model metric.

## Decision-time contract

Every feature row is anchored to an explicit score request. The decision time is `score_at`: the
instant at which the product would need the feature values in order to rank or prioritize a
session. Data that becomes known later must not be allowed to improve that score retroactively.

Two timestamps are needed for every page event:

| Timestamp | Meaning |
| --- | --- |
| `event_at` | When the user action occurred in event time |
| `available_at` | When the record became available to the feature computation |

A page event may contribute to a score only when both conditions are true:

```text
event_at < score_at
available_at <= score_at
```

The first comparison is deliberately strict. An event with `event_at == score_at` is excluded so
that equal timestamps cannot silently imply an ordering the data does not establish. The second
comparison permits a record available exactly at the scoring boundary, while excluding records that
arrive afterward.

| Event case | Included? | Reason |
| --- | --- | --- |
| Occurred before the score and was available by the score | Yes | It was observable at decision time |
| Occurred before the score but arrived after the score | No | Late-arriving data was not available to the decision |
| Occurred at or after the score | No | It is simultaneous or future behavior |
| `available_at` precedes `event_at` | No | The timestamp relationship is invalid and must fail validation |

The pipeline keeps late and future records in the raw synthetic event input so the exclusion logic
can be tested. It does not delete them first and then claim the remaining data were point-in-time
correct.

## Table grains and relationships

There are three input tables and one derived feature table. Their grains are part of the contract,
not implementation details.

| Table | Grain | Required identity and timing rule |
| --- | --- | --- |
| `sessions` | One row per synthetic session | `session_id` is complete and unique; every valid request satisfies `session_start_at <= score_at` |
| `score_requests` | One row per scoring decision | `score_id` is complete and unique; each row references one known `session_id` and has one `score_at` |
| `page_events` | One row per recorded page event | `event_id` is complete and unique; each row references one known `session_id` and has valid `event_at` and `available_at` values |
| `point_in_time_features` | One row per `score_id` | Contains the session-start fields and only page aggregates observable under the decision-time predicate |

The relationships are one session to many score requests and one session to many page events.
Valid page events also satisfy `session_start_at <= event_at <= available_at`.
Multiple score requests for the same session are allowed; each request must see only the event
prefix that was available at its own `score_at`. A page event can therefore contribute to a later
request without being visible to an earlier one.

The implementation must not join requests to events and then treat the expanded rows as scoring
units. It must aggregate back to `score_id` and verify that the output row count equals the input
score-request row count. Orphan requests, orphan events, incomplete keys, duplicate keys, invalid
timestamps, and join amplification fail closed rather than being silently dropped or deduplicated.

For a valid request with no eligible page events, `pages_observed_at_score_time` is zero. The final
session page total and the conversion outcome are not feature columns. They may be used separately
as retrospective descriptions or future labels, but never as inputs to the point-in-time score.

## Batch and scoring consistency

The Spark SQL/PySpark batch transformation and the small single-request reference computation must
implement the same cutoff rules, feature names, types, ordering, and missing-value behavior. Tests
compare them on deterministic synthetic fixtures, including sessions with:

- late-arriving page events;
- events exactly on the scoring boundary;
- future events that would change the final session total;
- no observable page events;
- more than one score request.

A poison-event test adds one record after the latest `score_at` in every session and verifies that
none of the session's feature rows change. This is evidence of engine parity and single-request
row-context invariance at the transformation boundary—not train/serve parity or evidence that an
online feature service has been built or operated.

## Isolation from the source benchmark

This workflow is synthetic-only. The educational source table has no event timestamps, availability
timestamps, stable session identifier, or score-request log, so it cannot be converted into this
schema without inventing facts. The point-in-time pipeline therefore must not:

- read the restricted source data;
- synthesize timestamps or identifiers for source rows;
- overwrite or reinterpret `reports/source-benchmark.json`;
- change any source-derived metric or deployment conclusion; or
- imply that the synthetic event process reconstructs the source population.

Synthetic row-level inputs and derived feature files remain in ignored synthetic directories.
Tracked reports, if any, are aggregate contract evidence only: row counts, exclusion counts, and
pass/fail results for key, timing, leakage, and parity gates.

## What this work can and cannot demonstrate

After the release gates pass, the repository may claim that it contains:

- an explicit prediction-time data contract;
- a reproducible Spark SQL/PySpark transformation over synthetic event data;
- fail-closed grain, timing, and join checks;
- protection against future and late-arriving feature leakage; and
- deterministic parity between batch and reference feature computation.

It must not claim:

- production scale, throughput, latency, reliability, or cost;
- a deployed streaming or online serving system;
- real-world train/serve parity;
- improved source-data or real-world model performance;
- validated product impact; or
- readiness to deploy the conversion score.

Local Spark execution demonstrates transformation correctness on controlled fixtures. Production
claims would require representative volume, infrastructure, monitoring, incident handling, service
level objectives, current labeled data, and prospective evaluation at the declared decision time.

## Release gates

This extension is complete only when automated tests verify all of the following:

1. the three input grains and the one-row-per-score output invariant;
2. strict `event_at < score_at` and `available_at <= score_at` behavior;
3. rejection of invalid keys, timestamps, orphan records, and join amplification;
4. boundary and exclusion behavior for late and simultaneous events, plus feature invariance to
   injected future poison events;
5. exact Spark SQL/PySpark parity with the reference computation on deterministic fixtures;
6. exclusion of outcome and final-session features from the scoring schema; and
7. continued isolation of private inputs and source benchmark artifacts.

Only after these gates pass should the profile describe SQL/PySpark as project evidence. The
evidence is point-in-time feature correctness—not a claim that adding Spark made the model more
accurate or the system production-ready.
