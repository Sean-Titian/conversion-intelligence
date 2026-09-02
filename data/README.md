# Data contract

The original educational dataset is not included because its redistribution terms are not documented. Place an authorized copy at `data/raw/conversion_project.csv`, or generate a schema-compatible synthetic dataset:

```bash
conversion-synthesize --output data/synthetic/conversion.csv
```

Expected columns:

| Column | Type | Meaning | Available at |
| --- | --- | --- | --- |
| `country` | categorical | Country inferred from IP | session start |
| `age` | integer | Self-reported age | session start |
| `new_user` | binary | Account created in this session | session start |
| `source` | categorical | Acquisition channel | session start |
| `total_pages_visited` | integer | Final pages viewed in the completed session | session end in this source |
| `converted` | binary | Purchase completed in the session | outcome |

The timing distinction is deliberate: `total_pages_visited` is unavailable for an acquisition-time
score and is also a proxy for latent purchase intent. Because the source has no page-event timestamps,
the full-feature result is reported only as a retrospective upper bound. A real-time model needs a
new field such as `pages_observed_at_score_time`, plus both event time and score time.

## Separate synthetic event contract

The point-in-time pipeline does not add invented timestamps or identifiers to this source-compatible
table. It uses a separate, authored synthetic contract:

| Frame | Exact public schema | Grain |
| --- | --- | --- |
| `sessions` | `session_id`, `user_id`, `session_start_at`, `country`, `age`, `new_user`, `source` | one row per synthetic session |
| `score_requests` | `score_id`, `session_id`, `score_at` | one row per scoring decision; several cutoffs per session are allowed |
| `page_events` | `event_id`, `session_id`, `event_at`, `available_at`, `page_category` | one row per synthetic page event |

All identifiers must be complete; primary keys must be unique; child session keys must exist; and
timestamps must carry a UTC offset. Scores before session start, events before session start, and
`available_at < event_at` fail validation. The feature query includes an event only when
`event_at < score_at` and `available_at <= score_at`, then restores exactly one row per `score_id`.

The public generator creates these frames in memory. Row-level synthetic inputs, feature dumps,
CSV/Parquet files, Spark shards, warehouses, and model artifacts are ignored. The only versioned
result is an aggregate verification record with counts and pass/fail gates. See
[`docs/point-in-time-features.md`](../docs/point-in-time-features.md) for the interpretation boundary.
