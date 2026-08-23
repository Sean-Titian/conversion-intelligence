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
