# Reports

Training writes metrics and diagnostic figures to a generated subdirectory. They are intentionally rebuilt rather than versioned:

```bash
conversion-train --data data/raw/conversion_project.csv --output reports/generated
conversion-audit --data data/raw/conversion_project.csv --output reports/robustness/realism_audit.json
conversion-decide --data data/raw/conversion_project.csv \
  --output reports/generated/decision-optimization.json \
  --benchmark-output reports/generated/decision-optimization-benchmark.json
```

The portfolio summary in the root README records the reviewed results and assumptions. Generated
reports are ignored because the restricted source data is not distributed and the outputs should
be recomputed with an authorized local copy.

`source-benchmark.json` is the exception: it is a reviewed, aggregate-only provenance record used
to reconcile the resume's original holdout with the stricter portfolio protocol. It contains no
source rows, identifiers, text, or local filesystem paths.

`decision-optimization-benchmark.json` is a second aggregate-only provenance record. It preserves
the reviewed five-seed 4:1 decision-policy result, fixed-0.50 and theoretical-cost references,
generation environment, and interpretation limits without exposing source rows. The public schema
is extracted by code from the full local report; the full cost/selection-ceiling sweep remains
ignored under `generated/`. Install `requirements-benchmark.txt` for the exact reviewed splitter
environment before regenerating source results.

`point-in-time-pipeline.json` is independent synthetic systems evidence. It records only fixture
sizes, event/request exclusion counts, engine metadata, and pass/fail gates for Spark SQL/reference,
future-poison, one-row-per-request, and batch/single-request parity. It contains no source-derived
rows or metrics and does not claim production scale or online serving.

Package 0.2.1 hardens source, scoring, fitted-pipeline, and point-in-time output schemas. It does not
change or regenerate the three tracked evidence records or any reported metric; estimator
algorithms, split logic, decision rules, and point-in-time transformations are unchanged.
