# Reports

Training writes metrics and diagnostic figures to a generated subdirectory. They are intentionally rebuilt rather than versioned:

```bash
conversion-train --data data/raw/conversion_project.csv --output reports/generated
conversion-audit --data data/raw/conversion_project.csv --output reports/robustness/realism_audit.json
```

The portfolio summary in the root README records the reviewed results and assumptions. Generated
reports are ignored because the licensed source data is not distributed and the outputs should be
recomputed with the local copy.

`source-benchmark.json` is the exception: it is a reviewed, aggregate-only provenance record used
to reconcile the resume's original holdout with the stricter portfolio protocol. It contains no
source rows, identifiers, text, or local filesystem paths.
