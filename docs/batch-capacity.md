# Fixed-capacity batch selection

## The decision

A hypothetical manual-review owner can process at most K scoring requests in one batch. Each
request has a conversion-likelihood score computed from the declared prediction-time features.
This module demonstrates the mechanical capacity boundary, not whether acting on that score helps
customers or changes conversion. It contains no model fit, threshold search, or outcome argument.

| Rule | What it controls | What it does not guarantee |
| --- | --- | --- |
| Validation-selected score threshold | An offline cost/rate objective on validation rows | K requests on a shifted future batch |
| Existing top-fraction evaluation | Offline ranking metrics, with stable-input-order ties | A serving policy or row-order-invariant ties |
| New `select_top_k` | At most K eligible requests in this single call | Business value, user-level or global queue capacity |

These remain distinct APIs. Existing source-model metrics are not recomputed with invented
request keys or the new tie policy.

## Runnable synthetic example

After installing the package as described in the README, run:

```python
import numpy as np

from conversion_intelligence.capacity import select_top_k

scores = np.array([0.9, 0.8, 0.8, 0.1])
keys = [f"synthetic_request_{letter}" for letter in "abcd"]
mask = select_top_k(scores, keys, capacity=2, minimum_score=0.5)

assert mask.dtype == np.dtype(bool)
assert mask.sum() == 2
assert mask[0] and not mask[3]  # exactly one of the two tied requests also fits
```

This is a hand-constructed contract example, not a customer-performance result. Do not use these
scores or the illustrative 0.5 eligibility floor as a business recommendation.

## Input and selection contract

- Probabilities are a one-dimensional real numeric sequence, finite and within `[0, 1]`.
  Booleans, text, complex values, and missing values fail closed. The policy ranks the float64
  representation of scores and the eligibility floor after checking their original range;
  higher-precision differences may become ties after conversion. NumPy masked-array containers
  are rejected before conversion
  so a hidden missing-value mask cannot be discarded for scores, keys, or the minimum score.
- `request_keys` is a same-length one-dimensional sequence of unique, non-empty UTF-8 strings.
  Whitespace-only or padded keys, non-string keys, and duplicates are rejected, not repaired.
  Keys are opaque scoring-request tokens, not model features or labels. The caller must correctly
  pair each key with its score; pandas indexes are not aligned automatically.
- `capacity` is a non-negative integer, not a boolean or floating-point value. Zero and values
  larger than the batch are allowed. Empty batches are allowed. Every row is validated even when
  K is zero or a row will be ineligible.
- `minimum_score=None` makes every valid row eligible, filling `min(K, n)` even for all-zero
  scores. A supplied finite real floor within `[0, 1]` uses `score >= minimum_score`, so the
  result contains `min(K, eligible rows)` requests and may leave capacity unused. The caller must
  choose and freeze any floor separately; this API does not optimize it using outcomes.
- Ranking is score descending, then ascending SHA-256 of the exact byte prefix
  `b"conversion-intelligence:batch-capacity:v1\x00"` concatenated with the key's UTF-8 bytes,
  then raw key bytes ascending if digests collide. Keys are not Unicode-normalized. This fixed
  rule gives the same selected key set for any permutation of the same score/key pairs.
- The returned NumPy boolean mask follows the input's positional order. It is not a ranked list,
  and no keys, scores, labels, or model objects are written to disk or logged. Errors do not echo
  input values. Inputs are not mutated.

## Verification and boundaries

`tests/test_capacity.py` checks a direct sorting oracle, exact cardinality and eligibility,
zero/oversized capacity, all-score ties, digest-collision fallback, row permutations, nested
capacities, input immutability, bad schemas, and score shift. A synthetic fitted-pipeline handoff
checks that scores and selection do not use extra outcome/metadata columns. Release CI runs these
tests from source and the installed wheel; the estimator, feature projection, split, existing
evaluation functions, SQL, and three tracked aggregate reports are unchanged.

The capacity guarantee is per call, not a distributed transaction. Splitting one queue into several
calls can consume K for every call. Reusing request keys in a later call is not prevented, and
multiple requests from one person are not user-level deduplication. Callers need separate eligibility,
consent, identity, idempotency, concurrency, and queue-state controls before real operations.

Deterministic hashing is neither anonymization nor random assignment. Keys may still identify
people and must stay local; no real keys or selected rows belong in this public repository.
Callers can manipulate key choices, and the rule does not prove fair selection among tied requests.
The helper does not validate score calibration or prediction-time provenance. A conversion-probability
ranking is not an incremental-treatment ranking: real value still needs measured costs, guardrails,
current labeled data, and a randomized or otherwise justified prospective evaluation.
