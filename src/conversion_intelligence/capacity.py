"""Label-free, request-level batch capacity; not a learned treatment policy."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from numbers import Integral, Real

import numpy as np

TIE_BREAK_DOMAIN = b"conversion-intelligence:batch-capacity:v1\x00"


def select_top_k(
    probabilities: Sequence[float] | np.ndarray,
    request_keys: Sequence[str] | np.ndarray,
    *,
    capacity: int,
    minimum_score: float | None = None,
) -> np.ndarray:
    """Return an input-aligned boolean mask selecting at most ``capacity`` requests.

    Inputs are positional, not pandas-index aligned. Scores must be finite real
    probabilities in [0, 1]; scores and floor use float64 for eligibility and ranking. Keys
    must be unique, non-empty, unpadded UTF-8 strings. All rows are validated,
    including ineligible rows and calls with zero capacity.

    Eligible requests are ordered by descending score, ascending SHA-256 of the
    versioned domain plus key, then ascending key bytes on a digest collision.
    Reordering a batch cannot change its selected key set. No labels are used.

    With no minimum score, the policy fills min(capacity, batch size), even for
    all-zero scores. An inclusive, caller-frozen minimum score permits unused
    capacity. This is a single-batch mechanical quota, not a guarantee across
    concurrent batches, a causal/value policy, or experiment randomization.
    Keys are neither returned nor logged; hashing them is not anonymization.
    """

    if any(
        np.ma.isMaskedArray(value)
        for value in (probabilities, request_keys, minimum_score)
    ):
        raise ValueError("Masked arrays are not supported; supply complete unmasked inputs")
    if isinstance(capacity, (bool, np.bool_)) or not isinstance(capacity, Integral):
        raise ValueError("capacity must be a non-negative integer")
    if capacity < 0:
        raise ValueError("capacity must be a non-negative integer")

    # Object conversion retains mixed booleans rather than silently coercing
    # e.g. [True, 0.5] to a floating-point array before validation.
    try:
        raw_scores = np.asarray(probabilities, dtype=object)
    except (TypeError, ValueError):
        raise ValueError("probabilities must be a one-dimensional real numeric sequence") from None
    if raw_scores.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for value in raw_scores
    ):
        raise ValueError("probabilities must contain only real numbers, not booleans or text")
    if any(not 0 <= value <= 1 for value in raw_scores):
        raise ValueError("probabilities must be finite numbers in [0, 1]")
    try:
        scores = np.asarray(raw_scores, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("probabilities must be finite numbers in [0, 1]") from None
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("probabilities must be finite numbers in [0, 1]")

    if minimum_score is not None:
        if (
            isinstance(minimum_score, (bool, np.bool_))
            or not isinstance(minimum_score, Real)
            or not 0 <= minimum_score <= 1
        ):
            raise ValueError("minimum_score must be None or a finite real number in [0, 1]")
        try:
            minimum_score = float(minimum_score)
        except (ValueError, OverflowError):
            raise ValueError(
                "minimum_score must be None or a finite real number in [0, 1]"
            ) from None
        if not np.isfinite(minimum_score) or not 0 <= minimum_score <= 1:
            raise ValueError("minimum_score must be None or a finite real number in [0, 1]")

    try:
        keys = np.asarray(request_keys, dtype=object)
    except (TypeError, ValueError):
        raise ValueError("request_keys must be a one-dimensional string sequence") from None
    if keys.ndim != 1 or len(keys) != len(scores):
        raise ValueError("request_keys must be one-dimensional and match the score count")
    if any(
        not isinstance(key, str) or not key or key != key.strip()
        for key in keys
    ):
        raise ValueError("request_keys must contain non-empty unpadded strings")
    if len(set(keys)) != len(keys):
        raise ValueError("request_keys must be unique within the batch")
    try:
        key_bytes = [key.encode("utf-8") for key in keys]
    except UnicodeError:
        raise ValueError("request_keys must be valid UTF-8 strings") from None

    eligible = np.flatnonzero(
        np.ones(len(scores), dtype=bool)
        if minimum_score is None
        else scores >= minimum_score
    )
    ranked = sorted(
        eligible,
        key=lambda index: (
            -float(scores[index]),
            hashlib.sha256(TIE_BREAK_DOMAIN + key_bytes[index]).digest(),
            key_bytes[index],
        ),
    )
    selected = np.zeros(len(scores), dtype=bool)
    selected[ranked[: min(int(capacity), len(ranked))]] = True
    return selected
