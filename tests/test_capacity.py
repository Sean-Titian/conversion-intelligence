"""Label-free batch-capacity contracts on authored synthetic requests."""

from __future__ import annotations

from fractions import Fraction
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

import conversion_intelligence.capacity as capacity_module
from conversion_intelligence.capacity import select_top_k
from conversion_intelligence.features import features_for_scenario
from conversion_intelligence.modeling import build_model_pipeline
from conversion_intelligence.synthetic import generate_synthetic_conversion_data

DOMAIN = b"conversion-intelligence:batch-capacity:v1\x00"


def _oracle(
    scores: list[float],
    keys: list[str],
    limit: int,
    minimum: float | None = None,
) -> np.ndarray:
    ranked = sorted(
        (
            (-float(score), sha256(DOMAIN + key.encode("utf-8")).digest(),
             key.encode("utf-8"), index)
            for index, (score, key) in enumerate(zip(scores, keys, strict=True))
            if minimum is None or score >= minimum
        )
    )
    mask = np.zeros(len(scores), dtype=bool)
    for _, _, _, index in ranked[:limit]:
        mask[index] = True
    return mask


def _selected_keys(keys: list[str], mask: np.ndarray) -> set[str]:
    return {key for key, selected in zip(keys, mask, strict=True) if selected}


def _assert_rejected(scores: object, keys: object, **kwargs: object) -> None:
    with pytest.raises(ValueError) as error:
        select_top_k(scores, keys, **kwargs)
    assert str(error.value)
    if isinstance(keys, (list, tuple)):
        for value in keys:
            if isinstance(value, str) and len(value) > 12:
                assert value not in str(error.value)


def test_domain_and_sha256_order_match_independent_oracle() -> None:
    assert capacity_module.TIE_BREAK_DOMAIN == DOMAIN
    scores = [0.8, 0.9, 0.8, 0.8, 0.1]
    keys = ["synthetic_b", "synthetic_a", "synthetic_z", "synthetic_c", "synthetic_d"]
    mask = select_top_k(scores, keys, capacity=3)
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.dtype(bool)
    assert mask.shape == (5,)
    assert mask[1] and not mask[4]
    np.testing.assert_array_equal(mask, _oracle(scores, keys, 3))


@pytest.mark.parametrize("limit", [0, 1, 2, 4, 5, 10**100])
def test_capacity_selects_exactly_minimum_of_limit_and_batch_size(limit: int) -> None:
    scores = [0.0, 0.3, 0.3, 1.0]
    keys = [f"synthetic_request_{index}" for index in range(4)]
    mask = select_top_k(scores, keys, capacity=limit)
    assert int(mask.sum()) == min(limit, len(scores))
    np.testing.assert_array_equal(mask, _oracle(scores, keys, limit))


@pytest.mark.parametrize("limit", [0, 3])
def test_empty_batch_returns_empty_boolean_mask(limit: int) -> None:
    mask = select_top_k([], [], capacity=limit)
    assert mask.dtype == np.dtype(bool)
    assert mask.shape == (0,)


def test_numpy_integer_capacity_and_real_numeric_arrays_are_supported() -> None:
    scores = np.array([0, 1], dtype=np.int64)
    mask = select_top_k(scores, ["synthetic_low", "synthetic_high"], capacity=np.int64(1))
    np.testing.assert_array_equal(mask, [False, True])


def test_zero_scores_still_fill_capacity_without_a_minimum_score() -> None:
    scores = [0.0] * 4
    keys = [f"synthetic_zero_{index}" for index in range(4)]
    mask = select_top_k(scores, keys, capacity=2)
    assert int(mask.sum()) == 2
    np.testing.assert_array_equal(mask, _oracle(scores, keys, 2))


@pytest.mark.parametrize("seed", range(16))
def test_selected_key_set_is_invariant_to_batch_row_order(seed: int) -> None:
    scores = np.array([0.9] * 3 + [0.5] * 14 + [0.1] * 5)
    keys = [f"synthetic_reordered_{index:03d}" for index in range(len(scores))]
    baseline = select_top_k(scores, keys, capacity=9)
    permutation = np.random.default_rng(seed).permutation(len(scores))
    shuffled_keys = [keys[index] for index in permutation]
    shuffled = select_top_k(scores[permutation], shuffled_keys, capacity=9)
    assert _selected_keys(keys, baseline) == _selected_keys(shuffled_keys, shuffled)


def test_increasing_capacity_produces_nested_selected_key_sets() -> None:
    scores = [0.1, 0.8, 0.8, 0.8, 0.0]
    keys = [f"synthetic_nested_{index}" for index in range(5)]
    previous: set[str] = set()
    for limit in range(7):
        current = _selected_keys(keys, select_top_k(scores, keys, capacity=limit))
        assert previous <= current
        assert len(current) == min(limit, len(scores))
        previous = current


def test_score_shift_can_overfill_threshold_but_not_batch_capacity() -> None:
    original = np.array([0.1, 0.2, 0.3, 0.4, 0.8])
    shifted = original + 0.2
    keys = [f"synthetic_shift_{index}" for index in range(len(original))]
    assert int((original >= 0.5).sum()) == 1
    assert int((shifted >= 0.5).sum()) > 2
    first = select_top_k(original, keys, capacity=2)
    second = select_top_k(shifted, keys, capacity=2)
    assert int(first.sum()) == int(second.sum()) == 2
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("minimum", [0.0, 0.5, 1.0])
def test_minimum_score_is_inclusive_and_respects_capacity(minimum: float) -> None:
    scores = [0.0, 0.5, 0.5, 0.9, 1.0]
    keys = [f"synthetic_floor_{index}" for index in range(len(scores))]
    mask = select_top_k(scores, keys, capacity=3, minimum_score=minimum)
    np.testing.assert_array_equal(mask, _oracle(scores, keys, 3, minimum))
    assert all(score >= minimum for score, selected in zip(scores, mask, strict=True) if selected)


def test_minimum_score_can_leave_capacity_unused_or_select_nothing() -> None:
    scores = [0.1, 0.2, 0.7]
    keys = [f"synthetic_unused_{index}" for index in range(len(scores))]
    one = select_top_k(scores, keys, capacity=10, minimum_score=0.7)
    none = select_top_k(scores, keys, capacity=10, minimum_score=0.8)
    np.testing.assert_array_equal(one, [False, False, True])
    assert not none.any()


def test_digest_collision_falls_back_to_raw_utf8_keys_not_input_order(monkeypatch) -> None:
    class CollidingDigest:
        def digest(self) -> bytes:
            return b"\x00" * 32

    monkeypatch.setattr(capacity_module.hashlib, "sha256", lambda _value: CollidingDigest())
    keys = ["synthetic_z", "synthetic_é", "synthetic_a", "synthetic_b"]
    mask = select_top_k([0.5] * 4, keys, capacity=2)
    expected_keys = set(sorted(keys, key=lambda key: key.encode("utf-8"))[:2])
    assert _selected_keys(keys, mask) == expected_keys


@pytest.mark.parametrize(
    "invalid",
    [True, np.bool_(False), -1, 1.0, 1.5, float("nan"), float("inf"), "1", None, 1j],
)
def test_invalid_capacity_is_rejected(invalid: object) -> None:
    _assert_rejected([0.2], ["synthetic_request_do_not_echo"], capacity=invalid)


@pytest.mark.parametrize(
    "invalid",
    [
        [float("nan")], [float("inf")], [float("-inf")], [-0.01], [1.01],
        [True], [np.bool_(False)], [0.2, True], [0.2, np.bool_(True)],
        ["0.2"], [None], [1j], [10**1000], [[0.2]], 0.2,
    ],
)
def test_invalid_scores_are_rejected_without_echoing_keys(invalid: object) -> None:
    keys = ["synthetic_request_do_not_echo"]
    if isinstance(invalid, list) and len(invalid) == 2:
        keys.append("synthetic_other_request_do_not_echo")
    _assert_rejected(invalid, keys, capacity=1)


@pytest.mark.parametrize(
    "invalid",
    [True, np.bool_(False), -0.01, 1.01, float("nan"), float("inf"), "0.5", 1j],
)
def test_invalid_minimum_score_is_rejected(invalid: object) -> None:
    _assert_rejected(
        [0.2], ["synthetic_request_do_not_echo"], capacity=1, minimum_score=invalid
    )


@pytest.mark.parametrize(
    "invalid",
    [Fraction(-1, 10**1000), Fraction(10**1000 + 1, 10**1000)],
)
def test_out_of_range_exact_real_scores_are_rejected_before_float64_rounding(
    invalid: Fraction,
) -> None:
    assert float(invalid) in (0.0, 1.0)
    _assert_rejected([invalid], ["synthetic_request_do_not_echo"], capacity=1)


@pytest.mark.parametrize(
    "invalid",
    [Fraction(-1, 10**1000), Fraction(10**1000 + 1, 10**1000)],
)
def test_out_of_range_exact_real_floor_is_rejected_before_float64_rounding(
    invalid: Fraction,
) -> None:
    assert float(invalid) in (0.0, 1.0)
    _assert_rejected(
        [0.5], ["synthetic_request_do_not_echo"], capacity=1, minimum_score=invalid
    )


def test_valid_exact_real_scores_and_floor_remain_supported() -> None:
    selected = select_top_k(
        [Fraction(1, 4), Fraction(3, 4)],
        ["synthetic_fraction_low", "synthetic_fraction_high"],
        capacity=2,
        minimum_score=Fraction(3, 4),
    )
    np.testing.assert_array_equal(selected, [False, True])


@pytest.mark.parametrize("limit", [0, 1])
@pytest.mark.parametrize("field", ["scores", "keys", "minimum_score"])
@pytest.mark.parametrize("masked", [False, True])
def test_masked_array_containers_are_rejected_even_without_an_active_mask(
    limit: int, field: str, masked: bool,
) -> None:
    scores: object = [0.5]
    keys: object = ["synthetic_request_do_not_echo"]
    minimum: object = None
    if field == "scores":
        scores = np.ma.array([0.5], mask=[masked])
    elif field == "keys":
        keys = np.ma.array(["synthetic_request_do_not_echo"], mask=[masked])
    else:
        minimum = np.ma.array(0.2, mask=masked)
    _assert_rejected(scores, keys, capacity=limit, minimum_score=minimum)


@pytest.mark.parametrize("limit", [0, 1])
@pytest.mark.parametrize("field", ["scores", "keys", "minimum_score"])
def test_masked_scalar_inside_sequence_cannot_bypass_validation(
    limit: int, field: str,
) -> None:
    scores: object = [0.5]
    keys: object = ["synthetic_request_do_not_echo"]
    minimum: object = None
    if field == "scores":
        scores = [np.ma.masked]
    elif field == "keys":
        keys = [np.ma.masked]
    else:
        minimum = [np.ma.masked]
    _assert_rejected(scores, keys, capacity=limit, minimum_score=minimum)


@pytest.mark.parametrize(
    "keys",
    [
        [""], ["   "], [" synthetic_request_do_not_echo"],
        ["synthetic_request_do_not_echo "], ["synthetic_request_do_not_echo\n"],
        [None], [1], [True], [b"synthetic_bytes"], ["synthetic_invalid_\ud800"],
        [["synthetic_nested"]], "synthetic_scalar", [],
    ],
)
def test_invalid_or_misaligned_keys_are_rejected(keys: object) -> None:
    _assert_rejected([0.5], keys, capacity=1)


def test_duplicate_keys_are_rejected_across_the_whole_batch() -> None:
    _assert_rejected(
        [0.9, 0.1], ["synthetic_duplicate_do_not_echo"] * 2, capacity=1
    )


@pytest.mark.parametrize("limit", [0, 1])
def test_validation_is_not_bypassed_when_nothing_would_be_selected(limit: int) -> None:
    _assert_rejected(
        [float("nan")], ["synthetic_request_do_not_echo"], capacity=limit,
        minimum_score=1.0,
    )
    _assert_rejected(
        [0.0, 0.0], ["synthetic_duplicate_do_not_echo"] * 2, capacity=limit,
        minimum_score=1.0,
    )


def test_inputs_are_not_mutated() -> None:
    scores = np.array([0.1, 0.8, 0.8])
    keys = ["synthetic_immutable_c", "synthetic_immutable_b", "synthetic_immutable_a"]
    before_scores = scores.copy()
    before_keys = keys.copy()
    select_top_k(scores, keys, capacity=2, minimum_score=0.5)
    np.testing.assert_array_equal(scores, before_scores)
    assert keys == before_keys


def test_fitted_pipeline_scores_feed_label_free_capacity_without_metadata_poison() -> None:
    fixture = generate_synthetic_conversion_data(400, seed=20261111, target_rate=0.15)
    features = features_for_scenario("acquisition")
    model = build_model_pipeline(features, "logistic_regression")
    model.fit(fixture.iloc[:300], fixture["converted"].iloc[:300])
    requests = fixture.iloc[300:].copy()
    scores = model.predict_proba(requests)[:, 1]
    poisoned = requests.copy(deep=True)
    poisoned["converted"] = 1 - poisoned["converted"]
    poisoned["total_pages_visited"] = 999_999
    poisoned["synthetic_metadata"] = "ignored_not_a_feature"
    poisoned_scores = model.predict_proba(poisoned)[:, 1]
    np.testing.assert_array_equal(scores, poisoned_scores)
    keys = [f"synthetic_pipeline_request_{index:03d}" for index in range(len(requests))]
    selected = select_top_k(scores, keys, capacity=7)
    poisoned_selected = select_top_k(poisoned_scores, keys, capacity=7)
    assert int(selected.sum()) == 7
    np.testing.assert_array_equal(selected, poisoned_selected)
    pd.testing.assert_frame_equal(requests, fixture.iloc[300:])
