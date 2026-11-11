"""Semantic output gates for source-free point-in-time engine fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from conversion_intelligence.point_in_time import (
    OUTPUT_COLUMNS,
    canonicalize_feature_frame,
)


def _valid_output() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "score_id": ["synthetic_private_score_do_not_echo"],
            "session_id": ["synthetic_private_session_do_not_echo"],
            "user_id": ["synthetic_private_user_do_not_echo"],
            "score_at_us": [5_000_000],
            "country": ["synthetic_country"],
            "age": [35],
            "new_user": [1],
            "source": ["synthetic_source"],
            "seconds_since_session_start": [5],
            "pages_observed_at_score_time": [2],
            "unique_page_categories_observed": [2],
            "feature_max_event_at_us": pd.array([3_750_000], dtype="Int64"),
            "seconds_since_last_page_view": pd.array([1], dtype="Int64"),
        }
    )
    return frame.loc[:, OUTPUT_COLUMNS]


def _assert_rejected_without_identifiers(frame: pd.DataFrame) -> None:
    before = frame.copy(deep=True)
    with pytest.raises(ValueError) as error:
        canonicalize_feature_frame(frame)
    message = str(error.value)
    assert message
    for column in ("score_id", "session_id", "user_id"):
        assert str(frame.iloc[0][column]) not in message
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("age", 12),
        ("age", 101),
        ("new_user", -1),
        ("new_user", 2),
        ("seconds_since_session_start", -1),
        ("pages_observed_at_score_time", -1),
        ("unique_page_categories_observed", -1),
        ("unique_page_categories_observed", 3),
    ],
)
def test_output_rejects_invalid_domains_without_echoing_identifiers(
    column: str, invalid_value: int
) -> None:
    frame = _valid_output()
    frame.loc[0, column] = invalid_value
    _assert_rejected_without_identifiers(frame)


@pytest.mark.parametrize("age", [13, 100])
@pytest.mark.parametrize("new_user", [0, 1])
def test_output_accepts_closed_age_and_binary_user_boundaries(
    age: int, new_user: int
) -> None:
    frame = _valid_output()
    frame.loc[0, "age"] = age
    frame.loc[0, "new_user"] = new_user
    result = canonicalize_feature_frame(frame)
    assert result.loc[0, "age"] == age
    assert result.loc[0, "new_user"] == new_user


@pytest.mark.parametrize(
    ("categories", "maximum_event", "recency"),
    [
        (1, None, None),
        (0, 3_750_000, None),
        (0, None, 1),
        (0, 3_750_000, 1),
    ],
)
def test_zero_history_rejects_category_or_timestamp_evidence(
    categories: int, maximum_event: int | None, recency: int | None
) -> None:
    frame = _valid_output()
    frame.loc[0, "pages_observed_at_score_time"] = 0
    frame.loc[0, "unique_page_categories_observed"] = categories
    frame["feature_max_event_at_us"] = pd.array([maximum_event], dtype="Int64")
    frame["seconds_since_last_page_view"] = pd.array([recency], dtype="Int64")
    _assert_rejected_without_identifiers(frame)


@pytest.mark.parametrize(
    ("categories", "maximum_event", "recency"),
    [
        (0, 3_750_000, 1),
        (2, None, 1),
        (2, 3_750_000, None),
        (2, None, None),
    ],
)
def test_positive_history_requires_categories_and_both_temporal_fields(
    categories: int, maximum_event: int | None, recency: int | None
) -> None:
    frame = _valid_output()
    frame.loc[0, "unique_page_categories_observed"] = categories
    frame["feature_max_event_at_us"] = pd.array([maximum_event], dtype="Int64")
    frame["seconds_since_last_page_view"] = pd.array([recency], dtype="Int64")
    _assert_rejected_without_identifiers(frame)


@pytest.mark.parametrize("recency", [-1, 0, 2, 6])
def test_output_rejects_negative_inconsistent_or_excess_recency(recency: int) -> None:
    frame = _valid_output()
    frame.loc[0, "seconds_since_last_page_view"] = recency
    _assert_rejected_without_identifiers(frame)


def test_output_rejects_consistent_recency_exceeding_session_elapsed() -> None:
    frame = _valid_output()
    frame.loc[0, "seconds_since_session_start"] = 0
    _assert_rejected_without_identifiers(frame)


@pytest.mark.parametrize("maximum_event", [5_000_000, 5_000_001])
def test_output_preserves_strict_event_before_score_boundary(maximum_event: int) -> None:
    frame = _valid_output()
    frame.loc[0, "feature_max_event_at_us"] = maximum_event
    frame.loc[0, "seconds_since_last_page_view"] = 0
    _assert_rejected_without_identifiers(frame)


def test_subsecond_history_permits_zero_recency_and_zero_elapsed_seconds() -> None:
    frame = _valid_output()
    frame.loc[0, "seconds_since_session_start"] = 0
    frame.loc[0, "feature_max_event_at_us"] = 4_999_999
    frame.loc[0, "seconds_since_last_page_view"] = 0
    result = canonicalize_feature_frame(frame)
    assert result.loc[0, "seconds_since_last_page_view"] == 0
    assert result.loc[0, "feature_max_event_at_us"] < result.loc[0, "score_at_us"]


def test_pre_epoch_timestamps_are_allowed_when_temporally_consistent() -> None:
    frame = _valid_output()
    frame.loc[0, "score_at_us"] = -1_000_000
    frame.loc[0, "feature_max_event_at_us"] = -2_250_000
    result = canonicalize_feature_frame(frame)
    assert result.loc[0, "seconds_since_last_page_view"] == 1


def test_recency_uses_python_integer_difference_without_int64_wraparound() -> None:
    frame = _valid_output()
    score_at = 9_000_000_000_000_000_000
    maximum_event = -9_000_000_000_000_000_000
    expected_recency = (score_at - maximum_event) // 1_000_000
    frame.loc[0, "score_at_us"] = score_at
    frame.loc[0, "feature_max_event_at_us"] = maximum_event
    frame.loc[0, "seconds_since_last_page_view"] = expected_recency
    frame.loc[0, "seconds_since_session_start"] = expected_recency
    result = canonicalize_feature_frame(frame)
    assert result.loc[0, "seconds_since_last_page_view"] == expected_recency


def test_zero_history_accepts_nullable_engine_float_columns() -> None:
    frame = _valid_output()
    frame.loc[0, "pages_observed_at_score_time"] = 0
    frame.loc[0, "unique_page_categories_observed"] = 0
    frame["feature_max_event_at_us"] = [float("nan")]
    frame["seconds_since_last_page_view"] = [float("nan")]
    result = canonicalize_feature_frame(frame)
    assert result.loc[0, "pages_observed_at_score_time"] == 0
    assert result.loc[0, "unique_page_categories_observed"] == 0
    assert pd.isna(result.loc[0, "feature_max_event_at_us"])
    assert pd.isna(result.loc[0, "seconds_since_last_page_view"])
    assert str(result["feature_max_event_at_us"].dtype) == "Int64"
    assert str(result["seconds_since_last_page_view"].dtype) == "Int64"


def test_nonmissing_nullable_engine_float_values_are_normalized_without_truncation() -> None:
    frame = _valid_output()
    frame["feature_max_event_at_us"] = frame["feature_max_event_at_us"].astype("float64")
    frame["seconds_since_last_page_view"] = frame[
        "seconds_since_last_page_view"
    ].astype("float64")
    result = canonicalize_feature_frame(frame)
    assert result.loc[0, "feature_max_event_at_us"] == 3_750_000
    assert result.loc[0, "seconds_since_last_page_view"] == 1
    assert str(result["feature_max_event_at_us"].dtype) == "Int64"
    assert str(result["seconds_since_last_page_view"].dtype) == "Int64"


def test_empty_output_retains_canonical_schema() -> None:
    empty = _valid_output().iloc[:0].copy()
    result = canonicalize_feature_frame(empty)
    assert result.empty
    assert list(result.columns) == list(OUTPUT_COLUMNS)
    assert str(result["feature_max_event_at_us"].dtype) == "Int64"


def test_reordered_rows_and_columns_are_canonicalized_without_input_mutation() -> None:
    later = _valid_output()
    earlier = _valid_output()
    later.loc[0, "score_id"] = "synthetic_score_b"
    earlier.loc[0, "score_id"] = "synthetic_score_a"
    frame = pd.concat([later, earlier], ignore_index=True)
    frame.index = [9, 4]
    frame = frame.loc[:, list(reversed(OUTPUT_COLUMNS))]
    before = frame.copy(deep=True)
    result = canonicalize_feature_frame(frame)
    assert list(result.columns) == list(OUTPUT_COLUMNS)
    assert list(result["score_id"]) == ["synthetic_score_a", "synthetic_score_b"]
    assert list(result.index) == [0, 1]
    pd.testing.assert_frame_equal(frame, before)
