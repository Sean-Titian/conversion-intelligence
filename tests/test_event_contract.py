import re

import pandas as pd
import pytest

from conversion_intelligence.event_contract import (
    EventContractError,
    validate_point_in_time_inputs,
)


def valid_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sessions = pd.DataFrame(
        {
            "session_id": ["synthetic_session_001"],
            "user_id": ["synthetic_user_001"],
            "session_start_at": [pd.Timestamp("2026-01-01T12:00:00Z")],
            "country": ["US"],
            "age": [31],
            "new_user": [1],
            "source": ["Ads"],
        }
    )
    score_requests = pd.DataFrame(
        {
            "score_id": ["synthetic_score_001"],
            "session_id": ["synthetic_session_001"],
            "score_at": [pd.Timestamp("2026-01-01T12:05:00Z")],
        }
    )
    page_events = pd.DataFrame(
        {
            "event_id": ["synthetic_event_001"],
            "session_id": ["synthetic_session_001"],
            "event_at": [pd.Timestamp("2026-01-01T12:01:00Z")],
            "available_at": [pd.Timestamp("2026-01-01T12:02:00Z")],
            "page_category": ["product"],
        }
    )
    return sessions, score_requests, page_events


def _identifier_values(*frames: pd.DataFrame) -> set[str]:
    values: set[str] = set()
    for frame in frames:
        for column in ("session_id", "user_id", "score_id", "event_id"):
            if column not in frame:
                continue
            values.update(
                str(value)
                for value in frame[column].dropna()
                if str(value).strip()
            )
    return values


def _assert_count_only_error(
    sessions: pd.DataFrame,
    score_requests: pd.DataFrame,
    page_events: pd.DataFrame,
) -> str:
    identifiers = _identifier_values(sessions, score_requests, page_events)
    with pytest.raises(EventContractError) as caught:
        validate_point_in_time_inputs(sessions, score_requests, page_events)

    message = str(caught.value)
    assert re.search(r"\bcount=\d+\b", message), message
    for identifier in identifiers:
        assert identifier not in message
    return message


def test_valid_point_in_time_inputs_pass_contract() -> None:
    validate_point_in_time_inputs(*valid_inputs())


@pytest.mark.parametrize(
    ("table_name", "duplicate_column"),
    [
        ("sessions", "country"),
        ("score_requests", "score_at"),
        ("page_events", "page_category"),
    ],
)
def test_duplicate_column_names_fail_closed(
    table_name: str,
    duplicate_column: str,
) -> None:
    sessions, score_requests, page_events = valid_inputs()
    frames = {
        "sessions": sessions,
        "score_requests": score_requests,
        "page_events": page_events,
    }
    frames[table_name] = pd.concat(
        [frames[table_name], frames[table_name][[duplicate_column]]],
        axis=1,
    )

    message = _assert_count_only_error(
        frames["sessions"],
        frames["score_requests"],
        frames["page_events"],
    )
    assert "duplicate column names" in message


@pytest.mark.parametrize(
    ("table_name", "primary_key"),
    [
        ("sessions", "session_id"),
        ("score_requests", "score_id"),
        ("page_events", "event_id"),
    ],
)
def test_duplicate_primary_ids_fail_closed(table_name: str, primary_key: str) -> None:
    sessions, score_requests, page_events = valid_inputs()
    frames = {
        "sessions": sessions,
        "score_requests": score_requests,
        "page_events": page_events,
    }
    frames[table_name] = pd.concat(
        [frames[table_name], frames[table_name].copy()],
        ignore_index=True,
    )

    message = _assert_count_only_error(
        frames["sessions"],
        frames["score_requests"],
        frames["page_events"],
    )
    assert primary_key in message


@pytest.mark.parametrize(
    ("table_name", "id_column"),
    [
        ("sessions", "session_id"),
        ("sessions", "user_id"),
        ("score_requests", "score_id"),
        ("score_requests", "session_id"),
        ("page_events", "event_id"),
        ("page_events", "session_id"),
    ],
)
@pytest.mark.parametrize("invalid_value", [None, "   "])
def test_null_or_empty_ids_fail_closed(
    table_name: str,
    id_column: str,
    invalid_value: object,
) -> None:
    sessions, score_requests, page_events = valid_inputs()
    frames = {
        "sessions": sessions,
        "score_requests": score_requests,
        "page_events": page_events,
    }
    frames[table_name].loc[0, id_column] = invalid_value

    message = _assert_count_only_error(
        frames["sessions"],
        frames["score_requests"],
        frames["page_events"],
    )
    assert id_column in message


@pytest.mark.parametrize("child_table", ["score_requests", "page_events"])
def test_orphan_session_foreign_keys_fail_closed(child_table: str) -> None:
    sessions, score_requests, page_events = valid_inputs()
    if child_table == "score_requests":
        score_requests.loc[0, "session_id"] = "synthetic_session_orphan_score"
    else:
        page_events.loc[0, "session_id"] = "synthetic_session_orphan_event"

    message = _assert_count_only_error(sessions, score_requests, page_events)
    assert "session_id" in message


@pytest.mark.parametrize(
    ("table_name", "timestamp_column"),
    [
        ("sessions", "session_start_at"),
        ("score_requests", "score_at"),
        ("page_events", "event_at"),
        ("page_events", "available_at"),
    ],
)
def test_naive_timestamps_fail_closed(table_name: str, timestamp_column: str) -> None:
    sessions, score_requests, page_events = valid_inputs()
    frames = {
        "sessions": sessions,
        "score_requests": score_requests,
        "page_events": page_events,
    }
    aware = frames[table_name].loc[0, timestamp_column]
    frames[table_name][timestamp_column] = frames[table_name][
        timestamp_column
    ].astype(object)
    frames[table_name].loc[0, timestamp_column] = aware.tz_localize(None)

    message = _assert_count_only_error(
        frames["sessions"],
        frames["score_requests"],
        frames["page_events"],
    )
    assert timestamp_column in message


def test_score_before_session_start_fails_closed() -> None:
    sessions, score_requests, page_events = valid_inputs()
    score_requests.loc[0, "score_at"] = pd.Timestamp("2026-01-01T11:59:00Z")

    message = _assert_count_only_error(sessions, score_requests, page_events)
    assert "score_at" in message


def test_event_before_session_start_fails_closed() -> None:
    sessions, score_requests, page_events = valid_inputs()
    page_events.loc[0, "event_at"] = pd.Timestamp("2026-01-01T11:59:00Z")

    message = _assert_count_only_error(sessions, score_requests, page_events)
    assert "event_at" in message


def test_event_available_before_it_occurs_fails_closed() -> None:
    sessions, score_requests, page_events = valid_inputs()
    page_events.loc[0, "available_at"] = pd.Timestamp("2026-01-01T12:00:59Z")

    message = _assert_count_only_error(sessions, score_requests, page_events)
    assert "available_at" in message


@pytest.mark.parametrize("table_name", ["sessions", "score_requests", "page_events"])
def test_unexpected_converted_column_fails_closed(table_name: str) -> None:
    sessions, score_requests, page_events = valid_inputs()
    frames = {
        "sessions": sessions,
        "score_requests": score_requests,
        "page_events": page_events,
    }
    frames[table_name]["converted"] = 0

    message = _assert_count_only_error(
        frames["sessions"],
        frames["score_requests"],
        frames["page_events"],
    )
    assert "converted" in message
