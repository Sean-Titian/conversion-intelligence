"""Fail-closed contracts for synthetic point-in-time session features.

The source conversion table has neither entity keys nor timestamps.  This module
therefore defines a separate event-level contract for an authored synthetic
systems demonstration; it does not retrofit timing information onto source rows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


class EventContractError(ValueError):
    """Raised when point-in-time inputs cannot be interpreted safely."""


SESSION_COLUMNS = (
    "session_id",
    "user_id",
    "session_start_at",
    "country",
    "age",
    "new_user",
    "source",
)
SCORE_REQUEST_COLUMNS = ("score_id", "session_id", "score_at")
PAGE_EVENT_COLUMNS = (
    "event_id",
    "session_id",
    "event_at",
    "available_at",
    "page_category",
)


@dataclass(frozen=True)
class EventContractReport:
    session_rows: int
    unique_users: int
    score_request_rows: int
    page_event_rows: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ValidatedPointInTimeInputs:
    """Validated, minimized inputs with UTC timestamps encoded as epoch microseconds."""

    sessions: pd.DataFrame
    score_requests: pd.DataFrame
    page_events: pd.DataFrame
    report: EventContractReport


def _require_exact_columns(
    frame: pd.DataFrame,
    expected: tuple[str, ...],
    *,
    frame_name: str,
) -> pd.DataFrame:
    duplicate_count = int(frame.columns.duplicated(keep=False).sum())
    if duplicate_count:
        raise EventContractError(
            f"{frame_name} has duplicate column names; count={duplicate_count}"
        )
    missing = sorted(set(expected) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(expected))
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing columns: {missing}")
        if unexpected:
            details.append(f"unexpected columns: {unexpected}")
        issue_count = len(missing) + len(unexpected)
        raise EventContractError(
            f"{frame_name} has " + "; ".join(details) + f"; count={issue_count}"
        )
    return frame.loc[:, expected].copy()


def _normalize_identifier(
    frame: pd.DataFrame,
    column: str,
    *,
    frame_name: str,
    unique: bool,
) -> None:
    values = frame[column].astype("string").str.strip()
    invalid_count = int((values.isna() | values.eq("")).sum())
    if invalid_count:
        raise EventContractError(
            f"{frame_name}.{column} has missing or empty identifiers; count={invalid_count}"
        )
    frame[column] = values
    if unique:
        duplicate_count = int(values.duplicated(keep=False).sum())
        if duplicate_count:
            raise EventContractError(
                f"{frame_name}.{column} has rows in duplicate keys; count={duplicate_count}"
            )


def _to_utc_epoch_microseconds(
    series: pd.Series,
    *,
    frame_name: str,
    column: str,
) -> pd.Series:
    missing_count = int(series.isna().sum())
    if missing_count:
        raise EventContractError(
            f"{frame_name}.{column} has missing timestamps; count={missing_count}"
        )

    invalid_count = 0
    naive_count = 0
    parsed_values: list[pd.Timestamp] = []
    for value in series:
        try:
            parsed = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            invalid_count += 1
            continue
        if parsed.tzinfo is None:
            naive_count += 1
        parsed_values.append(parsed)
    if invalid_count:
        raise EventContractError(
            f"{frame_name}.{column} has invalid timestamps; count={invalid_count}"
        )
    if naive_count:
        raise EventContractError(
            f"{frame_name}.{column} has timestamps without a UTC offset; count={naive_count}"
        )

    converted = pd.Series(
        pd.to_datetime(parsed_values, utc=True, errors="raise"),
        index=series.index,
    )
    # ``Series.astype('int64')`` follows the dtype's current resolution, which
    # can be microseconds in newer pandas releases. ``Timestamp.value`` is
    # explicitly nanoseconds and keeps the encoded contract version-stable.
    epoch_nanoseconds = converted.map(lambda timestamp: timestamp.value).astype("int64")
    submicrosecond_count = int((epoch_nanoseconds.mod(1_000) != 0).sum())
    if submicrosecond_count:
        raise EventContractError(
            f"{frame_name}.{column} has timestamps finer than microseconds; "
            f"count={submicrosecond_count}"
        )
    return epoch_nanoseconds.floordiv(1_000).astype("int64")


def _normalize_nonempty_string(
    frame: pd.DataFrame,
    column: str,
    *,
    frame_name: str,
) -> None:
    values = frame[column].astype("string").str.strip()
    invalid_count = int((values.isna() | values.eq("")).sum())
    if invalid_count:
        raise EventContractError(
            f"{frame_name}.{column} has missing or empty values; count={invalid_count}"
        )
    frame[column] = values


def _validate_session_attributes(sessions: pd.DataFrame) -> None:
    for column in ("country", "source"):
        _normalize_nonempty_string(sessions, column, frame_name="sessions")

    age = pd.to_numeric(sessions["age"], errors="coerce")
    invalid_age = age.isna() | ~age.between(13, 100) | age.mod(1).ne(0)
    invalid_age_count = int(invalid_age.sum())
    if invalid_age_count:
        raise EventContractError(
            f"sessions.age has invalid values; count={invalid_age_count}"
        )
    sessions["age"] = age.astype("int64")

    new_user = pd.to_numeric(sessions["new_user"], errors="coerce")
    invalid_new_user = new_user.isna() | ~new_user.isin([0, 1])
    invalid_new_user_count = int(invalid_new_user.sum())
    if invalid_new_user_count:
        raise EventContractError(
            f"sessions.new_user has non-binary values; count={invalid_new_user_count}"
        )
    sessions["new_user"] = new_user.astype("int64")


def _require_known_foreign_keys(
    values: pd.Series,
    known: set[str],
    *,
    frame_name: str,
    column: str,
) -> None:
    unknown_count = int((~values.isin(known)).sum())
    if unknown_count:
        raise EventContractError(
            f"{frame_name}.{column} has unknown foreign keys; count={unknown_count}"
        )


def validate_point_in_time_inputs(
    sessions: pd.DataFrame,
    score_requests: pd.DataFrame,
    page_events: pd.DataFrame,
) -> ValidatedPointInTimeInputs:
    """Validate event inputs and return only fields used by the feature query.

    Inclusion is evaluated later per score request as ``event_at < score_at``
    and ``available_at <= score_at``.  The first boundary is strict because an
    event recorded at the same instant as the request has ambiguous ordering.
    """

    session_work = _require_exact_columns(
        sessions, SESSION_COLUMNS, frame_name="sessions"
    )
    request_work = _require_exact_columns(
        score_requests, SCORE_REQUEST_COLUMNS, frame_name="score_requests"
    )
    event_work = _require_exact_columns(
        page_events, PAGE_EVENT_COLUMNS, frame_name="page_events"
    )

    _normalize_identifier(
        session_work, "session_id", frame_name="sessions", unique=True
    )
    _normalize_identifier(
        session_work, "user_id", frame_name="sessions", unique=False
    )
    _normalize_identifier(
        request_work, "score_id", frame_name="score_requests", unique=True
    )
    _normalize_identifier(
        request_work, "session_id", frame_name="score_requests", unique=False
    )
    _normalize_identifier(
        event_work, "event_id", frame_name="page_events", unique=True
    )
    _normalize_identifier(
        event_work, "session_id", frame_name="page_events", unique=False
    )
    _normalize_nonempty_string(
        event_work, "page_category", frame_name="page_events"
    )
    _validate_session_attributes(session_work)

    session_work["session_start_at_us"] = _to_utc_epoch_microseconds(
        session_work.pop("session_start_at"),
        frame_name="sessions",
        column="session_start_at",
    )
    request_work["score_at_us"] = _to_utc_epoch_microseconds(
        request_work.pop("score_at"),
        frame_name="score_requests",
        column="score_at",
    )
    event_work["event_at_us"] = _to_utc_epoch_microseconds(
        event_work.pop("event_at"),
        frame_name="page_events",
        column="event_at",
    )
    event_work["available_at_us"] = _to_utc_epoch_microseconds(
        event_work.pop("available_at"),
        frame_name="page_events",
        column="available_at",
    )

    known_sessions = set(session_work["session_id"])
    _require_known_foreign_keys(
        request_work["session_id"],
        known_sessions,
        frame_name="score_requests",
        column="session_id",
    )
    _require_known_foreign_keys(
        event_work["session_id"],
        known_sessions,
        frame_name="page_events",
        column="session_id",
    )

    starts = session_work[["session_id", "session_start_at_us"]]
    request_times = request_work.merge(
        starts, on="session_id", how="left", validate="many_to_one"
    )
    before_start_count = int(
        (request_times["score_at_us"] < request_times["session_start_at_us"]).sum()
    )
    if before_start_count:
        raise EventContractError(
            "score_requests.score_at is before sessions.session_start_at; "
            f"count={before_start_count}"
        )

    event_times = event_work.merge(
        starts, on="session_id", how="left", validate="many_to_one"
    )
    pre_session_count = int(
        (event_times["event_at_us"] < event_times["session_start_at_us"]).sum()
    )
    if pre_session_count:
        raise EventContractError(
            "page_events.event_at is before sessions.session_start_at; "
            f"count={pre_session_count}"
        )
    impossible_availability_count = int(
        (event_work["available_at_us"] < event_work["event_at_us"]).sum()
    )
    if impossible_availability_count:
        raise EventContractError(
            "page_events.available_at is before page_events.event_at; "
            f"count={impossible_availability_count}"
        )

    session_internal = session_work.loc[
        :,
        (
            "session_id",
            "user_id",
            "session_start_at_us",
            "country",
            "age",
            "new_user",
            "source",
        ),
    ]
    request_internal = request_work.loc[
        :, ("score_id", "session_id", "score_at_us")
    ]
    event_internal = event_work.loc[
        :,
        (
            "event_id",
            "session_id",
            "event_at_us",
            "available_at_us",
            "page_category",
        ),
    ]
    report = EventContractReport(
        session_rows=len(session_internal),
        unique_users=int(session_internal["user_id"].nunique()),
        score_request_rows=len(request_internal),
        page_event_rows=len(event_internal),
    )
    return ValidatedPointInTimeInputs(
        sessions=session_internal.reset_index(drop=True),
        score_requests=request_internal.reset_index(drop=True),
        page_events=event_internal.reset_index(drop=True),
        report=report,
    )
