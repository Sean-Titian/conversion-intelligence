"""Reference implementation and audits for point-in-time session features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import resources
from numbers import Integral, Real

import numpy as np
import pandas as pd

from .event_contract import ValidatedPointInTimeInputs

OUTPUT_COLUMNS = (
    "score_id",
    "session_id",
    "user_id",
    "score_at_us",
    "country",
    "age",
    "new_user",
    "source",
    "seconds_since_session_start",
    "pages_observed_at_score_time",
    "unique_page_categories_observed",
    "feature_max_event_at_us",
    "seconds_since_last_page_view",
)
IDENTIFIER_COLUMNS = frozenset({"score_id", "session_id", "user_id"})
MODEL_FEATURE_COLUMNS = tuple(
    column
    for column in OUTPUT_COLUMNS
    if column
    not in IDENTIFIER_COLUMNS
    | {"score_at_us", "feature_max_event_at_us"}
)


@dataclass(frozen=True)
class PointInTimeAudit:
    session_rows: int
    score_request_rows: int
    page_event_rows: int
    candidate_event_score_pairs: int
    eligible_event_score_pairs: int
    exact_score_boundary_pairs_excluded: int
    exact_availability_boundary_pairs_included: int
    future_event_score_pairs_excluded: int
    late_event_score_pairs_excluded: int
    zero_history_score_requests: int
    output_rows: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class PointInTimeFeatureResult:
    features: pd.DataFrame
    audit: PointInTimeAudit


def load_point_in_time_sql() -> str:
    """Load the versioned Spark SQL query from the installed package."""

    return (
        resources.files("conversion_intelligence")
        .joinpath("sql", "point_in_time_features.sql")
        .read_text(encoding="utf-8")
    )


def canonicalize_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize engine output to declared canonical contract types."""

    missing = sorted(set(OUTPUT_COLUMNS) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(OUTPUT_COLUMNS))
    if missing or unexpected:
        raise ValueError(
            f"Unexpected feature schema; missing={missing}, unexpected={unexpected}"
        )
    result = frame.loc[:, OUTPUT_COLUMNS].copy().sort_values("score_id", kind="stable")
    result = result.reset_index(drop=True)
    for column in ("score_id", "session_id", "user_id", "country", "source"):
        invalid_value = result[column].map(
            lambda value: not isinstance(value, str) or not value.strip()
        ).astype("bool")
        invalid = result[column].isna() | invalid_value
        if invalid.any():
            raise ValueError(
                f"{column} violates the non-empty string contract; count={int(invalid.sum())}"
            )
        result[column] = result[column].astype("string")
    for column in (
        "score_at_us",
        "age",
        "new_user",
        "seconds_since_session_start",
        "pages_observed_at_score_time",
        "unique_page_categories_observed",
    ):
        invalid_type = result[column].map(
            lambda value: isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
        )
        if invalid_type.any():
            raise ValueError(
                f"{column} violates the integer contract; count={int(invalid_type.sum())}"
            )
        result[column] = result[column].astype("int64")
    for column in ("feature_max_event_at_us", "seconds_since_last_page_view"):
        non_missing = result[column].notna()
        invalid_type = non_missing & result[column].map(
            lambda value: isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
        )
        if invalid_type.any():
            raise ValueError(
                f"{column} violates the nullable numeric contract; "
                f"count={int(invalid_type.sum())}"
            )
        numeric = pd.to_numeric(result[column], errors="raise")
        non_finite = non_missing & ~numeric.map(
            lambda value: bool(np.isfinite(value)) if pd.notna(value) else True
        )
        non_integral = non_missing & numeric.map(
            lambda value: bool(value % 1 != 0) if pd.notna(value) else False
        )
        invalid_numeric = non_finite | non_integral
        if invalid_numeric.any():
            raise ValueError(
                f"{column} violates the finite integer contract; "
                f"count={int(invalid_numeric.sum())}"
            )
        result[column] = numeric.astype("Int64")
    if result["score_id"].duplicated().any():
        raise ValueError("Feature output contains duplicate score_id rows")
    if (result["feature_max_event_at_us"] >= result["score_at_us"]).fillna(False).any():
        raise ValueError("Feature output contains an event at or after score time")
    if set(result.columns).intersection({"converted", "outcome", "outcome_at"}):
        raise ValueError("Feature output must not contain outcome fields")
    return result


def _event_score_pairs(inputs: ValidatedPointInTimeInputs) -> pd.DataFrame:
    return inputs.score_requests.merge(
        inputs.page_events,
        on="session_id",
        how="left",
        validate="many_to_many",
    )


def build_reference_features(
    inputs: ValidatedPointInTimeInputs,
) -> PointInTimeFeatureResult:
    """Build a small-batch Pandas reference for the Spark SQL contract."""

    pairs = _event_score_pairs(inputs)
    has_event = pairs["event_id"].notna()
    occurred_before_score = has_event & (pairs["event_at_us"] < pairs["score_at_us"])
    available_by_score = has_event & (
        pairs["available_at_us"] <= pairs["score_at_us"]
    )
    eligible = occurred_before_score & available_by_score
    exact_boundary = has_event & (pairs["event_at_us"] == pairs["score_at_us"])
    availability_boundary = eligible & (
        pairs["available_at_us"] == pairs["score_at_us"]
    )
    future = has_event & (pairs["event_at_us"] > pairs["score_at_us"])
    late = occurred_before_score & ~available_by_score

    eligible_pairs = pairs.loc[eligible, ["score_id", "event_at_us", "page_category"]]
    if eligible_pairs.empty:
        aggregates = pd.DataFrame(
            columns=(
                "score_id",
                "pages_observed_at_score_time",
                "unique_page_categories_observed",
                "feature_max_event_at_us",
            )
        )
    else:
        aggregates = (
            eligible_pairs.groupby("score_id", sort=False)
            .agg(
                pages_observed_at_score_time=("event_at_us", "size"),
                unique_page_categories_observed=("page_category", "nunique"),
                feature_max_event_at_us=("event_at_us", "max"),
            )
            .reset_index()
        )

    base = inputs.score_requests.merge(
        inputs.sessions,
        on="session_id",
        how="inner",
        validate="many_to_one",
    )
    output = base.merge(aggregates, on="score_id", how="left", validate="one_to_one")
    output["pages_observed_at_score_time"] = (
        output["pages_observed_at_score_time"].fillna(0).astype("int64")
    )
    output["unique_page_categories_observed"] = (
        output["unique_page_categories_observed"].fillna(0).astype("int64")
    )
    output["seconds_since_session_start"] = (
        (output["score_at_us"] - output["session_start_at_us"]) // 1_000_000
    ).astype("int64")
    output["seconds_since_last_page_view"] = (
        (output["score_at_us"] - output["feature_max_event_at_us"]) // 1_000_000
    ).astype("Int64")
    features = canonicalize_feature_frame(output.loc[:, OUTPUT_COLUMNS])

    audit = PointInTimeAudit(
        session_rows=len(inputs.sessions),
        score_request_rows=len(inputs.score_requests),
        page_event_rows=len(inputs.page_events),
        candidate_event_score_pairs=int(has_event.sum()),
        eligible_event_score_pairs=int(eligible.sum()),
        exact_score_boundary_pairs_excluded=int(exact_boundary.sum()),
        exact_availability_boundary_pairs_included=int(availability_boundary.sum()),
        future_event_score_pairs_excluded=int(future.sum()),
        late_event_score_pairs_excluded=int(late.sum()),
        zero_history_score_requests=int(
            (features["pages_observed_at_score_time"] == 0).sum()
        ),
        output_rows=len(features),
    )
    if audit.output_rows != audit.score_request_rows:
        raise ValueError("Point-in-time join changed the score-request grain")
    return PointInTimeFeatureResult(features=features, audit=audit)
