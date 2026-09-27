from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from conversion_intelligence.event_contract import validate_point_in_time_inputs
from conversion_intelligence.point_in_time import (
    IDENTIFIER_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    OUTPUT_COLUMNS,
    build_reference_features,
    canonicalize_feature_frame,
    load_point_in_time_sql,
)
from conversion_intelligence.spark_features import (
    _uniform_rows_per_session,
    assert_exact_feature_parity,
)
from conversion_intelligence.synthetic_events import (
    SyntheticEventConfig,
    generate_synthetic_event_bundle,
)


def _build_reference(sessions: int = 2):
    bundle = generate_synthetic_event_bundle(
        SyntheticEventConfig(sessions=sessions, users=1, seed=11)
    )
    inputs = validate_point_in_time_inputs(
        bundle.sessions, bundle.score_requests, bundle.page_events
    )
    return bundle, build_reference_features(inputs)


def test_reference_features_respect_boundary_availability_and_grain() -> None:
    _, result = _build_reference(sessions=2)
    features = result.features
    assert list(features.columns) == list(OUTPUT_COLUMNS)
    assert list(features["pages_observed_at_score_time"]) == [2, 4, 0, 2, 4, 0]
    recency = features["seconds_since_last_page_view"].astype("Int64")
    assert list(recency.dropna().astype(int)) == [30, 60, 30, 60]
    assert list(recency.isna()) == [False, False, True, False, False, True]
    assert features["score_id"].is_unique
    assert len(features) == 6
    assert result.audit.output_rows == result.audit.score_request_rows
    assert result.audit.exact_score_boundary_pairs_excluded == 2
    assert result.audit.exact_availability_boundary_pairs_included == 2
    assert result.audit.late_event_score_pairs_excluded == 2
    assert result.audit.zero_history_score_requests == 2


def test_feature_output_rejects_duplicate_column_names() -> None:
    _, result = _build_reference(sessions=1)
    duplicated = pd.concat(
        [result.features, result.features[["country"]]],
        axis=1,
    )

    with pytest.raises(
        ValueError,
        match=r"Point-in-time feature output has duplicate column names; count=2",
    ):
        canonicalize_feature_frame(duplicated)


def test_reference_features_are_invariant_to_future_poison_event() -> None:
    bundle, baseline = _build_reference(sessions=1)
    session_id = bundle.sessions.loc[0, "session_id"]
    future = pd.DataFrame(
        {
            "event_id": ["synthetic_event_future_poison"],
            "session_id": [session_id],
            "event_at": [pd.Timestamp("2026-01-01T00:20:00Z")],
            "available_at": [pd.Timestamp("2026-01-01T00:20:00Z")],
            "page_category": ["future_poison"],
        }
    )
    poisoned_events = pd.concat([bundle.page_events, future], ignore_index=True)
    poisoned_inputs = validate_point_in_time_inputs(
        bundle.sessions, bundle.score_requests, poisoned_events
    )
    poisoned = build_reference_features(poisoned_inputs)
    pd.testing.assert_frame_equal(baseline.features, poisoned.features)


def test_single_request_matches_its_batch_feature_row() -> None:
    bundle, batch = _build_reference(sessions=2)
    request = bundle.score_requests.sort_values("score_id", kind="stable").head(1)
    session_id = request.iloc[0]["session_id"]
    inputs = validate_point_in_time_inputs(
        bundle.sessions.loc[bundle.sessions["session_id"] == session_id],
        request,
        bundle.page_events.loc[bundle.page_events["session_id"] == session_id],
    )
    single = build_reference_features(inputs).features
    expected = batch.features.loc[batch.features["score_id"] == single.iloc[0]["score_id"]]
    pd.testing.assert_frame_equal(
        expected.reset_index(drop=True), single.reset_index(drop=True)
    )


def test_model_feature_allowlist_excludes_keys_audit_times_and_outcomes() -> None:
    assert not IDENTIFIER_COLUMNS.intersection(MODEL_FEATURE_COLUMNS)
    assert "score_at_us" not in MODEL_FEATURE_COLUMNS
    assert "feature_max_event_at_us" not in MODEL_FEATURE_COLUMNS
    assert "converted" not in MODEL_FEATURE_COLUMNS
    assert "total_pages_visited" not in MODEL_FEATURE_COLUMNS
    assert "pages_observed_at_score_time" in MODEL_FEATURE_COLUMNS


def test_fixture_metadata_rejects_nonuniform_rows_per_session() -> None:
    session_ids = pd.Series(["synthetic_session_a", "synthetic_session_b"])
    rows = pd.DataFrame(
        {
            "session_id": [
                "synthetic_session_a",
                "synthetic_session_a",
                "synthetic_session_b",
            ]
        }
    )

    with pytest.raises(AssertionError, match="not uniform"):
        _uniform_rows_per_session(
            session_ids,
            rows,
            table_name="score request",
        )


def test_synthetic_event_generator_is_deterministic_and_explicitly_synthetic() -> None:
    config = SyntheticEventConfig(sessions=3, users=2, seed=19)
    first = generate_synthetic_event_bundle(config)
    second = generate_synthetic_event_bundle(config)
    for first_frame, second_frame in zip(
        (first.sessions, first.score_requests, first.page_events),
        (second.sessions, second.score_requests, second.page_events),
        strict=True,
    ):
        pd.testing.assert_frame_equal(first_frame, second_frame)
    assert first.sessions["session_id"].str.startswith("synthetic_").all()
    assert first.sessions["user_id"].str.startswith("synthetic_").all()
    assert first.score_requests["score_id"].str.startswith("synthetic_").all()
    assert first.page_events["event_id"].str.startswith("synthetic_").all()


def test_packaged_sql_encodes_bitemporal_cutoff_without_outcome_fields() -> None:
    query = load_point_in_time_sql().lower()
    assert "page_events.event_at_us < score_requests.score_at_us" in query
    assert "page_events.available_at_us <= score_requests.score_at_us" in query
    assert "left join page_features" in query
    assert "converted" not in query
    assert "outcome" not in query


def test_exact_parity_rejects_fractional_value_in_integer_contract() -> None:
    _, reference = _build_reference(sessions=1)
    candidate = reference.features.copy()
    candidate["pages_observed_at_score_time"] = candidate[
        "pages_observed_at_score_time"
    ].astype(float)
    candidate.loc[0, "pages_observed_at_score_time"] = 2.9
    with pytest.raises(ValueError, match="integer contract"):
        assert_exact_feature_parity(reference.features, candidate)


def test_exact_parity_rejects_invalid_nonmissing_nullable_value() -> None:
    _, reference = _build_reference(sessions=1)
    candidate = reference.features.copy()
    candidate["feature_max_event_at_us"] = candidate[
        "feature_max_event_at_us"
    ].astype(object)
    missing_index = candidate["feature_max_event_at_us"].isna().idxmax()
    candidate.loc[missing_index, "feature_max_event_at_us"] = "not-a-timestamp"
    with pytest.raises(ValueError, match="nullable numeric contract"):
        assert_exact_feature_parity(reference.features, candidate)


def test_tracked_pipeline_report_is_current_aggregate_only_contract_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    report_path = root / "reports" / "point-in-time-pipeline.json"
    report_text = report_path.read_text(encoding="utf-8")
    report = json.loads(report_text)
    assert set(report) == {
        "artifact_type",
        "claim_boundary",
        "config",
        "contract",
        "counts",
        "data_classification",
        "engine",
        "gates",
        "report_scope",
        "schema_version",
    }
    assert report["data_classification"] == "synthetic"
    assert report["report_scope"] == "aggregate_only"
    assert report["contract"]["outcome_fields_in_features"] is False
    assert report["engine"]["spark_version"] == "4.2.0"
    assert all(report["gates"].values())
    assert "synthetic_session_" not in report_text
    assert "synthetic_user_" not in report_text
    assert "synthetic_event_" not in report_text

    query = load_point_in_time_sql()
    assert report["engine"]["query_sha256"] == hashlib.sha256(
        query.encode("utf-8")
    ).hexdigest()
    config = SyntheticEventConfig(
        sessions=report["config"]["sessions"],
        users=report["config"]["users"],
        seed=report["config"]["seed"],
    )
    bundle = generate_synthetic_event_bundle(config)
    inputs = validate_point_in_time_inputs(
        bundle.sessions, bundle.score_requests, bundle.page_events
    )
    assert report["counts"] == build_reference_features(inputs).audit.to_dict()
