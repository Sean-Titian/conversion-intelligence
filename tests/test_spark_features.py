from __future__ import annotations

import json
from pathlib import Path

import pytest

pyspark = pytest.importorskip("pyspark")

from conversion_intelligence.event_contract import (  # noqa: E402
    validate_point_in_time_inputs,
)
from conversion_intelligence.point_in_time import (  # noqa: E402
    OUTPUT_COLUMNS,
    build_reference_features,
)
from conversion_intelligence.spark_features import (  # noqa: E402
    assert_exact_feature_parity,
    build_spark_sql_features,
    build_verification_report,
    write_verification_report,
)
from conversion_intelligence.synthetic_events import (  # noqa: E402
    SyntheticEventConfig,
    generate_synthetic_event_bundle,
)


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("conversion-point-in-time-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def test_spark_sql_rebuilds_the_tracked_aggregate_report_exactly(
    spark, tmp_path: Path
) -> None:
    root = Path(__file__).resolve().parents[1]
    tracked_path = root / "reports" / "point-in-time-pipeline.json"
    tracked_bytes = tracked_path.read_bytes()
    tracked = json.loads(tracked_bytes)
    config = SyntheticEventConfig(
        sessions=tracked["config"]["sessions"],
        users=tracked["config"]["users"],
        seed=tracked["config"]["seed"],
    )
    report = build_verification_report(spark, config)
    assert report == tracked
    assert all(report["gates"].values())
    assert report["counts"]["output_rows"] == 1_500
    assert report["counts"]["score_request_rows"] == 1_500
    assert report["counts"]["exact_score_boundary_pairs_excluded"] == 500
    assert report["counts"]["exact_availability_boundary_pairs_included"] == 500
    assert report["counts"]["late_event_score_pairs_excluded"] == 500
    assert report["contract"]["outcome_fields_in_features"] is False
    serialized = json.dumps(report, sort_keys=True)
    assert "synthetic_session_" not in serialized
    assert "synthetic_user_" not in serialized
    assert "synthetic_event_" not in serialized
    first = write_verification_report(tmp_path / "first.json", report)
    second = write_verification_report(tmp_path / "second.json", report)
    assert first.read_bytes() == second.read_bytes() == tracked_bytes


def test_spark_sql_preserves_the_empty_score_request_contract(spark) -> None:
    bundle = generate_synthetic_event_bundle(
        SyntheticEventConfig(sessions=2, users=1, seed=20260901)
    )
    inputs = validate_point_in_time_inputs(
        bundle.sessions,
        bundle.score_requests.head(0),
        bundle.page_events,
    )

    reference = build_reference_features(inputs).features
    candidate = build_spark_sql_features(spark, inputs)

    assert list(candidate.columns) == list(OUTPUT_COLUMNS)
    assert candidate.empty
    assert_exact_feature_parity(reference, candidate)
