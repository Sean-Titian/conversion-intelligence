"""Spark SQL execution and aggregate verification for point-in-time features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .event_contract import ValidatedPointInTimeInputs, validate_point_in_time_inputs
from .point_in_time import (
    OUTPUT_COLUMNS,
    PointInTimeFeatureResult,
    build_reference_features,
    canonicalize_feature_frame,
    load_point_in_time_sql,
)
from .synthetic_events import (
    SyntheticEventBundle,
    SyntheticEventConfig,
    generate_synthetic_event_bundle,
)


def _python_records(frame: pd.DataFrame) -> list[tuple[object, ...]]:
    records: list[tuple[object, ...]] = []
    for row in frame.itertuples(index=False, name=None):
        records.append(
            tuple(value.item() if hasattr(value, "item") else value for value in row)
        )
    return records


def build_spark_sql_features(
    spark: Any,
    inputs: ValidatedPointInTimeInputs,
) -> pd.DataFrame:
    """Execute the packaged SQL in a caller-owned Spark session."""

    from pyspark.sql.types import LongType, StringType, StructField, StructType

    spark.conf.set("spark.sql.session.timeZone", "UTC")
    if spark.conf.get("spark.sql.session.timeZone") != "UTC":
        raise RuntimeError("Spark SQL session timezone must be UTC")

    sessions_schema = StructType(
        [
            StructField("session_id", StringType(), False),
            StructField("user_id", StringType(), False),
            StructField("session_start_at_us", LongType(), False),
            StructField("country", StringType(), False),
            StructField("age", LongType(), False),
            StructField("new_user", LongType(), False),
            StructField("source", StringType(), False),
        ]
    )
    requests_schema = StructType(
        [
            StructField("score_id", StringType(), False),
            StructField("session_id", StringType(), False),
            StructField("score_at_us", LongType(), False),
        ]
    )
    events_schema = StructType(
        [
            StructField("event_id", StringType(), False),
            StructField("session_id", StringType(), False),
            StructField("event_at_us", LongType(), False),
            StructField("available_at_us", LongType(), False),
            StructField("page_category", StringType(), False),
        ]
    )

    spark.createDataFrame(
        _python_records(inputs.sessions), schema=sessions_schema
    ).createOrReplaceTempView("sessions")
    spark.createDataFrame(
        _python_records(inputs.score_requests), schema=requests_schema
    ).createOrReplaceTempView("score_requests")
    spark.createDataFrame(
        _python_records(inputs.page_events), schema=events_schema
    ).createOrReplaceTempView("page_events")

    rows = spark.sql(load_point_in_time_sql()).orderBy("score_id").collect()
    output = pd.DataFrame(
        [row.asDict(recursive=True) for row in rows],
        columns=OUTPUT_COLUMNS,
    )
    return canonicalize_feature_frame(output)


def assert_exact_feature_parity(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> None:
    """Fail when engines disagree after strict canonical-contract validation."""

    pd.testing.assert_frame_equal(
        canonicalize_feature_frame(reference),
        canonicalize_feature_frame(candidate),
        check_exact=True,
        check_dtype=True,
    )


def _future_poison_bundle(bundle: SyntheticEventBundle) -> SyntheticEventBundle:
    latest_scores = (
        bundle.score_requests.groupby("session_id", sort=False)["score_at"].max()
    )
    poison = pd.DataFrame(
        {
            "event_id": [
                f"synthetic_future_poison_{index:08d}"
                for index in range(len(latest_scores))
            ],
            "session_id": list(latest_scores.index),
            "event_at": [value + pd.Timedelta(minutes=10) for value in latest_scores],
            "available_at": [
                value + pd.Timedelta(minutes=10) for value in latest_scores
            ],
            "page_category": ["future_poison"] * len(latest_scores),
        }
    )
    return SyntheticEventBundle(
        sessions=bundle.sessions.copy(),
        score_requests=bundle.score_requests.copy(),
        page_events=pd.concat([bundle.page_events, poison], ignore_index=True),
    )


def _single_request_bundle(bundle: SyntheticEventBundle) -> SyntheticEventBundle:
    request = bundle.score_requests.sort_values("score_id", kind="stable").head(1)
    session_id = request.iloc[0]["session_id"]
    return SyntheticEventBundle(
        sessions=bundle.sessions.loc[bundle.sessions["session_id"] == session_id].copy(),
        score_requests=request.copy(),
        page_events=bundle.page_events.loc[
            bundle.page_events["session_id"] == session_id
        ].copy(),
    )


def _validated(bundle: SyntheticEventBundle) -> ValidatedPointInTimeInputs:
    return validate_point_in_time_inputs(
        bundle.sessions,
        bundle.score_requests,
        bundle.page_events,
    )


def _uniform_rows_per_session(
    session_ids: pd.Series,
    rows: pd.DataFrame,
    *,
    table_name: str,
) -> int:
    counts = rows.groupby("session_id", sort=False).size().reindex(
        session_ids.tolist(), fill_value=0
    )
    if counts.nunique(dropna=False) != 1:
        raise AssertionError(
            f"Synthetic {table_name} fixture is not uniform across sessions"
        )
    return int(counts.iloc[0])


def build_verification_report(
    spark: Any,
    config: SyntheticEventConfig | None = None,
) -> dict[str, object]:
    """Run deterministic leakage and engine-parity gates and return aggregates."""

    if config is None:
        config = SyntheticEventConfig()
    bundle = generate_synthetic_event_bundle(config)
    inputs = _validated(bundle)
    reference: PointInTimeFeatureResult = build_reference_features(inputs)
    spark_features = build_spark_sql_features(spark, inputs)
    assert_exact_feature_parity(reference.features, spark_features)

    poisoned = _validated(_future_poison_bundle(bundle))
    poisoned_reference = build_reference_features(poisoned)
    poisoned_spark = build_spark_sql_features(spark, poisoned)
    assert_exact_feature_parity(reference.features, poisoned_reference.features)
    assert_exact_feature_parity(reference.features, poisoned_spark)

    single = _validated(_single_request_bundle(bundle))
    single_reference = build_reference_features(single)
    single_spark = build_spark_sql_features(spark, single)
    assert_exact_feature_parity(single_reference.features, single_spark)
    batch_row = reference.features.loc[
        reference.features["score_id"] == single_reference.features.iloc[0]["score_id"]
    ]
    assert_exact_feature_parity(batch_row, single_reference.features)

    audit = reference.audit
    if audit.exact_score_boundary_pairs_excluded <= 0:
        raise AssertionError("Synthetic fixture did not exercise the score boundary")
    if audit.exact_availability_boundary_pairs_included <= 0:
        raise AssertionError("Synthetic fixture did not exercise the availability boundary")
    if audit.future_event_score_pairs_excluded <= 0:
        raise AssertionError("Synthetic fixture did not exercise future events")
    if audit.late_event_score_pairs_excluded <= 0:
        raise AssertionError("Synthetic fixture did not exercise late events")
    if audit.output_rows != audit.score_request_rows:
        raise AssertionError("Feature output changed the score-request grain")

    score_requests_per_session = _uniform_rows_per_session(
        inputs.sessions["session_id"],
        inputs.score_requests,
        table_name="score request",
    )
    page_events_per_session = _uniform_rows_per_session(
        inputs.sessions["session_id"],
        inputs.page_events,
        table_name="page event",
    )
    query = load_point_in_time_sql()
    return {
        "schema_version": "1.0",
        "artifact_type": "synthetic_point_in_time_pipeline_verification",
        "data_classification": "synthetic",
        "report_scope": "aggregate_only",
        "contract": {
            "analysis_unit": "one row per score_id",
            "event_time_rule": "event_at < score_at",
            "availability_rule": "available_at <= score_at",
            "outcome_fields_in_features": False,
        },
        "config": {
            "sessions": config.sessions,
            "users": config.users,
            "score_requests_per_session": score_requests_per_session,
            "page_events_per_session": page_events_per_session,
            "seed": config.seed,
        },
        "counts": audit.to_dict(),
        "gates": {
            "exact_score_boundary_excluded": True,
            "exact_availability_boundary_included": True,
            "future_events_excluded": True,
            "late_events_excluded": True,
            "future_poison_invariance": True,
            "one_row_per_score_request": True,
            "spark_sql_reference_exact_parity": True,
            "batch_single_request_exact_parity": True,
        },
        "engine": {
            "spark_version": str(spark.version),
            "spark_sql_timezone": str(spark.conf.get("spark.sql.session.timeZone")),
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        },
        "claim_boundary": (
            "Synthetic transformation evidence only; source metrics are unchanged and this "
            "does not establish production scale, online serving, or real-world performance."
        ),
    }


def write_verification_report(path: str | Path, report: dict[str, object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="conversion-pit-verify")
    parser.add_argument(
        "--output", default="reports/generated/point-in-time-pipeline.json"
    )
    parser.add_argument("--sessions", type=int, default=500)
    parser.add_argument("--users", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("conversion-point-in-time-verification")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        report = build_verification_report(
            spark,
            SyntheticEventConfig(
                sessions=args.sessions,
                users=args.users,
                seed=args.seed,
            ),
        )
        write_verification_report(args.output, report)
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
