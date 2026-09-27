import pandas as pd
import pytest

from conversion_intelligence.features import (
    contract_for_scenario,
    features_for_scenario,
    prepare_feature_frame,
)


def test_acquisition_scenario_excludes_in_session_feature() -> None:
    assert "total_pages_visited" not in features_for_scenario("acquisition")
    assert "total_pages_visited" in features_for_scenario("in_session")


def test_full_session_contract_is_not_claimed_as_realtime_deployable() -> None:
    contract = contract_for_scenario("in_session")
    assert contract.decision_time == "session_end_in_source_data"
    assert contract.deployment_status == "retrospective_upper_bound_only"


def test_unknown_scenario_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown scenario"):
        features_for_scenario("after_purchase")


def test_scoring_contract_orders_columns_and_accepts_unseen_categories() -> None:
    frame = pd.DataFrame(
        {
            "irrelevant": [123],
            "source": ["NewPartner"],
            "new_user": [1],
            "age": [31],
            "country": ["Atlantis"],
        }
    )
    prepared = prepare_feature_frame(frame, "acquisition")
    assert list(prepared.columns) == ["country", "age", "new_user", "source"]
    assert prepared.loc[0, "country"] == "Atlantis"


def test_scoring_contract_uses_imputer_compatible_missing_categories() -> None:
    frame = pd.DataFrame(
        {"country": [None], "age": [31], "new_user": [1], "source": ["Seo"]}
    )
    prepared = prepare_feature_frame(frame, "acquisition")
    assert pd.isna(prepared.loc[0, "country"])
    assert prepared["country"].dtype == object


@pytest.mark.parametrize(
    ("scenario", "column"),
    [
        ("acquisition", "age"),
        ("acquisition", "new_user"),
        ("in_session", "total_pages_visited"),
    ],
)
def test_scoring_contract_allows_missing_numeric_values_for_fitted_imputation(
    scenario: str,
    column: str,
) -> None:
    frame = pd.DataFrame(
        {
            "country": ["US"],
            "age": [31],
            "new_user": [1],
            "source": ["Seo"],
            "total_pages_visited": [3],
        }
    )
    frame.loc[0, column] = None

    prepared = prepare_feature_frame(frame, scenario)

    assert pd.isna(prepared.loc[0, column])


@pytest.mark.parametrize(
    ("scenario", "column"),
    [
        ("acquisition", "country"),
        ("acquisition", "age"),
        ("in_session", "total_pages_visited"),
    ],
)
def test_scoring_contract_rejects_duplicate_required_columns(
    scenario: str,
    column: str,
) -> None:
    frame = pd.DataFrame(
        {
            "country": ["US"],
            "age": [31],
            "new_user": [1],
            "source": ["Seo"],
            "total_pages_visited": [3],
        }
    )
    duplicated = pd.concat([frame, frame[[column]]], axis=1)

    with pytest.raises(
        ValueError,
        match=r"Scoring request has duplicate column names; count=2",
    ):
        prepare_feature_frame(duplicated, scenario)


def test_scoring_contract_rejects_duplicate_irrelevant_metadata_columns() -> None:
    frame = pd.DataFrame(
        {
            "country": ["US"],
            "age": [31],
            "new_user": [1],
            "source": ["Seo"],
            "trace": ["a"],
        }
    )
    duplicated = pd.concat([frame, frame[["trace"]]], axis=1)

    with pytest.raises(ValueError, match=r"duplicate column names; count=2"):
        prepare_feature_frame(duplicated, "acquisition")


@pytest.mark.parametrize(
    ("scenario", "column", "value", "message"),
    [
        ("acquisition", "age", 31.5, "whole numbers"),
        ("acquisition", "age", float("inf"), "finite"),
        ("in_session", "total_pages_visited", 3.5, "whole numbers"),
        ("in_session", "total_pages_visited", float("inf"), "finite"),
        ("in_session", "total_pages_visited", float("-inf"), "finite"),
        ("acquisition", "age", 31 + 1j, "real numbers"),
        ("acquisition", "new_user", 1 + 0j, "real numbers"),
        ("in_session", "total_pages_visited", 3 + 1j, "real numbers"),
    ],
)
def test_scoring_contract_rejects_nonfinite_or_fractional_integer_features(
    scenario: str,
    column: str,
    value: complex | float,
    message: str,
) -> None:
    frame = pd.DataFrame(
        {
            "country": ["US"],
            "age": [31],
            "new_user": [1],
            "source": ["Seo"],
            "total_pages_visited": [3],
        }
    )
    frame[column] = frame[column].astype(complex if isinstance(value, complex) else float)
    frame.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        prepare_feature_frame(frame, scenario)


@pytest.mark.parametrize(
    ("scenario", "column"),
    [
        ("acquisition", "age"),
        ("in_session", "total_pages_visited"),
    ],
)
def test_scoring_contract_rejects_boolean_integer_features(
    scenario: str,
    column: str,
) -> None:
    frame = pd.DataFrame(
        {
            "country": ["US"],
            "age": [31],
            "new_user": [1],
            "source": ["Seo"],
            "total_pages_visited": [3],
        }
    )
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = True

    with pytest.raises(ValueError, match=rf"{column} scoring values must not be boolean"):
        prepare_feature_frame(frame, scenario)


def test_scoring_contract_rejects_missing_or_invalid_required_values() -> None:
    frame = pd.DataFrame(
        {"country": ["US"], "age": [31], "new_user": [1], "source": ["Seo"]}
    )
    with pytest.raises(ValueError, match="Missing scoring features"):
        prepare_feature_frame(frame, "in_session")
    frame.loc[0, "new_user"] = 3
    with pytest.raises(ValueError, match="must be binary"):
        prepare_feature_frame(frame, "acquisition")
