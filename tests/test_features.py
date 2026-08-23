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


def test_scoring_contract_rejects_missing_or_invalid_required_values() -> None:
    frame = pd.DataFrame(
        {"country": ["US"], "age": [31], "new_user": [1], "source": ["Seo"]}
    )
    with pytest.raises(ValueError, match="Missing scoring features"):
        prepare_feature_frame(frame, "in_session")
    frame.loc[0, "new_user"] = 3
    with pytest.raises(ValueError, match="must be binary"):
        prepare_feature_frame(frame, "acquisition")
