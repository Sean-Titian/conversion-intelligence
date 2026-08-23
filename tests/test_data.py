import pandas as pd
import pytest

from conversion_intelligence.data import validate_conversion_data


def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "country": ["US", "UK", "Germany"],
            "age": [25, 111, 36],
            "new_user": [1, 0, 1],
            "source": ["Ads", "Seo", "Direct"],
            "total_pages_visited": [3, 4, 8],
            "converted": [0, 1, 1],
        }
    )


def test_validation_drops_impossible_age_and_reports_it() -> None:
    cleaned, report = validate_conversion_data(valid_frame())
    assert len(cleaned) == 2
    assert report.invalid_age_rows == 1
    assert report.output_rows == 2
    assert report.unique_feature_profiles == 2


def test_validation_rejects_missing_column() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_conversion_data(valid_frame().drop(columns="source"))


def test_validation_rejects_non_binary_label() -> None:
    frame = valid_frame()
    frame.loc[0, "converted"] = 2
    with pytest.raises(ValueError, match="converted must be binary"):
        validate_conversion_data(frame)


@pytest.mark.parametrize("column", ["country", "source"])
def test_validation_rejects_blank_categories(column: str) -> None:
    frame = valid_frame()
    frame.loc[0, column] = "   "
    with pytest.raises(ValueError, match=f"{column} must not contain blank values"):
        validate_conversion_data(frame)


@pytest.mark.parametrize("column", ["age", "total_pages_visited"])
def test_validation_rejects_fractional_integer_fields(column: str) -> None:
    frame = valid_frame()
    frame[column] = frame[column].astype(float)
    frame.loc[0, column] = 25.5
    with pytest.raises(ValueError, match=f"{column} must contain whole numbers"):
        validate_conversion_data(frame)
