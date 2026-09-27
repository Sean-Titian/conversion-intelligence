import numpy as np
import pandas as pd
import pytest

from conversion_intelligence.data import (
    REQUIRED_COLUMNS,
    load_conversion_data,
    validate_conversion_data,
)


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


@pytest.mark.parametrize("column", REQUIRED_COLUMNS)
def test_validation_rejects_duplicate_column_names(column: str) -> None:
    frame = valid_frame()
    duplicated = pd.concat([frame, frame[[column]]], axis=1)

    with pytest.raises(
        ValueError,
        match=r"Training data has duplicate column names; count=2",
    ):
        validate_conversion_data(duplicated)


@pytest.mark.parametrize("column", ["age", "converted"])
def test_loader_rejects_duplicate_raw_csv_headers_before_pandas_mangling(
    tmp_path,
    column: str,
) -> None:
    headers = list(REQUIRED_COLUMNS)
    values = ["US", "25", "1", "Ads", "3", "0"]
    position = headers.index(column) + 1
    headers.insert(position, column)
    values.insert(position, "99" if column == "age" else "1")
    path = tmp_path / f"duplicate-{column}.csv"
    path.write_text(
        ",".join(f'"{name}"' for name in headers)
        + "\n"
        + ",".join(values)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"Training CSV header has duplicate column names; count=2",
    ):
        load_conversion_data(path)


def test_validation_canonicalizes_reordered_columns_and_ignores_unique_metadata() -> None:
    frame = valid_frame().assign(request_trace=["a", "b", "c"])
    reordered = frame[
        [
            "request_trace",
            "converted",
            "source",
            "new_user",
            "age",
            "country",
            "total_pages_visited",
        ]
    ]

    cleaned, _ = validate_conversion_data(reordered)

    assert list(cleaned.columns) == list(REQUIRED_COLUMNS)
    assert "request_trace" not in cleaned


def test_validation_rejects_non_binary_label() -> None:
    frame = valid_frame()
    frame.loc[0, "converted"] = 2
    with pytest.raises(ValueError, match="converted must be binary"):
        validate_conversion_data(frame)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("age", 31 + 1j),
        ("new_user", 1 + 0j),
        ("total_pages_visited", 3 + 1j),
        ("converted", np.complex64(1 + 0j)),
    ],
)
def test_validation_rejects_complex_numeric_fields(
    column: str,
    value: complex,
) -> None:
    frame = valid_frame()
    frame[column] = frame[column].astype(complex)
    frame.loc[0, column] = value

    with pytest.raises(ValueError, match=rf"{column} must contain real values"):
        validate_conversion_data(frame)


@pytest.mark.parametrize("column", ["age", "total_pages_visited"])
def test_validation_rejects_boolean_integer_fields(column: str) -> None:
    frame = valid_frame()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = True

    with pytest.raises(ValueError, match=rf"{column} must not contain boolean values"):
        validate_conversion_data(frame)


def test_validation_accepts_boolean_binary_fields_and_normalizes_them() -> None:
    frame = valid_frame()
    frame["age"] = [25, 33, 36]
    frame["new_user"] = [True, False, True]
    frame["converted"] = [False, True, True]

    cleaned, _ = validate_conversion_data(frame)

    assert cleaned["new_user"].dtype == "int8"
    assert cleaned["converted"].dtype == "int8"
    assert cleaned["new_user"].tolist() == [1, 0, 1]
    assert cleaned["converted"].tolist() == [0, 1, 1]


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
