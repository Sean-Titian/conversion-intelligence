from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "country",
    "age",
    "new_user",
    "source",
    "total_pages_visited",
    "converted",
)


@dataclass(frozen=True)
class DataQualityReport:
    input_rows: int
    output_rows: int
    invalid_age_rows: int
    invalid_page_rows: int
    exact_duplicate_rows: int
    unique_feature_profiles: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def load_conversion_data(path: str | Path) -> tuple[pd.DataFrame, DataQualityReport]:
    """Load and validate a conversion dataset without mutating the source file."""
    frame = pd.read_csv(path)
    return validate_conversion_data(frame)


def validate_conversion_data(
    frame: pd.DataFrame,
    *,
    minimum_age: int = 13,
    maximum_age: int = 100,
) -> tuple[pd.DataFrame, DataQualityReport]:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    data = frame.loc[:, REQUIRED_COLUMNS].copy()
    for column in ("age", "new_user", "total_pages_visited", "converted"):
        data[column] = pd.to_numeric(data[column], errors="raise")

    for column in ("new_user", "converted"):
        values = set(data[column].dropna().unique())
        if not values.issubset({0, 1}):
            raise ValueError(f"{column} must be binary; found {sorted(values)}")
        data[column] = data[column].astype("int8")

    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        missing_counts = data.isna().sum()
        details = ", ".join(f"{k}={v}" for k, v in missing_counts.items() if v)
        raise ValueError(f"Missing values are not supported in the source contract: {details}")

    data["country"] = data["country"].astype("string").str.strip()
    data["source"] = data["source"].astype("string").str.strip()
    for column in ("country", "source"):
        if data[column].eq("").any():
            raise ValueError(f"{column} must not contain blank values")

    for column in ("age", "total_pages_visited"):
        values = data[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must contain finite values")
        if not np.equal(np.mod(values, 1), 0).all():
            raise ValueError(f"{column} must contain whole numbers")

    invalid_age = ~data["age"].between(minimum_age, maximum_age)
    invalid_pages = data["total_pages_visited"] < 1
    exact_duplicates = int(data.duplicated().sum())
    keep = ~(invalid_age | invalid_pages)
    cleaned = data.loc[keep].reset_index(drop=True)
    cleaned["age"] = cleaned["age"].astype("int16")
    cleaned["total_pages_visited"] = cleaned["total_pages_visited"].astype("int32")

    report = DataQualityReport(
        input_rows=len(data),
        output_rows=len(cleaned),
        invalid_age_rows=int(invalid_age.sum()),
        invalid_page_rows=int(invalid_pages.sum()),
        exact_duplicate_rows=exact_duplicates,
        unique_feature_profiles=int(cleaned.drop(columns="converted").drop_duplicates().shape[0]),
    )
    return cleaned, report
