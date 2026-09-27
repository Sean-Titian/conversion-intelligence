from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .schema import require_unique_column_names


@dataclass(frozen=True)
class PredictionContract:
    """Features and decision timing that give a model score a valid meaning."""

    decision_time: str
    features: tuple[str, ...]
    deployment_status: str
    limitation: str

    def to_dict(self) -> dict[str, str | list[str]]:
        result: dict[str, str | list[str]] = asdict(self)
        result["features"] = list(self.features)
        return result


SCENARIO_CONTRACTS: dict[str, PredictionContract] = {
    "acquisition": PredictionContract(
        decision_time="session_start",
        features=("country", "age", "new_user", "source"),
        deployment_status="candidate_for_prospective_validation",
        limitation=(
            "The source has no timestamp or stable user identifier, so temporal and user-level "
            "generalization cannot be verified."
        ),
    ),
    "in_session": PredictionContract(
        decision_time="session_end_in_source_data",
        features=(
            "country",
            "age",
            "new_user",
            "source",
            "total_pages_visited",
        ),
        deployment_status="retrospective_upper_bound_only",
        limitation=(
            "The source records final total pages, not pages observed at a timestamped "
            "intervention point. This score cannot be claimed as a real-time model without "
            "event-level data."
        ),
    ),
}

SCENARIO_FEATURES: dict[str, tuple[str, ...]] = {
    name: contract.features for name, contract in SCENARIO_CONTRACTS.items()
}

CATEGORICAL_FEATURES = frozenset({"country", "source"})


def features_for_scenario(scenario: str) -> tuple[str, ...]:
    try:
        return SCENARIO_FEATURES[scenario]
    except KeyError as exc:
        choices = ", ".join(sorted(SCENARIO_FEATURES))
        raise ValueError(f"Unknown scenario {scenario!r}; choose one of: {choices}") from exc


def contract_for_scenario(scenario: str) -> PredictionContract:
    try:
        return SCENARIO_CONTRACTS[scenario]
    except KeyError as exc:
        choices = ", ".join(sorted(SCENARIO_CONTRACTS))
        raise ValueError(f"Unknown scenario {scenario!r}; choose one of: {choices}") from exc


def _prepare_feature_columns(
    frame: pd.DataFrame,
    features: tuple[str, ...],
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Scoring input must be a pandas DataFrame")
    require_unique_column_names(frame.columns, context="Scoring request")
    require_unique_column_names(features, context="Model feature contract")

    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing scoring features: {', '.join(missing)}")

    prepared = frame.loc[:, list(features)].copy()
    for column in CATEGORICAL_FEATURES.intersection(features):
        values = prepared[column].astype("string").str.strip()
        values = values.mask(values.eq(""))
        # sklearn's default imputer expects np.nan rather than pandas.NA for object columns.
        prepared[column] = values.astype(object).where(values.notna(), np.nan)

    numeric = [name for name in features if name not in CATEGORICAL_FEATURES]
    for column in numeric:
        if column in {"age", "total_pages_visited"}:
            boolean_values = prepared[column].map(
                lambda value: isinstance(value, (bool, np.bool_))
            )
            if boolean_values.any():
                raise ValueError(f"{column} scoring values must not be boolean")
        original_missing = prepared[column].isna()
        converted = pd.to_numeric(prepared[column], errors="coerce")
        invalid = converted.isna() & ~original_missing
        if invalid.any():
            raise ValueError(f"{column} contains non-numeric scoring values")
        if np.iscomplexobj(converted.to_numpy()):
            raise ValueError(f"{column} scoring values must be real numbers")
        prepared[column] = converted

    for column in numeric:
        present = prepared[column].notna()
        finite = np.isfinite(prepared[column].where(present, 0.0))
        if (present & ~finite).any():
            raise ValueError(f"{column} scoring values must be finite")

    for column in ("age", "total_pages_visited"):
        if column not in prepared:
            continue
        present = prepared[column].notna()
        whole = np.equal(np.mod(prepared[column].where(present, 0.0), 1), 0)
        if (present & ~whole).any():
            raise ValueError(f"{column} scoring values must be whole numbers")

    if "new_user" in prepared:
        invalid_binary = prepared["new_user"].notna() & ~prepared["new_user"].isin([0, 1])
        if invalid_binary.any():
            raise ValueError("new_user scoring values must be binary")
    if "age" in prepared:
        invalid_age = prepared["age"].notna() & ~prepared["age"].between(13, 100)
        if invalid_age.any():
            raise ValueError("age scoring values must be between 13 and 100")
    if "total_pages_visited" in prepared:
        invalid_pages = (
            prepared["total_pages_visited"].notna()
            & (prepared["total_pages_visited"] < 1)
        )
        if invalid_pages.any():
            raise ValueError("total_pages_visited scoring values must be positive")
    return prepared


def prepare_feature_frame(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Validate a scoring request and return features in training order.

    Missing values are allowed because the fitted preprocessing pipeline owns imputation. Unknown
    categorical values are also allowed and are handled by the one-hot encoder. Invalid numeric
    strings and impossible known values fail fast instead of silently changing the score.
    """

    return _prepare_feature_columns(frame, features_for_scenario(scenario))


class FeatureContractTransformer(TransformerMixin, BaseEstimator):
    """Embed the same feature projection and domain checks in fit and predict."""

    def __init__(self, feature_names: tuple[str, ...]):
        self.feature_names = feature_names

    def fit(
        self,
        frame: pd.DataFrame,
        _labels: object = None,
    ) -> FeatureContractTransformer:
        prepared = _prepare_feature_columns(frame, self.feature_names)
        self.n_features_in_ = prepared.shape[1]
        self.feature_names_in_ = np.asarray(self.feature_names, dtype=object)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return _prepare_feature_columns(frame, self.feature_names)

    def get_feature_names_out(
        self,
        _input_features: object = None,
    ) -> np.ndarray:
        return np.asarray(self.feature_names, dtype=object)
