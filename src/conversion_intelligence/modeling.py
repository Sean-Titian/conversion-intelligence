from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    expected_cost_per_1000: float
    false_positive_cost: float
    false_negative_cost: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def build_model_pipeline(feature_names: tuple[str, ...], model_name: str) -> Pipeline:
    categorical = [name for name in feature_names if name in CATEGORICAL_FEATURES]
    numeric = [name for name in feature_names if name not in CATEGORICAL_FEATURES]

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )

    if model_name == "logistic_regression":
        estimator = LogisticRegression(max_iter=1_000, random_state=42)
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=12,
            n_jobs=-1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(steps=[("preprocess", preprocessing), ("model", estimator)])


def select_cost_sensitive_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 10.0,
) -> ThresholdChoice:
    """Choose an operating threshold on validation data using explicit costs."""
    if false_positive_cost < 0 or false_negative_cost < 0:
        raise ValueError("Misclassification costs must be non-negative")
    if len(y_true) != len(probabilities) or len(probabilities) == 0:
        raise ValueError("Labels and probabilities must have the same non-zero length")

    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("Probabilities must be between zero and one")

    candidates = np.unique(np.concatenate(([0.0], np.linspace(0.01, 0.99, 197), [1.0])))
    best: tuple[float, float] | None = None
    for threshold in candidates:
        predicted = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
        del tn, tp
        cost = (fp * false_positive_cost + fn * false_negative_cost) / len(labels) * 1_000
        candidate = (float(cost), -float(threshold))
        if best is None or candidate < best:
            best = candidate

    assert best is not None
    return ThresholdChoice(
        threshold=-best[1],
        expected_cost_per_1000=best[0],
        false_positive_cost=float(false_positive_cost),
        false_negative_cost=float(false_negative_cost),
    )


def evaluate_probabilities(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    *,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 10.0,
    budget_fraction: float = 0.05,
) -> dict[str, float | int]:
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    expected_cost = (fp * false_positive_cost + fn * false_negative_cost) / len(labels) * 1_000
    budget = evaluate_ranking_at_budget(labels, scores, budget_fraction=budget_fraction)
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
        "calibration_bias": float(scores.mean() - labels.mean()),
        "expected_calibration_error": expected_calibration_error(labels, scores),
        "threshold": float(threshold),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "expected_cost_per_1000": float(expected_cost),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        **budget,
    }


def evaluate_ranking_at_budget(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    budget_fraction: float = 0.05,
) -> dict[str, float | int]:
    """Evaluate a fixed-capacity outreach policy using the top-scored rows."""
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must be in (0, 1]")
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if len(labels) != len(scores) or len(scores) == 0:
        raise ValueError("Labels and probabilities must have the same non-zero length")

    selected_rows = max(1, int(np.ceil(len(labels) * budget_fraction)))
    # mergesort is stable, which makes tied-score behavior deterministic and testable.
    selected = np.argsort(-scores, kind="mergesort")[:selected_rows]
    selected_positives = int(labels[selected].sum())
    total_positives = int(labels.sum())
    prevalence = float(labels.mean())
    precision = selected_positives / selected_rows
    recall = selected_positives / total_positives if total_positives else 0.0
    lift = precision / prevalence if prevalence else 0.0
    return {
        "budget_fraction": float(budget_fraction),
        "selected_rows": selected_rows,
        "selected_positives": selected_positives,
        "precision_at_budget": float(precision),
        "recall_at_budget": float(recall),
        "lift_at_budget": float(lift),
    }


def expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Return equal-width expected calibration error (ECE)."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if len(labels) != len(scores) or len(scores) == 0:
        raise ValueError("Labels and probabilities must have the same non-zero length")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("Probabilities must be between zero and one")

    bin_ids = np.minimum((scores * bins).astype(int), bins - 1)
    error = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if mask.any():
            error += float(mask.mean()) * abs(float(labels[mask].mean() - scores[mask].mean()))
    return float(error)
