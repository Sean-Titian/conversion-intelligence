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
    selected_rows: int
    selection_rate: float
    objective: str
    max_selection_fraction: float | None
    max_selected_rows: int | None
    policy_unused_capacity_rows: int | None
    capacity_binding: bool
    max_feasible_selected_rows: int | None
    frontier_unused_capacity_rows: int | None
    cutoff_tie_rows: int
    next_infeasible_tie_rows: int | None

    def to_dict(self) -> dict[str, float | int | str | None]:
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


def _validated_binary_scores(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw_labels = np.asarray(y_true)
    try:
        scores = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Probabilities must be numeric") from exc
    if raw_labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("Labels and probabilities must be one-dimensional")
    if len(raw_labels) != len(scores) or len(scores) == 0:
        raise ValueError("Labels and probabilities must have the same non-zero length")
    try:
        numeric_labels = raw_labels.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Labels must be binary") from exc
    if not np.isfinite(numeric_labels).all() or not np.isin(
        numeric_labels, [0.0, 1.0]
    ).all():
        raise ValueError("Labels must be binary")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Probabilities must be between zero and one")
    return numeric_labels.astype(int), scores


def _validate_misclassification_costs(
    false_positive_cost: float,
    false_negative_cost: float,
) -> None:
    if not np.isfinite(false_positive_cost) or not np.isfinite(false_negative_cost):
        raise ValueError("Misclassification costs must be finite")
    if false_positive_cost < 0 or false_negative_cost < 0:
        raise ValueError("Misclassification costs must be non-negative")
    if false_positive_cost == 0 and false_negative_cost == 0:
        raise ValueError("At least one misclassification cost must be positive")


def select_cost_sensitive_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 10.0,
    max_selection_fraction: float | None = None,
) -> ThresholdChoice:
    """Choose a validation operating threshold under explicit loss and capacity.

    Every distinct score boundary is evaluated, so the result is not an artifact of a coarse
    threshold grid. Scores tied at a boundary are kept together. If admitting a tied block would
    exceed ``max_selection_fraction``, that candidate is infeasible and the policy can leave some
    capacity unused rather than break ties using an unavailable session identifier.
    """
    _validate_misclassification_costs(false_positive_cost, false_negative_cost)
    if max_selection_fraction is not None and not 0 < max_selection_fraction <= 1:
        raise ValueError("max_selection_fraction must be in (0, 1]")
    labels, scores = _validated_binary_scores(y_true, probabilities)

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cumulative_true_positives = np.cumsum(sorted_labels)

    # Each group end represents a feasible threshold that includes the full tied-score block.
    group_ends = np.flatnonzero(
        np.concatenate((sorted_scores[:-1] != sorted_scores[1:], [True]))
    )
    selected_rows = group_ends + 1
    true_positives = cumulative_true_positives[group_ends]
    false_positives = selected_rows - true_positives
    false_negatives = int(labels.sum()) - true_positives
    thresholds = sorted_scores[group_ends]
    group_starts = np.concatenate(([0], group_ends[:-1] + 1))
    boundary_tie_rows = group_ends - group_starts + 1

    # A select-none policy is needed when the first score block exceeds a hard capacity.
    select_none_threshold = np.nextafter(float(sorted_scores[0]), np.inf)
    thresholds = np.concatenate(([select_none_threshold], thresholds))
    selected_rows = np.concatenate(([0], selected_rows))
    false_positives = np.concatenate(([0], false_positives))
    false_negatives = np.concatenate(([int(labels.sum())], false_negatives))
    boundary_tie_rows = np.concatenate(([0], boundary_tie_rows))

    maximum_rows = (
        None
        if max_selection_fraction is None
        else int(np.floor(len(labels) * max_selection_fraction))
    )
    feasible = np.ones(len(thresholds), dtype=bool)
    if maximum_rows is not None:
        feasible &= selected_rows <= maximum_rows

    costs = (
        false_positives * false_positive_cost
        + false_negatives * false_negative_cost
    ) / len(labels) * 1_000
    unconstrained_best_index = min(
        range(len(thresholds)),
        key=lambda index: (float(costs[index]), -float(thresholds[index])),
    )
    feasible_indices = np.flatnonzero(feasible)
    # Resolve equal loss toward the higher threshold / smaller selected population.
    best_index = min(
        feasible_indices,
        key=lambda index: (float(costs[index]), -float(thresholds[index])),
    )
    chosen_rows = int(selected_rows[best_index])
    policy_unused_capacity = (
        None if maximum_rows is None else maximum_rows - chosen_rows
    )
    if maximum_rows is None:
        capacity_binding = False
        max_feasible_rows = None
        frontier_unused_capacity = None
        next_infeasible_tie_rows = None
    else:
        capacity_binding = int(selected_rows[unconstrained_best_index]) > maximum_rows
        frontier_index = int(feasible_indices[-1])
        max_feasible_rows = int(selected_rows[frontier_index])
        frontier_unused_capacity = maximum_rows - max_feasible_rows
        next_infeasible_tie_rows = (
            int(boundary_tie_rows[frontier_index + 1])
            if frontier_index + 1 < len(boundary_tie_rows)
            else None
        )
    return ThresholdChoice(
        threshold=float(thresholds[best_index]),
        expected_cost_per_1000=float(costs[best_index]),
        false_positive_cost=float(false_positive_cost),
        false_negative_cost=float(false_negative_cost),
        selected_rows=chosen_rows,
        selection_rate=float(chosen_rows / len(labels)),
        objective="minimize_expected_misclassification_cost",
        max_selection_fraction=max_selection_fraction,
        max_selected_rows=maximum_rows,
        policy_unused_capacity_rows=policy_unused_capacity,
        capacity_binding=capacity_binding,
        max_feasible_selected_rows=max_feasible_rows,
        frontier_unused_capacity_rows=frontier_unused_capacity,
        cutoff_tie_rows=int(boundary_tie_rows[best_index]),
        next_infeasible_tie_rows=next_infeasible_tie_rows,
    )


def evaluate_probabilities(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    *,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 10.0,
    budget_fraction: float = 0.05,
) -> dict[str, float | int | str]:
    _validate_misclassification_costs(false_positive_cost, false_negative_cost)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    labels, scores = _validated_binary_scores(y_true, probabilities)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    expected_cost = (fp * false_positive_cost + fn * false_negative_cost) / len(labels) * 1_000
    selected_rows = int(predicted.sum())
    budget = evaluate_ranking_at_budget(labels, scores, budget_fraction=budget_fraction)
    budget_selected_rows = int(budget.pop("selected_rows"))
    budget_selected_positives = int(budget.pop("selected_positives"))
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
        "selected_rows": selected_rows,
        "selection_rate": float(selected_rows / len(labels)),
        "budget_selected_rows": budget_selected_rows,
        "budget_selected_positives": budget_selected_positives,
        **budget,
    }


def evaluate_ranking_at_budget(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    budget_fraction: float = 0.05,
) -> dict[str, float | int | str]:
    """Evaluate a fixed-capacity outreach policy using the top-scored rows."""
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must be in (0, 1]")
    labels, scores = _validated_binary_scores(y_true, probabilities)

    selected_rows = max(1, int(np.ceil(len(labels) * budget_fraction)))
    # mergesort is stable, which makes tied-score behavior deterministic and testable.
    selected = np.argsort(-scores, kind="mergesort")[:selected_rows]
    boundary_score = float(scores[selected[-1]])
    boundary_tie_rows = int(np.count_nonzero(scores == boundary_score))
    selected_from_boundary_tie = int(
        np.count_nonzero(scores[selected] == boundary_score)
    )
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
        "budget_tie_break_policy": "stable_input_order",
        "budget_boundary_score": boundary_score,
        "budget_boundary_tie_rows": boundary_tie_rows,
        "budget_selected_from_boundary_tie": selected_from_boundary_tie,
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
    labels, scores = _validated_binary_scores(y_true, probabilities)

    bin_ids = np.minimum((scores * bins).astype(int), bins - 1)
    error = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if mask.any():
            error += float(mask.mean()) * abs(float(labels[mask].mean() - scores[mask].mean()))
    return float(error)
