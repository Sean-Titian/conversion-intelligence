import numpy as np
import pandas as pd

from conversion_intelligence.features import prepare_feature_frame
from conversion_intelligence.modeling import (
    build_model_pipeline,
    evaluate_probabilities,
    evaluate_ranking_at_budget,
    select_cost_sensitive_threshold,
)


def test_higher_false_negative_cost_selects_no_higher_threshold() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1])
    probabilities = np.array([0.05, 0.10, 0.25, 0.40, 0.35, 0.80])
    balanced = select_cost_sensitive_threshold(
        labels,
        probabilities,
        false_positive_cost=1,
        false_negative_cost=1,
    )
    recall_weighted = select_cost_sensitive_threshold(
        labels,
        probabilities,
        false_positive_cost=1,
        false_negative_cost=10,
    )
    assert recall_weighted.threshold <= balanced.threshold


def test_probability_evaluation_returns_confusion_counts() -> None:
    metrics = evaluate_probabilities(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.7, 0.4, 0.9]),
        0.5,
    )
    assert metrics["true_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 1
    assert "brier_score" in metrics
    assert "expected_calibration_error" in metrics


def test_ranking_metrics_measure_top_capacity() -> None:
    metrics = evaluate_ranking_at_budget(
        np.array([0, 1, 0, 1, 0]),
        np.array([0.1, 0.9, 0.2, 0.8, 0.3]),
        budget_fraction=0.4,
    )
    assert metrics["selected_rows"] == 2
    assert metrics["precision_at_budget"] == 1.0
    assert metrics["recall_at_budget"] == 1.0
    assert metrics["lift_at_budget"] == 2.5


def test_training_and_scoring_column_order_have_prediction_parity() -> None:
    train = pd.DataFrame(
        {
            "country": ["US", "UK", "US", "China", "UK", "China"],
            "age": [24, 33, 41, 28, 36, 52],
            "new_user": [1, 0, 1, 1, 0, 0],
            "source": ["Ads", "Seo", "Direct", "Ads", "Direct", "Seo"],
        }
    )
    labels = np.array([0, 1, 0, 0, 1, 1])
    model = build_model_pipeline(tuple(train.columns), "logistic_regression")
    model.fit(train, labels)
    request = train[["source", "new_user", "age", "country"]].copy()
    request["ignored"] = "safe"
    prepared = prepare_feature_frame(request, "acquisition")
    np.testing.assert_allclose(model.predict_proba(train), model.predict_proba(prepared))

    degraded_request = request.copy()
    degraded_request.loc[0, "country"] = "Atlantis"
    degraded_request.loc[1, "source"] = None
    degraded = prepare_feature_frame(degraded_request, "acquisition")
    assert model.predict_proba(degraded).shape == (len(degraded), 2)
