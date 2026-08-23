import numpy as np
import pandas as pd

from conversion_intelligence.audit import (
    bootstrap_metric_interval,
    profile_group_split_indices,
)


def test_profile_group_split_has_no_profile_overlap() -> None:
    frame = pd.DataFrame(
        {
            "country": np.repeat(["US", "UK", "China", "Germany"], 30),
            "age": np.tile(np.repeat([20, 30, 40], 10), 4),
            "new_user": np.tile([0, 1], 60),
        }
    )
    features = ["country", "age", "new_user"]
    train, validation, test = profile_group_split_indices(frame, features, seed=7)

    def profiles(indices: np.ndarray) -> set[tuple[object, ...]]:
        return set(frame.iloc[indices][features].itertuples(index=False, name=None))

    assert profiles(train).isdisjoint(profiles(validation))
    assert profiles(train).isdisjoint(profiles(test))
    assert profiles(validation).isdisjoint(profiles(test))


def test_bootstrap_interval_contains_point_estimate() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1] * 20)
    probabilities = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9] * 20)
    interval = bootstrap_metric_interval(
        labels,
        probabilities,
        lambda y, scores: float(((scores >= 0.5) == y).mean()),
        samples=30,
        seed=11,
    )
    assert interval["lower_95"] <= interval["estimate"] <= interval["upper_95"]
