from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from .data import load_conversion_data
from .features import SCENARIO_CONTRACTS, SCENARIO_FEATURES, prepare_feature_frame
from .modeling import (
    build_model_pipeline,
    evaluate_probabilities,
    select_cost_sensitive_threshold,
)
from .train import split_indices

AUDIT_METRICS = (
    "average_precision",
    "brier_score",
    "roc_auc",
    "precision_at_budget",
    "recall_at_budget",
    "lift_at_budget",
)


def profile_group_split_indices(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create 60/20/20 splits with no exact feature profile shared across splits."""
    groups = pd.util.hash_pandas_object(frame.loc[:, list(features)], index=False).to_numpy()
    all_indices = np.arange(len(frame))
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_validation, test = next(outer.split(all_indices, groups=groups))

    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_relative, validation_relative = next(
        inner.split(train_validation, groups=groups[train_validation])
    )
    train = train_validation[train_relative]
    validation = train_validation[validation_relative]
    return np.asarray(train), np.asarray(validation), np.asarray(test)


def profile_diagnostics(frame: pd.DataFrame, features: Sequence[str]) -> dict[str, int | float]:
    grouped = frame.groupby(list(features), dropna=False, observed=True)["converted"]
    group_sizes = grouped.size()
    conflicting = grouped.nunique().gt(1)
    return {
        "unique_profiles": int(len(group_sizes)),
        "rows_in_repeated_profiles": int(group_sizes[group_sizes > 1].sum()),
        "repeated_profile_fraction": float(group_sizes[group_sizes > 1].sum() / len(frame)),
        "profiles_with_both_labels": int(conflicting.sum()),
        "rows_in_profiles_with_both_labels": int(group_sizes[conflicting].sum()),
    }


def _fit_evaluate(
    features: Sequence[str],
    feature_frame: pd.DataFrame,
    labels: pd.Series,
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    training_labels: pd.Series | np.ndarray | None = None,
) -> tuple[dict[str, float | int], Pipeline, np.ndarray]:
    train, validation, test = indices
    pipeline = build_model_pipeline(tuple(features), "logistic_regression")
    fit_labels = labels.iloc[train] if training_labels is None else np.asarray(training_labels)
    pipeline.fit(feature_frame.iloc[train][list(features)], fit_labels)
    validation_probabilities = pipeline.predict_proba(
        feature_frame.iloc[validation][list(features)]
    )[:, 1]
    threshold = select_cost_sensitive_threshold(labels.iloc[validation], validation_probabilities)
    test_probabilities = pipeline.predict_proba(feature_frame.iloc[test][list(features)])[:, 1]
    metrics = evaluate_probabilities(labels.iloc[test], test_probabilities, threshold.threshold)
    return metrics, pipeline, test_probabilities


def _summarize_runs(runs: list[dict[str, float | int]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric in AUDIT_METRICS:
        values = np.asarray([float(run[metric]) for run in runs])
        summary[metric] = {
            "mean": float(values.mean()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
        }
    return summary


def bootstrap_metric_interval(
    labels: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    samples: int = 200,
    seed: int = 42,
) -> dict[str, float | int | str]:
    """Naive row bootstrap; dependence cannot be handled without a stable entity identifier."""
    if samples < 20:
        raise ValueError("samples must be at least 20")
    y = np.asarray(labels, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(samples):
        selected = rng.integers(0, len(y), len(y))
        if np.unique(y[selected]).size < 2:
            continue
        estimates.append(float(metric(y[selected], scores[selected])))
    if len(estimates) < samples * 0.9:
        raise ValueError("Too few valid bootstrap samples")
    return {
        "estimate": float(metric(y, scores)),
        "lower_95": float(np.quantile(estimates, 0.025)),
        "upper_95": float(np.quantile(estimates, 0.975)),
        "resamples": len(estimates),
        "method": "iid_row_bootstrap",
    }


def _input_corruption_results(
    scenario: str,
    clean_test: pd.DataFrame,
    labels: pd.Series,
    pipeline: Pipeline,
    *,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    variants: dict[str, pd.DataFrame] = {}

    missing = clean_test.copy()
    mask = rng.random(missing.shape) < 0.05
    missing = missing.mask(mask)
    variants["five_percent_cells_missing"] = missing

    unseen = clean_test.copy()
    for column in ("country", "source"):
        replace = rng.random(len(unseen)) < 0.05
        unseen.loc[replace, column] = "__UNSEEN__"
    variants["five_percent_unseen_categories"] = unseen

    noisy_age = clean_test.copy()
    noisy_age["age"] = np.clip(
        np.rint(noisy_age["age"] + rng.normal(0, 3, len(noisy_age))),
        13,
        100,
    )
    variants["age_measurement_noise_sd_3"] = noisy_age

    if scenario == "in_session":
        noisy_pages = clean_test.copy()
        noisy_pages["total_pages_visited"] = np.maximum(
            1,
            np.rint(
                noisy_pages["total_pages_visited"]
                + rng.normal(0, 2, len(noisy_pages))
            ),
        )
        variants["page_count_measurement_noise_sd_2"] = noisy_pages

    results: dict[str, dict[str, float | int]] = {}
    for name, variant in variants.items():
        probabilities = pipeline.predict_proba(variant)[:, 1]
        # Threshold-independent metrics are the focus of corruption sensitivity.
        results[name] = evaluate_probabilities(labels, probabilities, threshold=0.5)
    return results


def run_realism_audit(
    data_path: Path,
    output_path: Path,
    *,
    seeds: Sequence[int] = (7, 21, 42, 84, 2026),
    bootstrap_samples: int = 200,
) -> dict[str, object]:
    data, quality = load_conversion_data(data_path)
    labels = data["converted"]
    scenario_frames = {
        scenario: prepare_feature_frame(data, scenario) for scenario in SCENARIO_FEATURES
    }

    report: dict[str, object] = {
        "protocol": {
            "primary_metric": "average_precision",
            "secondary_metrics": [
                "brier_score",
                "roc_auc",
                "lift_at_5_percent",
                "recall_at_5_percent",
            ],
            "multi_seed_summary": "observed range, not a confidence interval",
            "confidence_interval": (
                "iid row bootstrap only; user/time/group uncertainty is not identifiable"
            ),
            "profile_holdout": (
                "sensitivity analysis that forbids exact feature profiles across splits; profiles "
                "are not proven users and repeated profiles are not automatically leakage"
            ),
            "input_corruption": (
                "measurement robustness only; it is not a substitute for a true out-of-time test"
            ),
        },
        "dataset": {
            "rows": len(data),
            "conversion_rate": float(labels.mean()),
            "quality": quality.to_dict(),
        },
        "scenarios": {},
    }

    base_seed = 42 if 42 in seeds else int(seeds[0])
    for scenario, features in SCENARIO_FEATURES.items():
        feature_frame = scenario_frames[scenario]
        runs: list[dict[str, float | int]] = []
        seed_42_pipeline: Pipeline | None = None
        seed_42_probabilities: np.ndarray | None = None
        seed_42_indices: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        for seed in seeds:
            indices = split_indices(labels, int(seed))
            metrics, pipeline, probabilities = _fit_evaluate(
                features,
                feature_frame,
                labels,
                indices,
            )
            metrics = {"seed": int(seed), **metrics}
            runs.append(metrics)
            if seed == base_seed:
                seed_42_pipeline = pipeline
                seed_42_probabilities = probabilities
                seed_42_indices = indices

        assert seed_42_pipeline is not None
        assert seed_42_probabilities is not None
        assert seed_42_indices is not None
        group_indices = profile_group_split_indices(data, features, seed=base_seed)
        group_metrics, _, _ = _fit_evaluate(
            features,
            feature_frame,
            labels,
            group_indices,
        )
        test_labels = labels.iloc[seed_42_indices[2]]
        bootstrap = {
            "average_precision": bootstrap_metric_interval(
                test_labels,
                seed_42_probabilities,
                average_precision_score,
                samples=bootstrap_samples,
                seed=base_seed,
            ),
            "roc_auc": bootstrap_metric_interval(
                test_labels,
                seed_42_probabilities,
                roc_auc_score,
                samples=bootstrap_samples,
                seed=base_seed + 1,
            ),
            "brier_score": bootstrap_metric_interval(
                test_labels,
                seed_42_probabilities,
                brier_score_loss,
                samples=bootstrap_samples,
                seed=base_seed + 2,
            ),
        }
        clean_test = feature_frame.iloc[seed_42_indices[2]].copy()
        corruptions = _input_corruption_results(
            scenario,
            clean_test,
            test_labels,
            seed_42_pipeline,
            seed=base_seed,
        )
        report["scenarios"][scenario] = {  # type: ignore[index]
            "prediction_contract": SCENARIO_CONTRACTS[scenario].to_dict(),
            "profile_diagnostics": profile_diagnostics(data, features),
            "row_random_split": {
                "runs": runs,
                "summary": _summarize_runs(runs),
                "bootstrap_on_base_seed_test": bootstrap,
            },
            "exact_profile_group_holdout": {
                "train_rows": len(group_indices[0]),
                "validation_rows": len(group_indices[1]),
                "test_rows": len(group_indices[2]),
                "test_prevalence": float(labels.iloc[group_indices[2]].mean()),
                "metrics": group_metrics,
            },
            "input_corruption_on_base_seed_test": corruptions,
        }

    base_indices = split_indices(labels, base_seed)
    train, validation, test = base_indices
    prevalence = float(labels.iloc[train].mean())
    constant_validation = np.full(len(validation), prevalence)
    constant_test = np.full(len(test), prevalence)
    constant_threshold = select_cost_sensitive_threshold(
        labels.iloc[validation], constant_validation
    ).threshold

    rng = np.random.default_rng(base_seed)
    random_validation = np.clip(
        prevalence + rng.normal(0, 1e-4, len(validation)), 0, 1
    )
    random_test = np.clip(prevalence + rng.normal(0, 1e-4, len(test)), 0, 1)
    random_threshold = select_cost_sensitive_threshold(
        labels.iloc[validation], random_validation
    ).threshold

    page_features = ("total_pages_visited",)
    page_metrics, _, _ = _fit_evaluate(page_features, data, labels, base_indices)
    permuted_training_labels = labels.iloc[train].sample(
        frac=1,
        random_state=base_seed,
    ).to_numpy()
    permutation_metrics, _, _ = _fit_evaluate(
        SCENARIO_FEATURES["acquisition"],
        scenario_frames["acquisition"],
        labels,
        base_indices,
        training_labels=permuted_training_labels,
    )
    report["baselines_and_negative_control"] = {
        "training_prevalence_constant": evaluate_probabilities(
            labels.iloc[test], constant_test, constant_threshold
        ),
        "calibrated_random_ranking": evaluate_probabilities(
            labels.iloc[test], random_test, random_threshold
        ),
        "pages_only_logistic_regression": page_metrics,
        "permuted_training_labels_acquisition": permutation_metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote realism audit to {output_path}")
    for scenario in SCENARIO_FEATURES:
        details = report["scenarios"][scenario]  # type: ignore[index]
        summary = details["row_random_split"]["summary"]  # type: ignore[index]
        grouped = details["exact_profile_group_holdout"]["metrics"]  # type: ignore[index]
        print(
            f"{scenario}: row-split AP mean={summary['average_precision']['mean']:.4f} "
            f"range={summary['average_precision']['minimum']:.4f}-"
            f"{summary['average_precision']['maximum']:.4f}; "
            f"profile-holdout AP={grouped['average_precision']:.4f}"
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run conversion-model realism stress tests")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/robustness/realism_audit.json"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 21, 42, 84, 2026])
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_realism_audit(
        args.data,
        args.output,
        seeds=args.seeds,
        bootstrap_samples=args.bootstrap_samples,
    )


if __name__ == "__main__":
    main()
