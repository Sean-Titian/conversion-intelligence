from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split

from .data import load_conversion_data
from .features import SCENARIO_CONTRACTS, SCENARIO_FEATURES, prepare_feature_frame
from .modeling import (
    build_model_pipeline,
    evaluate_probabilities,
    select_cost_sensitive_threshold,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_NAMES = ("logistic_regression", "random_forest")


def split_indices(y: pd.Series, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    train_validation, test = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed,
        stratify=y,
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        random_state=seed,
        stratify=y.iloc[train_validation],
    )
    return np.asarray(train), np.asarray(validation), np.asarray(test)


def plot_model_curves(
    scenario: str,
    curves: list[tuple[str, np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model_name, labels, probabilities in curves:
        display_name = model_name.replace("_", " ").title()
        RocCurveDisplay.from_predictions(labels, probabilities, name=display_name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(
            labels,
            probabilities,
            name=display_name,
            ax=axes[1],
        )
    axes[0].set_title(f"ROC curve - {scenario.replace('_', ' ')}")
    axes[1].set_title(f"Precision-recall curve - {scenario.replace('_', ' ')}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{scenario}_model_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_descriptive_rates(data: pd.DataFrame, output_dir: Path) -> None:
    age_bins = pd.cut(data["age"], bins=[12, 20, 30, 40, 50, 60, 100])
    panels = [
        (data.groupby("country", observed=True)["converted"].mean().sort_values(), "Country"),
        (data.groupby(age_bins, observed=True)["converted"].mean(), "Age band"),
        (
            data.groupby("total_pages_visited", observed=True)["converted"].mean(),
            "Pages visited",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis, (rates, label) in zip(axes, panels, strict=True):
        rates.plot(kind="bar", ax=axis, color="#7c3aed")
        axis.set_xlabel(label)
        axis.set_ylabel("Observed conversion rate")
        axis.tick_params(axis="x", rotation=45)
    fig.suptitle("Descriptive associations - not causal effects", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "descriptive_conversion_rates.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_training(
    data_path: Path,
    output_dir: Path,
    *,
    model_names: tuple[str, ...] = MODEL_NAMES,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 10.0,
    seed: int = 42,
) -> dict[str, object]:
    data, quality = load_conversion_data(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    y = data["converted"]
    train_indices, validation_indices, test_indices = split_indices(y, seed)

    report: dict[str, object] = {
        "dataset": {
            "rows": len(data),
            "conversion_rate": float(y.mean()),
            "train_rows": len(train_indices),
            "validation_rows": len(validation_indices),
            "test_rows": len(test_indices),
        },
        "data_quality": quality.to_dict(),
        "cost_assumptions": {
            "false_positive": false_positive_cost,
            "false_negative": false_negative_cost,
        },
        "evaluation_contract": {
            "split": "stratified_row_random_60_20_20",
            "selection": "threshold selected on validation, test evaluated once",
            "primary_imbalanced_metric": "average_precision",
            "ranking_budget_fraction": 0.05,
            "known_gap": "no timestamp or stable user identifier for temporal/group holdout",
        },
        "scenarios": {},
    }

    for scenario, features in SCENARIO_FEATURES.items():
        print(f"\nScenario: {scenario} ({', '.join(features)})")
        scenario_frame = prepare_feature_frame(data, scenario)
        runs: dict[str, object] = {}
        curves: list[tuple[str, np.ndarray, np.ndarray]] = []
        for model_name in model_names:
            print(f"  Training {model_name}...")
            pipeline = build_model_pipeline(features, model_name)
            pipeline.fit(scenario_frame.iloc[train_indices], y.iloc[train_indices])

            validation_probabilities = pipeline.predict_proba(
                scenario_frame.iloc[validation_indices]
            )[:, 1]
            threshold = select_cost_sensitive_threshold(
                y.iloc[validation_indices],
                validation_probabilities,
                false_positive_cost=false_positive_cost,
                false_negative_cost=false_negative_cost,
            )
            test_probabilities = pipeline.predict_proba(
                scenario_frame.iloc[test_indices]
            )[:, 1]
            validation_metrics = evaluate_probabilities(
                y.iloc[validation_indices],
                validation_probabilities,
                threshold.threshold,
                false_positive_cost=false_positive_cost,
                false_negative_cost=false_negative_cost,
            )
            test_metrics = evaluate_probabilities(
                y.iloc[test_indices],
                test_probabilities,
                threshold.threshold,
                false_positive_cost=false_positive_cost,
                false_negative_cost=false_negative_cost,
            )
            runs[model_name] = {
                "validation": validation_metrics,
                "test": test_metrics,
                "threshold_selection": threshold.to_dict(),
            }
            curves.append((model_name, y.iloc[test_indices].to_numpy(), test_probabilities))
            print(
                "    test ROC-AUC={roc_auc:.4f}, AP={average_precision:.4f}, "
                "precision={precision:.3f}, recall={recall:.3f}, threshold={threshold:.3f}".format(
                    **test_metrics
                )
            )

        champion = min(
            runs,
            key=lambda name: runs[name]["validation"]["expected_cost_per_1000"],  # type: ignore[index]
        )
        report["scenarios"][scenario] = {  # type: ignore[index]
            "features": list(features),
            "prediction_contract": SCENARIO_CONTRACTS[scenario].to_dict(),
            "champion_by_validation_cost": champion,
            "models": runs,
        }
        plot_model_curves(scenario, curves, output_dir)

    plot_descriptive_rates(data, output_dir)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote metrics and figures to {output_dir}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train decision-aware conversion models")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/generated"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(MODEL_NAMES),
    )
    parser.add_argument("--false-positive-cost", type=float, default=1.0)
    parser.add_argument("--false-negative-cost", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(
        args.data,
        args.output,
        model_names=tuple(args.models),
        false_positive_cost=args.false_positive_cost,
        false_negative_cost=args.false_negative_cost,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
