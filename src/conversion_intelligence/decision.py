from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .data import load_conversion_data
from .features import SCENARIO_CONTRACTS, SCENARIO_FEATURES, prepare_feature_frame
from .modeling import (
    build_model_pipeline,
    evaluate_probabilities,
    select_cost_sensitive_threshold,
)
from .train import split_indices

DEFAULT_SEEDS = (7, 21, 42, 84, 2026)
DEFAULT_FALSE_NEGATIVE_COSTS = (1.0, 2.0, 4.0, 10.0, 20.0)
DEFAULT_CAPACITY_FRACTIONS = (0.05, 0.10)
PRIMARY_FALSE_NEGATIVE_COST = 4.0


def _runtime_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scipy": version("scipy"),
        "scikit_learn": version("scikit-learn"),
        "joblib": version("joblib"),
        "threadpoolctl": version("threadpoolctl"),
    }


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _policy_name(capacity: float | None) -> str:
    if capacity is None:
        return "unconstrained"
    return f"validation_selection_ceiling_{capacity:.3f}"


def _cost_key(false_negative_cost: float) -> str:
    return f"{false_negative_cost:g}"


def _compare_to_reference(
    selected: dict[str, float | int | str],
    reference: dict[str, float | int | str],
    *,
    reference_threshold: float,
) -> dict[str, float | int]:
    selected_cost = float(selected["expected_cost_per_1000"])
    reference_cost = float(reference["expected_cost_per_1000"])
    savings = reference_cost - selected_cost
    relative_savings = savings / reference_cost if reference_cost else 0.0
    return {
        "reference_threshold": float(reference_threshold),
        "reference_cost_per_1000": reference_cost,
        "reference_precision": float(reference["precision"]),
        "reference_recall": float(reference["recall"]),
        "reference_f1": float(reference["f1"]),
        "reference_selection_rate": float(reference["selection_rate"]),
        "selected_cost_per_1000": selected_cost,
        "absolute_cost_reduction_per_1000": savings,
        "relative_cost_reduction": relative_savings,
        "false_negatives_avoided": int(reference["false_negatives"])
        - int(selected["false_negatives"]),
        "additional_false_positives": int(selected["false_positives"])
        - int(reference["false_positives"]),
    }


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    fields = {
        "threshold": lambda run: run["threshold_selection"]["threshold"],
        "validation_selection_rate": lambda run: run["threshold_selection"][
            "selection_rate"
        ],
        "test_selection_rate": lambda run: run["test_metrics"]["selection_rate"],
        "test_precision": lambda run: run["test_metrics"]["precision"],
        "test_recall": lambda run: run["test_metrics"]["recall"],
        "test_f1": lambda run: run["test_metrics"]["f1"],
        "test_brier_score": lambda run: run["test_metrics"]["brier_score"],
        "test_calibration_error": lambda run: run["test_metrics"][
            "expected_calibration_error"
        ],
        "fixed_0_50_precision": lambda run: run["comparisons"]["vs_fixed_0_50"][
            "reference_precision"
        ],
        "fixed_0_50_recall": lambda run: run["comparisons"]["vs_fixed_0_50"][
            "reference_recall"
        ],
        "fixed_0_50_selection_rate": lambda run: run["comparisons"][
            "vs_fixed_0_50"
        ]["reference_selection_rate"],
        "fixed_0_50_cost_per_1000": lambda run: run["comparisons"][
            "vs_fixed_0_50"
        ]["reference_cost_per_1000"],
        "theoretical_threshold": lambda run: run["comparisons"][
            "vs_theoretical_cost_threshold"
        ]["reference_threshold"],
        "theoretical_precision": lambda run: run["comparisons"][
            "vs_theoretical_cost_threshold"
        ]["reference_precision"],
        "theoretical_recall": lambda run: run["comparisons"][
            "vs_theoretical_cost_threshold"
        ]["reference_recall"],
        "theoretical_selection_rate": lambda run: run["comparisons"][
            "vs_theoretical_cost_threshold"
        ]["reference_selection_rate"],
        "theoretical_cost_per_1000": lambda run: run["comparisons"][
            "vs_theoretical_cost_threshold"
        ]["reference_cost_per_1000"],
        "selected_cost_per_1000": lambda run: run["comparisons"][
            "vs_fixed_0_50"
        ]["selected_cost_per_1000"],
        "cost_reduction_vs_fixed_0_50": lambda run: run["comparisons"][
            "vs_fixed_0_50"
        ]["relative_cost_reduction"],
        "cost_reduction_vs_theoretical": lambda run: run["comparisons"][
            "vs_theoretical_cost_threshold"
        ]["relative_cost_reduction"],
        "absolute_reduction_vs_fixed_0_50": lambda run: run["comparisons"][
            "vs_fixed_0_50"
        ]["absolute_cost_reduction_per_1000"],
        "absolute_reduction_vs_theoretical": lambda run: run["comparisons"][
            "vs_theoretical_cost_threshold"
        ]["absolute_cost_reduction_per_1000"],
        "false_negatives_avoided_vs_fixed_0_50": lambda run: run["comparisons"][
            "vs_fixed_0_50"
        ]["false_negatives_avoided"],
        "additional_false_positives_vs_fixed_0_50": lambda run: run["comparisons"][
            "vs_fixed_0_50"
        ]["additional_false_positives"],
        "false_negatives_avoided_vs_theoretical": lambda run: run["comparisons"][
            "vs_theoretical_cost_threshold"
        ]["false_negatives_avoided"],
        "additional_false_positives_vs_theoretical": lambda run: run["comparisons"][
            "vs_theoretical_cost_threshold"
        ]["additional_false_positives"],
    }
    return {
        name: _summary([float(getter(run)) for run in runs])
        for name, getter in fields.items()
    }


def acquisition_profile_split_indices(
    data: pd.DataFrame,
    labels: pd.Series,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create nominal 60/20/20 splits without acquisition-profile overlap.

    The same split is used for both feature scenarios so the retrospective model cannot gain a
    more favorable holdout definition. This is a stress test: the source has no user/session ID,
    so an acquisition-feature profile is only a conservative proxy for an entity group.
    """
    features = list(SCENARIO_FEATURES["acquisition"])
    groups = pd.util.hash_pandas_object(data[features], index=False).to_numpy()
    placeholders = np.zeros(len(labels), dtype=np.int8)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_validation, test = next(outer.split(placeholders, labels, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    train_relative, validation_relative = next(
        inner.split(
            placeholders[train_validation],
            labels.iloc[train_validation],
            groups[train_validation],
        )
    )
    train = train_validation[train_relative]
    validation = train_validation[validation_relative]
    return np.asarray(train), np.asarray(validation), np.asarray(test)


def _build_policies(
    probability_cache: dict[int, dict[str, Any]],
    *,
    seeds: tuple[int, ...],
    false_negative_costs: tuple[float, ...],
    capacity_fractions: tuple[float, ...],
) -> dict[str, Any]:
    policy_capacities: tuple[float | None, ...] = (None, *capacity_fractions)
    policies: dict[str, Any] = {}
    for capacity in policy_capacities:
        ratios: dict[str, Any] = {}
        for false_negative_cost in false_negative_costs:
            theoretical_threshold = float(1.0 / (1.0 + false_negative_cost))
            runs: list[dict[str, Any]] = []
            for seed in seeds:
                cached = probability_cache[seed]
                choice = select_cost_sensitive_threshold(
                    cached["validation_labels"],
                    cached["validation_probabilities"],
                    false_positive_cost=1.0,
                    false_negative_cost=false_negative_cost,
                    max_selection_fraction=capacity,
                )
                selected = evaluate_probabilities(
                    cached["test_labels"],
                    cached["test_probabilities"],
                    choice.threshold,
                    false_positive_cost=1.0,
                    false_negative_cost=false_negative_cost,
                )
                fixed_0_50 = evaluate_probabilities(
                    cached["test_labels"],
                    cached["test_probabilities"],
                    0.50,
                    false_positive_cost=1.0,
                    false_negative_cost=false_negative_cost,
                )
                theoretical = evaluate_probabilities(
                    cached["test_labels"],
                    cached["test_probabilities"],
                    theoretical_threshold,
                    false_positive_cost=1.0,
                    false_negative_cost=false_negative_cost,
                )
                selected_ceiling_exceeded = bool(
                    capacity is not None
                    and float(selected["selection_rate"]) > capacity
                )
                fixed_0_50_ceiling_exceeded = bool(
                    capacity is not None
                    and float(fixed_0_50["selection_rate"]) > capacity
                )
                theoretical_ceiling_exceeded = bool(
                    capacity is not None
                    and float(theoretical["selection_rate"]) > capacity
                )
                runs.append(
                    {
                        "seed": seed,
                        "threshold_selection": choice.to_dict(),
                        "test_metrics": selected,
                        "references": {
                            "fixed_0_50": fixed_0_50,
                            "theoretical_cost_threshold": theoretical,
                        },
                        "comparisons": {
                            "vs_fixed_0_50": _compare_to_reference(
                                selected,
                                fixed_0_50,
                                reference_threshold=0.50,
                            ),
                            "vs_theoretical_cost_threshold": _compare_to_reference(
                                selected,
                                theoretical,
                                reference_threshold=theoretical_threshold,
                            ),
                        },
                        "test_selection_ceiling_exceeded": {
                            "validation_selected": selected_ceiling_exceeded,
                            "fixed_0_50": fixed_0_50_ceiling_exceeded,
                            "theoretical_cost_threshold": theoretical_ceiling_exceeded,
                        },
                        "selection_ceiling_comparison_feasible": {
                            "vs_fixed_0_50": not (
                                selected_ceiling_exceeded
                                or fixed_0_50_ceiling_exceeded
                            ),
                            "vs_theoretical_cost_threshold": not (
                                selected_ceiling_exceeded
                                or theoretical_ceiling_exceeded
                            ),
                        },
                    }
                )
            ratios[_cost_key(false_negative_cost)] = {
                "false_positive_cost": 1.0,
                "false_negative_cost": false_negative_cost,
                "theoretical_cost_threshold_if_calibrated": theoretical_threshold,
                "validation_selection_ceiling": capacity,
                "runs": runs,
                "summary": _summarize_runs(runs),
                "selection_ceiling_diagnostics": {
                    "validation_selected_exceeded_runs": sum(
                        int(
                            run["test_selection_ceiling_exceeded"][
                                "validation_selected"
                            ]
                        )
                        for run in runs
                    ),
                    "fixed_0_50_exceeded_runs": sum(
                        int(run["test_selection_ceiling_exceeded"]["fixed_0_50"])
                        for run in runs
                    ),
                    "theoretical_cost_threshold_exceeded_runs": sum(
                        int(
                            run["test_selection_ceiling_exceeded"][
                                "theoretical_cost_threshold"
                            ]
                        )
                        for run in runs
                    ),
                    "total_runs": len(runs),
                    "vs_fixed_0_50_all_feasible": all(
                        bool(
                            run["selection_ceiling_comparison_feasible"][
                                "vs_fixed_0_50"
                            ]
                        )
                        for run in runs
                    ),
                    "vs_theoretical_cost_threshold_all_feasible": all(
                        bool(
                            run["selection_ceiling_comparison_feasible"][
                                "vs_theoretical_cost_threshold"
                            ]
                        )
                        for run in runs
                    ),
                },
            }
        policies[_policy_name(capacity)] = ratios
    return policies


def _render_markdown(report: dict[str, Any]) -> str:
    primary_cost = float(report["evaluation_contract"]["primary_false_negative_cost"])
    primary_key = _cost_key(primary_cost)
    bayes_threshold = 1.0 / (1.0 + primary_cost)
    lines = [
        "# Decision optimization sensitivity report",
        "",
        "Thresholds are selected on validation data and evaluated on the held-out test partition "
        "within each split. Costs are unitless scenario assumptions, not measured dollars or "
        "causal uplift.",
        "",
        f"The primary reporting scenario assigns a false negative {primary_cost:g} times the "
        "loss of a false positive. For calibrated probabilities and no capacity constraint, that "
        f"cost ratio implies a theoretical Bayes threshold of 1 / (1 + {primary_cost:g}) = "
        f"{bayes_threshold:.3f}; the empirical threshold is still selected independently on each "
        "validation split.",
        "Seed ranges below are split-sensitivity ranges, not confidence intervals; test rows can "
        "overlap across seeds.",
        "",
    ]
    for scenario, details in report["scenarios"].items():
        contract = details["prediction_contract"]
        scenario_name = (
            "Retrospective Full Session"
            if scenario == "in_session"
            else scenario.replace("_", " ").title()
        )
        lines.extend(
            [
                f"## {scenario_name}",
                "",
                f"Decision time: `{contract['decision_time']}`. Deployment status: "
                f"`{contract['deployment_status']}`.",
                "",
            ]
        )
        lines.extend(
            [
                f"### Primary {primary_cost:g}:1 decision scenario",
                "",
                "| Holdout | Mean validation-selected threshold | Fixed 0.50 loss | "
                f"Theoretical {bayes_threshold:.2f} loss | Selected loss | vs 0.50 | "
                f"vs {bayes_threshold:.2f} | Selected P / R |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for split_name, split_details in details["split_strategies"].items():
            split_title = (
                "Acquisition-profile grouped"
                if split_name == "acquisition_profile_grouped"
                else "Row-random sensitivity"
            )
            primary = split_details["policies"]["unconstrained"][primary_key]["summary"]
            lines.append(
                "| {split} | {threshold:.3f} ({threshold_min:.3f}-{threshold_max:.3f}) | "
                "{fixed_cost:.1f} | {theoretical_cost:.1f} | {selected_cost:.1f} | "
                "{reduction_fixed:.1%} | {reduction_theoretical:.1%} | "
                "{precision:.3f} / {recall:.3f} |".format(
                    split=split_title,
                    threshold=primary["threshold"]["mean"],
                    threshold_min=primary["threshold"]["minimum"],
                    threshold_max=primary["threshold"]["maximum"],
                    fixed_cost=primary["fixed_0_50_cost_per_1000"]["mean"],
                    theoretical_cost=primary["theoretical_cost_per_1000"]["mean"],
                    selected_cost=primary["selected_cost_per_1000"]["mean"],
                    reduction_fixed=primary["cost_reduction_vs_fixed_0_50"]["mean"],
                    reduction_theoretical=primary[
                        "cost_reduction_vs_theoretical"
                    ]["mean"],
                    precision=primary["test_precision"]["mean"],
                    recall=primary["test_recall"]["mean"],
                )
            )
        lines.append("")
        for split_name, split_details in details["split_strategies"].items():
            split_title = (
                "Acquisition-profile grouped holdout"
                if split_name == "acquisition_profile_grouped"
                else "Row-random holdout sensitivity"
            )
            lines.extend(
                [
                    f"### {split_title}",
                    "",
                    "| FN:FP ratio | Theoretical threshold | Validation-selected threshold | "
                    "Loss reduction vs 0.50 | Loss reduction vs theoretical | P / R | "
                    "Selection rate |",
                    "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            policies = split_details["policies"]["unconstrained"]
            for ratio, result in policies.items():
                summary = result["summary"]
                lines.append(
                    "| {ratio}:1 | {theoretical:.3f} | {threshold:.3f} | "
                    "{reduction_fixed:.1%} | {reduction_theoretical:.1%} | "
                    "{precision:.3f} / {recall:.3f} | {selection:.1%} |".format(
                        ratio=ratio,
                        theoretical=summary["theoretical_threshold"]["mean"],
                        threshold=summary["threshold"]["mean"],
                        reduction_fixed=summary[
                            "cost_reduction_vs_fixed_0_50"
                        ]["mean"],
                        reduction_theoretical=summary[
                            "cost_reduction_vs_theoretical"
                        ]["mean"],
                        precision=summary["test_precision"]["mean"],
                        recall=summary["test_recall"]["mean"],
                        selection=summary["test_selection_rate"]["mean"],
                    )
                )
            lines.append("")
            capacity_policies = [
                policies
                for name, policies in split_details["policies"].items()
                if name != "unconstrained"
            ]
            if capacity_policies:
                lines.extend(
                    [
                        "Validation-selection-ceiling sensitivity. This is not a hard batch "
                        "capacity: a frozen threshold can exceed the ceiling on test data under "
                        "score shift, and test labels are never used to retune. A comparison is "
                        "shown as N/A if either policy exceeds the test ceiling.",
                        "",
                        "| Validation ceiling | FN:FP ratio | Selected threshold | Validation / "
                        "test selection | Loss vs 0.50 | Loss vs theoretical | Test ceiling "
                        "violations (selected/0.50/theoretical) |",
                        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                    ]
                )
                for capacity_policy in capacity_policies:
                    capacity = capacity_policy[next(iter(capacity_policy))][
                        "validation_selection_ceiling"
                    ]
                    for ratio, result in capacity_policy.items():
                        summary = result["summary"]
                        diagnostics = result["selection_ceiling_diagnostics"]
                        reduction_fixed = (
                            f"{summary['cost_reduction_vs_fixed_0_50']['mean']:.1%}"
                            if diagnostics["vs_fixed_0_50_all_feasible"]
                            else "N/A*"
                        )
                        reduction_theoretical = (
                            f"{summary['cost_reduction_vs_theoretical']['mean']:.1%}"
                            if diagnostics[
                                "vs_theoretical_cost_threshold_all_feasible"
                            ]
                            else "N/A*"
                        )
                        lines.append(
                            "| {capacity:.1%} | {ratio}:1 | {threshold:.3f} | "
                            "{validation:.1%} / {test:.1%} | {reduction_fixed} | "
                            "{reduction_theoretical} | {selected_violations}/"
                            "{fixed_violations}/{theoretical_violations} |".format(
                                capacity=capacity,
                                ratio=ratio,
                                threshold=summary["threshold"]["mean"],
                                validation=summary["validation_selection_rate"]["mean"],
                                test=summary["test_selection_rate"]["mean"],
                                reduction_fixed=reduction_fixed,
                                reduction_theoretical=reduction_theoretical,
                                selected_violations=diagnostics[
                                    "validation_selected_exceeded_runs"
                                ],
                                fixed_violations=diagnostics[
                                    "fixed_0_50_exceeded_runs"
                                ],
                                theoretical_violations=diagnostics[
                                    "theoretical_cost_threshold_exceeded_runs"
                                ],
                            )
                        )
                lines.append("")
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "- The acquisition policy is an offline candidate for prospective validation, not a "
            "production result.",
            "- The full-session policy uses final cumulative page views and is a retrospective "
            "upper bound, not a deployable late-session policy.",
            "- The observational labels do not identify intervention lift, ROI, or treatment "
            "effects.",
            "- A real deployment must replace scenario costs and the validation selection ceiling "
            "with measured business inputs, use a separate top-K policy for a hard batch capacity, "
            "and revalidate over time.",
            "",
        ]
    )
    return "\n".join(lines)


def build_public_benchmark(
    report: dict[str, Any],
    *,
    scenario: str = "in_session",
    split_strategy: str = "acquisition_profile_grouped",
    validation_selection_ceiling: float = 0.05,
) -> dict[str, Any]:
    """Extract a reviewed aggregate-only benchmark from the full local report."""
    primary_cost = float(report["evaluation_contract"]["primary_false_negative_cost"])
    primary_key = _cost_key(primary_cost)
    split = report["scenarios"][scenario]["split_strategies"][split_strategy]
    unconstrained = split["policies"]["unconstrained"][primary_key]
    ceiling_key = _policy_name(validation_selection_ceiling)
    try:
        ceiling = split["policies"][ceiling_key][primary_key]
    except KeyError as exc:
        raise ValueError(
            "The requested validation selection ceiling is absent from the report"
        ) from exc
    summary = unconstrained["summary"]
    ceiling_summary = ceiling["summary"]
    ceiling_diagnostics = ceiling["selection_ceiling_diagnostics"]
    contract = report["evaluation_contract"]
    return {
        "schema_version": "2.0",
        "source_rows_after_validation": int(report["dataset"]["rows"]),
        "source_data_included": False,
        "benchmark_environment": report["environment"],
        "model": contract["model"],
        "feature_scenario": scenario,
        "prediction_contract": report["scenarios"][scenario]["prediction_contract"],
        "evaluation_contract": {
            "split": split_strategy,
            "seeds": list(contract["seeds"]),
            "model_fit": "train_only",
            "threshold_selection": "validation_only_exact_score_boundaries",
            "test_use": (
                "evaluate frozen validation-selected and reference policies on each held-out "
                "test partition"
            ),
            "cross_seed_interpretation": contract["cross_seed_interpretation"],
            "reference_thresholds": [
                "fixed_0.50",
                "C_FP/(C_FP+C_FN)_if_probabilities_are_calibrated",
            ],
        },
        "primary_sensitivity_scenario": {
            "false_positive_loss": 1.0,
            "false_negative_loss": primary_cost,
            "loss_units": contract["cost_units"],
            "theoretical_cost_threshold_if_calibrated": float(
                summary["theoretical_threshold"]["mean"]
            ),
        },
        "five_seed_summary": {
            "validation_selected_threshold": summary["threshold"],
            "validation_selected_precision": summary["test_precision"],
            "validation_selected_recall": summary["test_recall"],
            "validation_selected_selection_rate": summary["test_selection_rate"],
            "fixed_0_50_modeled_loss_per_1000": summary[
                "fixed_0_50_cost_per_1000"
            ],
            "theoretical_threshold_modeled_loss_per_1000": summary[
                "theoretical_cost_per_1000"
            ],
            "validation_selected_modeled_loss_per_1000": summary[
                "selected_cost_per_1000"
            ],
            "relative_modeled_loss_reduction_vs_fixed_0_50": summary[
                "cost_reduction_vs_fixed_0_50"
            ],
            "relative_modeled_loss_reduction_vs_theoretical_threshold": summary[
                "cost_reduction_vs_theoretical"
            ],
        },
        "validation_selection_ceiling_sensitivity": {
            "ceiling": validation_selection_ceiling,
            "meaning": (
                "applied during validation threshold selection; not a hard top-K test capacity"
            ),
            "validation_selected_threshold": ceiling_summary["threshold"],
            "validation_selection_rate": ceiling_summary[
                "validation_selection_rate"
            ],
            "test_selection_rate": ceiling_summary["test_selection_rate"],
            "relative_modeled_loss_reduction_vs_fixed_0_50": (
                ceiling_summary["cost_reduction_vs_fixed_0_50"]
                if ceiling_diagnostics["vs_fixed_0_50_all_feasible"]
                else None
            ),
            "relative_modeled_loss_reduction_vs_theoretical_threshold": (
                ceiling_summary["cost_reduction_vs_theoretical"]
                if ceiling_diagnostics[
                    "vs_theoretical_cost_threshold_all_feasible"
                ]
                else None
            ),
            "raw_descriptive_relative_change_vs_fixed_0_50": ceiling_summary[
                "cost_reduction_vs_fixed_0_50"
            ],
            "raw_descriptive_relative_change_vs_theoretical_threshold": ceiling_summary[
                "cost_reduction_vs_theoretical"
            ],
            "test_ceiling_diagnostics": ceiling_diagnostics,
        },
        "interpretation_limits": [
            "The loss ratio is an explicit sensitivity assumption, not a measured business cost.",
            "The result is modeled classification loss, not ROI, savings, or causal uplift.",
            "The theoretical threshold is optimal only for calibrated probabilities and the "
            "stated cost matrix.",
            "A ceiling comparison is null when either policy exceeds the test ceiling; raw "
            "descriptive changes are retained separately and are not capacity-feasible claims.",
            "Final cumulative page views are unavailable at a verified pre-conversion decision "
            "time.",
            "The grouped holdout uses acquisition-feature profiles because no stable user or "
            "session ID exists.",
        ],
    }


def run_decision_optimization(
    data_path: Path,
    output_path: Path,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    false_negative_costs: tuple[float, ...] = DEFAULT_FALSE_NEGATIVE_COSTS,
    capacity_fractions: tuple[float, ...] = DEFAULT_CAPACITY_FRACTIONS,
    benchmark_output_path: Path | None = None,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("At least one split seed is required")
    if not false_negative_costs or any(
        not np.isfinite(cost) or cost <= 0 for cost in false_negative_costs
    ):
        raise ValueError("false_negative_costs must contain finite positive values")
    if any(
        not np.isfinite(fraction) or not 0 < fraction <= 1
        for fraction in capacity_fractions
    ):
        raise ValueError("capacity_fractions must be in (0, 1]")
    cost_keys = [_cost_key(cost) for cost in false_negative_costs]
    if len(set(cost_keys)) != len(cost_keys):
        raise ValueError("false_negative_costs collide after report-key formatting")
    capacity_keys = [_policy_name(fraction) for fraction in capacity_fractions]
    if len(set(capacity_keys)) != len(capacity_keys):
        raise ValueError("capacity_fractions collide after report-key formatting")

    data, quality = load_conversion_data(data_path)
    labels = data["converted"]
    primary_match = next(
        (
            cost
            for cost in false_negative_costs
            if np.isclose(cost, PRIMARY_FALSE_NEGATIVE_COST)
        ),
        None,
    )
    primary_cost = false_negative_costs[0] if primary_match is None else primary_match
    report: dict[str, Any] = {
        "schema_version": "2.0",
        "environment": _runtime_environment(),
        "dataset": {
            "rows": len(data),
            "conversion_rate": float(labels.mean()),
            "source_data_included": False,
        },
        "data_quality": quality.to_dict(),
        "evaluation_contract": {
            "split_strategies": [
                "acquisition_profile_grouped_outer5_inner4_nominal_60_20_20",
                "stratified_row_random_60_20_20_sensitivity",
            ],
            "seeds": list(seeds),
            "model": "logistic_regression",
            "selection": (
                "model fit on train; each explicitly enumerated policy threshold selected on "
                "validation and evaluated once on the held-out partition within each split; no "
                "policy selected by test metrics"
            ),
            "reference_policies": [
                "fixed_0.50_threshold",
                "theoretical_cost_threshold_C_FP_over_C_FP_plus_C_FN",
            ],
            "decision_definition": (
                "binary classification/flagging policy; not an intervention-allocation or "
                "incremental-response policy"
            ),
            "cost_matrix": {
                "true_negative": 0,
                "false_positive": 1,
                "false_negative": "varies_by_policy",
                "true_positive": 0,
            },
            "cost_units": "illustrative unit loss, not measured dollars",
            "primary_false_negative_cost": primary_cost,
            "primary_cost_rationale": (
                "explicit sensitivity scenario; the Bayes threshold equals C_FP / "
                "(C_FP + C_FN) only for calibrated probabilities without a capacity constraint"
            ),
            "cross_seed_interpretation": (
                "mean/min/max are split sensitivity, not confidence intervals; test rows can "
                "overlap across seeds"
            ),
            "validation_selection_ceiling_contract": (
                "keep equal-score rows together and allow unused validation capacity; a frozen "
                "threshold can exceed the ceiling on test data, so this is not a hard top-K batch "
                "capacity"
            ),
        },
        "scenarios": {},
    }

    splitters = {
        "acquisition_profile_grouped": lambda seed: acquisition_profile_split_indices(
            data,
            labels,
            seed=seed,
        ),
        "row_random": lambda seed: split_indices(labels, seed),
    }
    for scenario, features in SCENARIO_FEATURES.items():
        frame = prepare_feature_frame(data, scenario)
        split_strategies: dict[str, Any] = {}
        for split_name, splitter in splitters.items():
            probability_cache: dict[int, dict[str, Any]] = {}
            split_rows: list[dict[str, int | float]] = []
            for seed in seeds:
                train, validation, test = splitter(seed)
                model = build_model_pipeline(features, "logistic_regression")
                model.fit(frame.iloc[train], labels.iloc[train])
                probability_cache[seed] = {
                    "validation_labels": labels.iloc[validation],
                    "validation_probabilities": model.predict_proba(
                        frame.iloc[validation]
                    )[:, 1],
                    "test_labels": labels.iloc[test],
                    "test_probabilities": model.predict_proba(frame.iloc[test])[:, 1],
                }
                split_rows.append(
                    {
                        "seed": seed,
                        "train_rows": len(train),
                        "validation_rows": len(validation),
                        "test_rows": len(test),
                        "validation_prevalence": float(labels.iloc[validation].mean()),
                        "test_prevalence": float(labels.iloc[test].mean()),
                    }
                )
            split_strategies[split_name] = {
                "splits": split_rows,
                "policies": _build_policies(
                    probability_cache,
                    seeds=seeds,
                    false_negative_costs=false_negative_costs,
                    capacity_fractions=capacity_fractions,
                ),
            }
        report["scenarios"][scenario] = {
            "features": list(features),
            "prediction_contract": SCENARIO_CONTRACTS[scenario].to_dict(),
            "split_strategies": split_strategies,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    outputs = [output_path, markdown_path]
    if benchmark_output_path is not None:
        benchmark_output_path.parent.mkdir(parents=True, exist_ok=True)
        benchmark_output_path.write_text(
            json.dumps(build_public_benchmark(report), indent=2),
            encoding="utf-8",
        )
        outputs.append(benchmark_output_path)
    print("Wrote decision outputs to " + ", ".join(str(path) for path in outputs))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize and stress-test conversion decision policies"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/generated/decision-optimization.json"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--false-negative-costs",
        type=float,
        nargs="+",
        default=list(DEFAULT_FALSE_NEGATIVE_COSTS),
    )
    parser.add_argument(
        "--validation-selection-ceilings",
        "--capacity-fractions",
        dest="capacity_fractions",
        type=float,
        nargs="*",
        default=list(DEFAULT_CAPACITY_FRACTIONS),
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        help="Optional aggregate-only benchmark JSON extracted from the full local report",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_decision_optimization(
        args.data,
        args.output,
        seeds=tuple(args.seeds),
        false_negative_costs=tuple(args.false_negative_costs),
        capacity_fractions=tuple(args.capacity_fractions),
        benchmark_output_path=args.benchmark_output,
    )


if __name__ == "__main__":
    main()
