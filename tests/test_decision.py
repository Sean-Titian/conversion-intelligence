import json
import re
from pathlib import Path

import numpy as np

from conversion_intelligence.decision import (
    _build_policies,
    acquisition_profile_split_indices,
    build_public_benchmark,
    run_decision_optimization,
)
from conversion_intelligence.features import SCENARIO_FEATURES
from conversion_intelligence.synthetic import generate_synthetic_conversion_data


def test_grouped_decision_split_has_no_acquisition_profile_overlap() -> None:
    data = generate_synthetic_conversion_data(rows=3_000, seed=9, target_rate=0.05)
    train, validation, test = acquisition_profile_split_indices(
        data,
        data["converted"],
        seed=7,
    )
    features = list(SCENARIO_FEATURES["acquisition"])

    def profiles(indices):
        return set(data.iloc[indices][features].itertuples(index=False, name=None))

    assert profiles(train).isdisjoint(profiles(validation))
    assert profiles(train).isdisjoint(profiles(test))
    assert profiles(validation).isdisjoint(profiles(test))

    # This golden contract intentionally catches dependency-driven splitter drift. The exact
    # benchmark environment is recorded and pinned separately.
    assert (len(train), len(validation), len(test)) == (1_812, 651, 537)
    assert np.isclose(data.iloc[train]["converted"].mean(), 0.04415011037527594)
    assert np.isclose(data.iloc[validation]["converted"].mean(), 0.059907834101382486)
    assert np.isclose(data.iloc[test]["converted"].mean(), 0.055865921787709494)


def test_decision_report_freezes_validation_policy_before_test(tmp_path) -> None:
    data_path = tmp_path / "synthetic.csv"
    output_path = tmp_path / "decision.json"
    benchmark_path = tmp_path / "benchmark.json"
    generate_synthetic_conversion_data(rows=3_000, seed=11, target_rate=0.05).to_csv(
        data_path,
        index=False,
    )

    report = run_decision_optimization(
        data_path,
        output_path,
        seeds=(7,),
        false_negative_costs=(3.0,),
        capacity_fractions=(0.05,),
        benchmark_output_path=benchmark_path,
    )

    assert output_path.exists()
    assert output_path.with_suffix(".md").exists()
    assert benchmark_path.exists()
    serialized = json.loads(output_path.read_text(encoding="utf-8"))
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    assert serialized["schema_version"] == "2.0"
    assert serialized["environment"]["scikit_learn"]
    assert serialized["evaluation_contract"]["selection"].startswith("model fit on train")
    assert serialized["evaluation_contract"]["primary_false_negative_cost"] == 3.0
    assert benchmark == build_public_benchmark(report)
    assert benchmark["source_data_included"] is False
    assert benchmark["benchmark_environment"] == serialized["environment"]
    markdown = output_path.with_suffix(".md").read_text(encoding="utf-8")
    assert "Primary 3:1 decision scenario" in markdown
    assert "not measured dollars or causal uplift" in markdown
    assert "Loss reduction vs theoretical" in markdown
    for scenario in ("acquisition", "in_session"):
        strategies = report["scenarios"][scenario]["split_strategies"]
        for strategy in ("acquisition_profile_grouped", "row_random"):
            unconstrained = strategies[strategy]["policies"]["unconstrained"]["3"]
            run = unconstrained["runs"][0]
            assert run["threshold_selection"]["false_negative_cost"] == 3.0
            assert unconstrained["theoretical_cost_threshold_if_calibrated"] == 0.25
            assert (
                run["comparisons"]["vs_theoretical_cost_threshold"][
                    "reference_threshold"
                ]
                == 0.25
            )
            assert "relative_cost_reduction" in run["comparisons"]["vs_fixed_0_50"]
            capacity = strategies[strategy]["policies"][
                "validation_selection_ceiling_0.050"
            ]["3"]
            assert capacity["validation_selection_ceiling"] == 0.05
            assert (
                "validation_selected_exceeded_runs"
                in capacity["selection_ceiling_diagnostics"]
            )


def test_test_data_cannot_change_validation_selected_threshold() -> None:
    validation = {
        "validation_labels": np.array([0, 0, 1, 1]),
        "validation_probabilities": np.array([0.1, 0.3, 0.4, 0.8]),
    }
    cache_a = {
        7: {
            **validation,
            "test_labels": np.array([0, 0, 1, 1]),
            "test_probabilities": np.array([0.1, 0.2, 0.7, 0.9]),
        }
    }
    cache_b = {
        7: {
            **validation,
            "test_labels": np.array([1, 1, 0, 0]),
            "test_probabilities": np.array([0.9, 0.7, 0.2, 0.1]),
        }
    }
    policies_a = _build_policies(
        cache_a,
        seeds=(7,),
        false_negative_costs=(4.0,),
        capacity_fractions=(),
    )
    policies_b = _build_policies(
        cache_b,
        seeds=(7,),
        false_negative_costs=(4.0,),
        capacity_fractions=(),
    )
    threshold_a = policies_a["unconstrained"]["4"]["runs"][0][
        "threshold_selection"
    ]
    threshold_b = policies_b["unconstrained"]["4"]["runs"][0][
        "threshold_selection"
    ]
    assert threshold_a == threshold_b


def test_theoretical_reference_uses_cost_ratio_and_tracks_ceiling_feasibility() -> None:
    cache = {
        7: {
            "validation_labels": np.array([0, 0, 0, 1]),
            "validation_probabilities": np.array([0.9, 0.8, 0.7, 0.6]),
            "test_labels": np.array([0, 0, 1, 1]),
            "test_probabilities": np.array([0.9, 0.8, 0.1, 0.1]),
        }
    }
    policies = _build_policies(
        cache,
        seeds=(7,),
        false_negative_costs=(4.0,),
        capacity_fractions=(0.25,),
    )
    result = policies["validation_selection_ceiling_0.250"]["4"]
    run = result["runs"][0]

    assert result["theoretical_cost_threshold_if_calibrated"] == 0.2
    assert (
        run["references"]["theoretical_cost_threshold"]["selection_rate"]
        == 0.5
    )
    assert not run["test_selection_ceiling_exceeded"]["validation_selected"]
    assert run["test_selection_ceiling_exceeded"]["theoretical_cost_threshold"]
    assert not run["selection_ceiling_comparison_feasible"][
        "vs_theoretical_cost_threshold"
    ]
    assert not result["selection_ceiling_diagnostics"][
        "vs_theoretical_cost_threshold_all_feasible"
    ]


def test_tracked_benchmark_is_public_safe_pinned_and_reflected_in_readme() -> None:
    root = Path(__file__).resolve().parents[1]
    benchmark_path = root / "reports" / "decision-optimization-benchmark.json"
    benchmark_text = benchmark_path.read_text(encoding="utf-8")
    benchmark = json.loads(benchmark_text)

    assert benchmark["schema_version"] == "2.0"
    assert benchmark["source_data_included"] is False
    assert not re.search(r"[A-Za-z]:\\\\", benchmark_text)
    for forbidden_key in ("data_path", "source_path", "records", "identifiers", "text_samples"):
        assert f'"{forbidden_key}"' not in benchmark_text

    pins = {}
    for line in (root / "requirements-benchmark.txt").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            package, pinned_version = line.split("==", maxsplit=1)
            pins[package] = pinned_version
    environment = benchmark["benchmark_environment"]
    assert environment["numpy"] == pins["numpy"]
    assert environment["pandas"] == pins["pandas"]
    assert environment["scipy"] == pins["scipy"]
    assert environment["scikit_learn"] == pins["scikit-learn"]
    assert environment["joblib"] == pins["joblib"]
    assert environment["threadpoolctl"] == pins["threadpoolctl"]

    ceiling = benchmark["validation_selection_ceiling_sensitivity"]
    diagnostics = ceiling["test_ceiling_diagnostics"]
    assert not diagnostics["vs_theoretical_cost_threshold_all_feasible"]
    assert ceiling["relative_modeled_loss_reduction_vs_theoretical_threshold"] is None
    assert isinstance(
        ceiling["raw_descriptive_relative_change_vs_theoretical_threshold"], dict
    )

    summary = benchmark["five_seed_summary"]
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert f"{summary['fixed_0_50_modeled_loss_per_1000']['mean']:.3f}" in readme
    assert (
        f"{summary['theoretical_threshold_modeled_loss_per_1000']['mean']:.3f}"
        in readme
    )
    assert f"{summary['validation_selected_modeled_loss_per_1000']['mean']:.3f}" in readme
    assert (
        f"{summary['relative_modeled_loss_reduction_vs_fixed_0_50']['mean']:.3%}"
        in readme
    )
    assert (
        f"{abs(summary['relative_modeled_loss_reduction_vs_theoretical_threshold']['mean']):.3%}"
        in readme
    )
