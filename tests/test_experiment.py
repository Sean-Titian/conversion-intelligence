from __future__ import annotations

import math
import sys
from fractions import Fraction

import numpy as np
import pytest

from conversion_intelligence.experiment import build_parser, main, two_proportion_sample_size


def _sample_size(**overrides: object) -> int:
    values = {
        "baseline": 0.0323,
        "treatment": 0.0323 * 1.1,
        "alpha": 0.05,
        "power": 0.8,
        "attrition": 0.05,
    }
    values.update(overrides)
    return two_proportion_sample_size(**values)


def _literal_quantile_reference(baseline: float, treatment: float, attrition: float) -> int:
    # Independent literal standard-normal quantiles for two-sided alpha=.05,
    # power=.80; do not import or call the implementation's NormalDist helper.
    z_alpha = 1.959963984540054
    z_power = 0.8416212335729143
    pooled = (baseline + treatment) / 2
    standard_errors = (
        z_alpha * math.sqrt(2 * pooled * (1 - pooled))
        + z_power * math.sqrt(baseline * (1 - baseline) + treatment * (1 - treatment))
    )
    return math.ceil(standard_errors**2 / (treatment - baseline) ** 2 / (1 - attrition))


def test_sample_size_is_positive_and_increases_for_smaller_effect() -> None:
    large_effect = two_proportion_sample_size(0.03, 0.04)
    small_effect = two_proportion_sample_size(0.03, 0.033)
    assert large_effect > 0
    assert small_effect > large_effect


def test_sample_size_rejects_equal_rates() -> None:
    with pytest.raises(ValueError, match="must differ"):
        two_proportion_sample_size(0.03, 0.03)


def test_illustrative_reference_is_unchanged_not_an_observed_baseline() -> None:
    assert _sample_size() == 51_893


@pytest.mark.parametrize("baseline,treatment", [(0.03, 0.04), (0.2, 0.25), (0.7, 0.65)])
@pytest.mark.parametrize("attrition", [0.0, 0.05, 0.2])
def test_formula_matches_independent_literal_quantile_reference(
    baseline: float, treatment: float, attrition: float
) -> None:
    expected = _literal_quantile_reference(baseline, treatment, attrition)
    assert _sample_size(baseline=baseline, treatment=treatment, attrition=attrition) == expected


@pytest.mark.parametrize("baseline,treatment", [(0.03, 0.04), (0.2, 0.25), (0.7, 0.65)])
def test_equal_allocation_formula_is_symmetric_to_rate_swap(
    baseline: float, treatment: float
) -> None:
    assert _sample_size(baseline=baseline, treatment=treatment) == _sample_size(
        baseline=treatment, treatment=baseline
    )


def test_stricter_alpha_higher_power_and_more_attrition_need_more_units() -> None:
    assert _sample_size(alpha=0.01) > _sample_size(alpha=0.05) > _sample_size(alpha=0.1)
    assert _sample_size(power=0.9) > _sample_size(power=0.8) > _sample_size(power=0.6)
    assert _sample_size(attrition=0.2) > _sample_size(attrition=0.05) > _sample_size(attrition=0)


def test_numpy_real_inputs_and_exact_fraction_inputs_are_supported() -> None:
    assert _sample_size(
        baseline=np.float64(0.0323), treatment=np.float64(0.0323 * 1.1),
        alpha=np.float64(0.05), power=np.float64(0.8), attrition=np.float64(0.05),
    ) == 51_893
    assert _sample_size(
        baseline=Fraction(323, 10_000), treatment=Fraction(3553, 100_000),
        alpha=Fraction(1, 20), power=Fraction(4, 5), attrition=Fraction(1, 20),
    ) == 51_893


@pytest.mark.parametrize("field", ["baseline", "treatment", "alpha", "power", "attrition"])
@pytest.mark.parametrize(
    "invalid",
    [True, np.bool_(False), "0.2", None, 1j, float("nan"), float("inf"),
     float("-inf"), [0.2], np.array(0.2), np.array([0.2]), np.ma.masked],
)
def test_all_inputs_reject_non_real_non_finite_or_boolean_values(
    field: str, invalid: object
) -> None:
    with pytest.raises(ValueError):
        _sample_size(**{field: invalid})


@pytest.mark.parametrize("field", ["baseline", "treatment", "alpha", "power"])
@pytest.mark.parametrize("invalid", [0, 1, -0.01, 1.01])
def test_open_interval_parameters_reject_endpoints_and_out_of_range_values(
    field: str, invalid: float
) -> None:
    with pytest.raises(ValueError):
        _sample_size(**{field: invalid})


@pytest.mark.parametrize("invalid", [-0.01, 1, 1.01])
def test_attrition_rejects_values_outside_its_half_open_interval(invalid: float) -> None:
    with pytest.raises(ValueError):
        _sample_size(attrition=invalid)


@pytest.mark.parametrize("field", ["baseline", "treatment", "alpha", "power", "attrition"])
@pytest.mark.parametrize(
    "invalid",
    [Fraction(-1, 10**1000), Fraction(10**1000 + 1, 10**1000)],
)
def test_raw_fraction_bounds_are_checked_before_float_rounding(
    field: str, invalid: Fraction
) -> None:
    with pytest.raises(ValueError):
        _sample_size(**{field: invalid})


@pytest.mark.parametrize("field", ["baseline", "treatment", "alpha", "power"])
def test_positive_open_interval_values_that_round_to_zero_are_rejected(field: str) -> None:
    with pytest.raises(ValueError):
        _sample_size(**{field: Fraction(1, 10**1000)})


@pytest.mark.parametrize("field", ["baseline", "treatment", "alpha", "power", "attrition"])
def test_values_that_round_to_forbidden_one_are_rejected(field: str) -> None:
    with pytest.raises(ValueError):
        _sample_size(**{field: Fraction(10**1000 - 1, 10**1000)})


def test_unrepresentable_quantile_and_underflowing_effect_square_fail_closed() -> None:
    with pytest.raises(ValueError):
        _sample_size(alpha=1e-30)
    with pytest.raises(ValueError):
        _sample_size(baseline=1e-200, treatment=2e-200)


def test_non_positive_normal_root_is_not_squared_into_a_sample_size() -> None:
    with pytest.raises(ValueError):
        _sample_size(baseline=0.03, treatment=0.04, power=0.01)


def test_cli_defaults_retain_the_declared_illustrative_assumptions() -> None:
    args = build_parser().parse_args(["--baseline", "0.0323", "--relative-lift", "0.10"])
    assert args.alpha == 0.05
    assert args.power == 0.8
    assert args.attrition == 0.05


def test_cli_reports_planning_assumptions_and_independent_units_not_required_users(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["experiment", "--baseline", "0.0323", "--relative-lift", "0.10"]
    )
    main()
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Planning only" in captured.out
    assert "caller-supplied assumptions" in captured.out
    assert "3.230%" in captured.out and "3.553%" in captured.out
    assert "0.323" in captured.out and "pp" in captured.out
    assert "two-sided" in captured.out.lower()
    assert "alpha" in captured.out.lower() and "power" in captured.out.lower()
    assert "attrition" in captured.out.lower()
    assert (
        "Approximate enrollment: 51,893 independent units per arm (103,786 total)"
        in captured.out
    )
    assert "Required sample" not in captured.out
    assert "users per arm" not in captured.out
    assert "Attained power:" not in captured.out
    assert "Achieved power:" not in captured.out


def test_cli_accepts_negative_relative_effect_as_two_sided_planning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["experiment", "--baseline", "0.0323", "--relative-lift", "-0.10"]
    )
    main()
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "2.907%" in captured.out
    assert "-0.323" in captured.out and "pp" in captured.out
    assert "independent units per arm" in captured.out


@pytest.mark.parametrize(
    "arguments",
    [
        ["--baseline", "0", "--relative-lift", "0.10"],
        ["--baseline", "nan", "--relative-lift", "0.10"],
        ["--baseline", "0.0323", "--relative-lift", "0"],
        ["--baseline", "0.0323", "--relative-lift", "-1"],
        ["--baseline", "0.0323", "--relative-lift", "100"],
        ["--baseline", "0.0323", "--relative-lift", "nan"],
        ["--baseline", "0.0323", "--relative-lift", "inf"],
        ["--baseline", "0.0323", "--relative-lift", "0.10", "--alpha", "0"],
        ["--baseline", "0.0323", "--relative-lift", "0.10", "--power", "0.01"],
        ["--baseline", "0.0323", "--relative-lift", "0.10", "--attrition", "1"],
        ["--baseline", "not-a-number", "--relative-lift", "0.10"],
    ],
)
def test_cli_invalid_inputs_exit_two_without_traceback_or_partial_plan(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["experiment", *arguments])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err
    assert "Traceback" not in captured.err
