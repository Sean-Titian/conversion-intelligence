from __future__ import annotations

import argparse
import math
from numbers import Real
from statistics import NormalDist, StatisticsError


def _planning_fraction(value: float, name: str, *, allow_zero: bool = False) -> float:
    """Validate the original scalar range before normalizing to a Python float."""

    message = (
        f"{name} must be a finite real number in "
        + ("[0, 1)" if allow_zero else "(0, 1)")
    )
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(message)
    if not (0 <= value < 1 if allow_zero else 0 < value < 1):
        raise ValueError(message)
    try:
        normalized = float(value)
    except (ValueError, OverflowError):
        raise ValueError(message) from None
    if not math.isfinite(normalized) or not (
        0 <= normalized < 1 if allow_zero else 0 < normalized < 1
    ):
        raise ValueError(message)
    return normalized


def two_proportion_sample_size(
    baseline: float,
    treatment: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    attrition: float = 0.0,
) -> int:
    """Approximate equal-allocation enrollment per arm for independent binary units.

    Rates are caller-supplied assumptions, not verified population/endpoint
    baselines. The normal approximation ignores the distant rejection tail;
    target power is not empirically attained power or a full launch decision.
    Attrition inflates enrollment only; it cannot fix biased missing outcomes.
    """

    baseline = _planning_fraction(baseline, "baseline")
    treatment = _planning_fraction(treatment, "treatment")
    alpha = _planning_fraction(alpha, "alpha")
    power = _planning_fraction(power, "power")
    attrition = _planning_fraction(attrition, "attrition", allow_zero=True)
    if math.isclose(baseline, treatment):
        raise ValueError("Baseline and treatment rates must differ")

    pooled = (baseline + treatment) / 2
    try:
        z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
        z_power = NormalDist().inv_cdf(power)
    except StatisticsError:
        raise ValueError("alpha or power is too extreme for this normal approximation") from None
    root = (
        z_alpha * math.sqrt(2 * pooled * (1 - pooled))
        + z_power
        * math.sqrt(baseline * (1 - baseline) + treatment * (1 - treatment))
    )
    squared_difference = (treatment - baseline) ** 2
    if root <= 0 or squared_difference == 0:
        raise ValueError("Inputs are outside the supported normal-approximation range")
    enrollment = root**2 / squared_difference / (1 - attrition)
    if not math.isfinite(enrollment) or enrollment <= 0:
        raise ValueError("Approximate enrollment must be positive and finite")
    return math.ceil(enrollment)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Illustrative normal-approximation planning for independent binary units"
    )
    parser.add_argument("--baseline", type=float, required=True)
    parser.add_argument("--relative-lift", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--attrition", type=float, default=0.05)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    treatment = args.baseline * (1 + args.relative_lift)
    try:
        per_arm = two_proportion_sample_size(
            args.baseline,
            treatment,
            alpha=args.alpha,
            power=args.power,
            attrition=args.attrition,
        )
    except ValueError as error:
        parser.error(str(error))
    print("Planning only: caller-supplied assumptions; endpoint compatibility is not verified.")
    print(f"Assumed baseline rate: {args.baseline:.3%}")
    print(f"Assumed target rate: {treatment:.3%}")
    print(f"Absolute target difference: {(treatment - args.baseline) * 100:.3f} pp")
    print(
        f"Two-sided alpha: {args.alpha:g}; target power: {args.power:g}; "
        f"attrition: {args.attrition:g}"
    )
    print(f"Approximate enrollment: {per_arm:,} independent units per arm ({per_arm * 2:,} total)")
    print(
        "Equal allocation; normal approximation, not verified attained power or rollout approval."
    )


if __name__ == "__main__":
    main()
