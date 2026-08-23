from __future__ import annotations

import argparse
import math
from statistics import NormalDist


def two_proportion_sample_size(
    baseline: float,
    treatment: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    attrition: float = 0.0,
) -> int:
    """Approximate equal-sized per-arm sample for a two-sided difference in proportions."""
    if not 0 < baseline < 1 or not 0 < treatment < 1:
        raise ValueError("Rates must be strictly between zero and one")
    if math.isclose(baseline, treatment):
        raise ValueError("Baseline and treatment rates must differ")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must be strictly between zero and one")
    if not 0 <= attrition < 1:
        raise ValueError("attrition must be in [0, 1)")

    pooled = (baseline + treatment) / 2
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(power)
    numerator = (
        z_alpha * math.sqrt(2 * pooled * (1 - pooled))
        + z_power
        * math.sqrt(baseline * (1 - baseline) + treatment * (1 - treatment))
    ) ** 2
    raw = numerator / (treatment - baseline) ** 2
    return math.ceil(raw / (1 - attrition))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Power an equal-allocation conversion experiment")
    parser.add_argument("--baseline", type=float, required=True)
    parser.add_argument("--relative-lift", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--attrition", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    treatment = args.baseline * (1 + args.relative_lift)
    per_arm = two_proportion_sample_size(
        args.baseline,
        treatment,
        alpha=args.alpha,
        power=args.power,
        attrition=args.attrition,
    )
    print(f"Baseline rate: {args.baseline:.3%}")
    print(f"Treatment rate: {treatment:.3%}")
    print(f"Absolute lift: {treatment - args.baseline:.3%}")
    print(f"Required sample: {per_arm:,} users per arm ({per_arm * 2:,} total)")


if __name__ == "__main__":
    main()
