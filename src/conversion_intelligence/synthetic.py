from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_conversion_data(
    rows: int = 50_000,
    *,
    seed: int = 42,
    target_rate: float = 0.0323,
) -> pd.DataFrame:
    if rows < 100:
        raise ValueError("rows must be at least 100")
    if not 0 < target_rate < 1:
        raise ValueError("target_rate must be strictly between zero and one")

    rng = np.random.default_rng(seed)
    countries = rng.choice(["US", "China", "UK", "Germany"], rows, p=[0.56, 0.24, 0.15, 0.05])
    sources = rng.choice(["Seo", "Ads", "Direct"], rows, p=[0.49, 0.28, 0.23])
    age = np.clip(np.rint(rng.normal(30.5, 8.0, rows)), 17, 80).astype(int)
    new_user = rng.binomial(1, 0.685, rows)
    pages = np.maximum(1, rng.negative_binomial(3, 0.39, rows) + 1)

    linear = (
        -0.035 * (age - 30)
        - 0.65 * new_user
        + 0.43 * (pages - 5)
        + 0.35 * (countries == "Germany")
        + 0.22 * (countries == "UK")
        - 0.70 * (countries == "China")
        + 0.08 * (sources == "Direct")
    )

    low, high = -15.0, 5.0
    for _ in range(80):
        intercept = (low + high) / 2
        mean_rate = np.mean(1 / (1 + np.exp(-(linear + intercept))))
        if mean_rate < target_rate:
            low = intercept
        else:
            high = intercept
    probabilities = 1 / (1 + np.exp(-(linear + (low + high) / 2)))
    converted = rng.binomial(1, probabilities)

    return pd.DataFrame(
        {
            "country": countries,
            "age": age,
            "new_user": new_user,
            "source": sources,
            "total_pages_visited": pages,
            "converted": converted,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate schema-compatible synthetic data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-rate", type=float, default=0.0323)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = generate_synthetic_conversion_data(
        args.rows,
        seed=args.seed,
        target_rate=args.target_rate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(
        f"Wrote {len(data):,} rows to {args.output} "
        f"(conversion rate {data.converted.mean():.2%})"
    )


if __name__ == "__main__":
    main()
