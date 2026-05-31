#!/usr/bin/env python
"""Compute OOF MAE improvement relative to each model baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


INPUT_NAME = "oof_delta_vs_baseline.csv"
OUTPUT_NAMES = {
    "all": "oof_mae_improvement_vs_baseline_all.csv",
    "unique_prototype": "oof_mae_improvement_vs_baseline_unique_prototype.csv",
}
KEEP_COLUMNS = ["baseline_model", "representation", "subset", "MAE"]
IMPROVEMENT_COLUMN = "relative_MAE_improvement_vs_baseline_%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / INPUT_NAME,
        help=f"Input OOF delta CSV. Default: script directory/{INPUT_NAME}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Output directory. Default: script directory.",
    )
    return parser.parse_args()


def baseline_representation_name(baseline_model: object) -> str:
    return f"{str(baseline_model)}-baseline"


def validate_columns(df: pd.DataFrame, path: Path) -> None:
    missing = set(KEEP_COLUMNS) - set(df.columns)
    if missing:
        raise KeyError(f"{path} is missing required columns: {sorted(missing)}")


def add_improvement(df: pd.DataFrame) -> pd.DataFrame:
    working = df[KEEP_COLUMNS].copy()
    working["baseline_representation"] = working["baseline_model"].map(baseline_representation_name)

    baseline_rows = working[
        working["representation"].astype(str).str.lower()
        == working["baseline_representation"].astype(str).str.lower()
    ][["baseline_model", "subset", "MAE"]].rename(columns={"MAE": "baseline_MAE"})

    if baseline_rows.empty:
        raise ValueError("Could not find any baseline representation rows.")

    duplicate_mask = baseline_rows.duplicated(["baseline_model", "subset"], keep=False)
    if duplicate_mask.any():
        duplicates = baseline_rows.loc[duplicate_mask, ["baseline_model", "subset"]].drop_duplicates()
        raise ValueError(
            "Found duplicate baseline rows for these baseline_model/subset pairs:\n"
            + duplicates.to_string(index=False)
        )

    merged = working.merge(baseline_rows, on=["baseline_model", "subset"], how="left")
    if merged["baseline_MAE"].isna().any():
        missing = merged.loc[
            merged["baseline_MAE"].isna(),
            ["baseline_model", "subset"],
        ].drop_duplicates()
        raise ValueError(
            "Missing baseline MAE for these baseline_model/subset pairs:\n"
            + missing.to_string(index=False)
        )

    merged[IMPROVEMENT_COLUMN] = (
        (merged["baseline_MAE"] - merged["MAE"]) / merged["baseline_MAE"] * 100.0
    )

    result = merged[KEEP_COLUMNS + [IMPROVEMENT_COLUMN]].copy()
    return result.sort_values(
        ["baseline_model", "subset", IMPROVEMENT_COLUMN, "representation"],
        ascending=[True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    validate_columns(df, input_path)
    result = add_improvement(df)

    for subset, output_name in OUTPUT_NAMES.items():
        subset_df = result[result["subset"].astype(str) == subset].copy()
        if subset_df.empty:
            print(f"WARNING: no rows found for subset={subset!r}")
        output_path = output_dir / output_name
        subset_df.to_csv(output_path, index=False)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
