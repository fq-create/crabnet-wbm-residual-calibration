#!/usr/bin/env python
"""Compute MAE improvement of each representation relative to its baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SUMMARY_NAME = "residual_calibration_full_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root directory containing <baseline_model>/crabnet_frontend summary files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output root. Defaults to <input-root>/mae_improvement_vs_baseline. "
            "Each baseline_model is written to its own subdirectory."
        ),
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split to use from residual_calibration_full_summary.csv. Default: test.",
    )
    return parser.parse_args()


def find_summary_files(input_root: Path) -> list[Path]:
    return sorted(input_root.glob(f"*/crabnet_frontend/{SUMMARY_NAME}"))


def load_required_rows(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"baseline_model", "representation", "subset", "MAE_mean"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{path} is missing required columns: {sorted(missing)}")

    if "split" in df.columns:
        df = df[df["split"].astype(str) == split].copy()

    return df[["baseline_model", "representation", "subset", "MAE_mean"]].copy()


def baseline_representation_name(baseline_model: object) -> str:
    return f"{str(baseline_model)}-baseline"


def add_relative_improvement(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["baseline_representation"] = working["baseline_model"].map(baseline_representation_name)

    baseline_rows = (
        working[
            working["representation"].astype(str).str.lower()
            == working["baseline_representation"].astype(str).str.lower()
        ]
        [["baseline_model", "subset", "MAE_mean"]]
        .rename(columns={"MAE_mean": "baseline_MAE_mean"})
    )

    if baseline_rows.empty:
        raise ValueError("Could not find any baseline representation rows.")

    merged = working.merge(baseline_rows, on=["baseline_model", "subset"], how="left")

    if merged["baseline_MAE_mean"].isna().any():
        bad = merged.loc[
            merged["baseline_MAE_mean"].isna(),
            ["baseline_model", "subset"],
        ].drop_duplicates()
        raise ValueError(
            "Missing baseline MAE for these baseline/subset pairs:\n"
            + bad.to_string(index=False)
        )

    improvement_col = "relative_improvement_vs_baseline_%"
    merged[improvement_col] = (
        (merged["baseline_MAE_mean"] - merged["MAE_mean"])
        / merged["baseline_MAE_mean"]
        * 100.0
    )

    return merged.sort_values(
        ["baseline_model", "subset", improvement_col, "representation"],
        ascending=[True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = (args.output_dir or input_root / "mae_improvement_vs_baseline").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [load_required_rows(path, split=args.split) for path in find_summary_files(input_root)]
    if not frames:
        raise SystemExit(f"No {SUMMARY_NAME} files found under {input_root}")

    combined = pd.concat(frames, ignore_index=True)
    result = add_relative_improvement(combined)

    for baseline_model, model_df in result.groupby("baseline_model", sort=True):
        safe_model = str(baseline_model).replace("/", "_")
        model_output_dir = output_dir / safe_model
        model_output_dir.mkdir(parents=True, exist_ok=True)

        model_df = model_df.reset_index(drop=True)
        model_df.to_csv(model_output_dir / "mae_improvement_vs_baseline.csv", index=False)

        for subset, subset_df in model_df.groupby("subset", sort=True):
            safe_subset = str(subset).replace("/", "_")
            subset_df.to_csv(
                model_output_dir / f"mae_improvement_vs_baseline_{safe_subset}.csv",
                index=False,
            )


if __name__ == "__main__":
    main()
