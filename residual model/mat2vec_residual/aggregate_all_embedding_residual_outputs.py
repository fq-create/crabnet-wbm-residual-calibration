#!/usr/bin/env python
"""Aggregate per-model all-embedding residual outputs into comparison tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESIDUAL_OUTPUT_ROOT = SCRIPT_DIR / "outputs_all_embeddings_residual"


SUMMARY_FILES = ["residual_calibration_full_summary.csv"]

FOLD_FILES = ["residual_calibration_full_fold_metrics.csv"]

OTHER_FILES = [
    "residual_calibration_full_projection_summary.csv",
    "residual_calibration_full_embedding_availability.csv",
    "residual_calibration_full_missing_embedding_elements.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_RESIDUAL_OUTPUT_ROOT,
        help="Root directory containing per-model residual outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for combined CSV outputs. Defaults to <input-root>/combined.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional explicit model subdirectories to aggregate.",
    )
    return parser.parse_args()


def iter_model_dirs(input_root: Path, selected_models: list[str] | None) -> list[Path]:
    if selected_models:
        return [input_root / name for name in selected_models]
    return sorted(
        path
        for path in input_root.iterdir()
        if path.is_dir() and not path.name.startswith("_") and path.name != "combined"
    )


def collect_csvs(model_dirs: list[Path], relative_name: str) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    for model_dir in model_dirs:
        for route_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            csv_path = route_dir / relative_name
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if "baseline_model" not in df.columns:
                df["baseline_model"] = model_dir.name
            if "residual_route" not in df.columns:
                df["residual_route"] = route_dir.name
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def maybe_sort(df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        column
        for column in [
            "baseline_model",
            "residual_route",
            "metrics_variant",
            "split",
            "subset",
            "rank_by_MAE",
            "representation",
            "fold",
        ]
        if column in df.columns
    ]
    if sort_columns:
        return df.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    return df


def build_leaderboard(summary_df: pd.DataFrame) -> pd.DataFrame:
    leaderboard = summary_df.copy()
    if "split" in leaderboard.columns:
        leaderboard = leaderboard[leaderboard["split"] == "test"]
    if "subset" in leaderboard.columns:
        leaderboard = leaderboard[leaderboard["subset"] == "all"]
    sort_columns = [column for column in ["MAE_mean", "RMSE_mean", "rank_by_MAE"] if column in leaderboard.columns]
    if sort_columns:
        leaderboard = leaderboard.sort_values(sort_columns, kind="stable")
    return leaderboard.reset_index(drop=True)


def write_combined_tables(output_dir: Path, model_dirs: list[Path]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in SUMMARY_FILES + FOLD_FILES + OTHER_FILES:
        combined_df = collect_csvs(model_dirs, filename)
        if combined_df is None:
            continue
        combined_df = maybe_sort(combined_df)
        combined_path = output_dir / filename.replace(".csv", "_combined.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"Wrote {combined_path}")

        if filename in SUMMARY_FILES:
            leaderboard = build_leaderboard(combined_df)
            leaderboard_name = filename.replace(".csv", "_leaderboard_test_all.csv")
            leaderboard_path = output_dir / leaderboard_name
            leaderboard.to_csv(leaderboard_path, index=False)
            print(f"Wrote {leaderboard_path}")


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_dir = (args.output_dir or (input_root / "combined")).resolve()
    model_dirs = iter_model_dirs(input_root, args.models)
    if not model_dirs:
        raise SystemExit(f"No model directories found under {input_root}")
    write_combined_tables(output_dir, model_dirs)


if __name__ == "__main__":
    main()
