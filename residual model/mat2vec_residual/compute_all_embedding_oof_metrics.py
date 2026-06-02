#!/usr/bin/env python
"""Compute full out-of-fold metrics from all-embedding residual predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matbench_discovery.metrics import stable_metrics
from sklearn.metrics import mean_absolute_error, r2_score


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOTS = [str(SCRIPT_DIR / "outputs_all_embeddings_residual")]
DEFAULT_OUTPUT_DIR = str(SCRIPT_DIR / "outputs_all_embeddings_residual" / "oof_metrics")
PREDICTION_FILE = "residual_calibration_full_predictions.csv.gz"
METRIC_COLUMNS = [
    "MAE",
    "RMSE",
    "R2",
    "F1",
    "DAF",
    "Precision",
    "Recall",
    "Accuracy",
    "TPR",
    "FPR",
    "TNR",
    "FNR",
    "TP",
    "FP",
    "TN",
    "FN",
]
DISCOVERY_COLUMNS = [key for key in METRIC_COLUMNS if key not in {"MAE", "RMSE", "R2"}]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-roots",
        nargs="*",
        default=DEFAULT_INPUT_ROOTS,
        help="Residual output roots to scan.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--full-baseline-metrics",
        default=None,
        help="Optional full baseline metrics CSV for consistency checks.",
    )
    parser.add_argument("--expected-total", type=int, default=256_963)
    return parser.parse_args()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def discovery_metrics(each_true: np.ndarray, each_pred: np.ndarray) -> dict[str, float]:
    metrics = stable_metrics(each_true, each_pred, fillna=True)
    return {key: float(value) for key, value in metrics.items() if key in DISCOVERY_COLUMNS}


def compute_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["e_form_true"].to_numpy(dtype=float)
    y_pred = frame["e_form_pred"].to_numpy(dtype=float)
    each_true = frame["e_above_hull_true"].to_numpy(dtype=float)
    each_pred = frame["e_above_hull_pred"].to_numpy(dtype=float)
    metrics = regression_metrics(y_true, y_pred)
    metrics.update(discovery_metrics(each_true, each_pred))
    return metrics


def find_prediction_files(input_roots: list[str]) -> list[Path]:
    paths: list[Path] = []
    for root_name in input_roots:
        root = Path(root_name)
        if not root.exists():
            continue
        paths.extend(root.glob(f"*/crabnet_frontend/{PREDICTION_FILE}"))
    return sorted(paths)


def route_from_path(path: Path) -> tuple[str, str, str]:
    # <root>/<model>/<route>/residual_calibration_full_predictions.csv.gz
    return str(path.parents[2]), path.parents[1].name, path.parents[0].name


def subset_frame(frame: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "all":
        return frame
    raise KeyError(f"Unknown subset {subset!r}")


def add_metric_prefix(row: dict[str, Any], prefix: str, metrics: dict[str, float]) -> None:
    for key in METRIC_COLUMNS:
        if key in metrics:
            row[f"{prefix}{key}"] = metrics[key]


def load_full_baseline_metrics(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        return pd.DataFrame()
    if path.is_dir():
        return pd.DataFrame()
    df = pd.read_csv(path)
    model_col = "model" if "model" in df.columns else "baseline_model"
    df = df.rename(columns={model_col: "baseline_model"})
    return df


def compute_oof_metrics_for_file(
    path: Path,
    *,
    expected_total: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    input_root, path_model, path_route = route_from_path(path)
    preferred_cols = [
        "baseline_model",
        "residual_route",
        "material_id",
        "split",
        "fold",
        "representation",
        "e_form_true",
        "e_form_base",
        "e_form_pred",
        "e_above_hull_true",
        "e_above_hull_pred",
    ]
    header = pd.read_csv(path, nrows=0)
    usecols = [col for col in preferred_cols if col in header.columns]
    df = pd.read_csv(path, usecols=usecols)
    if "baseline_model" not in df.columns:
        df.insert(0, "baseline_model", path_model)
    if "residual_route" not in df.columns:
        df.insert(1, "residual_route", path_route)
    if "split" not in df.columns:
        # Legacy prediction files contain only out-of-fold test predictions.
        df.insert(3, "split", "test")
    if "test" in set(df["split"].astype(str)):
        df = df[df["split"] == "test"].copy()

    if df.empty:
        raise ValueError(f"No OOF/test rows found in {path}")

    baseline_model = str(df["baseline_model"].iloc[0])
    residual_route = str(df["residual_route"].iloc[0])
    baseline_candidate = f"{baseline_model}-baseline"
    representation_index = pd.Index(df["representation"].dropna().astype(str).unique())
    baseline_matches = [
        representation
        for representation in representation_index
        if representation.lower() == baseline_candidate.lower()
    ]
    baseline_representation = baseline_matches[0] if baseline_matches else baseline_candidate
    subsets = ["all"]

    metric_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    for subset in subsets:
        subset_df = subset_frame(df, subset)
        baseline_rows = subset_df[subset_df["representation"] == baseline_representation].copy()
        baseline_by_id = baseline_rows.drop_duplicates("material_id").set_index("material_id")
        if not baseline_rows.empty:
            baseline_metrics_full_subset = compute_metrics(baseline_rows)
        else:
            baseline_metrics_full_subset = {}

        for representation, rep_df in subset_df.groupby("representation", sort=True):
            rep_df = rep_df.copy()
            duplicate_count = int(rep_df["material_id"].duplicated().sum())
            unique_materials = int(rep_df["material_id"].nunique())
            n_rows = int(len(rep_df))
            n_folds = int(rep_df["fold"].nunique())
            metrics = compute_metrics(rep_df)

            same_ids_baseline = baseline_by_id.reindex(rep_df["material_id"].astype(str))
            has_same_id_baseline = not same_ids_baseline.empty and same_ids_baseline["e_form_pred"].notna().all()
            if has_same_id_baseline:
                baseline_same_rows = same_ids_baseline.reset_index(drop=True).copy()
                baseline_same_rows["material_id"] = rep_df["material_id"].to_numpy()
                baseline_same_metrics = compute_metrics(baseline_same_rows)
            else:
                baseline_same_metrics = {}

            row: dict[str, Any] = {
                "baseline_model": baseline_model,
                "residual_route": residual_route,
                "input_root": input_root,
                "path_model": path_model,
                "path_route": path_route,
                "representation": representation,
                "subset": subset,
                "n_rows": n_rows,
                "n_unique_material_ids": unique_materials,
                "n_duplicate_material_ids": duplicate_count,
                "n_folds": n_folds,
                "expected_total": expected_total if subset == "all" else np.nan,
                "coverage_fraction": n_rows / expected_total if subset == "all" else np.nan,
                "prediction_file": str(path),
            }
            row.update(metrics)
            add_metric_prefix(row, "baseline_all_subset_", baseline_metrics_full_subset)
            add_metric_prefix(row, "baseline_same_rows_", baseline_same_metrics)

            for key in METRIC_COLUMNS:
                if key in metrics and f"baseline_same_rows_{key}" in row:
                    row[f"delta_{key}_vs_baseline_same_rows"] = row[key] - row[f"baseline_same_rows_{key}"]
                if key in metrics and f"baseline_all_subset_{key}" in row:
                    row[f"delta_{key}_vs_baseline_all_subset"] = row[key] - row[f"baseline_all_subset_{key}"]

            if "MAE" in metrics and "baseline_same_rows_MAE" in row:
                row["relative_MAE_improvement_vs_baseline_same_rows_%"] = (
                    (row["baseline_same_rows_MAE"] - row["MAE"]) / row["baseline_same_rows_MAE"] * 100
                )
            metric_rows.append(row)

        validation_rows.append(
            {
                "baseline_model": baseline_model,
                "residual_route": residual_route,
                "input_root": input_root,
                "subset": subset,
                "baseline_representation": baseline_representation,
                "baseline_n_rows": int(len(baseline_rows)),
                "baseline_n_unique_material_ids": int(baseline_rows["material_id"].nunique()),
                "baseline_duplicate_material_ids": int(baseline_rows["material_id"].duplicated().sum()),
                "baseline_n_folds": int(baseline_rows["fold"].nunique()) if not baseline_rows.empty else 0,
                "expected_total": expected_total if subset == "all" else np.nan,
                "baseline_has_expected_total": bool(len(baseline_rows) == expected_total)
                if subset == "all"
                else np.nan,
                "prediction_file": str(path),
            }
        )

    return metric_rows, validation_rows


def add_best_tables(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    all_subset = metrics_df[metrics_df["subset"] == "all"].copy()
    all_subset["is_baseline_representation"] = (
        all_subset["representation"].astype(str).str.lower()
        == (all_subset["baseline_model"].astype(str) + "-baseline").str.lower()
    )
    best_mae = (
        all_subset.sort_values(["baseline_model", "MAE", "RMSE"], kind="stable")
        .groupby("baseline_model", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_f1 = (
        all_subset.sort_values(["baseline_model", "F1", "MAE"], ascending=[True, False, True], kind="stable")
        .groupby("baseline_model", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_mae.to_csv(output_dir / "oof_best_by_model_mae.csv", index=False)
    best_f1.to_csv(output_dir / "oof_best_by_model_f1.csv", index=False)

    nonbaseline = all_subset[~all_subset["is_baseline_representation"]].copy()
    best_nonbaseline_mae = (
        nonbaseline.sort_values(["baseline_model", "MAE", "RMSE"], kind="stable")
        .groupby("baseline_model", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_nonbaseline_f1 = (
        nonbaseline.sort_values(["baseline_model", "F1", "MAE"], ascending=[True, False, True], kind="stable")
        .groupby("baseline_model", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best_nonbaseline_mae.to_csv(output_dir / "oof_best_nonbaseline_by_model_mae.csv", index=False)
    best_nonbaseline_f1.to_csv(output_dir / "oof_best_nonbaseline_by_model_f1.csv", index=False)

    baseline = all_subset[all_subset["is_baseline_representation"]].copy()
    baseline = baseline.sort_values(["baseline_model", "residual_route"], kind="stable").drop_duplicates(
        "baseline_model",
        keep="first",
    )
    main = baseline[
        [
            "baseline_model",
            "MAE",
            "RMSE",
            "R2",
            "F1",
            "DAF",
            "Precision",
            "Recall",
            "Accuracy",
        ]
    ].rename(
        columns={
            "MAE": "baseline_OOF_MAE",
            "RMSE": "baseline_OOF_RMSE",
            "R2": "baseline_OOF_R2",
            "F1": "baseline_OOF_F1",
            "DAF": "baseline_OOF_DAF",
            "Precision": "baseline_OOF_Precision",
            "Recall": "baseline_OOF_Recall",
            "Accuracy": "baseline_OOF_Accuracy",
        }
    )
    mae_cols = [
        "baseline_model",
        "residual_route",
        "representation",
        "MAE",
        "RMSE",
        "R2",
        "F1",
        "DAF",
        "Precision",
        "Recall",
        "Accuracy",
        "delta_MAE_vs_baseline_same_rows",
        "relative_MAE_improvement_vs_baseline_same_rows_%",
        "delta_F1_vs_baseline_same_rows",
        "delta_DAF_vs_baseline_same_rows",
    ]
    mae_cols = [col for col in mae_cols if col in best_nonbaseline_mae.columns]
    best_mae_for_main = best_nonbaseline_mae[mae_cols].rename(
        columns={
            "residual_route": "best_MAE_residual_route",
            "representation": "best_MAE_representation",
            "MAE": "best_residual_OOF_MAE",
            "RMSE": "best_residual_OOF_RMSE",
            "R2": "best_residual_OOF_R2",
            "F1": "best_residual_OOF_F1",
            "DAF": "best_residual_OOF_DAF",
            "Precision": "best_residual_OOF_Precision",
            "Recall": "best_residual_OOF_Recall",
            "Accuracy": "best_residual_OOF_Accuracy",
            "delta_MAE_vs_baseline_same_rows": "best_residual_delta_MAE",
            "relative_MAE_improvement_vs_baseline_same_rows_%": "best_residual_relative_MAE_improvement_%",
            "delta_F1_vs_baseline_same_rows": "best_residual_delta_F1",
            "delta_DAF_vs_baseline_same_rows": "best_residual_delta_DAF",
        }
    )
    main = main.merge(best_mae_for_main, on="baseline_model", how="left")
    main = main.sort_values("baseline_OOF_MAE", kind="stable").reset_index(drop=True)
    main.to_csv(output_dir / "oof_main_baseline_vs_best_nonbaseline_mae.csv", index=False)

    delta_cols = [
        col
        for col in metrics_df.columns
        if col.startswith("delta_") or col.startswith("relative_MAE_improvement")
    ]
    keep_cols = [
        "baseline_model",
        "residual_route",
        "input_root",
        "representation",
        "subset",
        "n_rows",
        "MAE",
        "RMSE",
        "R2",
        "F1",
        "DAF",
        "Precision",
        "Recall",
        "Accuracy",
        *delta_cols,
    ]
    keep_cols = [col for col in keep_cols if col in metrics_df.columns]
    metrics_df[keep_cols].to_csv(output_dir / "oof_delta_vs_baseline.csv", index=False)


def add_full_baseline_check(
    metrics_df: pd.DataFrame,
    full_baseline_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    if full_baseline_df.empty:
        return
    baseline_rows = metrics_df[
        (metrics_df["subset"] == "all")
        & (
            metrics_df["representation"].astype(str).str.lower()
            == (metrics_df["baseline_model"].astype(str) + "-baseline").str.lower()
        )
    ].copy()
    if baseline_rows.empty:
        return
    merged = baseline_rows.merge(
        full_baseline_df,
        on="baseline_model",
        how="left",
        suffixes=("_oof", "_full"),
    )
    for key in ["MAE", "RMSE", "R2", "F1", "DAF", "Precision"]:
        oof_col = f"{key}_oof"
        full_col = f"{key}_full"
        if oof_col in merged.columns and full_col in merged.columns:
            merged[f"delta_{key}_oof_minus_full"] = merged[oof_col] - merged[full_col]
    merged.to_csv(output_dir / "oof_baseline_vs_full_baseline_check.csv", index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_files = find_prediction_files(args.input_roots)
    if not prediction_files:
        raise SystemExit(f"No {PREDICTION_FILE} files found under {args.input_roots}")

    metric_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for path in prediction_files:
        print(f"Processing {path}")
        rows, validation = compute_oof_metrics_for_file(path, expected_total=args.expected_total)
        metric_rows.extend(rows)
        validation_rows.extend(validation)

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["baseline_model", "residual_route", "subset", "MAE", "representation"],
        kind="stable",
    )
    validation_df = pd.DataFrame(validation_rows).sort_values(
        ["baseline_model", "residual_route", "subset"],
        kind="stable",
    )

    metrics_df.to_csv(output_dir / "oof_metrics_all_representations.csv", index=False)
    validation_df.to_csv(output_dir / "oof_validation_summary.csv", index=False)
    add_best_tables(metrics_df, output_dir)
    full_baseline_path = Path(args.full_baseline_metrics) if args.full_baseline_metrics else None
    full_baseline_df = load_full_baseline_metrics(full_baseline_path)
    add_full_baseline_check(metrics_df, full_baseline_df, output_dir)

    print(f"Wrote OOF metrics to {output_dir}")


if __name__ == "__main__":
    main()
