#!/usr/bin/env python
"""Prototype-blocked residual calibration for every embedding table."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from matbench_discovery.metrics import stable_metrics


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
BASELINE_OUTPUT_ROOT = PROJECT_ROOT / "frozen baseline" / "outputs"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "wbm_summary.csv.gz"
DEFAULT_PREDICTIONS = BASELINE_OUTPUT_ROOT / "chgnet" / "eform" / "chgnet_wbm_computed_full.csv"
DEFAULT_EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
DEFAULT_OUTPUT_DIR = SCRIPT_PATH.parent / "outputs_all_embeddings_residual" / "chgnet" / "crabnet_frontend"

RESIDUAL_ROUTE = "crabnet_frontend"
METRICS_VARIANT = "full"
FRONTEND_NAME = "crabnet_src_frac_linear_pooling_all_embeddings_no_projection"
ALPHAS = np.logspace(-6, 6, 25)
GAMMA_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)

BASELINE_ALIASES = {
    "chgnet-0.3.0": "chgnet",
    "mace": "mace-mpa-0",
    "matgl-m3gnet": "m3gnet",
    "matgl": "m3gnet",
    "mattersim": "mattersim-v1-5m",
    "orb": "orb-v3",
    "sevenn": "sevennet-l3i5",
    "sevennet": "sevennet-l3i5",
    "7net-l3i5": "sevennet-l3i5",
}

SUPPORTED_BASELINES = {
    "chgnet",
    "m3gnet",
    "mace-mpa-0",
    "mattersim-v1-5m",
    "orb-v3",
    "sevennet-l3i5",
}

LEGACY_EFORM_PREFIX = {"chgnet": "chgnet"}


def normalize_baseline_model(name: str) -> str:
    key = str(name).strip().lower()
    normalized = BASELINE_ALIASES.get(key, key)
    if normalized not in SUPPORTED_BASELINES:
        supported = ", ".join(sorted(SUPPORTED_BASELINES))
        raise KeyError(f"Unsupported baseline model {name!r}. Supported values: {supported}")
    return normalized


def baseline_representation_name(baseline_model: str) -> str:
    return f"{normalize_baseline_model(baseline_model)}-baseline"


def detect_eform_column(df: pd.DataFrame, baseline_model: str) -> str:
    candidates = []
    prefix = LEGACY_EFORM_PREFIX.get(normalize_baseline_model(baseline_model))
    if prefix:
        candidates.append(f"{prefix}_e_form_per_atom")
    candidates.append("e_form_per_atom")
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError(
        "Could not locate formation-energy column. "
        f"Available columns: {sorted(df.columns.tolist())}"
    )


def ensure_result_metadata(
    df: pd.DataFrame,
    *,
    baseline_model: str,
    residual_route: str,
    metrics_variant: str | None = None,
) -> pd.DataFrame:
    enriched = df.copy()
    enriched.insert(0, "residual_route", residual_route)
    enriched.insert(0, "baseline_model", normalize_baseline_model(baseline_model))
    if metrics_variant is not None:
        enriched.insert(2, "metrics_variant", metrics_variant)
    return enriched


@dataclass(frozen=True)
class FormulaSequence:
    src: np.ndarray
    frac: np.ndarray
    elements: tuple[str, ...]


@dataclass(frozen=True)
class ProjectionSummary:
    representation: str
    raw_dimension: int
    projected_dimension: int
    projection_kind: str
    explained_variance_ratio_sum: float | None
    projector_is_trained: bool
    projector_scope: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-model", default="chgnet")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--embeddings-dir", default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--label-col", default="e_form_per_atom_mp2020_corrected")
    parser.add_argument(
        "--each-col",
        default="e_above_hull_mp2020_corrected_ppd_mp",
    )
    parser.add_argument("--group-col", default="wyckoff_spglib_initial_structure")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--expected-total", type=int, default=256_963)
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=32,
        help="Accepted for CLI compatibility; ignored because precomputed embeddings are not projected.",
    )
    parser.add_argument(
        "--embedding-glob",
        default="*.csv",
        help="Glob pattern under --embeddings-dir. The default runs every CSV embedding table.",
    )
    return parser.parse_args()


def is_element_symbol(value: object) -> bool:
    try:
        Element(str(value))
        return True
    except Exception:
        return False


def load_embedding(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    element_col = None
    for col in df.columns:
        normalized = str(col).strip().lstrip("\ufeff").lower()
        if normalized in {"element", "symbol", "element_symbol", "elem"}:
            element_col = col
            break
    if element_col is None and df.shape[1] > 1 and df.iloc[:, 0].map(is_element_symbol).mean() > 0.8:
        element_col = df.columns[0]
    if element_col is None:
        df = pd.read_csv(path, index_col=0)
    else:
        df = df.set_index(element_col)
    df.index = df.index.map(str)
    numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.loc[:, numeric.notna().all(axis=0)]
    numeric = numeric.dropna(axis=0, how="any")
    return numeric


def discover_embedding_files(embeddings_dir: Path, pattern: str) -> list[tuple[str, Path]]:
    paths = sorted(embeddings_dir.glob(pattern), key=lambda path: path.name.lower())
    paths = [path for path in paths if path.is_file()]
    if not paths:
        raise FileNotFoundError(f"No embedding CSV files matching {pattern!r} under {embeddings_dir}")
    return [(path.stem, path) for path in paths]


def canonicalize_prototype(value: object) -> tuple[str, bool]:
    if pd.isna(value):
        return "__MISSING_WYCKOFF__", True
    text = str(value).strip()
    if not text:
        return "__EMPTY_WYCKOFF__", True
    if ":" in text:
        template = text.split(":", 1)[0].strip()
        if template:
            return template, False
    return text, True


def parse_formula_sequences(formulas: pd.Series) -> list[FormulaSequence]:
    parsed: list[FormulaSequence] = []
    for formula in tqdm(formulas.astype(str), desc="Parsing formulas"):
        comp = Composition(formula)
        items = sorted(comp.get_el_amt_dict().items())
        elements = tuple(symbol for symbol, _ in items)
        frac = np.array([amount for _, amount in items], dtype=np.float64)
        frac /= frac.sum()
        src = np.array([Element(symbol).Z for symbol in elements], dtype=np.int64)
        parsed.append(FormulaSequence(src=src, frac=frac.astype(np.float32), elements=elements))
    return parsed



def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = math.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def discovery_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    each_true: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    each_pred = np.asarray(each_true, dtype=float) + np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    metrics = stable_metrics(each_true, each_pred, fillna=True)
    return {"F1": float(metrics["F1"]), "DAF": float(metrics["DAF"])}, each_pred


def choose_gamma(
    base_holdout: np.ndarray,
    y_holdout: np.ndarray,
    residual_hat_holdout: np.ndarray,
) -> tuple[float, float]:
    best_gamma = 0.0
    best_mae = math.inf
    for gamma in GAMMA_GRID:
        pred = base_holdout + gamma * residual_hat_holdout
        mae = float(mean_absolute_error(y_holdout, pred))
        if mae < best_mae - 1e-12 or (abs(mae - best_mae) <= 1e-12 and gamma < best_gamma):
            best_gamma = float(gamma)
            best_mae = mae
    return best_gamma, best_mae


def fit_ridge_model(x_train: np.ndarray, residual_train: np.ndarray):
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    model.fit(x_train, residual_train)
    alpha = float(model.named_steps["ridgecv"].alpha_)
    return model, alpha


def summarize_embedding_table(
    representation: str,
    embedding: pd.DataFrame,
) -> ProjectionSummary:
    raw_dim = int(embedding.shape[1])
    return ProjectionSummary(
        representation=representation,
        raw_dimension=raw_dim,
        projected_dimension=raw_dim,
        projection_kind="precomputed_embedding_no_projection",
        explained_variance_ratio_sum=1.0,
        projector_is_trained=False,
        projector_scope="embedding_table",
    )


def build_crabnet_frontend_features(
    representation: str,
    embedding: pd.DataFrame,
    formula_sequences: list[FormulaSequence],
    material_ids: pd.Series,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, list[str]], ProjectionSummary]:
    projected_embedding = embedding
    projection_summary = summarize_embedding_table(
        representation=representation,
        embedding=embedding,
    )

    features: list[np.ndarray] = []
    usable_idx: list[int] = []
    missing_counts: dict[str, int] = defaultdict(int)
    missing_examples: dict[str, list[str]] = defaultdict(list)
    available_symbols = set(projected_embedding.index)
    projected_values = projected_embedding.to_numpy(dtype=np.float32)
    projected_lookup = {
        symbol: projected_values[idx] for idx, symbol in enumerate(projected_embedding.index)
    }
    feature_dim = projected_embedding.shape[1] * 2

    iterator = zip(formula_sequences, material_ids.astype(str), strict=False)
    for idx, (sequence, material_id) in enumerate(
        tqdm(iterator, desc=f"Frontend {representation}", total=len(formula_sequences))
    ):
        symbols = tuple(Element.from_Z(int(z)).symbol for z in sequence.src)
        missing = [symbol for symbol in symbols if symbol not in available_symbols]
        if missing:
            for symbol in missing:
                missing_counts[symbol] += 1
                if len(missing_examples[symbol]) < 10:
                    missing_examples[symbol].append(material_id)
            continue

        values = np.stack([projected_lookup[symbol] for symbol in symbols], axis=0).astype(np.float32)
        weights = sequence.frac.astype(np.float32)
        mean = np.average(values, axis=0, weights=weights).astype(np.float32)
        var = np.average((values - mean) ** 2, axis=0, weights=weights).astype(np.float32)
        std = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
        features.append(np.concatenate([mean, std]).astype(np.float32))
        usable_idx.append(idx)

    if features:
        x = np.vstack(features).astype(np.float32)
    else:
        x = np.empty((0, feature_dim), dtype=np.float32)
    return x, np.asarray(usable_idx, dtype=int), dict(missing_counts), dict(missing_examples), projection_summary


def evaluate_split_subset(
    *,
    fold: int,
    representation: str,
    split: str,
    subset: str,
    row_mask: np.ndarray,
    y: np.ndarray,
    e_form_pred: np.ndarray,
    e_base: np.ndarray,
    each_true: np.ndarray,
    gamma: float,
    alpha: float | None,
    inner_alpha: float | None,
    inner_holdout_mae: float | None,
    n_dropped_total: int,
    n_total_representation: int,
    n_train_representation: int,
    n_test_representation: int,
    baseline_name: str,
) -> dict[str, Any]:
    n_rows = int(row_mask.sum())
    if n_rows == 0:
        return {
            "fold": fold,
            "representation": representation,
            "split": split,
            "subset": subset,
            "n_rows": 0,
            "gamma": gamma,
            "alpha": alpha,
            "inner_alpha": inner_alpha,
            "inner_holdout_mae": inner_holdout_mae,
            "n_dropped_total": n_dropped_total,
            "n_total_representation": n_total_representation,
            "n_train_representation": n_train_representation,
            "n_test_representation": n_test_representation,
            "baseline_representation": baseline_name,
            "MAE": math.nan,
            "RMSE": math.nan,
            "R2": math.nan,
            "baseline_MAE_on_eval_rows": math.nan,
            "baseline_RMSE_on_eval_rows": math.nan,
            "baseline_R2_on_eval_rows": math.nan,
            "delta_MAE_vs_baseline": math.nan,
            "delta_RMSE_vs_baseline": math.nan,
            "delta_R2_vs_baseline": math.nan,
            "F1": math.nan,
            "DAF": math.nan,
            "baseline_F1_on_eval_rows": math.nan,
            "baseline_DAF_on_eval_rows": math.nan,
            "delta_F1_vs_baseline": math.nan,
            "delta_DAF_vs_baseline": math.nan,
        }

    y_rows = y[row_mask]
    pred_rows = e_form_pred[row_mask]
    base_rows = e_base[row_mask]
    each_rows = each_true[row_mask]

    reg = regression_metrics(y_rows, pred_rows)
    base_reg = regression_metrics(y_rows, base_rows)
    disc, _ = discovery_metrics(y_rows, pred_rows, each_rows)
    base_disc, _ = discovery_metrics(y_rows, base_rows, each_rows)

    return {
        "fold": fold,
        "representation": representation,
        "split": split,
        "subset": subset,
        "n_rows": n_rows,
        "gamma": gamma,
        "alpha": alpha,
        "inner_alpha": inner_alpha,
        "inner_holdout_mae": inner_holdout_mae,
        "n_dropped_total": n_dropped_total,
        "n_total_representation": n_total_representation,
        "n_train_representation": n_train_representation,
        "n_test_representation": n_test_representation,
        "baseline_representation": baseline_name,
        **reg,
        "baseline_MAE_on_eval_rows": base_reg["MAE"],
        "baseline_RMSE_on_eval_rows": base_reg["RMSE"],
        "baseline_R2_on_eval_rows": base_reg["R2"],
        "delta_MAE_vs_baseline": reg["MAE"] - base_reg["MAE"],
        "delta_RMSE_vs_baseline": reg["RMSE"] - base_reg["RMSE"],
        "delta_R2_vs_baseline": reg["R2"] - base_reg["R2"],
        **disc,
        "baseline_F1_on_eval_rows": base_disc["F1"],
        "baseline_DAF_on_eval_rows": base_disc["DAF"],
        "delta_F1_vs_baseline": disc["F1"] - base_disc["F1"],
        "delta_DAF_vs_baseline": disc["DAF"] - base_disc["DAF"],
    }


def make_inner_split(groups: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = pd.Index(groups).nunique()
    if unique_groups < 2:
        raise ValueError("Need at least 2 unique groups for an inner holdout split")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    dummy_x = np.zeros(len(groups))
    inner_train, inner_holdout = next(splitter.split(dummy_x, groups=groups))
    if len(inner_train) == 0 or len(inner_holdout) == 0:
        raise ValueError("GroupShuffleSplit produced an empty inner split")
    return inner_train, inner_holdout


def baseline_prediction_rows(
    data: pd.DataFrame,
    fold_assignments: np.ndarray,
    y: np.ndarray,
    e_base: np.ndarray,
    each_true: np.ndarray,
    baseline_representation: str,
) -> list[dict[str, Any]]:
    rows = []
    for fold in sorted(np.unique(fold_assignments)):
        test_mask = fold_assignments == fold
        each_pred = each_true[test_mask] + e_base[test_mask] - y[test_mask]
        for df_row, pred_each in zip(
            data.loc[test_mask].itertuples(index=False),
            each_pred,
            strict=False,
        ):
            rows.append(
                {
                    "material_id": df_row.material_id,
                    "formula": df_row.formula,
                    "fold": int(fold),
                    "representation": baseline_representation,
                    "frontend": FRONTEND_NAME,
                    "prototype_group": df_row.prototype_group,
                    "wyckoff_spglib_initial_structure": df_row.wyckoff_spglib_initial_structure,
                    "e_form_true": float(df_row.label),
                    "e_form_base": float(df_row.E_base),
                    "e_form_pred": float(df_row.E_base),
                    "residual_true": float(df_row.residual),
                    "residual_hat": 0.0,
                    "gamma": 0.0,
                    "alpha": math.nan,
                    "inner_alpha": math.nan,
                    "e_above_hull_true": float(df_row.e_above_hull_true),
                    "e_above_hull_pred": float(pred_each),
                }
            )
    return rows

def main() -> None:
    args = parse_args()
    baseline_model = normalize_baseline_model(args.baseline_model)
    baseline_representation = baseline_representation_name(baseline_model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Baseline model: {baseline_model}")
    print(f"Residual route: {RESIDUAL_ROUTE}")

    summary_cols = [
        "material_id",
        "formula",
        args.label_col,
        args.each_col,
        args.group_col,
    ]
    summary = pd.read_csv(args.summary, usecols=summary_cols)
    summary["material_id"] = summary["material_id"].astype(str)

    preds = pd.read_csv(args.predictions)
    preds["material_id"] = preds["material_id"].astype(str)
    merged = summary.merge(preds, on="material_id", how="inner", suffixes=("_summary", "_pred"))
    print(f"Summary rows: {len(summary)}")
    print(f"Prediction rows: {len(preds)}")
    print(f"Merged rows before cleanup: {len(merged)}")
    print(f"Frontend variant: {FRONTEND_NAME}")
    print("Embedding projection: disabled for precomputed embedding tables")

    if len(summary) != args.expected_total:
        print(
            "WARNING: unexpected summary row count. "
            f"Expected {args.expected_total}, got {len(summary)}."
        )
    if len(preds) != args.expected_total:
        print(
            "WARNING: unexpected prediction row count. "
            f"Expected {args.expected_total}, got {len(preds)}."
        )

    if "formula_summary" in merged.columns:
        formula_mismatch = (
            merged["formula_pred"].notna()
            & merged["formula_summary"].notna()
            & (merged["formula_summary"].astype(str) != merged["formula_pred"].astype(str))
        )
        mismatch_count = int(formula_mismatch.sum())
        print(f"Formula mismatches between summary and prediction files: {mismatch_count}")
        merged["formula"] = merged["formula_summary"].astype(str)
        merged = merged.drop(columns=["formula_summary", "formula_pred"])
    else:
        merged["formula"] = merged["formula"].astype(str)

    merged = merged.rename(
        columns={
            args.label_col: "label",
            args.each_col: "e_above_hull_true",
            args.group_col: "wyckoff_spglib_initial_structure",
        }
    )
    eform_col = detect_eform_column(merged, baseline_model)
    merged = merged.dropna(subset=["label", "e_above_hull_true", eform_col]).copy()
    merged["label"] = merged["label"].astype(float)
    merged["e_above_hull_true"] = merged["e_above_hull_true"].astype(float)
    merged["E_base"] = merged[eform_col].astype(float)
    merged["residual"] = merged["label"] - merged["E_base"]
    canonicalized = merged["wyckoff_spglib_initial_structure"].map(canonicalize_prototype)
    merged["prototype_group"] = canonicalized.map(lambda pair: pair[0])
    fallback_count = int(canonicalized.map(lambda pair: pair[1]).sum())
    print(f"Merged rows after cleanup: {len(merged)}")
    print(f"Prototype canonicalization fallback count: {fallback_count}")
    print(f"Unique prototype groups: {merged['prototype_group'].nunique()}")

    formula_sequences = parse_formula_sequences(merged["formula"])
    fold_assignments = np.full(len(merged), -1, dtype=int)
    groups = merged["prototype_group"].astype(str).to_numpy()
    gkf = GroupKFold(n_splits=args.n_splits)
    for fold, (_, test_idx) in enumerate(gkf.split(np.zeros(len(merged)), groups=groups)):
        fold_assignments[test_idx] = fold
    if (fold_assignments < 0).any():
        raise RuntimeError("Not all samples received an outer fold assignment")
    merged["fold"] = fold_assignments

    assignment_path = output_dir / "prototype_groupkfold_full_assignments.csv"
    assignment_df = merged[
        [
            "material_id",
            "formula",
            "prototype_group",
            "wyckoff_spglib_initial_structure",
            "fold",
        ]
    ].copy()
    assignment_df = ensure_result_metadata(
        assignment_df,
        baseline_model=baseline_model,
        residual_route=RESIDUAL_ROUTE,
    )
    assignment_df.to_csv(assignment_path, index=False)
    print(f"Wrote fold assignments to {assignment_path}")

    for fold in range(args.n_splits):
        train_groups = set(merged.loc[merged["fold"] != fold, "prototype_group"])
        test_groups = set(merged.loc[merged["fold"] == fold, "prototype_group"])
        overlap = train_groups & test_groups
        print(
            f"Fold {fold}: train={int((merged['fold'] != fold).sum())} rows, "
            f"test={int((merged['fold'] == fold).sum())} rows, "
            f"train_groups={len(train_groups)}, test_groups={len(test_groups)}, "
            f"group_overlap={len(overlap)}"
        )
        if overlap:
            raise RuntimeError(f"Outer fold {fold} has group leakage: {sorted(list(overlap))[:10]}")

    y = merged["label"].to_numpy(dtype=float)
    e_base = merged["E_base"].to_numpy(dtype=float)
    residual = merged["residual"].to_numpy(dtype=float)
    each_true = merged["e_above_hull_true"].to_numpy(dtype=float)
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows = baseline_prediction_rows(
        merged,
        fold_assignments,
        y,
        e_base,
        each_true,
        baseline_representation=baseline_representation,
    )

    for fold in range(args.n_splits):
        train_mask = fold_assignments != fold
        test_mask = fold_assignments == fold
        for split_name, split_mask in (("train", train_mask), ("test", test_mask)):
            for subset_name, subset_mask in (("all", np.ones(len(merged), dtype=bool)),):
                row_mask = split_mask & subset_mask
                metrics_rows.append(
                    evaluate_split_subset(
                        fold=fold,
                        representation=baseline_representation,
                        split=split_name,
                        subset=subset_name,
                        row_mask=row_mask,
                        y=y,
                        e_form_pred=e_base,
                        e_base=e_base,
                        each_true=each_true,
                        gamma=0.0,
                        alpha=None,
                        inner_alpha=None,
                        inner_holdout_mae=None,
                        n_dropped_total=0,
                        n_total_representation=len(merged),
                        n_train_representation=int(train_mask.sum()),
                        n_test_representation=int(test_mask.sum()),
                        baseline_name=baseline_representation,
                    )
                )

    availability_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []

    embeddings_dir = Path(args.embeddings_dir)
    embedding_files = discover_embedding_files(embeddings_dir, args.embedding_glob)
    print(f"Discovered {len(embedding_files)} embedding table(s) under {embeddings_dir}")
    for representation, embedding_path in embedding_files:
        path_text = str(embedding_path)
        status = "found"
        if not embedding_path.is_file():
            availability_rows.append(
                {
                    "representation": representation,
                    "path": path_text,
                    "status": "missing",
                    "dimension": 0,
                    "projected_dimension": 0,
                    "projection_kind": "precomputed_embedding_no_projection",
                    "frontend": FRONTEND_NAME,
                    "n_elements": 0,
                    "n_total_samples": len(merged),
                    "n_usable_samples": 0,
                    "n_dropped_samples": len(merged),
                    "usable_fraction": 0.0,
                }
            )
            continue

        embedding = load_embedding(embedding_path)
        x, usable_idx, missing_counts, missing_examples, projection_summary = build_crabnet_frontend_features(
            representation,
            embedding,
            formula_sequences,
            merged["material_id"],
        )
        projection_rows.append(
            {
                "representation": projection_summary.representation,
                "raw_dimension": projection_summary.raw_dimension,
                "projected_dimension": projection_summary.projected_dimension,
                "projection_kind": projection_summary.projection_kind,
                "explained_variance_ratio_sum": projection_summary.explained_variance_ratio_sum,
                "projector_is_trained": projection_summary.projector_is_trained,
                "projector_scope": projection_summary.projector_scope,
                "frontend": FRONTEND_NAME,
            }
        )
        n_usable = len(usable_idx)
        n_dropped = len(merged) - n_usable
        availability_rows.append(
            {
                "representation": representation,
                "path": path_text,
                "status": status,
                "dimension": int(embedding.shape[1]),
                "projected_dimension": int(projection_summary.projected_dimension),
                "projection_kind": projection_summary.projection_kind,
                "frontend": FRONTEND_NAME,
                "n_elements": int(embedding.shape[0]),
                "n_total_samples": len(merged),
                "n_usable_samples": n_usable,
                "n_dropped_samples": n_dropped,
                "usable_fraction": n_usable / len(merged),
            }
        )
        for element, count in sorted(missing_counts.items()):
            missing_rows.append(
                {
                    "representation": representation,
                    "element": element,
                    "missing_count": count,
                    "example_material_ids": "|".join(missing_examples.get(element, [])),
                }
            )

        if n_usable == 0:
            print(f"{representation}: no usable samples after embedding filtering")
            continue

        usable_folds = fold_assignments[usable_idx]
        usable_groups = groups[usable_idx]
        usable_y = y[usable_idx]
        usable_e_base = e_base[usable_idx]
        usable_residual = residual[usable_idx]
        usable_each_true = each_true[usable_idx]
        usable_df = merged.iloc[usable_idx].reset_index(drop=True)

        print(
            f"{representation}: usable={n_usable}, dropped={n_dropped}, "
            f"raw_dim={embedding.shape[1]}, projected_dim={projection_summary.projected_dimension}, "
            f"projection={projection_summary.projection_kind}"
        )

        for fold in range(args.n_splits):
            train_local = np.flatnonzero(usable_folds != fold)
            test_local = np.flatnonzero(usable_folds == fold)
            if len(train_local) == 0 or len(test_local) == 0:
                raise RuntimeError(f"{representation} fold {fold} has empty train/test after filtering")

            train_groups = usable_groups[train_local]
            test_groups = usable_groups[test_local]
            overlap = set(train_groups) & set(test_groups)
            print(
                f"{representation} fold {fold}: train={len(train_local)}, test={len(test_local)}, "
                f"train_groups={len(set(train_groups))}, test_groups={len(set(test_groups))}, "
                f"group_overlap={len(overlap)}"
            )
            if overlap:
                raise RuntimeError(
                    f"{representation} fold {fold} has group leakage: {sorted(list(overlap))[:10]}"
                )

            inner_train_local, inner_holdout_local = make_inner_split(
                train_groups,
                random_state=args.random_state + 1000 + fold,
            )
            inner_train_groups = set(train_groups[inner_train_local])
            inner_holdout_groups = set(train_groups[inner_holdout_local])
            inner_overlap = inner_train_groups & inner_holdout_groups
            print(
                f"{representation} fold {fold}: inner_train={len(inner_train_local)}, "
                f"inner_holdout={len(inner_holdout_local)}, inner_group_overlap={len(inner_overlap)}"
            )
            if inner_overlap:
                raise RuntimeError(
                    f"{representation} fold {fold} inner split has group leakage: "
                    f"{sorted(list(inner_overlap))[:10]}"
                )

            model_inner, inner_alpha = fit_ridge_model(
                x[train_local][inner_train_local],
                usable_residual[train_local][inner_train_local],
            )
            holdout_residual_hat = model_inner.predict(x[train_local][inner_holdout_local])
            gamma, inner_holdout_mae = choose_gamma(
                usable_e_base[train_local][inner_holdout_local],
                usable_y[train_local][inner_holdout_local],
                holdout_residual_hat,
            )

            model_outer, alpha = fit_ridge_model(x[train_local], usable_residual[train_local])
            residual_hat_train = model_outer.predict(x[train_local])
            residual_hat_test = model_outer.predict(x[test_local])
            pred_full = usable_e_base.copy()
            pred_full[train_local] = usable_e_base[train_local] + gamma * residual_hat_train
            pred_full[test_local] = usable_e_base[test_local] + gamma * residual_hat_test

            for split_name, local_idx in (("train", train_local), ("test", test_local)):
                usable_row_mask = np.zeros(n_usable, dtype=bool)
                usable_row_mask[local_idx] = True
                for subset_name, subset_local_mask in (("all", np.ones(n_usable, dtype=bool)),):
                    local_mask = usable_row_mask & subset_local_mask
                    metrics_rows.append(
                        evaluate_split_subset(
                            fold=fold,
                            representation=representation,
                            split=split_name,
                            subset=subset_name,
                            row_mask=local_mask,
                            y=usable_y,
                            e_form_pred=pred_full,
                            e_base=usable_e_base,
                            each_true=usable_each_true,
                            gamma=float(gamma),
                            alpha=float(alpha),
                            inner_alpha=float(inner_alpha),
                            inner_holdout_mae=float(inner_holdout_mae),
                            n_dropped_total=n_dropped,
                            n_total_representation=n_usable,
                            n_train_representation=len(train_local),
                            n_test_representation=len(test_local),
                            baseline_name=baseline_representation,
                        )
                    )

            pred_test = pred_full[test_local]
            test_each_pred = usable_each_true[test_local] + pred_test - usable_y[test_local]
            for df_row, residual_hat, pred_e, pred_each in zip(
                usable_df.iloc[test_local].itertuples(index=False),
                residual_hat_test,
                pred_test,
                test_each_pred,
                strict=False,
            ):
                prediction_rows.append(
                    {
                        "material_id": df_row.material_id,
                        "formula": df_row.formula,
                        "fold": int(fold),
                        "representation": representation,
                        "frontend": FRONTEND_NAME,
                        "prototype_group": df_row.prototype_group,
                        "wyckoff_spglib_initial_structure": df_row.wyckoff_spglib_initial_structure,
                        "e_form_true": float(df_row.label),
                        "e_form_base": float(df_row.E_base),
                        "e_form_pred": float(pred_e),
                        "residual_true": float(df_row.residual),
                        "residual_hat": float(residual_hat),
                        "gamma": float(gamma),
                        "alpha": float(alpha),
                        "inner_alpha": float(inner_alpha),
                        "e_above_hull_true": float(df_row.e_above_hull_true),
                        "e_above_hull_pred": float(pred_each),
                    }
                )

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    availability_df = pd.DataFrame(availability_rows)
    missing_df = pd.DataFrame(missing_rows)
    projection_df = pd.DataFrame(projection_rows)
    if missing_df.empty:
        missing_df = pd.DataFrame(columns=["representation", "element", "missing_count", "example_material_ids"])
    if projection_df.empty:
        projection_df = pd.DataFrame(
            columns=[
                "representation",
                "raw_dimension",
                "projected_dimension",
                "projection_kind",
                "explained_variance_ratio_sum",
                "projector_is_trained",
                "projector_scope",
                "frontend",
            ]
        )
    metrics_df = ensure_result_metadata(
        metrics_df,
        baseline_model=baseline_model,
        residual_route=RESIDUAL_ROUTE,
        metrics_variant=METRICS_VARIANT,
    )
    predictions_df = ensure_result_metadata(
        predictions_df,
        baseline_model=baseline_model,
        residual_route=RESIDUAL_ROUTE,
    )
    if not predictions_df.empty and "split" not in predictions_df.columns:
        predictions_df.insert(4, "split", "test")
    availability_df = ensure_result_metadata(
        availability_df,
        baseline_model=baseline_model,
        residual_route=RESIDUAL_ROUTE,
    )
    missing_df = ensure_result_metadata(
        missing_df,
        baseline_model=baseline_model,
        residual_route=RESIDUAL_ROUTE,
    )
    projection_df = ensure_result_metadata(
        projection_df,
        baseline_model=baseline_model,
        residual_route=RESIDUAL_ROUTE,
    )

    metrics_path = output_dir / "residual_calibration_full_fold_metrics.csv"
    predictions_path = output_dir / "residual_calibration_full_predictions.csv.gz"
    availability_path = output_dir / "residual_calibration_full_embedding_availability.csv"
    missing_path = output_dir / "residual_calibration_full_missing_embedding_elements.csv"
    projection_path = output_dir / "residual_calibration_full_projection_summary.csv"
    summary_path = output_dir / "residual_calibration_full_summary.csv"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False, compression="gzip")
    availability_df.to_csv(availability_path, index=False)
    missing_df.to_csv(missing_path, index=False)
    projection_df.to_csv(projection_path, index=False)

    metric_cols = [
        "MAE",
        "RMSE",
        "R2",
        "delta_MAE_vs_baseline",
        "delta_RMSE_vs_baseline",
        "delta_R2_vs_baseline",
        "F1",
        "DAF",
        "delta_F1_vs_baseline",
        "delta_DAF_vs_baseline",
        "gamma",
        "alpha",
        "inner_alpha",
        "inner_holdout_mae",
        "n_rows",
    ]
    summary_df = (
        metrics_df.groupby(["representation", "split", "subset"], dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_df.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else str(col)
        for col in summary_df.columns
    ]
    summary_df = summary_df.rename(
        columns={
            "representation_": "representation",
            "split_": "split",
            "subset_": "subset",
        }
    )
    summary_df["rank_by_MAE"] = (
        summary_df.groupby(["split", "subset"])["MAE_mean"].rank(method="dense", ascending=True)
    )
    summary_df = summary_df.sort_values(["split", "subset", "rank_by_MAE", "representation"])
    summary_df = ensure_result_metadata(
        summary_df,
        baseline_model=baseline_model,
        residual_route=RESIDUAL_ROUTE,
        metrics_variant=METRICS_VARIANT,
    )
    summary_df.to_csv(summary_path, index=False)

    print(f"Wrote metrics to {metrics_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote predictions to {predictions_path}")
    print(f"Wrote embedding availability to {availability_path}")
    print(f"Wrote missing-element log to {missing_path}")
    print(f"Wrote projection summary to {projection_path}")


if __name__ == "__main__":
    main()
