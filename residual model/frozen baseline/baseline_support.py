#!/usr/bin/env python
"""Shared helpers for frozen-baseline WBM workflows under residual model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


FROZEN_BASELINE_ROOT = Path(__file__).resolve().parent
RESIDUAL_MODEL_ROOT = FROZEN_BASELINE_ROOT.parent
DEFAULT_DATA_ROOT = RESIDUAL_MODEL_ROOT / "data"
DEFAULT_SUMMARY = DEFAULT_DATA_ROOT / "wbm_summary.csv.gz"
DEFAULT_STRUCTURES = DEFAULT_DATA_ROOT / "2022-10-19-wbm-computed-structure-entries.json"
DEFAULT_EMBEDDINGS_DIR = RESIDUAL_MODEL_ROOT / "embeddings"
DEFAULT_OUTPUT_ROOT = FROZEN_BASELINE_ROOT / "outputs"
DEFAULT_RESIDUAL_OUTPUT_ROOT = FROZEN_BASELINE_ROOT / "outputs_residual"
WBM_STRUCTURE_TAG = "computed"

CRABNET_FRONTEND_ROUTE = "crabnet_frontend"
FULL_METRICS_VARIANT = "full"

GENERIC_DELTA_COLUMN_MAP = {
    "delta_MAE_vs_CHGNet": "delta_MAE_vs_baseline",
    "delta_RMSE_vs_CHGNet": "delta_RMSE_vs_baseline",
    "delta_R2_vs_CHGNet": "delta_R2_vs_baseline",
    "delta_F1_vs_CHGNet": "delta_F1_vs_baseline",
    "delta_DAF_vs_CHGNet": "delta_DAF_vs_baseline",
}


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    default_predict_env: str | None
    default_analysis_env: str | None
    default_model_name: str | None
    default_device: str | None = None
    legacy_energy_prefix: str | None = None
    apply_mp2020_correction_to_model_energy: bool = False
    apply_structure_dependent_mp2020_correction: bool = False


BASELINE_SPECS: dict[str, BaselineSpec] = {
    "chgnet": BaselineSpec(
        name="chgnet",
        default_predict_env="CHGNET",
        default_analysis_env="matbench-discovery",
        default_model_name="0.3.0",
        default_device="cuda",
        legacy_energy_prefix="chgnet",
    ),
    "mace-mpa-0": BaselineSpec(
        name="mace-mpa-0",
        default_predict_env="mace",
        default_analysis_env="matbench-discovery",
        default_model_name="medium-mpa-0",
        default_device="cuda",
        apply_mp2020_correction_to_model_energy=True,
    ),
    "m3gnet": BaselineSpec(
        name="m3gnet",
        default_predict_env="SLICES",
        default_analysis_env="matbench-discovery",
        default_model_name=None,
        default_device=None,
        apply_mp2020_correction_to_model_energy=True,
    ),
    "mattersim-v1-5m": BaselineSpec(
        name="mattersim-v1-5m",
        default_predict_env="mattersim",
        default_analysis_env="matbench-discovery",
        default_model_name="MatterSim-v1.0.0-5M.pth",
        default_device="cuda",
        apply_mp2020_correction_to_model_energy=True,
    ),
    "orb-v3": BaselineSpec(
        name="orb-v3",
        default_predict_env="orbital",
        default_analysis_env="matbench-discovery",
        default_model_name="orb_v3_conservative_inf_mpa",
        default_device="cuda",
        apply_mp2020_correction_to_model_energy=True,
    ),
    "sevennet-l3i5": BaselineSpec(
        name="sevennet-l3i5",
        default_predict_env="sevennet",
        default_analysis_env="matbench-discovery",
        default_model_name="sevennet-l3i5",
        default_device="auto",
        apply_mp2020_correction_to_model_energy=True,
    ),
}

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


def normalize_baseline_model(name: str) -> str:
    key = str(name).strip().lower()
    normalized = BASELINE_ALIASES.get(key, key)
    if normalized not in BASELINE_SPECS:
        supported = ", ".join(sorted(BASELINE_SPECS))
        raise KeyError(f"Unsupported baseline model {name!r}. Supported values: {supported}")
    return normalized


def get_baseline_spec(name: str) -> BaselineSpec:
    return BASELINE_SPECS[normalize_baseline_model(name)]


def baseline_representation_name(baseline_model: str) -> str:
    return f"{normalize_baseline_model(baseline_model)}-baseline"


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


def add_generic_delta_aliases(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    for legacy_name, generic_name in GENERIC_DELTA_COLUMN_MAP.items():
        if legacy_name in enriched.columns and generic_name not in enriched.columns:
            enriched[generic_name] = enriched[legacy_name]
    return enriched


def detect_energy_columns(df: pd.DataFrame, baseline_model: str) -> dict[str, str]:
    candidates: list[dict[str, str]] = []
    prefix = get_baseline_spec(baseline_model).legacy_energy_prefix
    if prefix:
        candidates.append(
            {
                "raw": f"{prefix}_energy_raw",
                "is_per_atom": f"{prefix}_energy_is_per_atom",
                "total": f"{prefix}_energy_total",
                "per_atom": f"{prefix}_energy_per_atom",
            }
        )
    candidates.append(
        {
            "raw": "energy_raw",
            "is_per_atom": "energy_is_per_atom",
            "total": "energy_total",
            "per_atom": "energy_per_atom",
        }
    )
    for column_map in candidates:
        if all(col in df.columns for col in column_map.values()):
            return column_map
    raise KeyError(
        "Could not locate raw energy columns. "
        f"Available columns: {sorted(df.columns.tolist())}"
    )


def detect_eform_column(df: pd.DataFrame, baseline_model: str) -> str:
    prefix = get_baseline_spec(baseline_model).legacy_energy_prefix
    candidates = []
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


def model_output_dirs(
    *,
    output_root: Path,
    residual_output_root: Path,
    baseline_model: str,
) -> dict[str, Path]:
    model_name = normalize_baseline_model(baseline_model)
    model_root = output_root / model_name
    residual_root = residual_output_root / model_name
    return {
        "model_root": model_root,
        "raw_predictions": model_root / "raw_predictions",
        "eform": model_root / "eform",
        "logs": model_root / "logs",
        "chunks": model_root / "chunks",
        "residual_root": residual_root,
        "route_crabnet_frontend": residual_root / CRABNET_FRONTEND_ROUTE,
    }


def default_raw_prediction_path(output_root: Path, baseline_model: str) -> Path:
    dirs = model_output_dirs(
        output_root=output_root,
        residual_output_root=DEFAULT_RESIDUAL_OUTPUT_ROOT,
        baseline_model=baseline_model,
    )
    model_name = normalize_baseline_model(baseline_model)
    return dirs["raw_predictions"] / f"{model_name}_wbm_{WBM_STRUCTURE_TAG}_full_raw.csv"


def default_eform_path(output_root: Path, baseline_model: str) -> Path:
    dirs = model_output_dirs(
        output_root=output_root,
        residual_output_root=DEFAULT_RESIDUAL_OUTPUT_ROOT,
        baseline_model=baseline_model,
    )
    model_name = normalize_baseline_model(baseline_model)
    return dirs["eform"] / f"{model_name}_wbm_{WBM_STRUCTURE_TAG}_full.csv"


def maybe_add_baseline_column(df: pd.DataFrame, baseline_model: str) -> pd.DataFrame:
    enriched = df.copy()
    if "baseline_model" not in enriched.columns:
        enriched.insert(0, "baseline_model", normalize_baseline_model(baseline_model))
    return enriched


def write_frame(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)
