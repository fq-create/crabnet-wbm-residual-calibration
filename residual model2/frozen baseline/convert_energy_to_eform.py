#!/usr/bin/env python
"""Convert baseline total energies to MP2020-style formation energy per atom."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from monty.serialization import loadfn

from baseline_support import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STRUCTURES,
    DEFAULT_SUMMARY,
    default_eform_path,
    default_raw_prediction_path,
    detect_energy_columns,
    get_baseline_spec,
    normalize_baseline_model,
)


def parse_args(default_baseline_model: str | None = None) -> argparse.Namespace:
    baseline_model = normalize_baseline_model(default_baseline_model or "chgnet")
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-model", default=baseline_model)
    parser.add_argument("--input", default=str(default_raw_prediction_path(DEFAULT_OUTPUT_ROOT, baseline_model)))
    parser.add_argument("--output", default=str(default_eform_path(DEFAULT_OUTPUT_ROOT, baseline_model)))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--computed-structures", default=str(DEFAULT_STRUCTURES))
    return parser.parse_args()


def _computed_structure_entries_by_id(path: str | Path) -> dict[str, object]:
    from pymatgen.entries.computed_entries import ComputedStructureEntry

    payload = loadfn(path)
    material_ids = payload["material_id"]
    cse_payloads = payload["computed_structure_entry"]
    entries = {}
    for idx in material_ids:
        material_id = str(material_ids[idx])
        cse_obj = cse_payloads[idx]
        if isinstance(cse_obj, ComputedStructureEntry):
            entries[material_id] = cse_obj
        else:
            entries[material_id] = ComputedStructureEntry.from_dict(cse_obj)
    return entries


def _apply_structure_dependent_corrections(df: pd.DataFrame, args: argparse.Namespace) -> list[float]:
    from pymatgen.core import Structure
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    from matbench_discovery.energy import get_e_form_per_atom

    structure_col = "relaxed_structure"
    if structure_col not in df.columns:
        raise KeyError(
            f"{structure_col!r} column is required for structure-dependent MP2020 "
            "corrections. Rerun the benchmark relaxer prediction step."
        )
    if df[structure_col].isna().any():
        missing_ids = df.loc[df[structure_col].isna(), "material_id"].head(10).tolist()
        raise ValueError(
            "Some predictions are missing relaxed structures needed for MP2020 "
            f"corrections. First few material IDs: {missing_ids}"
        )

    base_entries = _computed_structure_entries_by_id(args.computed_structures)
    entries = []
    for row in df[["material_id", "energy_total_for_eform", structure_col]].itertuples(index=False):
        material_id = str(row.material_id)
        if material_id not in base_entries:
            raise KeyError(f"Missing computed-structure-entry for {material_id!r}")
        cse = base_entries[material_id].copy()
        structure_dict = json.loads(row.relaxed_structure)
        cse._energy = float(row.energy_total_for_eform)  # noqa: SLF001
        cse._structure = Structure.from_dict(structure_dict)  # noqa: SLF001
        entries.append(cse)

    processed = MaterialsProject2020Compatibility().process_entries(
        entries,
        verbose=True,
        clean=True,
    )
    if len(processed) != len(entries):
        raise ValueError(
            "MP2020 compatibility removed some entries during processing: "
            f"{len(processed)} kept vs {len(entries)} input."
        )
    return [get_e_form_per_atom(entry) for entry in entries]


def main(default_baseline_model: str | None = None) -> None:
    args = parse_args(default_baseline_model=default_baseline_model)
    baseline_model = normalize_baseline_model(args.baseline_model)
    spec = get_baseline_spec(baseline_model)
    try:
        from matbench_discovery.energy import get_e_form_per_atom
    except Exception as exc:
        raise SystemExit(
            "matbench_discovery.energy.get_e_form_per_atom is unavailable. "
            "Stopping to avoid using mismatched elemental reference energies."
        ) from exc

    df = pd.read_csv(args.input)
    energy_cols = detect_energy_columns(df, baseline_model)
    required = {"material_id", "formula", "n_atoms", *energy_cols.values()}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in raw prediction file: {sorted(missing)}")

    df["material_id"] = df["material_id"].astype(str)
    df["energy_total_for_eform"] = df[energy_cols["total"]].astype(float)
    df["e_correction_per_atom_mp2020"] = 0.0
    if spec.apply_mp2020_correction_to_model_energy and spec.apply_structure_dependent_mp2020_correction:
        raise ValueError(f"{baseline_model} cannot use both scalar and structure-dependent MP2020 corrections.")
    if spec.apply_mp2020_correction_to_model_energy:
        summary = pd.read_csv(
            args.summary,
            usecols=["material_id", "n_sites", "e_correction_per_atom_mp2020"],
        )
        summary["material_id"] = summary["material_id"].astype(str)
        df = df.merge(summary, on="material_id", how="left", validate="one_to_one")
        missing_summary = df["e_correction_per_atom_mp2020_y"].isna()
        if missing_summary.any():
            missing_ids = df.loc[missing_summary, "material_id"].head(10).tolist()
            raise ValueError(
                f"Missing MP2020 correction rows for {int(missing_summary.sum())} predictions. "
                f"First few material IDs: {missing_ids}"
            )
        site_mismatch = df["n_sites"].astype(int) != df["n_atoms"].astype(int)
        if site_mismatch.any():
            mismatch_ids = df.loc[site_mismatch, "material_id"].head(10).tolist()
            raise ValueError(
                f"n_sites and n_atoms differ for {int(site_mismatch.sum())} predictions. "
                f"First few material IDs: {mismatch_ids}"
            )
        df["e_correction_per_atom_mp2020"] = df["e_correction_per_atom_mp2020_y"].astype(float)
        df["energy_total_for_eform"] = (
            df[energy_cols["total"]].astype(float)
            + df["e_correction_per_atom_mp2020"] * df["n_atoms"].astype(float)
        )
        df = df.drop(columns=["n_sites", "e_correction_per_atom_mp2020_x", "e_correction_per_atom_mp2020_y"])

    baseline_series = (
        df["baseline_model"].astype(str)
        if "baseline_model" in df.columns
        else pd.Series([baseline_model] * len(df), dtype="string")
    )
    eform_inputs = df[["formula", "energy_total_for_eform"]].rename(
        columns={"energy_total_for_eform": "energy_total"}
    )
    if spec.apply_structure_dependent_mp2020_correction:
        df["e_form_per_atom"] = _apply_structure_dependent_corrections(df, args)
    else:
        df["e_form_per_atom"] = [
            get_e_form_per_atom({"energy": row.energy_total, "composition": row.formula})
            for row in eform_inputs.itertuples(index=False)
        ]
    out = pd.DataFrame(
        {
            "baseline_model": baseline_series.astype(str),
            "material_id": df["material_id"].astype(str),
            "formula": df["formula"].astype(str),
            "n_atoms": df["n_atoms"],
            "energy_total": df[energy_cols["total"]],
            "energy_per_atom": df[energy_cols["per_atom"]],
            "e_correction_per_atom_mp2020": df["e_correction_per_atom_mp2020"],
            "energy_total_for_eform": df["energy_total_for_eform"],
            "e_form_per_atom": df["e_form_per_atom"],
        }
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}")
    print(out.head().to_string(index=False))


if __name__ == "__main__":
    main()
