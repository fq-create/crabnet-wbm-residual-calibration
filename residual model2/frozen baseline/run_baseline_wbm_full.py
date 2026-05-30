#!/usr/bin/env python
"""Run configurable baseline-model predictions for the full computed WBM OOD set."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from monty.serialization import loadfn
from pymatgen.core import Structure
from tqdm import tqdm

from baseline_support import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STRUCTURES,
    DEFAULT_SUMMARY,
    WBM_STRUCTURE_TAG,
    get_baseline_spec,
    maybe_add_baseline_column,
    normalize_baseline_model,
)


RAW_PREDICTIONS_DIRNAME = "raw_predictions"
CHUNK_PREDICTIONS_DIRNAME = "chunks"


@dataclass(frozen=True)
class EnergyPrediction:
    raw_energy: float
    energy_total: float
    energy_per_atom: float
    is_per_atom: bool


def parse_args(default_baseline_model: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-model",
        default=default_baseline_model or "chgnet",
        help="Baseline model key.",
    )
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument(
        "--structures",
        default=str(DEFAULT_STRUCTURES),
        help="WBM computed-structure-entry payload used for baseline raw-energy prediction.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for per-model raw-prediction outputs.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional merged raw-prediction CSV path. Defaults to output-root/<model>/raw_predictions/.",
    )
    parser.add_argument(
        "--chunk-dir",
        default=None,
        help="Optional chunk output directory. Defaults to output-root/<model>/chunks/raw_predictions/.",
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--expected-total", type=int, default=256_963)
    parser.add_argument(
        "--chunk-offset",
        type=int,
        default=0,
        help="Only run chunks where chunk_idx %% chunk_stride equals this offset.",
    )
    parser.add_argument(
        "--chunk-stride",
        type=int,
        default=1,
        help="Stride for sharding chunk prediction across multiple workers.",
    )
    parser.add_argument(
        "--skip-assemble",
        action="store_true",
        help="Write chunk files only. Use a later non-sharded run to assemble the merged CSV.",
    )
    parser.add_argument("--overwrite-chunks", action="store_true")
    parser.add_argument("--rebuild-merged", action="store_true")
    return parser.parse_args()


def coerce_structure(obj: Any) -> Structure:
    if isinstance(obj, Structure):
        return obj
    structure = getattr(obj, "structure", None)
    if isinstance(structure, Structure):
        return structure
    if isinstance(obj, dict):
        if "structure" in obj:
            return coerce_structure(obj["structure"])
        return Structure.from_dict(obj)
    raise TypeError(f"Cannot convert object of type {type(obj)} to Structure")


def load_structure_payload(path: Path) -> tuple[dict[str, int], dict[int, str], dict[int, Any], str]:
    data = loadfn(path)
    required = {"material_id", "formula_from_cse"}
    missing = required.difference(data)
    if missing:
        raise KeyError(f"Structure file is missing required columns: {sorted(missing)}")

    material_ids = data["material_id"]
    formulas = data["formula_from_cse"]
    if "computed_structure_entry" not in data:
        raise KeyError(
            "Structure file must contain 'computed_structure_entry' for computed WBM prediction."
        )
    structures = data["computed_structure_entry"]
    structure_source = "computed_structure_entry.structure"
    id_to_idx = {str(material_ids[idx]): idx for idx in material_ids}
    return id_to_idx, formulas, structures, structure_source


def model_output_paths(output_root: Path, baseline_model: str) -> tuple[Path, Path]:
    model_name = normalize_baseline_model(baseline_model)
    model_root = output_root / model_name
    raw_output = (
        model_root
        / RAW_PREDICTIONS_DIRNAME
        / f"{model_name}_wbm_{WBM_STRUCTURE_TAG}_full_raw.csv"
    )
    chunk_dir = model_root / CHUNK_PREDICTIONS_DIRNAME / RAW_PREDICTIONS_DIRNAME
    return raw_output, chunk_dir


def ensure_all_model_output_dirs(output_root: Path) -> None:
    for baseline_model in sorted(PREDICTOR_CLASSES):
        raw_output, chunk_dir = model_output_paths(output_root, baseline_model)
        raw_output.parent.mkdir(parents=True, exist_ok=True)
        chunk_dir.mkdir(parents=True, exist_ok=True)


def chunk_output_path(chunk_dir: Path, chunk_idx: int, start: int, stop: int) -> Path:
    return chunk_dir / f"chunk_{chunk_idx:04d}_{start:06d}_{stop - 1:06d}.csv"


def is_complete_chunk(path: Path, expected_rows: int) -> bool:
    if not path.is_file():
        return False
    try:
        df = pd.read_csv(path, usecols=["material_id"])
    except Exception as exc:
        print(f"Chunk read failed for {path}: {exc}")
        return False
    return len(df) == expected_rows


def assemble_chunks(chunk_paths: list[Path], output: Path, expected_rows: int) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in chunk_paths]
    merged = pd.concat(frames, ignore_index=True)
    if len(merged) != expected_rows:
        raise ValueError(f"Merged raw predictions have {len(merged)} rows, expected {expected_rows}")
    if merged["material_id"].duplicated().any():
        dupes = merged.loc[merged["material_id"].duplicated(), "material_id"].head(10).tolist()
        raise ValueError(f"Duplicate material IDs found in merged raw predictions: {dupes}")
    output.parent.mkdir(parents=True, exist_ok=True)
    maybe_add_baseline_column(merged, merged["baseline_model"].iloc[0]).to_csv(output, index=False)
    return merged


class BaselinePredictor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.baseline_model = normalize_baseline_model(args.baseline_model)
        self.spec = get_baseline_spec(self.baseline_model)
        self.model_name = args.model_name or self.spec.default_model_name
        self.device = args.device or self.spec.default_device

    def predict(self, structure: Structure) -> EnergyPrediction:
        raise NotImplementedError

    def predict_many(self, structures: list[Structure]) -> list[EnergyPrediction]:
        return [self.predict(structure) for structure in structures]


class CHGNetPredictor(BaselinePredictor):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        from chgnet.model.model import CHGNet
        import torch

        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("CUDA device:", torch.cuda.get_device_name(0))
        self.model = CHGNet.load(model_name=self.model_name)
        self.model.graph_converter.set_isolated_atom_response("warn")
        self.is_intensive = bool(getattr(self.model, "is_intensive", True))
        print("CHGNet isolated-atom handling:", getattr(self.model.graph_converter, "on_isolated_atoms", None))
        print("CHGNet is_intensive:", self.is_intensive)

    def predict(self, structure: Structure) -> EnergyPrediction:
        pred = self.model.predict_structure(structure, task="e")
        raw_energy = float(np.asarray(pred["e"]).item())
        n_atoms = len(structure)
        if self.is_intensive:
            energy_per_atom = raw_energy
            energy_total = raw_energy * n_atoms
        else:
            energy_total = raw_energy
            energy_per_atom = raw_energy / n_atoms
        return EnergyPrediction(
            raw_energy=raw_energy,
            energy_total=energy_total,
            energy_per_atom=energy_per_atom,
            is_per_atom=self.is_intensive,
        )


class ASECalculatorPredictor(BaselinePredictor):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        from pymatgen.io.ase import AseAtomsAdaptor

        self.adaptor = AseAtomsAdaptor()
        self.calculator = self.build_calculator()

    def build_calculator(self):
        raise NotImplementedError

    def predict(self, structure: Structure) -> EnergyPrediction:
        atoms = self.adaptor.get_atoms(structure)
        atoms.calc = self.calculator
        energy_total = float(atoms.get_potential_energy())
        n_atoms = len(structure)
        return EnergyPrediction(
            raw_energy=energy_total,
            energy_total=energy_total,
            energy_per_atom=energy_total / n_atoms,
            is_per_atom=False,
        )


class MacePredictor(ASECalculatorPredictor):
    def build_calculator(self):
        from mace.calculators import mace_mp

        print(f"Loading MACE calculator with model={self.model_name!r}, device={self.device!r}")
        return mace_mp(model=self.model_name, device=self.device or "")


class MatglPredictor(ASECalculatorPredictor):
    def build_calculator(self):
        import matgl
        from matgl.ext.ase import M3GNetCalculator

        print(f"Loading matgl model {self.model_name!r}")
        potential = matgl.load_model(self.model_name)
        return M3GNetCalculator(potential=potential)


class M3GNetOriginalPredictor(BaselinePredictor):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        from m3gnet.models import M3GNet

        if self.model_name:
            print(f"Loading original M3GNet model from {self.model_name!r}")
            self.model = M3GNet.load(self.model_name)
        else:
            print("Loading original M3GNet package default model")
            self.model = M3GNet.load()

    def predict(self, structure: Structure) -> EnergyPrediction:
        raw_energy = float(np.asarray(self.model.predict_structure(structure)).reshape(-1)[0])
        n_atoms = len(structure)
        return EnergyPrediction(
            raw_energy=raw_energy,
            energy_total=raw_energy,
            energy_per_atom=raw_energy / n_atoms,
            is_per_atom=False,
        )

    def predict_many(self, structures: list[Structure]) -> list[EnergyPrediction]:
        raw_energies = np.asarray(self.model.predict_structures(structures)).reshape(-1)
        predictions = []
        if len(raw_energies) != len(structures):
            raise ValueError(
                f"M3GNet returned {len(raw_energies)} energies for {len(structures)} structures"
            )
        for structure, raw_energy in zip(structures, raw_energies):
            energy_total = float(raw_energy)
            n_atoms = len(structure)
            predictions.append(
                EnergyPrediction(
                    raw_energy=energy_total,
                    energy_total=energy_total,
                    energy_per_atom=energy_total / n_atoms,
                    is_per_atom=False,
                )
            )
        return predictions


class MatterSimPredictor(ASECalculatorPredictor):
    def build_calculator(self):
        try:
            from mattersim.forcefield import MatterSimCalculator
        except Exception as exc:
            raise SystemExit(
                "MatterSim is not available in the current environment. "
                "Install the official MatterSim package and rerun with the corresponding conda env."
            ) from exc

        print(f"Loading MatterSim weights {self.model_name!r}")
        return MatterSimCalculator(load_path=self.model_name, device=self.device or "cuda")


class ORBPredictor(ASECalculatorPredictor):
    def build_calculator(self):
        from orb_models.forcefield.calculator import ORBCalculator
        import orb_models.forcefield.pretrained as orb_pretrained

        loader_name = self.model_name or "orb_v3_conservative_inf_mpa"
        loader = getattr(orb_pretrained, loader_name, None)
        if loader is None:
            available = sorted(name for name in dir(orb_pretrained) if name.startswith("orb_v3"))
            raise KeyError(f"Unknown ORB loader {loader_name!r}. Available examples: {available[:10]}")
        print(f"Loading ORB pretrained loader {loader_name!r}")
        model = loader(device=self.device, compile=False)
        return ORBCalculator(model, device=self.device)


class SevenNetPredictor(ASECalculatorPredictor):
    def build_calculator(self):
        from sevenn.sevennet_calculator import SevenNetCalculator

        print(f"Loading SevenNet model {self.model_name!r}")
        return SevenNetCalculator(model=self.model_name or "7net-0", device=self.device or "auto")


PREDICTOR_CLASSES = {
    "chgnet": CHGNetPredictor,
    "mace-mpa-0": MacePredictor,
    "m3gnet": M3GNetOriginalPredictor,
    "mattersim-v1-5m": MatterSimPredictor,
    "orb-v3": ORBPredictor,
    "sevennet-l3i5": SevenNetPredictor,
}


def build_predictor(args: argparse.Namespace) -> BaselinePredictor:
    baseline_model = normalize_baseline_model(args.baseline_model)
    predictor_cls = PREDICTOR_CLASSES[baseline_model]
    return predictor_cls(args)


def main(default_baseline_model: str | None = None) -> None:
    args = parse_args(default_baseline_model=default_baseline_model)
    baseline_model = normalize_baseline_model(args.baseline_model)
    baseline_spec = get_baseline_spec(baseline_model)
    if args.chunk_stride < 1:
        raise ValueError("--chunk-stride must be >= 1")
    if not 0 <= args.chunk_offset < args.chunk_stride:
        raise ValueError("--chunk-offset must satisfy 0 <= offset < chunk_stride")

    summary = pd.read_csv(args.summary, usecols=["material_id", "formula"])
    summary["material_id"] = summary["material_id"].astype(str)
    summary["formula"] = summary["formula"].astype(str)
    if args.limit is not None:
        summary = summary.head(args.limit).copy()

    requested_rows = len(summary)
    print(f"Baseline model: {baseline_model}")
    print(f"Requested WBM rows from summary: {requested_rows}")
    if args.limit is None and requested_rows != args.expected_total:
        print(
            "WARNING: unexpected summary row count. "
            f"Expected {args.expected_total}, got {requested_rows}."
        )

    output_root = Path(args.output_root)
    ensure_all_model_output_dirs(output_root)
    default_output, default_chunk_dir = model_output_paths(output_root, baseline_model)
    output = Path(args.output) if args.output else default_output
    chunk_dir = Path(args.chunk_dir) if args.chunk_dir else default_chunk_dir
    chunk_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = chunk_dir / "manifest.json"
    print(f"Output root: {output_root}")
    print(f"Raw prediction output: {output}")
    print(f"Chunk directory: {chunk_dir}")

    material_ids = summary["material_id"].tolist()
    summary_formula = dict(zip(summary["material_id"], summary["formula"]))

    print(f"Loading structure payload from {args.structures}")
    id_to_idx, formulas, structures, structure_source = load_structure_payload(Path(args.structures))
    print(f"Using structures from payload field: {structure_source}")
    missing_ids = [material_id for material_id in material_ids if material_id not in id_to_idx]
    if missing_ids:
        raise ValueError(f"Missing {len(missing_ids)} requested structures. First few: {missing_ids[:10]}")
    print(f"Structure lookup built for {len(id_to_idx)} materials")

    n_chunks = math.ceil(requested_rows / args.chunk_size)
    chunk_specs = []
    complete_rows = 0
    for chunk_idx in range(n_chunks):
        start = chunk_idx * args.chunk_size
        stop = min((chunk_idx + 1) * args.chunk_size, requested_rows)
        path = chunk_output_path(chunk_dir, chunk_idx, start, stop)
        expected_rows = stop - start
        complete = is_complete_chunk(path, expected_rows) and not args.overwrite_chunks
        if complete:
            complete_rows += expected_rows
        chunk_specs.append(
            {
                "chunk_idx": chunk_idx,
                "start": start,
                "stop": stop,
                "expected_rows": expected_rows,
                "path": path,
                "complete": complete,
            }
        )
    print(
        f"Chunk resume status: {sum(spec['complete'] for spec in chunk_specs)}/{n_chunks} "
        f"chunks complete, {complete_rows}/{requested_rows} rows already present"
    )
    assigned_specs = [
        spec
        for spec in chunk_specs
        if spec["chunk_idx"] % args.chunk_stride == args.chunk_offset
    ]
    assigned_rows = sum(spec["expected_rows"] for spec in assigned_specs)
    assigned_complete_rows = sum(
        spec["expected_rows"]
        for spec in assigned_specs
        if spec["complete"] and not args.overwrite_chunks
    )
    print(
        "Chunk shard: "
        f"offset={args.chunk_offset}, stride={args.chunk_stride}, "
        f"assigned_chunks={len(assigned_specs)}/{n_chunks}, "
        f"assigned_rows={assigned_rows}, assigned_complete_rows={assigned_complete_rows}"
    )

    pending_specs = [
        spec
        for spec in assigned_specs
        if not (spec["complete"] and not args.overwrite_chunks)
    ]
    predictor = build_predictor(args) if pending_specs else None
    progress = tqdm(
        total=assigned_rows,
        desc=f"{baseline_model} WBM full",
        initial=assigned_complete_rows,
    )
    predicted_rows_this_run = 0
    skipped_chunks = 0

    for spec in assigned_specs:
        if spec["complete"] and not args.overwrite_chunks:
            skipped_chunks += 1
            continue

        if predictor is None:
            raise RuntimeError("Internal error: missing predictor for incomplete chunks")

        start = spec["start"]
        stop = spec["stop"]
        chunk_ids = material_ids[start:stop]
        chunk_structures = [
            coerce_structure(structures[id_to_idx[material_id]])
            for material_id in chunk_ids
        ]
        predictions = predictor.predict_many(chunk_structures)
        if len(predictions) != len(chunk_ids):
            raise ValueError(
                f"Predictor returned {len(predictions)} rows for {len(chunk_ids)} requested structures"
            )
        rows = []
        for material_id, structure, pred in zip(chunk_ids, chunk_structures, predictions):
            n_atoms = len(structure)
            rows.append(
                {
                    "baseline_model": baseline_model,
                    "material_id": material_id,
                    "formula": summary_formula.get(material_id, str(formulas.get(id_to_idx[material_id], ""))),
                    "n_atoms": n_atoms,
                    "energy_raw": pred.raw_energy,
                    "energy_is_per_atom": pred.is_per_atom,
                    "energy_total": pred.energy_total,
                    "energy_per_atom": pred.energy_per_atom,
                }
            )
            progress.update(1)
            predicted_rows_this_run += 1

        chunk_df = pd.DataFrame(rows)
        if len(chunk_df) != spec["expected_rows"]:
            raise ValueError(
                f"Chunk {spec['chunk_idx']} produced {len(chunk_df)} rows, expected {spec['expected_rows']}"
            )
        tmp_path = spec["path"].with_suffix(".csv.tmp")
        chunk_df.to_csv(tmp_path, index=False)
        tmp_path.replace(spec["path"])
        print(f"Finished chunk {spec['chunk_idx'] + 1}/{n_chunks}: rows {start}:{stop} -> {spec['path']}")

    progress.close()

    chunk_paths = [Path(spec["path"]) for spec in chunk_specs]
    if args.skip_assemble:
        print("Skipping merged raw CSV assembly because --skip-assemble was set.")
    else:
        incomplete = [
            str(path)
            for path, spec in zip(chunk_paths, chunk_specs)
            if not is_complete_chunk(path, spec["expected_rows"])
        ]
        if incomplete:
            raise RuntimeError(
                f"Cannot assemble merged raw predictions because {len(incomplete)} chunks are incomplete. "
                f"First few: {incomplete[:5]}"
            )

        if output.is_file() and not args.rebuild_merged:
            existing = pd.read_csv(output, usecols=["material_id"])
            if len(existing) == requested_rows:
                print(f"Final raw output already exists with {len(existing)} rows: {output}")
            else:
                print(f"Existing final raw output has {len(existing)} rows, rebuilding merged file: {output}")
                assemble_chunks(chunk_paths, output, requested_rows)
        else:
            merged = assemble_chunks(chunk_paths, output, requested_rows)
            print(f"Wrote {len(merged)} rows to {output}")

    manifest = {
        "baseline_model": baseline_model,
        "summary": str(args.summary),
        "structures": str(args.structures),
        "structure_source": structure_source,
        "structure_tag": WBM_STRUCTURE_TAG,
        "output_root": str(output_root),
        "output": str(output),
        "chunk_dir": str(chunk_dir),
        "model_name": predictor.model_name if predictor else (args.model_name or baseline_spec.default_model_name),
        "device": predictor.device if predictor else (args.device or baseline_spec.default_device),
        "chunk_size": args.chunk_size,
        "chunk_offset": args.chunk_offset,
        "chunk_stride": args.chunk_stride,
        "skip_assemble": args.skip_assemble,
        "requested_rows": requested_rows,
        "assigned_rows": assigned_rows,
        "predicted_rows_this_run": predicted_rows_this_run,
        "skipped_chunks": skipped_chunks,
        "n_chunks": n_chunks,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
