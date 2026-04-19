#!/usr/bin/env python3
"""Compute DEG weights for all or specified datasets.

Generates cached DEG weight CSVs required by the benchmarking pipeline
(when using deg_weight_source='vscontrol' or 'synthetic') and by the
perturbation discrimination analyses.

Usage:
    # Both modes for all paper datasets
    python scripts/compute_deg_weights.py \
        --datasets adamson16 frangieh21 nadig25hepg2 norman19 replogle20 replogle22k562 wessels23 \
        --mode both

    # Control mode only for a single dataset
    python scripts/compute_deg_weights.py --datasets wessels23 --mode control

    # Force recomputation
    python scripts/compute_deg_weights.py --datasets wessels23 --mode both --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import scanpy as sc
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyses.perturbation_discrimination import deg_scanpy


def load_dataset_path(dataset_name: str) -> Path:
    config_path = PROJECT_ROOT / "cellsimbench" / "configs" / "dataset" / f"{dataset_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r") as handle:
        config = yaml.safe_load(handle)
    data_path = config.get("data_path")
    if not data_path:
        raise ValueError(f"No data_path in {config_path}")
    resolved = (PROJECT_ROOT / data_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset file not found: {resolved}")
    return resolved


def compute_for_dataset(dataset_name: str, modes: list[str], force: bool) -> None:
    results_root = PROJECT_ROOT / "analyses" / "perturbation_discrimination" / "results"
    h5ad_path = load_dataset_path(dataset_name)
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Path:    {h5ad_path}")
    print(f"  Modes:   {', '.join(modes)}")
    print(f"{'='*60}")

    print(f"  Loading {h5ad_path} ...")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  Loaded {adata.n_obs} cells, {adata.n_vars} genes")

    for mode in modes:
        t0 = time.time()
        csv_path = deg_scanpy.compute_deg_cache(
            adata=adata,
            dataset_name=dataset_name,
            results_root=results_root,
            perturbation_key="condition",
            mode=mode,
            min_cells=4,
            force=force,
        )
        elapsed = time.time() - t0
        print(f"  [{mode}] Done in {elapsed:.1f}s -> {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute DEG weights for benchmarking and discrimination analyses."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset names (matching configs in cellsimbench/configs/dataset/)",
    )
    parser.add_argument(
        "--mode",
        choices=["control", "synthetic", "both"],
        default="both",
        help="DEG mode: 'control', 'synthetic', or 'both' (default: both)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if cached results exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = ["control", "synthetic"] if args.mode == "both" else [args.mode]

    print(f"Computing DEG weights for {len(args.datasets)} dataset(s), mode(s): {', '.join(modes)}")

    for dataset_name in args.datasets:
        try:
            compute_for_dataset(dataset_name, modes, args.force)
        except Exception as e:
            print(f"\n  ERROR processing {dataset_name}: {e}", file=sys.stderr)
            continue

    print("\nAll done.")


if __name__ == "__main__":
    main()
