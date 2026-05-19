#!/usr/bin/env python3
"""
run_covariance.py
-----------------
Prepare OneCovariance inputs and (optionally) run OneCovariance.

This script:
  1. Loads the masked galaxy catalog for the chosen sample.
  2. Loads the observed 2PACF result (.npz) for that sample.
  3. Calls src.covariance.prepare_onecovariance_inputs to write all
     input ASCII files (n(z), bias, npair) and a config.ini.
  4. Optionally invokes OneCovariance with that config.

Usage
-----
    # Prepare inputs only
    python scripts/run_covariance.py --sample irac_footprint_masked

    # Prepare inputs and run OneCovariance
    python scripts/run_covariance.py \\
        --sample irac_footprint_masked \\
        --run \\
        --onecovariance-master /path/to/OneCovariance/config.ini

    # Point to a specific correlation result
    python scripts/run_covariance.py \\
        --sample irac_footprint_masked \\
        --corr-result results/correlation_functions/2pacf_irac_footprint_masked.npz

Prerequisites
-------------
- Run run_2pacf.py first to produce the 2PACF .npz file for the sample.
- OneCovariance must be installed if --run is used.
  See: https://github.com/rreischke/OneCovariance
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from astropy.table import Table

from src.config import cfg, resolve_path
from src.catalog import load_fits_catalog, get_radec, apply_sky_mask, filter_radec_range
from src.correlation import CorrelationResult
from src.covariance import prepare_onecovariance_inputs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sample_catalog(sample_name: str) -> Table:
    """Load and filter the galaxy catalog for the given sample.

    Mirrors the filtering logic in run_2pacf.py (flags + sky cut + mask).
    """
    if sample_name not in cfg["samples"]:
        available = list(cfg["samples"].keys())
        raise ValueError(f"Unknown sample '{sample_name}'. Available: {available}")

    sample_cfg  = cfg["samples"][sample_name]
    cols        = cfg["columns"]
    field_name  = sample_cfg["field"] if sample_cfg["field"] != "all" else "edf_s"
    irac_filter = sample_cfg.get("irac_filter", "none")
    survey      = cfg["fields"][field_name]

    print(f"\n{'='*60}")
    print(f"Sample: {sample_name}")
    print(f"  {sample_cfg['description']}")
    print(f"{'='*60}")

    table = load_fits_catalog(str(resolve_path(cfg["paths"]["raw_catalog"])))

    # IRAC filter
    if irac_filter == "footprint":
        sel = np.asarray(table[cols["irac_footprint"]], dtype=bool)
        table = table[sel]
    elif irac_filter == "detected":
        sel = np.asarray(table[cols["irac_detected"]], dtype=bool)
        table = table[sel]
    elif irac_filter == "no_footprint":
        sel = ~np.asarray(table[cols["irac_footprint"]], dtype=bool)
        table = table[sel]
    print(f"  After IRAC filter ({irac_filter}): {len(table):,} objects")

    # Sky cut
    table = filter_radec_range(
        table,
        ra_min=survey["ra_min"], ra_max=survey["ra_max"],
        dec_min=survey["dec_min"], dec_max=survey["dec_max"],
        ra_col=cols["ra"], dec_col=cols["dec"],
    )

    # HealSparse mask
    if sample_cfg.get("apply_mask", False):
        mask_rel = survey.get("mask_healsparse")
        if mask_rel:
            import healsparse
            mask_path = str(resolve_path(mask_rel))
            print(f"  Loading HealSparse mask: {mask_path}")
            mask_map = healsparse.HealSparseMap.read(mask_path)
            nside = mask_map.nside_sparse
            table = apply_sky_mask(
                table, mask_map, nside,
                ra_col=cols["ra"], dec_col=cols["dec"],
            )

    print(f"  Final sample size: {len(table):,} galaxies")
    return table


def run_onecovariance(config_path: str) -> None:
    """Invoke OneCovariance with the generated config file.

    Requires that the `covariance.py` script from the OneCovariance package
    is on the system PATH, or that OneCovariance is importable.

    Parameters
    ----------
    config_path : str
        Path to the generated config.ini.
    """
    print(f"\n{'='*60}")
    print("Running OneCovariance …")
    print(f"  Config: {config_path}")
    print(f"{'='*60}")

    # Try the command-line interface first
    cmd = ["python", "/home/k3vinpaul/OneCovariance/covariance.py", config_path]
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("OneCovariance completed successfully.")
    except FileNotFoundError:
        print(
            "Error: 'covariance.py' not found on PATH.\n"
            "Make sure OneCovariance is installed and the script is accessible.\n"
            f"You can also run manually:\n  python covariance.py {config_path}"
        )
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"OneCovariance returned a non-zero exit code: {exc.returncode}")
        sys.exit(exc.returncode)


def print_input_summary(out_dir: Path) -> None:
    """List generated input files for verification."""
    print(f"\nGenerated OneCovariance inputs in: {out_dir}")
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            size_kb = p.stat().st_size / 1024
            print(f"  {p.relative_to(out_dir)}  ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare OneCovariance inputs for a named sample",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sample", required=True,
        help="Sample name (must match a key in config/analysis.yml 'samples').",
    )
    parser.add_argument(
        "--corr-result", default=None,
        help="Path to observed 2PACF .npz. "
             "Default: results/correlation_functions/2pacf_{sample}.npz",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Directory for OneCovariance inputs and config. "
             "Default: results/covariance_matrices/{sample}/",
    )
    parser.add_argument(
        "--onecovariance-master", default="/home/k3vinpaul/OneCovariance/config.ini",
        help="Path to OneCovariance master config.ini.  "
             "If provided, HOD/halomodel sections are appended to the "
             "generated config so the strict parser does not fail.",
    )
    parser.add_argument(
        "--bias", type=float, default=None,
        help="Linear bias value b(z) = const.  "
             "Overrides bias.value in config/analysis.yml.",
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Run OneCovariance after preparing inputs (requires OneCovariance installed).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # --- Resolve paths ---
    corr_path = args.corr_result or str(
        resolve_path(cfg["paths"]["corr_dir"]) / f"2pacf_{args.sample}.npz"
    )
    out_dir = Path(args.out_dir) if args.out_dir else (
        resolve_path(cfg["paths"]["cov_dir"]) / args.sample
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load 2PACF result ---
    if not Path(corr_path).exists():
        print(
            f"Error: 2PACF result not found at {corr_path}\n"
            f"Run first:\n  python scripts/run_2pacf.py --sample {args.sample}"
        )
        sys.exit(1)
    print(f"\nLoading 2PACF result: {corr_path}")
    corr_result = CorrelationResult.load(corr_path)

    # --- Load galaxy catalog ---
    table = load_sample_catalog(args.sample)

    # Validate that the redshift column exists
    z_col = cfg["columns"]["redshift"]
    if z_col not in table.colnames:
        print(
            f"Error: redshift column '{z_col}' not found in catalog.\n"
            f"Available columns: {table.colnames}"
        )
        sys.exit(1)

    # --- Prepare OneCovariance inputs ---
    bias_val = args.bias or cfg["bias"]["value"]
    print(f"\nUsing linear bias b = {bias_val} (constant)")

    _sample_field = cfg["samples"][args.sample]["field"]
    _field_name   = _sample_field if _sample_field != "all" else "edf_s"
    _mask_rel     = cfg["fields"][_field_name].get("mask_healsparse")
    if not _mask_rel:
        raise ValueError(
            f"No HealSparse mask defined for field '{_field_name}'. "
            f"OneCovariance requires a mask."
        )

    config_path = prepare_onecovariance_inputs(
        galaxy_table=table,
        corr_result=corr_result,
        mask_path=str(resolve_path(_mask_rel)),
        out_dir=str(out_dir),
        onecovariance_master_config=args.onecovariance_master,
        bias_value=bias_val,
        nside_mask=1024,
        z_col=z_col,
        ra_col=cfg["columns"]["ra"],
        dec_col=cfg["columns"]["dec"],
    )

    # --- Summary ---
    print_input_summary(out_dir)

    print(f"\nNext step: run OneCovariance with:")
    print(f"  python covariance.py {config_path}")

    # --- Optionally run OneCovariance ---
    if args.run:
        run_onecovariance(config_path)

        # Check for output files
        output_dir = out_dir / "output"
        if output_dir.exists():
            outputs = list(output_dir.iterdir())
            if outputs:
                print(f"\nOneCovariance output files:")
                for f in sorted(outputs):
                    print(f"  {f}")
            else:
                print("\nWarning: OneCovariance output directory is empty.")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f} s")


if __name__ == "__main__":
    main()
