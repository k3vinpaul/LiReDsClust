#!/usr/bin/env python3
"""
run_2pacf.py
------------
Compute the two-point angular correlation function for a named sample.

Usage
-----
    # Single-field samples
    python scripts/run_2pacf.py --sample edf_s_irac_footprint
    python scripts/run_2pacf.py --sample edf_s_irac_detected
    python scripts/run_2pacf.py --sample edf_fornax_irac_footprint
    python scripts/run_2pacf.py --sample edf_n_all

    # All fields combined
    python scripts/run_2pacf.py --sample all_irac_footprint

    # Force regenerate randoms
    python scripts/run_2pacf.py --sample edf_s_irac_footprint --regen-randoms

    # Skip figures (batch / headless)
    python scripts/run_2pacf.py --sample edf_s_irac_footprint --no-plots

Samples are defined in config/analysis.yml.  Results are saved to
results/correlation_functions/ and figures to results/figures/.
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from astropy.table import Table, vstack

from src.config import cfg, resolve_path
from src import (
    load_fits_catalog,
    get_radec,
    apply_sky_mask,
    filter_radec_range,
    generate_random_catalog,
    compute_2pacf,
    plot_2pacf,
    plot_sky_distribution,
    plot_separation_histogram,
    plot_data_vs_randoms_density,
)

# Keys of all sub-fields used when field == "all"
_ALL_SUBFIELDS = ["edf_s", "edf_fornax", "edf_n"]


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------

def _apply_irac_filter(table: Table, irac_filter: str) -> Table:
    """Restrict catalog rows according to the IRAC filter setting."""
    cols = cfg["columns"]
    if irac_filter == "footprint":
        sel = np.asarray(table[cols["irac_footprint"]], dtype=bool)
        table = table[sel]
        print(f"  After IRAC-footprint=True: {len(table):,} objects")
    elif irac_filter == "detected":
        sel = np.asarray(table[cols["irac_detected"]], dtype=bool)
        table = table[sel]
        print(f"  After IRAC-detected=True: {len(table):,} objects")
    elif irac_filter == "no_footprint":
        sel = ~np.asarray(table[cols["irac_footprint"]], dtype=bool)
        table = table[sel]
        print(f"  After IRAC-footprint=False: {len(table):,} objects")
    # else: "none" — keep everything
    return table


def _filter_to_field(
    table: Table,
    field_name: str,
    apply_mask: bool,
) -> Table:
    """Restrict *table* to a single field's bounding box and (optionally) mask."""
    field_cfg = cfg["fields"][field_name]
    cols = cfg["columns"]

    # Bounding box
    t = filter_radec_range(
        table,
        ra_min=field_cfg["ra_min"], ra_max=field_cfg["ra_max"],
        dec_min=field_cfg["dec_min"], dec_max=field_cfg["dec_max"],
        ra_col=cols["ra"], dec_col=cols["dec"],
    )

    # HealSparse mask
    if apply_mask:
        mask_rel = field_cfg.get("mask_healsparse")
        if mask_rel:
            import healsparse
            mask_path = str(resolve_path(mask_rel))
            print(f"  Loading HealSparse mask: {mask_path}")
            mask_map = healsparse.HealSparseMap.read(mask_path)
            nside = mask_map.nside_sparse
            t = apply_sky_mask(t, mask_map, nside, ra_col=cols["ra"], dec_col=cols["dec"])
        else:
            print(f"  [Note] No mask defined for {field_name} — skipping mask")

    return t


def load_sample(sample_name: str) -> tuple:
    """Load and filter the catalog according to the sample definition in config.

    For field == "all", the three sub-fields are concatenated.  Per-field
    galaxy counts are returned for use in random generation.

    Returns
    -------
    table : Table
        Filtered galaxy catalog.
    ra, dec : np.ndarray
        Coordinate arrays in degrees.
    field_counts : dict
        Mapping of field_name → n_galaxies in that field (useful for "all").
    """
    if sample_name not in cfg["samples"]:
        available = list(cfg["samples"].keys())
        raise ValueError(
            f"Unknown sample '{sample_name}'. "
            f"Available:\n  " + "\n  ".join(available)
        )

    sample_cfg = cfg["samples"][sample_name]
    field_name = sample_cfg["field"]
    irac_filter = sample_cfg.get("irac_filter", "none")
    apply_mask = sample_cfg.get("apply_mask", False)

    print(f"\n{'='*60}")
    print(f"Sample : {sample_name}")
    print(f"  {sample_cfg['description']}")
    print(f"{'='*60}")

    # 1. Load full catalog
    full_table = load_fits_catalog(str(resolve_path(cfg["paths"]["raw_catalog"])))

    # 2. Apply IRAC filter (global — before the sky cut)
    filtered = _apply_irac_filter(full_table, irac_filter)

    # 3. Split by field
    field_counts: dict[str, int] = {}

    if field_name == "all":
        sub_tables = []
        for fname in _ALL_SUBFIELDS:
            t = _filter_to_field(filtered, fname, apply_mask)
            field_counts[fname] = len(t)
            print(f"  [{fname}] {len(t):,} galaxies")
            sub_tables.append(t)
        table = vstack(sub_tables) if sub_tables else filtered[:0]
    else:
        table = _filter_to_field(filtered, field_name, apply_mask)
        field_counts[field_name] = len(table)

    cols = cfg["columns"]
    ra, dec = get_radec(table, ra_col=cols["ra"], dec_col=cols["dec"])
    print(f"  Final sample size: {len(ra):,} galaxies")
    return table, ra, dec, field_counts


# ---------------------------------------------------------------------------
# Random catalog helpers
# ---------------------------------------------------------------------------

def _load_mask_for_field(field_name: str):
    """Return (mask_map, nside) for a field, or (None, None) if no mask."""
    mask_rel = cfg["fields"][field_name].get("mask_healsparse")
    if not mask_rel:
        return None, None
    import healsparse
    mask_map = healsparse.HealSparseMap.read(str(resolve_path(mask_rel)))
    return mask_map, mask_map.nside_sparse


def _generate_field_randoms(
    field_name: str,
    apply_mask: bool,
    n_galaxies: int,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random points for a single field."""
    field_cfg = cfg["fields"][field_name]
    factor = cfg["randoms"]["n_randoms_factor"]
    n_randoms = factor * n_galaxies

    mask_map, nside = _load_mask_for_field(field_name) if apply_mask else (None, None)

    rand_table = generate_random_catalog(
        n_randoms=n_randoms,
        ra_min=field_cfg["ra_min"], ra_max=field_cfg["ra_max"],
        dec_min=field_cfg["dec_min"], dec_max=field_cfg["dec_max"],
        mask_map=mask_map,
        nside=nside,
        batch_factor=cfg["randoms"]["batch_factor"],
        seed=seed,
    )
    return np.asarray(rand_table["RA"]), np.asarray(rand_table["DEC"])


def get_or_generate_randoms(
    sample_name: str,
    sample_cfg: dict,
    field_counts: dict,
    regen: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a cached random catalog or generate a new one.

    The catalog is cached at data/processed/randoms_{sample_name}.fits.
    Use --regen-randoms to force regeneration.

    For field == "all", randoms are generated per sub-field proportionally
    to the galaxy count in each sub-field, then concatenated.

    Returns
    -------
    ra_rand, dec_rand : np.ndarray
    """
    rand_path = resolve_path(f"data/processed/randoms_{sample_name}.fits")
    n_total = sum(field_counts.values())

    if rand_path.exists() and not regen:
        print(f"\nLoading cached random catalog: {rand_path.name}")
        rand = Table.read(str(rand_path))
        ra_rand = np.asarray(rand["RA"])
        dec_rand = np.asarray(rand["DEC"])
        factor = cfg["randoms"]["n_randoms_factor"]
        print(f"  {len(ra_rand):,} random points  "
              f"(ratio {len(ra_rand)/n_total:.1f}× expected {factor}×)")
        return ra_rand, dec_rand

    print("\nGenerating random catalog …")
    field_name = sample_cfg["field"]
    apply_mask = sample_cfg.get("apply_mask", False)
    seed = cfg["randoms"]["seed"]

    if field_name == "all":
        all_ra, all_dec = [], []
        for fname in _ALL_SUBFIELDS:
            n_in_field = field_counts.get(fname, 0)
            if n_in_field == 0:
                print(f"  [{fname}] 0 galaxies — skipping randoms")
                continue
            print(f"  [{fname}] generating randoms for {n_in_field:,} galaxies …")
            # Use different seed per field (deterministic offset)
            field_seed = None if seed is None else seed + _ALL_SUBFIELDS.index(fname)
            ra_f, dec_f = _generate_field_randoms(fname, apply_mask, n_in_field, field_seed)
            all_ra.append(ra_f)
            all_dec.append(dec_f)
        ra_rand = np.concatenate(all_ra)
        dec_rand = np.concatenate(all_dec)
    else:
        ra_rand, dec_rand = _generate_field_randoms(field_name, apply_mask, n_total, seed)

    # Cache to disk
    rand_table = Table([ra_rand, dec_rand], names=["RA", "DEC"])
    rand_path.parent.mkdir(parents=True, exist_ok=True)
    rand_table.write(str(rand_path), format="fits", overwrite=True)
    print(f"  Saved random catalog → {rand_path}")

    return ra_rand, dec_rand


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plots(result, sample_name: str, ra, dec, ra_rand, dec_rand) -> None:
    """Generate and save diagnostic and result figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = resolve_path(cfg["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    fmt = cfg["plotting"]["figformat"]
    dpi = cfg["plotting"]["dpi"]

    def _save(fig, name: str):
        for ext in ([fmt] if fmt != "both" else ["pdf", "png"]):
            out = fig_dir / f"{name}.{ext}"
            fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
            print(f"  Saved: {out}")
        plt.close(fig)

    fig, _ = plot_2pacf(
        result,
        title=f"2PACF — {sample_name.replace('_', ' ').title()}",
        label="Observed w(θ)",
    )
    _save(fig, f"2pacf_{sample_name}")

    fig, _ = plot_sky_distribution(ra, dec, ra_rand=ra_rand, dec_rand=dec_rand,
                                   title=f"Sky Distribution — {sample_name}")
    _save(fig, f"skyplot_{sample_name}")

    fig, _ = plot_separation_histogram(
        result.DD, result.theta_arcmin,
        title=f"DD Pair Counts — {sample_name}",
    )
    _save(fig, f"pair_counts_{sample_name}")

    fig, _ = plot_data_vs_randoms_density(
        ra, dec, ra_rand, dec_rand,
        n_rand_ratio=cfg["randoms"]["n_randoms_factor"],
    )
    _save(fig, f"data_vs_randoms_{sample_name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    available = list(cfg["samples"].keys())
    parser = argparse.ArgumentParser(
        description="Compute 2PACF for a named sample defined in config/analysis.yml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available samples:\n"
            + "\n".join(f"  {k:40s}  {cfg['samples'][k]['description']}"
                        for k in available)
        ),
    )
    parser.add_argument(
        "--sample", required=True,
        help="Sample name (must match a key under 'samples' in config/analysis.yml).",
    )
    parser.add_argument(
        "--regen-randoms", action="store_true",
        help="Force regeneration of the random catalog even if a cached one exists.",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip figure generation (useful for batch / headless runs).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Override output .npz path. "
             "Default: results/correlation_functions/2pacf_{sample}.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # --- Load and filter galaxy catalog ---
    table, ra, dec, field_counts = load_sample(args.sample)
    sample_cfg = cfg["samples"][args.sample]

    # --- Random catalog ---
    ra_rand, dec_rand = get_or_generate_randoms(
        args.sample, sample_cfg, field_counts, regen=args.regen_randoms
    )

    # --- 2PACF ---
    bins = cfg["bins"]
    print(f"\nComputing 2PACF …")
    print(f"  Bins: {bins['nbins']} log-spaced from "
          f"{bins['min_sep']}° to {bins['max_sep']}°")
    result = compute_2pacf(
        ra, dec, ra_rand, dec_rand,
        min_sep=bins["min_sep"],
        max_sep=bins["max_sep"],
        nbins=bins["nbins"],
        sep_units=bins["sep_units"],
        bin_slop=bins["bin_slop"],
    )

    # --- Save result ---
    if args.output:
        out_path = args.output
    else:
        corr_dir = resolve_path(cfg["paths"]["corr_dir"])
        corr_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(corr_dir / f"2pacf_{args.sample}.npz")

    result.save(out_path)

    # --- Summary table ---
    print(f"\n{'θ [arcmin]':>12}  {'w(θ)':>12}  {'σ_Poisson':>12}  {'DD pairs':>10}")
    print("-" * 52)
    for i in range(len(result.theta_arcmin)):
        w = result.w[i]
        e = result.w_err_poisson[i]
        neg = "  ← negative" if np.isfinite(w) and w < 0 else ""
        print(f"{result.theta_arcmin[i]:12.3f}  {w:12.5f}  {e:12.5f}  "
              f"{result.DD[i]:10.0f}{neg}")

    # --- Plots ---
    if not args.no_plots:
        print("\nGenerating figures …")
        save_plots(result, args.sample, ra, dec, ra_rand, dec_rand)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f} s")


if __name__ == "__main__":
    main()
