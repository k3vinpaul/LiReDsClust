#!/usr/bin/env python3
"""
run_simulations.py
------------------
Generate simulation-based error bars for the 2PACF.

The method rotates a large simulation sky catalog to random positions,
windows each rotation through the survey mask, and computes the 2PACF on
each resulting synthetic catalog.  The standard deviation across realizations
gives the simulation error bar per angular bin.

Usage
-----
    # Use the active simulation defined in config/analysis.yml
    python scripts/run_simulations.py --sample irac_footprint_masked

    # Pick a specific simulation by name (must be listed under simulations.available)
    python scripts/run_simulations.py --sample irac_footprint_masked --sim-name alpha1
    python scripts/run_simulations.py --sample irac_footprint_masked --sim-name alpha3

    # Compare ALL available simulations for a given sample
    python scripts/run_simulations.py --sample irac_footprint_masked --all-sims

    # Override number of synthetic catalogs
    python scripts/run_simulations.py --sample irac_footprint_masked --n-catalogs 200

    # Force regeneration even if cached result exists
    python scripts/run_simulations.py --sample irac_footprint_masked --regen-catalogs

    # Skip figure output
    python scripts/run_simulations.py --sample irac_footprint_masked --no-plots

Available simulation names (set in config/analysis.yml under simulations.available):
  alpha0.5  — α = 0.5  (default active)
  alpha1    — α = 1
  alpha3    — α = 3
  default   — original collaborator file, unlabelled α

Column-swap note
----------------
The simulation files have a known column swap: RA_sim_*.npy actually contains
DEC values and DEC_sim_*.npy contains RA values.  This is corrected
automatically — you do not need to swap the files manually.
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.config import cfg, resolve_path
from src.randoms import apply_periodic_boundaries
from src.errors import (
    generate_synthetic_catalogs,
    save_synthetic_catalogs,
    load_synthetic_catalogs,
    compute_simulation_errors,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_sim_config(sim_name: str | None) -> dict:
    """Resolve a simulation config dict from the YAML.

    If *sim_name* is None, uses ``simulations.active`` from config.

    Returns a dict with keys: ra_file, dec_file, label.
    """
    available = cfg["simulations"]["available"]
    active    = sim_name or cfg["simulations"].get("active", next(iter(available)))

    if active not in available:
        raise ValueError(
            f"Simulation '{active}' not found in config/analysis.yml.\n"
            f"Available: {list(available.keys())}"
        )
    return {"name": active, **available[active]}


def load_simulation_coords(sim_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load and correct simulation RA/Dec coordinate arrays from .npy files.

    The simulation files provided by the collaborator have a known column swap:
      - RA_sim_*.npy  → actually contains Dec values
      - DEC_sim_*.npy → actually contains RA values

    This function applies the correction and removes (0, 0) sentinel entries.

    Parameters
    ----------
    sim_cfg : dict
        Entry from simulations.available in config/analysis.yml.
        Must have keys: ra_file, dec_file, label.

    Returns
    -------
    ra, dec : np.ndarray
        Corrected coordinate arrays in degrees, with periodic boundaries
        applied and (0, 0) sentinel entries removed.
    """
    ra_path  = str(resolve_path(sim_cfg["ra_file"]))
    dec_path = str(resolve_path(sim_cfg["dec_file"]))
    label    = sim_cfg["label"]

    print(f"\nLoading simulation: {label}")
    print(f"  RA file  (note: actually contains Dec): {ra_path}")
    print(f"  Dec file (note: actually contains RA) : {dec_path}")

    ra_raw  = np.load(ra_path)
    dec_raw = np.load(dec_path)
    print(f"  RA  file shape: {ra_raw.shape}")
    print(f"  Dec file shape: {dec_raw.shape}")

    # Column swap correction (inherited from original analysis notebooks)
    ra_correct  = dec_raw   # DEC file holds the true RA
    dec_correct = ra_raw    # RA  file holds the true Dec

    # Remove sentinel (0, 0) pairs
    valid = ~((ra_correct == 0.0) & (dec_correct == 0.0))
    ra_correct  = ra_correct[valid]
    dec_correct = dec_correct[valid]
    print(f"  Valid points after removing (0,0) sentinels: {valid.sum():,}")

    # Apply periodic boundary conditions
    ra_correct, dec_correct = apply_periodic_boundaries(ra_correct, dec_correct)
    print(f"  RA  range after correction: {ra_correct.min():.2f}° – {ra_correct.max():.2f}°")
    print(f"  Dec range after correction: {dec_correct.min():.2f}° – {dec_correct.max():.2f}°")
    return ra_correct, dec_correct


def load_random_catalog(sample_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the cached random catalog for a sample (must already exist).

    If not found, run:
        python scripts/run_2pacf.py --sample <sample_name> --regen-randoms --no-plots
    """
    rand_path = resolve_path(f"data/processed/randoms_{sample_name}.fits")
    if not rand_path.exists():
        raise FileNotFoundError(
            f"Random catalog not found: {rand_path}\n"
            f"Generate it first with:\n"
            f"  python scripts/run_2pacf.py --sample {sample_name} "
            f"--regen-randoms --no-plots"
        )
    from astropy.table import Table
    rand     = Table.read(str(rand_path))
    ra_rand  = np.asarray(rand["RA"])
    dec_rand = np.asarray(rand["DEC"])
    print(f"Loaded random catalog: {len(ra_rand):,} points ({rand_path.name})")
    return ra_rand, dec_rand


def run_one_simulation(
    sim_name: str,
    sample_name: str,
    n_catalogs: int,
    regen: bool,
    no_plots: bool,
    corr_result_path: str | None,
) -> dict:
    """Run simulation error estimation for one simulation file.

    Returns the sim_errors dict.
    """
    sim_cfg  = get_sim_config(sim_name)
    bins     = cfg["bins"]
    sim_main = cfg["simulations"]

    cov_dir    = resolve_path(cfg["paths"]["cov_dir"])
    cov_dir.mkdir(parents=True, exist_ok=True)

    cache_tag  = f"{sample_name}_{sim_name}"
    synth_cache = cov_dir / f"synthetic_catalogs_{cache_tag}.npz"
    errors_out  = cov_dir / f"sim_errors_{cache_tag}.npz"

    # --- Load or generate synthetic catalogs ---
    if synth_cache.exists() and not regen:
        print(f"\nLoading cached synthetic catalogs: {synth_cache.name}")
        synthetic_catalogs = load_synthetic_catalogs(str(synth_cache))
        print(f"  {len(synthetic_catalogs)} catalogs loaded")
    else:
        ra_sim, dec_sim = load_simulation_coords(sim_cfg)

        import healsparse
        sample_cfg  = cfg["samples"][sample_name]
        field_name  = sample_cfg["field"] if sample_cfg["field"] != "all" else "edf_s"
        mask_rel    = cfg["fields"][field_name].get("mask_healsparse")
        if not mask_rel:
            raise ValueError(
                f"No HealSparse mask defined for field '{field_name}'. "
                f"Simulation error estimation requires a mask."
            )
        mask_path = str(resolve_path(mask_rel))
        print(f"\nLoading HealSparse mask: {mask_path}")
        mask_map = healsparse.HealSparseMap.read(mask_path)
        nside    = mask_map.nside_sparse

        print(f"\nGenerating {n_catalogs} synthetic catalogs "
              f"(min_galaxies={sim_main['min_galaxies']}) …")
        synthetic_catalogs = generate_synthetic_catalogs(
            ra_sim=ra_sim,
            dec_sim=dec_sim,
            mask_map=mask_map,
            nside=nside,
            n_catalogs=n_catalogs,
            min_galaxies=sim_main["min_galaxies"],
            seed=sim_main["seed"],
        )
        save_synthetic_catalogs(synthetic_catalogs, str(synth_cache))

    # --- Load random catalog ---
    ra_rand, dec_rand = load_random_catalog(sample_name)

    # --- Compute simulation errors ---
    print(f"\nComputing 2PACF for {len(synthetic_catalogs)} synthetic catalogs …")
    sim_errors = compute_simulation_errors(
        synthetic_catalogs,
        ra_rand, dec_rand,
        min_sep=bins["min_sep"],
        max_sep=bins["max_sep"],
        nbins=bins["nbins"],
        sep_units=bins["sep_units"],
        bin_slop=bins["bin_slop"],
    )

    # --- Save error arrays ---
    np.savez(
        str(errors_out),
        theta_deg=sim_errors["theta_deg"],
        theta_arcmin=sim_errors["theta_arcmin"],
        mean_w=sim_errors["mean_w"],
        std_w=sim_errors["std_w"],
        median_w=sim_errors["median_w"],
        all_w=sim_errors["all_w"],
        n_valid=sim_errors["n_valid"],
        n_catalogs=len(synthetic_catalogs),
        sim_name=sim_name,
        sim_label=sim_cfg["label"],
    )
    print(f"\nSaved simulation errors → {errors_out}")

    # --- Summary table ---
    print(f"\n{'θ [arcmin]':>12}  {'<w(θ)>':>12}  {'σ(θ)':>12}  {'N valid':>8}")
    print("-" * 50)
    for i in range(len(sim_errors["theta_arcmin"])):
        print(f"{sim_errors['theta_arcmin'][i]:12.3f}  "
              f"{sim_errors['mean_w'][i]:12.5f}  "
              f"{sim_errors['std_w'][i]:12.5f}  "
              f"{sim_errors['n_valid'][i]:8.0f}")

    # --- Plots ---
    if not no_plots:
        real_result = None
        corr_path   = corr_result_path or str(
            resolve_path(cfg["paths"]["corr_dir"]) / f"2pacf_{sample_name}.npz"
        )
        if Path(corr_path).exists():
            from src.correlation import CorrelationResult
            real_result = CorrelationResult.load(corr_path)

        _save_plots(sim_errors, real_result, sample_name, sim_name, sim_cfg["label"])

    return sim_errors


def _save_plots(sim_errors, real_result, sample_name, sim_name, sim_label) -> None:
    """Generate and save simulation error figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.plotting import plot_2pacf_with_sim_errors, plot_2pacf
    from src.correlation import CorrelationResult

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

    # Observed + simulation error band
    if real_result is not None:
        fig, _ = plot_2pacf_with_sim_errors(
            real_result, sim_errors,
            title=(f"2PACF with Simulation Errors ({sim_label}) — "
                   f"{sample_name.replace('_', ' ').title()}"),
        )
        _save(fig, f"2pacf_sim_errors_{sample_name}_{sim_name}")

    # Simulation mean standalone
    dummy = CorrelationResult(
        theta_deg=sim_errors["theta_deg"],
        theta_arcmin=sim_errors["theta_arcmin"],
        w=sim_errors["mean_w"],
        w_err_poisson=sim_errors["std_w"],
        DD=np.zeros_like(sim_errors["theta_deg"]),
        DR=np.zeros_like(sim_errors["theta_deg"]),
        RR=np.zeros_like(sim_errors["theta_deg"]),
        n_galaxies=0,
        n_randoms=0,
    )
    fig, _ = plot_2pacf(
        dummy,
        title=(f"2PACF Simulation Mean ± σ ({sim_label}) — "
               f"{sample_name.replace('_', ' ').title()}"),
        label=f"Sim mean ({sim_label}, N={int(sim_errors['all_w'].shape[0])})",
        fit_powerlaw=False,
    )
    _save(fig, f"2pacf_sim_mean_{sample_name}_{sim_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulation-based 2PACF error estimation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sample", required=True,
        help="Sample name (must match a key in config/analysis.yml 'samples').",
    )
    parser.add_argument(
        "--sim-name", default=None,
        help=(
            "Simulation to use (must match a key under simulations.available in "
            "config/analysis.yml). Default: the value of simulations.active. "
            "Available: alpha0.5, alpha1, alpha3, default."
        ),
    )
    parser.add_argument(
        "--all-sims", action="store_true",
        help="Run error estimation for ALL available simulations and compare.",
    )
    parser.add_argument(
        "--n-catalogs", type=int, default=None,
        help="Number of synthetic catalogs to generate. "
             "Overrides simulations.n_catalogs in config.",
    )
    parser.add_argument(
        "--regen-catalogs", action="store_true",
        help="Force regeneration of synthetic catalogs even if a cached .npz exists.",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip figure generation.",
    )
    parser.add_argument(
        "--corr-result", default=None,
        help="Path to the observed 2PACF .npz (for overlay plot). "
             "Default: results/correlation_functions/2pacf_{sample}.npz",
    )
    return parser.parse_args()


def main() -> None:
    args     = parse_args()
    t0       = time.time()
    n_cats   = args.n_catalogs or cfg["simulations"]["n_catalogs"]

    if args.all_sims:
        # Run all available simulations
        available = cfg["simulations"]["available"]
        print(f"\nRunning all {len(available)} simulations for sample '{args.sample}'")
        all_results: dict[str, dict] = {}
        for sim_name in available:
            print(f"\n{'='*60}")
            print(f"Simulation: {sim_name}  ({available[sim_name]['label']})")
            print(f"{'='*60}")
            all_results[sim_name] = run_one_simulation(
                sim_name=sim_name,
                sample_name=args.sample,
                n_catalogs=n_cats,
                regen=args.regen_catalogs,
                no_plots=True,  # skip per-sim plots; make comparison plot below
                corr_result_path=args.corr_result,
            )

        # Comparison plot: all sims on one figure
        if not args.no_plots:
            _save_comparison_plot(all_results, args.sample, args.corr_result)

    else:
        run_one_simulation(
            sim_name=args.sim_name,
            sample_name=args.sample,
            n_catalogs=n_cats,
            regen=args.regen_catalogs,
            no_plots=args.no_plots,
            corr_result_path=args.corr_result,
        )

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f} s")


def _save_comparison_plot(
    all_results: dict[str, dict],
    sample_name: str,
    corr_result_path: str | None,
) -> None:
    """Plot simulation error bars from all alpha values on one figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["steelblue", "darkorange", "green", "purple", "crimson"]
    fig, ax = plt.subplots(figsize=(10, 6))

    corr_path = corr_result_path or str(
        resolve_path(cfg["paths"]["corr_dir"]) / f"2pacf_{sample_name}.npz"
    )
    if Path(corr_path).exists():
        from src.correlation import CorrelationResult
        from src.errors import compute_poisson_errors
        result  = CorrelationResult.load(corr_path)
        poisson = compute_poisson_errors(result)
        ax.errorbar(
            result.theta_arcmin, result.w, yerr=poisson,
            fmt="ko", capsize=3, markersize=5, label="Observed (Poisson err)",
            zorder=5,
        )

    available = cfg["simulations"]["available"]
    for (sim_name, sim_errors), color in zip(all_results.items(), colors):
        label = available[sim_name]["label"]
        theta = sim_errors["theta_arcmin"]
        ax.fill_between(
            theta,
            sim_errors["mean_w"] - sim_errors["std_w"],
            sim_errors["mean_w"] + sim_errors["std_w"],
            alpha=0.20, color=color,
        )
        ax.plot(theta, sim_errors["mean_w"], "-", color=color,
                linewidth=1.5, label=f"Sim mean ± 1σ  ({label})")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\theta$ [arcmin]", fontsize=13)
    ax.set_ylabel(r"$w(\theta)$", fontsize=13)
    ax.set_title(
        f"2PACF Simulation Comparison — {sample_name.replace('_', ' ').title()}",
        fontsize=14,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()

    fig_dir = resolve_path(cfg["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    fmt = cfg["plotting"]["figformat"]
    dpi = cfg["plotting"]["dpi"]
    for ext in ([fmt] if fmt != "both" else ["pdf", "png"]):
        out = fig_dir / f"sim_comparison_{sample_name}.{ext}"
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
        print(f"  Saved comparison plot: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
