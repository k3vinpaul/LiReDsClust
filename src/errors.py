"""
errors.py
---------
Error estimation for the 2PACF via simulation-based realizations.

Strategy
--------
Given a large simulation sky catalogue (RA_sim, Dec_sim), apply random
rigid rotations to the full simulation and then window it through the
real survey mask.  Each rotation yields an independent synthetic galaxy
catalog with a realistic angular distribution.  Running the 2PACF on
*N* such realizations gives the sample standard deviation per bin,
which is our simulation-based uncertainty.

This is the method used in the thesis analysis (Part1 / Part2 notebooks).
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table

from .correlation import CorrelationResult, compute_2pacf


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def rotate_sky(
    ra: np.ndarray,
    dec: np.ndarray,
    delta_ra: float,
    delta_dec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a rigid rotation (translation) on the sky.

    RA is wrapped modulo 360°.  Dec is reflected at the poles to preserve
    the spherical topology.

    Parameters
    ----------
    ra, dec : np.ndarray
        Input coordinates in degrees.
    delta_ra, delta_dec : float
        Translation offsets in degrees.

    Returns
    -------
    ra_rot, dec_rot : np.ndarray
    """
    from .randoms import apply_periodic_boundaries

    ra_new = (ra + delta_ra) % 360.0
    dec_new = dec + delta_dec
    ra_new, dec_new = apply_periodic_boundaries(ra_new, dec_new)
    return ra_new, dec_new


# ---------------------------------------------------------------------------
# Synthetic catalog generation
# ---------------------------------------------------------------------------

def _apply_healpix_mask(
    ra: np.ndarray,
    dec: np.ndarray,
    mask_map,
    nside: int,
    batch_size: int = 100_000,
) -> np.ndarray:
    """Return boolean mask: True where (ra, dec) falls inside the HealSparse mask."""
    import healpy as hp

    n = len(ra)
    valid = np.zeros(n, dtype=bool)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        theta = np.radians(90.0 - dec[start:end])
        phi = np.radians(ra[start:end])
        pix = hp.ang2pix(nside, theta, phi, nest=True)
        valid[start:end] = mask_map.get_values_pix(pix) == 1
    return valid


def generate_synthetic_catalogs(
    ra_sim: np.ndarray,
    dec_sim: np.ndarray,
    mask_map,
    nside: int,
    n_catalogs: int = 100,
    min_galaxies: int = 2,
    seed: int | None = None,
) -> list[dict]:
    """Generate synthetic galaxy catalogs by rotating a simulation onto the footprint.

    Each catalog is produced by:
      1. Drawing a random (ΔRA, ΔDec) offset.
      2. Rotating the full simulation sky by that offset.
      3. Keeping only positions that fall inside the survey mask.

    Rotations that yield fewer than *min_galaxies* inside the footprint are
    discarded and re-drawn.

    Parameters
    ----------
    ra_sim, dec_sim : np.ndarray
        Full simulation sky coordinates (degrees).  These should already
        have periodic boundaries applied (use `randoms.apply_periodic_boundaries`).
    mask_map : HealSparseMap
        Survey mask.
    nside : int
        NSIDE of the sparse mask.
    n_catalogs : int
        Number of valid synthetic catalogs to produce.
    min_galaxies : int
        Minimum galaxies required for a catalog to be kept.
    seed : int or None
        Random seed.

    Returns
    -------
    list of dict, each with keys:
        ``RA``, ``DEC``  – coordinate arrays inside the footprint
        ``n_galaxies``   – number of galaxies
        ``delta_ra``     – RA offset applied
        ``delta_dec``    – Dec offset applied
    """
    rng = np.random.default_rng(seed)
    catalogs: list[dict] = []
    n_attempts = 0
    n_rejected = 0

    print(f"Generating {n_catalogs} synthetic catalogs …")
    while len(catalogs) < n_catalogs:
        delta_ra = rng.uniform(0, 360)
        delta_dec = rng.uniform(-10, 100)
        n_attempts += 1

        ra_rot, dec_rot = rotate_sky(ra_sim, dec_sim, delta_ra, delta_dec)
        in_fp = _apply_healpix_mask(ra_rot, dec_rot, mask_map, nside)

        ra_in = ra_rot[in_fp]
        dec_in = dec_rot[in_fp]

        if len(ra_in) >= min_galaxies:
            catalogs.append(
                {
                    "RA": ra_in,
                    "DEC": dec_in,
                    "n_galaxies": len(ra_in),
                    "delta_ra": delta_ra,
                    "delta_dec": delta_dec,
                }
            )
            print(f"  [{len(catalogs):>3}/{n_catalogs}] "
                  f"{len(ra_in)} galaxies  (attempt {n_attempts})")
        else:
            n_rejected += 1

    print(f"Done: {n_catalogs} valid catalogs "
          f"| {n_attempts} total attempts "
          f"| {n_rejected} rejected (<{min_galaxies} gal)")
    return catalogs


def save_synthetic_catalogs(catalogs: list[dict], path: str) -> None:
    """Save a list of synthetic catalogs to a compressed .npz file."""
    save_dict: dict = {
        "n_catalogs": len(catalogs),
        "n_galaxies_per_catalog": np.array([c["n_galaxies"] for c in catalogs]),
        "delta_ra_offsets": np.array([c["delta_ra"] for c in catalogs]),
        "delta_dec_offsets": np.array([c["delta_dec"] for c in catalogs]),
    }
    for i, cat in enumerate(catalogs):
        save_dict[f"RA_catalog_{i}"] = cat["RA"]
        save_dict[f"DEC_catalog_{i}"] = cat["DEC"]
    np.savez(path, **save_dict)
    print(f"Saved {len(catalogs)} synthetic catalogs → {path}")


def load_synthetic_catalogs(path: str) -> list[dict]:
    """Load synthetic catalogs previously saved with `save_synthetic_catalogs`."""
    data = np.load(path)
    n = int(data["n_catalogs"])
    catalogs = []
    for i in range(n):
        ra = data[f"RA_catalog_{i}"]
        dec = data[f"DEC_catalog_{i}"]
        catalogs.append(
            {
                "RA": ra,
                "DEC": dec,
                "n_galaxies": len(ra),
                "delta_ra": float(data["delta_ra_offsets"][i]),
                "delta_dec": float(data["delta_dec_offsets"][i]),
            }
        )
    print(f"Loaded {n} synthetic catalogs from {path}")
    return catalogs


# ---------------------------------------------------------------------------
# Error estimation
# ---------------------------------------------------------------------------

def compute_simulation_errors(
    synthetic_catalogs: list[dict],
    ra_rand: np.ndarray,
    dec_rand: np.ndarray,
    min_sep: float = 0.009,
    max_sep: float = 3.3,
    nbins: int = 15,
    sep_units: str = "deg",
    bin_slop: float = 0.01,
) -> dict:
    """Compute the 2PACF for each synthetic catalog and derive error bars.

    Parameters
    ----------
    synthetic_catalogs : list of dict
        Output of `generate_synthetic_catalogs` or `load_synthetic_catalogs`.
    ra_rand, dec_rand : np.ndarray
        Shared random catalog coordinates (degrees).
    min_sep, max_sep, nbins, sep_units, bin_slop :
        Bin configuration passed directly to `correlation.compute_2pacf`.

    Returns
    -------
    dict with keys:
        ``theta_deg``, ``theta_arcmin`` – angular bin centres
        ``mean_w``   – mean w(θ) over all realizations
        ``std_w``    – standard deviation (simulation error bars)
        ``median_w`` – median w(θ)
        ``all_w``    – array of shape (n_catalogs, nbins) with all w(θ)
        ``n_valid``  – number of valid (finite) realizations per bin
    """
    n_cats = len(synthetic_catalogs)
    all_w = np.full((n_cats, nbins), np.nan)
    theta_deg = None

    print(f"Computing 2PACF for {n_cats} synthetic catalogs …")
    for i, cat in enumerate(synthetic_catalogs):
        try:
            result = compute_2pacf(
                cat["RA"], cat["DEC"],
                ra_rand, dec_rand,
                min_sep=min_sep, max_sep=max_sep,
                nbins=nbins, sep_units=sep_units, bin_slop=bin_slop,
            )
            all_w[i] = result.w
            if theta_deg is None:
                theta_deg = result.theta_deg
        except Exception as exc:
            print(f"  Catalog {i}: skipped ({exc})")

    if theta_deg is None:
        raise RuntimeError("No valid 2PACF computed — check catalog inputs.")

    mean_w = np.nanmean(all_w, axis=0)
    std_w = np.nanstd(all_w, axis=0)
    median_w = np.nanmedian(all_w, axis=0)
    n_valid = np.sum(~np.isnan(all_w), axis=0)

    print(f"Error estimation complete: "
          f"mean σ = {std_w[np.isfinite(std_w)].mean():.4f}")

    return {
        "theta_deg": theta_deg,
        "theta_arcmin": theta_deg * 60.0,
        "mean_w": mean_w,
        "std_w": std_w,
        "median_w": median_w,
        "all_w": all_w,
        "n_valid": n_valid,
    }


def compute_poisson_errors(result: CorrelationResult) -> np.ndarray:
    """Return the Poisson error already stored in a CorrelationResult.

    Provided as a convenience wrapper so callers can treat all error methods
    uniformly.

    Returns
    -------
    np.ndarray
        w_err_poisson array (NaN for empty bins).
    """
    return result.w_err_poisson.copy()
