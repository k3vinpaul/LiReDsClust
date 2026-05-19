"""
correlation.py
--------------
Two-point angular correlation function (2PACF) using the Landy-Szalay estimator.

    w(θ) = (DD − 2·DR + RR) / RR

Pair counting is delegated to TreeCorr, which handles the spherical geometry
and is fast enough for catalogs of O(10^3–10^5) objects.

Reference
---------
Landy, S. D. & Szalay, A. S. (1993), ApJ, 412, 64.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import treecorr


@dataclass
class CorrelationResult:
    """Container for 2PACF output arrays.

    Attributes
    ----------
    theta_deg : np.ndarray
        Mean angular separation per bin (degrees).
    theta_arcmin : np.ndarray
        Same, in arcminutes (= theta_deg × 60).
    w : np.ndarray
        Landy-Szalay correlation function w(θ).
    w_err_poisson : np.ndarray
        Poisson error estimate: σ = sqrt((1 + w) / DD).
        NaN for empty bins.
    DD, DR, RR : np.ndarray
        Raw pair counts per angular bin.
    n_galaxies : int
        Number of galaxies in the data catalog.
    n_randoms : int
        Number of points in the random catalog.
    valid : np.ndarray
        Boolean mask — True where w and w_err are both finite.
    """

    theta_deg: np.ndarray
    theta_arcmin: np.ndarray
    w: np.ndarray
    w_err_poisson: np.ndarray
    DD: np.ndarray
    DR: np.ndarray
    RR: np.ndarray
    n_galaxies: int
    n_randoms: int
    valid: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.valid = np.isfinite(self.w) & np.isfinite(self.w_err_poisson)

    def save(self, path: str) -> None:
        """Save all arrays to a compressed NumPy archive (.npz)."""
        np.savez(
            path,
            theta_deg=self.theta_deg,
            theta_arcmin=self.theta_arcmin,
            correlation=self.w,
            correlation_poisson_err=self.w_err_poisson,
            DD_pairs=self.DD,
            DR_pairs=self.DR,
            RR_pairs=self.RR,
            n_galaxies=self.n_galaxies,
            n_randoms=self.n_randoms,
            valid_mask=self.valid,
        )
        print(f"Saved 2PACF result → {path}")

    @classmethod
    def load(cls, path: str) -> "CorrelationResult":
        """Load a previously saved .npz result file."""
        d = np.load(path)
        obj = cls(
            theta_deg=d["theta_deg"],
            theta_arcmin=d["theta_arcmin"],
            w=d["correlation"],
            w_err_poisson=d["correlation_poisson_err"],
            DD=d["DD_pairs"],
            DR=d["DR_pairs"],
            RR=d["RR_pairs"],
            n_galaxies=int(d["n_galaxies"]),
            n_randoms=int(d["n_randoms"]),
        )
        return obj


def _build_bin_config(
    min_sep: float,
    max_sep: float,
    nbins: int,
    sep_units: str = "deg",
    bin_slop: float = 0.01,
) -> dict:
    return {
        "min_sep": min_sep,
        "max_sep": max_sep,
        "nbins": nbins,
        "sep_units": sep_units,
        "bin_slop": bin_slop,
    }


def compute_2pacf(
    ra_data: np.ndarray,
    dec_data: np.ndarray,
    ra_rand: np.ndarray,
    dec_rand: np.ndarray,
    min_sep: float = 0.009,
    max_sep: float = 3.3,
    nbins: int = 15,
    sep_units: str = "deg",
    bin_slop: float = 0.01,
) -> CorrelationResult:
    """Compute the 2-point angular correlation function.

    Uses TreeCorr to count DD, DR, RR pairs and applies the Landy-Szalay
    estimator.  All pair counts are explicitly normalised before the
    estimator is evaluated.

    Parameters
    ----------
    ra_data, dec_data : np.ndarray
        Galaxy coordinates in degrees.
    ra_rand, dec_rand : np.ndarray
        Random catalog coordinates in degrees.
    min_sep : float
        Minimum angular separation (default 0.009°  ≈ 0.54 arcmin).
        Chosen to capture small-scale clustering while staying above the
        minimum observed pair separation in the Euclid EDFS.
    max_sep : float
        Maximum angular separation (default 3.3°  ≈ 198 arcmin).
        Conservative upper limit: max_sep / L_min = 3.3/7 ≈ 0.47 < 0.5,
        minimising edge effects (Landy & Szalay 1993).
    nbins : int
        Number of logarithmic angular bins (default 15).
    sep_units : str
        Unit of min_sep / max_sep passed to TreeCorr.
    bin_slop : float
        TreeCorr bin_slop parameter (fractional bin-edge tolerance).

    Returns
    -------
    CorrelationResult
    """
    n_gal = len(ra_data)
    n_rand = len(ra_rand)

    if n_gal < 2:
        raise ValueError(f"Need at least 2 galaxies, got {n_gal}.")

    bin_cfg = _build_bin_config(min_sep, max_sep, nbins, sep_units, bin_slop)

    cat_d = treecorr.Catalog(ra=ra_data, dec=dec_data, ra_units="deg", dec_units="deg")
    cat_r = treecorr.Catalog(ra=ra_rand, dec=dec_rand, ra_units="deg", dec_units="deg")

    dd = treecorr.NNCorrelation(**bin_cfg)
    dr = treecorr.NNCorrelation(**bin_cfg)
    rr = treecorr.NNCorrelation(**bin_cfg)

    print("  Computing DD …", end=" ", flush=True)
    dd.process(cat_d)
    print(f"{dd.npairs.sum():,.0f} pairs")

    print("  Computing DR …", end=" ", flush=True)
    dr.process(cat_d, cat_r)
    print(f"{dr.npairs.sum():,.0f} pairs")

    print("  Computing RR …", end=" ", flush=True)
    rr.process(cat_r)
    print(f"{rr.npairs.sum():,.0f} pairs")

    theta = np.exp(dd.meanlogr)

    DD = dd.npairs.astype(float)
    DR = dr.npairs.astype(float)
    RR = rr.npairs.astype(float)

    # Normalise
    DD_norm = DD / (n_gal * (n_gal - 1) / 2.0)
    DR_norm = DR / (n_gal * n_rand)
    RR_norm = RR / (n_rand * (n_rand - 1) / 2.0)

    # Landy-Szalay estimator
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(RR_norm > 0, (DD_norm - 2 * DR_norm + RR_norm) / RR_norm, np.nan)
        # Poisson error: σ = sqrt((1 + w) / DD)
        w_err = np.where(DD > 0, np.sqrt((1.0 + w) / DD), np.nan)

    result = CorrelationResult(
        theta_deg=theta,
        theta_arcmin=theta * 60.0,
        w=w,
        w_err_poisson=w_err,
        DD=DD,
        DR=DR,
        RR=RR,
        n_galaxies=n_gal,
        n_randoms=n_rand,
    )

    n_valid = result.valid.sum()
    print(f"  w(θ) computed: {n_valid}/{nbins} valid bins "
          f"(w range: {w[result.valid].min():.4f} – {w[result.valid].max():.4f})")
    return result
