"""
plotting.py
-----------
Publication-quality figures for the 2PACF analysis.

All functions return a ``(fig, ax)`` tuple so callers can further customise
or save with the format / DPI of their choice.  Default style targets
journal-quality output (300 dpi, Times-like fonts where available).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

# Use a clean, publication-ready style if available
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass  # fallback to matplotlib defaults

_EDFS_RA_MIN, _EDFS_RA_MAX = 56.0, 67.0
_EDFS_DEC_MIN, _EDFS_DEC_MAX = -52.0, -45.0


# ---------------------------------------------------------------------------
# 2PACF plots
# ---------------------------------------------------------------------------

def plot_2pacf(
    result,
    title: str = "Two-Point Angular Correlation Function",
    label: str = "Data",
    fit_powerlaw: bool = True,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot w(θ) with Poisson error bars and an optional power-law fit.

    Negative bins are included — they indicate anti-correlation and are
    physically meaningful.  The power-law fit is computed in log-log space
    on positive bins only (log of a negative number is undefined).

    Parameters
    ----------
    result : CorrelationResult
        Output of `correlation.compute_2pacf`.
    title : str
        Figure title.
    label : str
        Legend label for the data points.
    fit_powerlaw : bool
        Whether to overplot a power-law fit to positive bins.
    ax : plt.Axes or None
        Existing axes to plot into.  If None, a new figure is created.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.get_figure()

    theta = result.theta_arcmin
    w = result.w
    w_err = result.w_err_poisson

    ax.errorbar(
        theta, w, yerr=w_err,
        fmt="o", markersize=7, capsize=4, capthick=1.8,
        color="navy", ecolor="steelblue",
        label=label, zorder=3,
    )

    if fit_powerlaw:
        mask = (w > 0) & np.isfinite(w) & np.isfinite(w_err)
        if mask.sum() > 2:
            coeffs = np.polyfit(np.log10(theta[mask]), np.log10(w[mask]), 1)
            fit = 10 ** np.polyval(coeffs, np.log10(theta[mask]))
            ax.plot(
                theta[mask], fit,
                "--", color="crimson", linewidth=1.8,
                label=fr"Power-law (pos. bins): $\delta={coeffs[0]:.2f}$",
                zorder=2,
            )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel(r"Angular Separation $\theta$ [arcmin]", fontsize=13)
    ax.set_ylabel(r"$w(\theta)$", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    fig.tight_layout()
    return fig, ax


def plot_2pacf_comparison(
    results: list[tuple[object, str, str]],
    title: str = "2PACF Comparison",
) -> tuple[plt.Figure, plt.Axes]:
    """Overlay multiple 2PACF results on one axes.

    Parameters
    ----------
    results : list of (CorrelationResult, label, color)
        Each tuple provides one curve.
    title : str
        Figure title.

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for result, label, color in results:
        ax.errorbar(
            result.theta_arcmin, result.w, yerr=result.w_err_poisson,
            fmt="o-", markersize=5, capsize=3, linewidth=1.5,
            color=color, ecolor=color, alpha=0.8, label=label,
        )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel(r"Angular Separation $\theta$ [arcmin]", fontsize=13)
    ax.set_ylabel(r"$w(\theta)$", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    fig.tight_layout()
    return fig, ax


def plot_2pacf_with_sim_errors(
    result,
    sim_errors: dict,
    title: str = "2PACF with Simulation Error Bars",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the observed w(θ) alongside simulation-based error bars.

    Parameters
    ----------
    result : CorrelationResult
        Observed 2PACF.
    sim_errors : dict
        Output of `errors.compute_simulation_errors`.
    title : str

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    theta = result.theta_arcmin

    # Individual simulation realizations (light background lines)
    for w_sim in sim_errors["all_w"]:
        ax.plot(theta, w_sim, color="gray", linewidth=0.6, alpha=0.12, zorder=1)

    # Simulation mean ± std
    ax.fill_between(
        sim_errors["theta_arcmin"],
        sim_errors["mean_w"] - sim_errors["std_w"],
        sim_errors["mean_w"] + sim_errors["std_w"],
        color="steelblue", alpha=0.25, label="Sim. 1-σ band",
    )
    ax.plot(
        sim_errors["theta_arcmin"], sim_errors["mean_w"],
        "--", color="steelblue", linewidth=1.5, label="Sim. mean",
    )

    # Observed w(θ) with Poisson errors
    ax.errorbar(
        theta, result.w, yerr=result.w_err_poisson,
        fmt="o", markersize=7, capsize=4, capthick=1.8,
        color="navy", ecolor="navy", label="Observed (Poisson err.)", zorder=4,
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel(r"Angular Separation $\theta$ [arcmin]", fontsize=13)
    ax.set_ylabel(r"$w(\theta)$", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Sky distribution plots
# ---------------------------------------------------------------------------

def plot_sky_distribution(
    ra: np.ndarray,
    dec: np.ndarray,
    title: str = "Galaxy Sky Distribution",
    show_edfs_box: bool = True,
    ra_rand: np.ndarray | None = None,
    dec_rand: np.ndarray | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter plot of galaxy (and optionally random) sky positions.

    Parameters
    ----------
    ra, dec : np.ndarray
        Galaxy coordinates in degrees.
    title : str
    show_edfs_box : bool
        Overlay the Euclid Deep Field South bounding box.
    ra_rand, dec_rand : np.ndarray or None
        If provided, random catalog points are plotted in the background.

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    if ra_rand is not None and dec_rand is not None:
        ax.scatter(ra_rand, dec_rand, s=1, color="lightgray", alpha=0.3,
                   label=f"Randoms (N={len(ra_rand):,})", zorder=1)

    ax.scatter(ra, dec, s=20, color="navy", alpha=0.7, edgecolors="none",
               label=f"Galaxies (N={len(ra):,})", zorder=2)

    if show_edfs_box:
        rect = Rectangle(
            (_EDFS_RA_MIN, _EDFS_DEC_MIN),
            _EDFS_RA_MAX - _EDFS_RA_MIN,
            _EDFS_DEC_MAX - _EDFS_DEC_MIN,
            fill=False, edgecolor="crimson", linewidth=2.0,
            linestyle="--", label="Euclid Deep Field South",
        )
        ax.add_patch(rect)

    ax.set_xlabel("Right Ascension [deg]", fontsize=13)
    ax.set_ylabel("Declination [deg]", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def plot_separation_histogram(
    DD: np.ndarray,
    theta_arcmin: np.ndarray,
    title: str = "Galaxy Pair Separation Distribution",
) -> tuple[plt.Figure, plt.Axes]:
    """Bar plot of DD pair counts per angular bin.

    Parameters
    ----------
    DD : np.ndarray
        DD pair counts (from CorrelationResult).
    theta_arcmin : np.ndarray
        Bin centres in arcminutes.

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(theta_arcmin, DD, width=np.diff(np.append(theta_arcmin, theta_arcmin[-1] * 1.5)) * 0.8,
           color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"Angular Separation $\theta$ [arcmin]", fontsize=13)
    ax.set_ylabel("DD Pair Counts", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3, linestyle="--", axis="y")
    fig.tight_layout()
    return fig, ax


def plot_data_vs_randoms_density(
    ra_data: np.ndarray,
    dec_data: np.ndarray,
    ra_rand: np.ndarray,
    dec_rand: np.ndarray,
    n_rand_ratio: float | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Side-by-side RA and Dec histograms comparing data and randoms.

    Parameters
    ----------
    ra_data, dec_data : np.ndarray
        Data galaxy coordinates.
    ra_rand, dec_rand : np.ndarray
        Random catalog coordinates.
    n_rand_ratio : float or None
        If provided, the random histogram is normalised by this factor so
        both distributions have the same integral.

    Returns
    -------
    fig, axes
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    scale = 1.0 if n_rand_ratio is None else 1.0 / n_rand_ratio

    for ax, coord_d, coord_r, xlabel in zip(
        axes,
        [ra_data, dec_data],
        [ra_rand, dec_rand],
        ["Right Ascension [deg]", "Declination [deg]"],
    ):
        bins = np.linspace(coord_d.min(), coord_d.max(), 40)
        ax.hist(coord_d, bins=bins, color="navy", alpha=0.7,
                density=True, label="Galaxies")
        ax.hist(coord_r, bins=bins, color="crimson", alpha=0.5,
                density=True, label="Randoms (normalised)", histtype="step",
                linewidth=1.8)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Normalised Counts", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle("Data vs Randoms: Sky Coverage Comparison",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig, list(axes)
