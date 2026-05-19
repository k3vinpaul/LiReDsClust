"""
LRDs 2PACF pipeline
-------------------
Public API for the two-point angular correlation function analysis of
Little Red Dot galaxies in the Euclid Deep Field South.

Modules
-------
catalog    – FITS catalog loading, filtering, masking
randoms    – Uniform masked random catalog generation
correlation – 2PACF computation (Landy-Szalay via TreeCorr)
errors     – Simulation-based and Poisson error estimation
covariance – OneCovariance input preparation
plotting   – Publication-quality figures
"""

from .config import cfg, load_config, resolve_path
from .catalog import load_fits_catalog, get_radec, apply_sky_mask, filter_radec_range
from .correlation import compute_2pacf, CorrelationResult
from .randoms import generate_random_catalog
from .errors import (
    generate_synthetic_catalogs,
    compute_simulation_errors,
    compute_poisson_errors,
)
from .covariance import prepare_onecovariance_inputs
from .plotting import (
    plot_2pacf,
    plot_2pacf_comparison,
    plot_2pacf_with_sim_errors,
    plot_sky_distribution,
    plot_separation_histogram,
    plot_data_vs_randoms_density,
)

__all__ = [
    "cfg",
    "load_config",
    "resolve_path",
    "load_fits_catalog",
    "get_radec",
    "apply_sky_mask",
    "filter_radec_range",
    "compute_2pacf",
    "CorrelationResult",
    "generate_random_catalog",
    "generate_synthetic_catalogs",
    "compute_simulation_errors",
    "compute_poisson_errors",
    "prepare_onecovariance_inputs",
    "plot_2pacf",
    "plot_2pacf_comparison",
    "plot_2pacf_with_sim_errors",
    "plot_sky_distribution",
    "plot_separation_histogram",
    "plot_data_vs_randoms_density",
]
