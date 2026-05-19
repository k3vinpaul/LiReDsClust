"""
covariance.py
-------------
Covariance matrix preparation and interaction with the OneCovariance package.

OneCovariance (https://github.com/rreischke/OneCovariance) computes the
Gaussian + non-Gaussian covariance of angular clustering statistics.
This module:
  1. Prepares all input ASCII files required by OneCovariance.
  2. Generates a config.ini that points to those files.
  3. Provides a helper to merge the required HOD/halomodel defaults from
     the OneCovariance master config so the strict parser doesn't crash.

Usage
-----
    from src.covariance import prepare_onecovariance_inputs
    prepare_onecovariance_inputs(
        galaxy_table=table,
        corr_result=result,
        mask_path="data/raw/masks/mask_map_healsparse_EDFS_v1.fits",
        out_dir="results/covariance_matrices/",
        onecovariance_master_config="/path/to/OneCovariance/config.ini",
    )
"""

from __future__ import annotations

import configparser
import os

import numpy as np
from astropy.table import Table


# ---------------------------------------------------------------------------
# Individual input-file writers
# ---------------------------------------------------------------------------

def write_redshift_distribution(
    z: np.ndarray,
    out_path: str,
    n_bins: int = 50,
) -> None:
    """Write n(z) histogram to an ASCII file expected by OneCovariance.

    Parameters
    ----------
    z : np.ndarray
        Redshift values for the galaxy sample.
    out_path : str
        Destination ASCII file path.
    n_bins : int
        Number of histogram bins.
    """
    z_hist, z_edges = np.histogram(z, bins=n_bins, range=(z.min(), z.max()))
    z_mid = (z_edges[:-1] + z_edges[1:]) / 2.0
    np.savetxt(
        out_path,
        np.column_stack((z_mid, z_hist)),
        fmt="%.4f %.4e",
        header="z n(z)",
    )
    print(f"  n(z) → {out_path}")


def write_bias_file(
    z: np.ndarray,
    out_path: str,
    bias_value: float = 3.0,
    n_bins: int = 50,
) -> None:
    """Write a constant linear bias b(z) file.

    Parameters
    ----------
    z : np.ndarray
        Redshift values (used only to define the z grid).
    out_path : str
        Destination ASCII file path.
    bias_value : float
        Constant bias (default 3.0, a reasonable starting point for LRDs).
    n_bins : int
        Number of z bins.
    """
    _, z_edges = np.histogram(z, bins=n_bins, range=(z.min(), z.max()))
    z_mid = (z_edges[:-1] + z_edges[1:]) / 2.0
    bias = np.full_like(z_mid, bias_value)
    np.savetxt(
        out_path,
        np.column_stack((z_mid, bias)),
        fmt="%.4f %.4f",
        header="z b(z)",
    )
    print(f"  bias(z) → {out_path}  (constant b = {bias_value})")


def write_npair_file(
    theta_deg: np.ndarray,
    DD_pairs: np.ndarray,
    out_path: str,
) -> None:
    """Write pair-count file in the format expected by OneCovariance.

    Columns: r_nom, meanr, meanlogr, npairs_weighted

    Parameters
    ----------
    theta_deg : np.ndarray
        Mean angular separation per bin in degrees.
    DD_pairs : np.ndarray
        DD pair counts per bin.
    out_path : str
        Destination ASCII file path.
    """
    meanlogr = np.log(theta_deg)
    out = np.column_stack((theta_deg, theta_deg, meanlogr, DD_pairs))
    np.savetxt(out_path, out, fmt="%.6e", header="r_nom meanr meanlogr npairs_weighted")
    print(f"  npair → {out_path}")


def convert_healsparse_to_healpix(
    mask_path: str,
    out_path: str,
    nside_out: int = 1024,
) -> None:
    """Downgrade a HealSparse mask to a HEALPix FITS map.

    The output HEALPix map uses RING ordering with float32 weights in
    [0, 1] representing the fraction of the high-resolution pixel covered.

    Parameters
    ----------
    mask_path : str
        Path to the HealSparse mask FITS file.
    out_path : str
        Destination HEALPix FITS file path.
    nside_out : int
        Output NSIDE resolution (default 1024 — memory-safe).
    """
    import healsparse
    import healpy as hp

    print(f"  Converting mask: {mask_path}")
    hs_map = healsparse.HealSparseMap.read(mask_path)
    nside_high = hs_map.nside_sparse

    # Fast nested downgrade via integer bit shift
    shift = 2 * int(np.log2(nside_high) - np.log2(nside_out))
    low_res_pixels = hs_map.valid_pixels >> shift
    counts = np.bincount(low_res_pixels)
    unique_low = np.nonzero(counts)[0]
    weights = counts[unique_low] / (1 << shift)

    npix = hp.nside2npix(nside_out)
    hp_map = np.zeros(npix, dtype=np.float32)
    ring_idx = hp.nest2ring(nside_out, unique_low)
    hp_map[ring_idx] = weights.astype(np.float32)

    hp.write_map(out_path, hp_map, overwrite=True, dtype=np.float32)
    print(f"  HEALPix mask (nside={nside_out}) → {out_path}")


# ---------------------------------------------------------------------------
# Survey area helper
# ---------------------------------------------------------------------------

def survey_area_from_catalog(ra: np.ndarray, dec: np.ndarray) -> tuple[float, float]:
    """Estimate survey area from the catalog bounding box (fallback only).

    Returns
    -------
    area_deg2 : float
    area_arcmin2 : float
    """
    area = (ra.max() - ra.min()) * (dec.max() - dec.min())
    return float(area), float(area * 3600.0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prepare_onecovariance_inputs(
    galaxy_table: Table,
    corr_result,
    mask_path: str,
    out_dir: str,
    onecovariance_master_config: str | None = None,
    bias_value: float = 3.0,
    nside_mask: int = 1024,
    z_col: str = "z",
    ra_col: str = "RA",
    dec_col: str = "DEC",
) -> str:
    """Prepare all OneCovariance input files and generate config.ini.

    Parameters
    ----------
    galaxy_table : Table
        Masked galaxy catalog (used for n(z), survey area).
    corr_result : CorrelationResult
        Result from `correlation.compute_2pacf`.
    mask_path : str
        Path to the HealSparse mask file.
    out_dir : str
        Directory where all output files will be written.
    onecovariance_master_config : str or None
        Path to the OneCovariance master config.ini.  If provided, HOD /
        halomodel sections are appended so the strict parser does not fail.
    bias_value : float
        Constant linear bias for the galaxy sample.
    nside_mask : int
        NSIDE for the downgraded HEALPix mask.
    z_col, ra_col, dec_col : str
        Column names in galaxy_table.

    Returns
    -------
    str
        Path to the generated config.ini.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Sub-directories
    nz_dir = os.path.join(out_dir, "input", "redshift_distribution")
    bias_dir = os.path.join(out_dir, "input", "bias")
    npair_dir = os.path.join(out_dir, "input", "npair")
    mask_dir = os.path.join(out_dir, "input", "mask")
    output_dir = os.path.join(out_dir, "output")
    for d in (nz_dir, bias_dir, npair_dir, mask_dir, output_dir):
        os.makedirs(d, exist_ok=True)

    print("Preparing OneCovariance inputs …")

    # 1. n(z)
    z = np.asarray(galaxy_table[z_col], dtype=float)
    write_redshift_distribution(z, os.path.join(nz_dir, "nz_bin1.ascii"))

    # 2. bias
    write_bias_file(z, os.path.join(bias_dir, "bias.ascii"), bias_value=bias_value)

    # 3. pair counts
    write_npair_file(
        corr_result.theta_deg,
        corr_result.DD,
        os.path.join(npair_dir, "npair_gg_Bin1_Bin1.ascii"),
    )

    # 4. mask
    healpix_mask_file = f"mask_healpix_nside{nside_mask}.fits"
    convert_healsparse_to_healpix(
        mask_path,
        os.path.join(mask_dir, healpix_mask_file),
        nside_out=nside_mask,
    )

    # 5. Survey specs
    ra = np.asarray(galaxy_table[ra_col], dtype=float)
    dec = np.asarray(galaxy_table[dec_col], dtype=float)
    area_deg2, area_arcmin2 = survey_area_from_catalog(ra, dec)
    n_eff = len(galaxy_table) / area_arcmin2

    min_sep_arcmin = float(corr_result.theta_arcmin.min())
    max_sep_arcmin = float(corr_result.theta_arcmin.max())
    nbins = len(corr_result.theta_deg)

    print(f"  Survey area (bbox approx): {area_deg2:.2f} deg²")
    print(f"  n_eff: {n_eff:.5f} gal/arcmin²")

    # 6. config.ini
    config_path = os.path.join(out_dir, "config.ini")
    config_content = f"""[covariance terms]
gauss = True
nongauss = False
ssc = False

[observables]
cosmic_shear = False
ggl = False
clustering = True
est_clust = w
cross_terms = False

[covTHETAspace settings]
theta_min_clustering = {min_sep_arcmin:.2f}
theta_max_clustering = {max_sep_arcmin:.2f}
theta_bins_clustering = {nbins}
theta_type_clustering = log
theta_accuracy = 1e-2
integration_intervals = 400

[survey specs]
survey_area_clust_in_deg2 = {area_deg2:.2f}
survey_area_ggl_in_deg2 = {area_deg2:.2f}
survey_area_lensing_in_deg2 = {area_deg2:.2f}
n_eff_clust = {n_eff:.5f}
n_eff_lensing = {n_eff:.5f}
ellipticity_dispersion = 0.27
mask_directory = {mask_dir}/
mask_file_clust = {healpix_mask_file}

[redshift]
zclust_directory = {nz_dir}/
zclust_file = nz_bin1.ascii
value_loc_in_clustbin = mid

[bias]
bias_files = {bias_dir}/bias.ascii

[cosmo]
sigma8 = 0.81
h = 0.67
omega_m = 0.32
omega_b = 0.049
omega_de = 0.68
w0 = -1.0
wa = 0.0
ns = 0.965
neff = 3.046
m_nu = 0.06

[tabulated inputs files]
npair_directory = {npair_dir}/
npair_gg_file = npair_gg_Bin?_Bin?.ascii

[output settings]
directory = {output_dir}/
file = cov_list.dat, cov_matrix.mat
style = list, matrix

[misc]
num_cores = 4
"""

    with open(config_path, "w") as fh:
        fh.write(config_content)

    # Optionally append HOD defaults from master config
    if onecovariance_master_config and os.path.isfile(onecovariance_master_config):
        master = configparser.ConfigParser()
        master.read(onecovariance_master_config)
        with open(config_path, "a") as fh:
            for section in ["halomodel evaluation", "hod", "powspec evaluation",
                            "covELLspace settings"]:
                if master.has_section(section):
                    fh.write(f"\n[{section}]\n")
                    for key, val in master.items(section):
                        fh.write(f"{key} = {val}\n")
        print(f"  HOD defaults appended from {onecovariance_master_config}")

    print(f"  config.ini → {config_path}")
    print("OneCovariance inputs ready.")
    print(f"Run with:  python covariance.py {config_path}")
    return config_path
