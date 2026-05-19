"""
catalog.py
----------
Utilities for loading, filtering, and subsetting galaxy catalogs from FITS files.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.table import Table


def load_fits_catalog(path: str, hdu: int = 1) -> Table:
    """Load a FITS binary table catalog into an Astropy Table.

    Parameters
    ----------
    path : str
        Path to the FITS file.
    hdu : int
        HDU index of the binary table (default 1).

    Returns
    -------
    Table
        Astropy Table with all catalog columns.
    """
    table = Table.read(path, hdu=hdu)
    print(f"Loaded catalog: {path}")
    print(f"  Objects : {len(table):,}")
    print(f"  Columns : {table.colnames}")
    return table


def get_radec(table: Table, ra_col: str = "RA", dec_col: str = "DEC") -> tuple[np.ndarray, np.ndarray]:
    """Extract RA and Dec arrays from a catalog table.

    Parameters
    ----------
    table : Table
    ra_col, dec_col : str
        Column names for RA and Dec (degrees).

    Returns
    -------
    ra, dec : np.ndarray
        Coordinate arrays in degrees.
    """
    ra = np.asarray(table[ra_col], dtype=float)
    dec = np.asarray(table[dec_col], dtype=float)
    print(f"  RA  range : {ra.min():.4f} – {ra.max():.4f} deg")
    print(f"  Dec range : {dec.min():.4f} – {dec.max():.4f} deg")
    return ra, dec


def apply_sky_mask(
    table: Table,
    mask_map,
    nside: int,
    ra_col: str = "RA",
    dec_col: str = "DEC",
    batch_size: int = 100_000,
) -> Table:
    """Keep only catalog rows whose sky position falls inside a HealSparse mask.

    Parameters
    ----------
    table : Table
        Input galaxy catalog.
    mask_map : HealSparseMap
        Loaded HealSparse mask (pixel value == 1 means valid).
    nside : int
        NSIDE of the sparse mask.
    ra_col, dec_col : str
        Column names for coordinates.
    batch_size : int
        Number of points per processing batch (memory management).

    Returns
    -------
    Table
        Subset of *table* containing only galaxies inside the mask.
    """
    import healpy as hp

    ra = np.asarray(table[ra_col], dtype=float)
    dec = np.asarray(table[dec_col], dtype=float)
    n = len(ra)
    valid = np.zeros(n, dtype=bool)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        theta = np.radians(90.0 - dec[start:end])
        phi = np.radians(ra[start:end])
        pix = hp.ang2pix(nside, theta, phi, nest=True)
        valid[start:end] = mask_map.get_values_pix(pix) == 1

    masked = table[valid]
    print(f"Mask applied: {valid.sum():,} / {n:,} objects kept ({100*valid.mean():.1f}%)")
    return masked


def filter_radec_range(
    table: Table,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    ra_col: str = "RA",
    dec_col: str = "DEC",
) -> Table:
    """Rectangular sky cut on RA / Dec.

    Parameters
    ----------
    table : Table
    ra_min, ra_max : float
        Right Ascension bounds in degrees.
    dec_min, dec_max : float
        Declination bounds in degrees.

    Returns
    -------
    Table
        Subset of *table* inside the bounding box.
    """
    ra = np.asarray(table[ra_col])
    dec = np.asarray(table[dec_col])
    sel = (ra >= ra_min) & (ra <= ra_max) & (dec >= dec_min) & (dec <= dec_max)
    subset = table[sel]
    print(f"Sky cut [{ra_min},{ra_max}] × [{dec_min},{dec_max}]: "
          f"{sel.sum():,} / {len(table):,} objects kept")
    return subset


def save_fits_catalog(table: Table, path: str) -> None:
    """Write an Astropy Table to a FITS binary table file.

    Parameters
    ----------
    table : Table
    path : str
        Output file path.
    """
    table.write(path, format="fits", overwrite=True)
    print(f"Saved catalog ({len(table):,} objects) → {path}")
