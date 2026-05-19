"""
randoms.py
----------
Generation of uniform random catalogs respecting the survey mask.

The standard approach is:
  1. Draw candidate positions uniformly on the sphere inside the survey
     bounding box.
  2. Keep only candidates that fall inside valid HealSparse mask pixels.
  3. Repeat until the requested number of randoms is reached.

When no mask is available (mask_map=None), step 2 is skipped and positions
are drawn uniformly across the bounding box.

The output is an Astropy Table with RA/Dec columns, ready to be written to
FITS and used as input to `correlation.compute_2pacf`.
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table


def _draw_uniform_sphere(
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw *n* positions uniformly distributed on the sphere within a bbox.

    Uses the equal-area projection: cos(Dec) is uniform in [cos(dec_max),
    cos(dec_min)] rather than Dec being uniform — this avoids over-sampling
    near the equator.
    """
    ra = rng.uniform(ra_min, ra_max, size=n)
    cos_dec_min = np.cos(np.radians(90.0 - dec_min))
    cos_dec_max = np.cos(np.radians(90.0 - dec_max))
    cos_dec = rng.uniform(
        min(cos_dec_min, cos_dec_max),
        max(cos_dec_min, cos_dec_max),
        size=n,
    )
    dec = 90.0 - np.degrees(np.arccos(cos_dec))
    return ra, dec


def generate_random_catalog(
    n_randoms: int,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    mask_map=None,
    nside: int | None = None,
    batch_factor: int = 10,
    seed: int | None = None,
) -> Table:
    """Generate a random catalog inside a field bounding box.

    If *mask_map* is provided, positions are filtered through the HealSparse
    mask (only pixels with value == 1 are kept).  If *mask_map* is None,
    positions are drawn uniformly across the bounding box without any mask
    filtering — appropriate for fields without an official mask.

    Parameters
    ----------
    n_randoms : int
        Number of valid random points to generate.
    ra_min, ra_max : float
        RA bounding box in degrees.
    dec_min, dec_max : float
        Dec bounding box in degrees.
    mask_map : HealSparseMap or None
        Loaded HealSparse mask.  Pass None to skip mask filtering.
    nside : int or None
        NSIDE of the sparse mask.  Required when mask_map is not None.
    batch_factor : int
        Oversampling factor per batch to account for mask rejection.
        Default 10 means we draw 10× n_randoms per batch.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    Table
        Astropy Table with columns ``RA`` and ``DEC`` (degrees).
    """
    rng = np.random.default_rng(seed)

    # ── Unmasked case — simple uniform draw ──────────────────────────────────
    if mask_map is None:
        print(f"Generating {n_randoms:,} uniform random points (no mask) …")
        ra_out, dec_out = _draw_uniform_sphere(
            ra_min, ra_max, dec_min, dec_max, n_randoms, rng
        )
        print(f"Done. {len(ra_out):,} uniform random points generated.")
        return Table([ra_out, dec_out], names=["RA", "DEC"])

    # ── Masked case — iterative batch rejection ───────────────────────────────
    import healpy as hp

    ra_accepted: list[np.ndarray] = []
    dec_accepted: list[np.ndarray] = []
    n_collected = 0

    print(f"Generating {n_randoms:,} masked random points …")
    n_attempt = 0
    while n_collected < n_randoms:
        n_draw = batch_factor * n_randoms
        n_attempt += n_draw
        ra_cand, dec_cand = _draw_uniform_sphere(
            ra_min, ra_max, dec_min, dec_max, n_draw, rng
        )

        # Check mask
        theta = np.radians(90.0 - dec_cand)
        phi = np.radians(ra_cand)
        pix = hp.ang2pix(nside, theta, phi, nest=True)
        in_mask = mask_map.get_values_pix(pix) == 1

        ra_accepted.append(ra_cand[in_mask])
        dec_accepted.append(dec_cand[in_mask])
        n_collected += int(in_mask.sum())
        print(f"  batch: {in_mask.sum():,} accepted  (total so far: {n_collected:,})")

    ra_out = np.concatenate(ra_accepted)[:n_randoms]
    dec_out = np.concatenate(dec_accepted)[:n_randoms]

    print(f"Done. {len(ra_out):,} random points generated "
          f"(mask acceptance rate: {100*n_randoms/n_attempt:.1f}%)")

    return Table([ra_out, dec_out], names=["RA", "DEC"])


def apply_periodic_boundaries(
    ra: np.ndarray, dec: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Enforce valid celestial coordinate ranges on simulation outputs.

    RA is wrapped modulo 360°.  Dec is reflected at the poles (±90°)
    with a 180° flip of RA — the physically correct boundary condition.

    Parameters
    ----------
    ra, dec : np.ndarray
        Input coordinate arrays in degrees.

    Returns
    -------
    ra_clean, dec_clean : np.ndarray
        Coordinates with all values inside [0, 360) × [−90, 90].
    """
    ra_out = ra % 360.0
    dec_out = dec.copy()

    # Iterate until all Dec values are within [-90, 90]
    for _ in range(10):  # bounded loop — should converge in ≤2 iterations
        north = dec_out > 90.0
        if north.any():
            dec_out[north] = 180.0 - dec_out[north]
            ra_out[north] = (ra_out[north] + 180.0) % 360.0
        south = dec_out < -90.0
        if south.any():
            dec_out[south] = -180.0 - dec_out[south]
            ra_out[south] = (ra_out[south] + 180.0) % 360.0
        if not north.any() and not south.any():
            break

    return ra_out, dec_out
