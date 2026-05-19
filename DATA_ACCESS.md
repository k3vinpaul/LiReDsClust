# Data Access

The raw data files used in this project are **not publicly available** at this time.
They are proprietary to the Euclid Collaboration and/or to the collaborators who
provided the N-body simulation catalogs.

If you wish to reproduce or extend the analysis, you must request access to these
files directly. See the contact information below.

---

## Restricted files

### Galaxy catalog

| File | Location | Origin |
|---|---|---|
| `LRD_MarIRAC.fits` | `data/raw/` | Euclid Collaboration — provided by Dr. Laura Bisigello |

This is the primary Little Red Dot (LRD) catalog for the Euclid Deep Fields.
It contains 3,341 objects with photometric redshifts, UV spectral slopes, and
Spitzer/IRAC detection flags.

### Survey masks

| File | Location | Origin |
|---|---|---|
| `mask_map_healsparse_EDFS_v1.fits` | `data/raw/masks/` | Euclid Collaboration — Euclid Deep Field South v1 mask |
| `mask_healpix_nside1024.fits` | `data/raw/masks/` | Derived from the above (downgraded for OneCovariance) |

The HealSparse mask encodes the observed footprint of the Euclid Deep Field South (EDF-S).
It is used to generate masked random catalogs and to prepare the OneCovariance covariance matrix.

### N-body simulation files

| File pair | Config key | alpha value |
|---|---|---|
| `RA_sim_alpha0.5.npy` / `DEC_sim_alpha0.5.npy` | `alpha0.5` | 0.5 |
| `RA_sim_alpha1.npy` / `DEC_sim_alpha1.npy` | `alpha1` | 1 |
| `RA_sim_alpha3.npy` / `DEC_sim_alpha3.npy` | `alpha3` | 3 |
| `RA_sim_noalpha.npy` / `DEC_sim_noalpha.npy` | `default` | unlabelled |

These sky catalogs were generated with the ALPT code (Kitaura & Sinigaglia 2025)
and provided by Prof. Francesco Sinigaglia. They are used to produce simulation-based
error bars for the 2PACF measurement.

---

## How to request access

Send an email to **Kevin Espinosa de los Monteros** at:

**k3vinpaul4502p@gmail.com**

Please include in your message:
- Your name and institutional affiliation
- A brief description of how you intend to use the data
- Whether you are requesting the catalog, the masks, the simulation files, or all of them

Access to the Euclid catalog and masks is subject to Euclid Collaboration data policy
and will require approval from the data custodians. Access to the simulation files
is subject to approval by Prof. Sinigaglia. Kevin will forward your request and keep
you informed of the response.

---

## Once you have access

Place the files in the following locations relative to the project root:

```
data/
└── raw/
    ├── LRD_MarIRAC.fits
    ├── masks/
    │   ├── mask_map_healsparse_EDFS_v1.fits
    │   └── mask_healpix_nside1024.fits
    └── simulations/
        ├── RA_sim_alpha0.5.npy
        ├── DEC_sim_alpha0.5.npy
        ├── RA_sim_alpha1.npy
        ├── DEC_sim_alpha1.npy
        ├── RA_sim_alpha3.npy
        ├── DEC_sim_alpha3.npy
        ├── RA_sim_noalpha.npy
        └── DEC_sim_noalpha.npy
```

These paths are already configured in `config/analysis.yml`. No changes to the
code are needed once the files are in place. Run `ls data/raw/` to verify.

---

## Expected timeline for public release

The Euclid Deep Field data are expected to become publicly available following
the Euclid Collaboration's standard data release schedule. When the data are
officially released, this repository will be updated accordingly.
