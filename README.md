# LRD 2PACF Pipeline

Two-point angular correlation function (2PACF) analysis of **Little Red Dot (LRD)**
galaxies in the **Euclid Deep Field South (EDFS)**.

LRDs are compact, red galaxies discovered at $z \sim 4$–8 by JWST.
Measuring their angular clustering constrains their host halo masses and
large-scale environment — key inputs to early galaxy-formation and AGN
feedback models.

> **Data access:** The raw catalog, survey masks, and N-body simulation files
> used in this analysis are proprietary to the Euclid Collaboration and collaborators
> and are **not included in this repository**. To request access, contact
> Kevin Espinosa de los Monteros at **k3vinpaul4502p@gmail.com**.
> See [DATA_ACCESS.md](DATA_ACCESS.md) for full details.

---

## Table of contents

1. [Repository layout](#1-repository-layout)
2. [Installation](#2-installation)
3. [Data setup](#3-data-setup)
4. [Configuration: all parameters in one place](#4-configuration-all-parameters-in-one-place)
5. [Step-by-step: running the pipeline](#5-step-by-step-running-the-pipeline)
   - [Step 1 — Compute the 2PACF](#step-1--compute-the-2pacf)
   - [Step 2 — Simulation-based errors](#step-2--simulation-based-errors)
   - [Step 3 — OneCovariance matrix](#step-3--onecovariance-matrix)
6. [Notebooks: interactive exploration](#6-notebooks-interactive-exploration)
7. [Changing analysis parameters](#7-changing-analysis-parameters)
8. [Simulation files and alpha values](#8-simulation-files-and-alpha-values)
9. [Understanding the outputs](#9-understanding-the-outputs)
10. [Scientific methods summary](#10-scientific-methods-summary)

---

## 1. Repository layout

```
LRDs/
│
├── config/
│   ├── analysis.yml               ← ALL parameters live here (edit this!)
│   └── onecovariance_template.ini ← Template config for OneCovariance
│
├── data/
│   ├── raw/
│   │   ├── LRD_MarIRAC.fits              ← Main galaxy catalog (3 341 objects)
│   │   ├── masks/
│   │   │   ├── mask_map_healsparse_EDFS_v1.fits  ← HealSparse survey mask
│   │   │   └── mask_healpix_nside1024.fits        ← Downgraded mask for OneCovariance
│   │   └── simulations/
│   │       ├── RA_sim_alpha0.5.npy   ← N-body sim, α=0.5  (see §8)
│   │       ├── DEC_sim_alpha0.5.npy
│   │       ├── RA_sim_alpha1.npy     ← N-body sim, α=1
│   │       ├── DEC_sim_alpha1.npy
│   │       ├── RA_sim_alpha3.npy     ← N-body sim, α=3
│   │       ├── DEC_sim_alpha3.npy
│   │       ├── RA_sim_noalpha.npy    ← Original sim (unlabelled α)
│   │       └── DEC_sim_noalpha.npy
│   └── processed/                   ← Cached random catalogs (auto-generated)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    ← Catalog, flags, sky distribution
│   ├── 02_mask_analysis.ipynb       ← HealSparse mask, effective area
│   ├── 03_2pacf_irac_footprint.ipynb ← Primary 2PACF, step-by-step
│   ├── 04_2pacf_irac_detected.ipynb  ← IRAC-detected sub-sample
│   ├── 05_error_estimation.ipynb     ← Poisson vs simulation errors
│   ├── 06_covariance_analysis.ipynb  ← OneCovariance matrix
│   └── 07_final_comparison.ipynb     ← All samples + science summary
│
├── results/
│   ├── correlation_functions/       ← 2PACF .npz files
│   ├── covariance_matrices/         ← OneCovariance inputs/outputs
│   └── figures/                     ← All figures (PDF/PNG)
│
├── scripts/
│   ├── run_2pacf.py                 ← Compute 2PACF for one sample
│   ├── run_simulations.py           ← Simulation-based error bars
│   └── run_covariance.py            ← Prepare OneCovariance inputs
│
├── src/                             ← Reusable Python modules
│   ├── catalog.py                   ← FITS loading, filtering, masking
│   ├── randoms.py                   ← Masked random catalog generation
│   ├── correlation.py               ← Landy-Szalay 2PACF via TreeCorr
│   ├── errors.py                    ← Poisson + simulation errors
│   ├── covariance.py                ← OneCovariance input preparation
│   ├── plotting.py                  ← Publication figures
│   └── config.py                    ← YAML config loader (singleton)
│
├── archive/                         ← Original notebooks (preserved, not used)
├── environment.yml
└── README.md                        ← This file
```

---

## 2. Installation

### Step 2.1 — Clone or locate the repository

```bash
cd /home/k3vinpaul/LRDs    # already done
```

### Step 2.2 — Create the conda environment

```bash
conda env create -f environment.yml
conda activate lrds
```

This installs Python 3.10, NumPy, SciPy, Matplotlib, Astropy, healpy,
HealSparse, TreeCorr, Jupyter, and PyYAML.

### Step 2.3 — (Optional) Install OneCovariance

OneCovariance is only needed for Step 3 (covariance matrix).
It is not on PyPI — install from source:

```bash
git clone https://github.com/rreischke/OneCovariance
cd OneCovariance
pip install -e .
cd ..
```

### Step 2.4 — Verify the installation

```bash
python -c "import treecorr, healsparse, healpy; print('OK')"
```

---

## 3. Data setup

**The raw data files are not included in this repository.**
They are proprietary to the Euclid Collaboration and collaborators.
See [DATA_ACCESS.md](DATA_ACCESS.md) for the full list of restricted files
and instructions for requesting access.

Once you have been granted access, place the files as follows:

### Required files

| File | Location | Description |
|---|---|---|
| `LRD_MarIRAC.fits` | `data/raw/` | Main LRD galaxy catalog (Euclid Collaboration) |
| `mask_map_healsparse_EDFS_v1.fits` | `data/raw/masks/` | HealSparse EDF-S survey mask (Euclid Collaboration) |
| `mask_healpix_nside1024.fits` | `data/raw/masks/` | Downgraded HEALPix mask for OneCovariance (derived) |
| `RA_sim_alpha0.5.npy` | `data/raw/simulations/` | Sim alpha=0.5, RA file — contains Dec (see §8) |
| `DEC_sim_alpha0.5.npy` | `data/raw/simulations/` | Sim alpha=0.5, Dec file — contains RA (see §8) |
| *(other sim files)* | `data/raw/simulations/` | See §8 and DATA_ACCESS.md |

### Check that files are in place

```bash
ls data/raw/
ls data/raw/masks/
ls data/raw/simulations/
```

---

## 4. Configuration: all parameters in one place

**`config/analysis.yml` is the only file you should edit** to change analysis
settings. Nothing is hardcoded in scripts or notebooks.

### Most important user-configurable sections

#### Angular bins

```yaml
bins:
  min_sep:   0.009   # degrees ≈ 0.54 arcmin  — minimum angular scale
  max_sep:   3.3     # degrees ≈ 198 arcmin   — maximum angular scale
  nbins:     15      # number of log-spaced bins
  sep_units: deg     # unit passed to TreeCorr
  bin_slop:  0.01    # TreeCorr bin-edge tolerance (1%)
```

**How to choose `max_sep`**: The Landy & Szalay (1993) edge-effect criterion
is $\theta_{\rm max} < L_{\rm min}/3$. For the EDFS, $L_{\rm min} = 7°$ (Dec
direction), giving a strict limit of 2.33°. Using 3.3° (≈ $L_{\rm min}/2.1$)
extends the dynamic range at modest edge risk. The original notebooks also
tested `max_sep = 6.36°` — increase if you need a wider range.

#### Random catalog

```yaml
randoms:
  n_randoms_factor: 100   # N_rand = 100 × N_galaxies
  seed: 42
```

Higher `n_randoms_factor` reduces shot noise in pair counts but takes longer.
The thesis analysis used 100×.

#### Active simulation

```yaml
simulations:
  active: alpha0.5   # which simulation to use by default
```

Change `active` to `alpha1`, `alpha3`, or `default` to switch simulation.
See §8 for details.

---

## 5. Step-by-step: running the pipeline

### Step 1 — Compute the 2PACF

Run this for the primary sample (IRAC-footprint, Euclid mask applied):

```bash
python scripts/run_2pacf.py --sample irac_footprint_masked
```

**What it does:**
1. Loads `data/raw/LRD_MarIRAC.fits`
2. Filters to `IRAC-footprint = True`
3. Applies the EDFS bounding box (RA 56–67°, Dec −52° to −45°)
4. Applies the HealSparse mask
5. Loads or generates a random catalog (100× galaxies, cached to `data/processed/`)
6. Computes DD, DR, RR pair counts with TreeCorr
7. Applies the Landy-Szalay estimator: $w(\theta) = (DD - 2DR + RR)/RR$
8. Saves the result to `results/correlation_functions/2pacf_irac_footprint_masked.npz`
9. Saves four diagnostic figures to `results/figures/`

**Other samples:**

```bash
python scripts/run_2pacf.py --sample irac_footprint_unmasked
python scripts/run_2pacf.py --sample irac_detected_masked
python scripts/run_2pacf.py --sample irac_detected_unmasked
python scripts/run_2pacf.py --sample no_irac_footprint_masked
```

**Useful flags:**

```bash
# Force regeneration of the random catalog (e.g. after changing n_randoms_factor)
python scripts/run_2pacf.py --sample irac_footprint_masked --regen-randoms

# Skip figure output (faster, good for batch runs)
python scripts/run_2pacf.py --sample irac_footprint_masked --no-plots

# Save 2PACF result to a custom path
python scripts/run_2pacf.py --sample irac_footprint_masked --output my_result.npz
```

**Expected output (console):**
```
============================================================
Sample: irac_footprint_masked
  Galaxies inside the IRAC imaging footprint, Euclid mask applied
============================================================
  After IRAC-footprint=True: 1509 objects
  After sky cut : 1509
  Loading HealSparse mask: data/raw/masks/mask_map_healsparse_EDFS_v1.fits
  Final sample size: 1490 galaxies

Loading cached random catalog: data/processed/randoms_irac_footprint_masked.fits
  149000 random points loaded (ratio: 100.0×)

Computing 2PACF …
  Bins: 15 log-spaced from 0.009° to 3.3°
  Computing DD … 4321 pairs
  Computing DR … 843201 pairs
  Computing RR … 11032451 pairs
  w(θ) computed: 11/15 valid bins (w range: -0.0412 – 0.3821)
```

---

### Step 2 — Simulation-based errors

#### 2a. Run for the active simulation (set in config)

```bash
python scripts/run_simulations.py --sample irac_footprint_masked
```

**What it does:**
1. Loads the simulation file specified by `simulations.active` in `config/analysis.yml`
2. Applies the column-swap correction (RA file contains Dec and vice versa)
3. Generates 100 synthetic catalogs by random rigid sky rotations
4. For each catalog: filters through the survey mask, computes the 2PACF
5. Computes the standard deviation per bin as the simulation error bar
6. Saves results to `results/covariance_matrices/sim_errors_{sample}_{sim_name}.npz`

#### 2b. Pick a specific simulation

```bash
python scripts/run_simulations.py --sample irac_footprint_masked --sim-name alpha1
python scripts/run_simulations.py --sample irac_footprint_masked --sim-name alpha3
python scripts/run_simulations.py --sample irac_footprint_masked --sim-name default
```

#### 2c. Run ALL simulations and produce a comparison plot

```bash
python scripts/run_simulations.py --sample irac_footprint_masked --all-sims
```

This runs all four simulations (alpha0.5, alpha1, alpha3, default) and saves
a comparison figure to `results/figures/sim_comparison_{sample}.pdf`.

#### 2d. Useful flags

```bash
# Force regeneration of synthetic catalogs (e.g. after changing n_catalogs)
python scripts/run_simulations.py --sample irac_footprint_masked --regen-catalogs

# Use more synthetic catalogs for a smoother error estimate
python scripts/run_simulations.py --sample irac_footprint_masked --n-catalogs 500

# Skip figures
python scripts/run_simulations.py --sample irac_footprint_masked --no-plots
```

**Note**: Random catalogs must exist before running simulations.
If you get a FileNotFoundError, run Step 1 first (it generates the random catalog).

---

### Step 3 — OneCovariance matrix

#### 3a. Prepare inputs

```bash
python scripts/run_covariance.py --sample irac_footprint_masked
```

**What it does:**
1. Loads the galaxy catalog for the sample
2. Loads the 2PACF result from Step 1
3. Writes ASCII input files for OneCovariance:
   - `n(z)` — redshift distribution
   - `bias.txt` — constant linear bias $b(z) = 3.0$ (set in config)
   - `npair.txt` — DD pair counts per bin
   - `mask_healpix_nside1024.fits` — downgraded HEALPix mask
4. Generates a `config.ini` for OneCovariance
5. Saves everything to `results/covariance_matrices/{sample}/`

#### 3b. Run OneCovariance

```bash
# Run OneCovariance automatically after preparing inputs
python scripts/run_covariance.py --sample irac_footprint_masked --run

# Or point to the OneCovariance master config for advanced settings
python scripts/run_covariance.py \
    --sample irac_footprint_masked \
    --run \
    --onecovariance-master /path/to/OneCovariance/config.ini

# Override the bias value
python scripts/run_covariance.py --sample irac_footprint_masked --bias 5.0
```

---

## 6. Notebooks: interactive exploration

Open JupyterLab from the project root:

```bash
conda activate lrds
cd /home/k3vinpaul/LRDs
jupyter lab
```

Then open the notebooks in order:

| Notebook | What you'll see |
|---|---|
| `01_data_exploration.ipynb` | Catalog columns, IRAC flag counts, redshift and sky distributions |
| `02_mask_analysis.ipynb` | HealSparse mask visualisation, effective area, mask effect on catalog |
| `03_2pacf_irac_footprint.ipynb` | Full 2PACF computation, results table, power-law fit |
| `04_2pacf_irac_detected.ipynb` | Same for the tiny IRAC-detected sub-sample, S/N discussion |
| `05_error_estimation.ipynb` | Poisson vs simulation errors; multi-simulation comparison |
| `06_covariance_analysis.ipynb` | OneCovariance matrix, correlation heatmap, three-way error comparison |
| `07_final_comparison.ipynb` | All samples on one plot, power-law fits, science conclusions |

**Note**: Notebooks 03–07 check for pre-computed results and skip the expensive
computation if results already exist. Run the scripts first (Steps 1–3) or let
the notebook compute on-the-fly.

---

## 7. Changing analysis parameters

All parameters are in `config/analysis.yml`. After changing them:

1. **Bin parameters** (`bins.min_sep`, `bins.max_sep`, `bins.nbins`):
   Rerun Step 1 with `--no-plots` to get the new 2PACF quickly, then
   rerun Step 2 with `--regen-catalogs` to recompute simulation errors.

   ```bash
   python scripts/run_2pacf.py --sample irac_footprint_masked
   python scripts/run_simulations.py --sample irac_footprint_masked --regen-catalogs
   ```

2. **Random factor** (`randoms.n_randoms_factor`):
   Rerun Step 1 with `--regen-randoms`:

   ```bash
   python scripts/run_2pacf.py --sample irac_footprint_masked --regen-randoms
   ```

3. **Active simulation** (`simulations.active`):
   Change the value to `alpha1`, `alpha3`, or `default`, then rerun Step 2.

4. **Galaxy bias** (`bias.value`):
   Rerun Step 3 — this only affects the OneCovariance model.

---

## 8. Simulation files and alpha values

The collaborator provided four N-body simulation sky catalogs,
each with a different clustering amplitude parameter α.

| Config key | File (RA) | File (Dec) | α value | Notes |
|---|---|---|---|---|
| `alpha0.5` | `RA_sim_alpha0.5.npy` | `DEC_sim_alpha0.5.npy` | 0.5 | Most used in PostMeting analysis |
| `alpha1` | `RA_sim_alpha1.npy` | `DEC_sim_alpha1.npy` | 1 | |
| `alpha3` | `RA_sim_alpha3.npy` | `DEC_sim_alpha3.npy` | 3 | |
| `default` | `RA_sim_noalpha.npy` | `DEC_sim_noalpha.npy` | unlabelled | Original collaborator file |

### Column-swap bug

Despite their filenames, **RA_sim_*.npy actually contains Dec values** and
**DEC_sim_*.npy contains RA values**. This is a known quirk of the collaborator's
output format. The swap is corrected automatically in `src/errors.py` and
`scripts/run_simulations.py` — you do not need to do anything manually.

### Selecting a simulation

In `config/analysis.yml`:
```yaml
simulations:
  active: alpha0.5   # ← change this
```

Or on the command line:
```bash
python scripts/run_simulations.py --sample irac_footprint_masked --sim-name alpha1
```

### Adding a new simulation

1. Copy the `.npy` files to `data/raw/simulations/`
2. Add an entry under `simulations.available` in `config/analysis.yml`:

```yaml
simulations:
  available:
    my_new_sim:
      ra_file:  data/raw/simulations/RA_sim_mynew.npy
      dec_file: data/raw/simulations/DEC_sim_mynew.npy
      label:    "My new simulation (α = X)"
```

3. Set `active: my_new_sim` or use `--sim-name my_new_sim` on the CLI.

---

## 9. Understanding the outputs

### 2PACF result file (`.npz`)

Located at `results/correlation_functions/2pacf_{sample}.npz`.
Load it in Python:

```python
import numpy as np
from src.correlation import CorrelationResult

result = CorrelationResult.load("results/correlation_functions/2pacf_irac_footprint_masked.npz")

result.theta_arcmin    # angular bin centres [arcmin]
result.w               # w(θ) — can be negative (physically meaningful)
result.w_err_poisson   # Poisson error per bin (NaN for empty bins)
result.DD              # raw galaxy-galaxy pair counts
result.n_galaxies      # number of galaxies in sample
result.valid           # boolean mask: True where w and w_err are finite
```

### Simulation errors file (`.npz`)

Located at `results/covariance_matrices/sim_errors_{sample}_{sim_name}.npz`.

```python
data = np.load("results/covariance_matrices/sim_errors_irac_footprint_masked_alpha0.5.npz")
data["theta_arcmin"]   # angular bin centres
data["mean_w"]         # mean w(θ) across all synthetic catalogs
data["std_w"]          # standard deviation = simulation error bar
data["all_w"]          # shape (n_catalogs, n_bins) — all individual w(θ)
data["n_catalogs"]     # number of synthetic catalogs used
```

### Figures

All figures are saved to `results/figures/`. Key outputs:

| File | Description |
|---|---|
| `2pacf_{sample}.pdf` | Main 2PACF plot with Poisson errors and power-law fit |
| `skyplot_{sample}.pdf` | Galaxy and random sky positions |
| `data_vs_randoms_{sample}.pdf` | Density comparison (data vs randoms) |
| `2pacf_sim_errors_{sample}_{sim}.pdf` | Observed 2PACF with simulation error band |
| `sim_comparison_{sample}.pdf` | All simulations on one plot |

---

## 10. Mathematical methods summary

### Estimator

The Landy-Szalay (1993) estimator:

$$w(\theta) = \frac{DD - 2\,DR + RR}{RR}$$

where $DD$, $DR$, $RR$ are **normalised** pair counts:

$$DD_{\rm norm} = \frac{DD}{N_g(N_g-1)/2}, \quad
DR_{\rm norm} = \frac{DR}{N_g\,N_r}, \quad
RR_{\rm norm} = \frac{RR}{N_r(N_r-1)/2}$$

Pair counting uses [TreeCorr](https://github.com/rmjarvis/TreeCorr) with
logarithmic bin spacing.

### Angular bins

15 log-spaced bins from $0.009°$ to $3.3°$ (0.54–198 arcmin).  
Dynamic range: $3.3/0.009 \approx 367\times$ — $\approx 0.17$ dex per bin.

The maximum separation follows the Landy & Szalay (1993) edge-effect criterion:
$\theta_{\rm max} < L_{\rm min}/3 = 7°/3 = 2.33°$ (strict).
We use $3.3°$ as a compromise to extend the dynamic range.

### Poisson error

$$\sigma_w(\theta) = \sqrt{\frac{1 + w(\theta)}{DD(\theta)}}$$

Valid for uncorrelated bins; underestimates when cosmic variance is significant.

### Simulation error

Standard deviation of $w(\theta)$ across $N_{\rm cat} = 100$ synthetic catalogs.
Each catalog is produced by:

1. Drawing a random rigid offset $(\Delta\alpha, \Delta\delta)$.
2. Rotating the full simulation by that offset.
3. Filtering through the Euclid mask.

Captures cosmic variance that Poisson errors miss.

### Survey

| Property | Value |
|---|---|
| Field | Euclid Deep Field South (EDFS) |
| RA range | 56° – 67° |
| Dec range | −52° – −45° |
| $L_{\rm min}$ | 7° (Dec direction) |
| Mask | HealSparse EDFS v1 |
| Catalog | LRD_MarIRAC.fits (3 341 objects) |
| Randoms | 100× $N_{\rm gal}$, uniform, masked |

### Samples

| Key | Selection | Mask | N gal (approx.) |
|---|---|---|---|
| `irac_footprint_masked` | IRAC-footprint=True | Yes | ~1490 |
| `irac_footprint_unmasked` | IRAC-footprint=True | No | ~1509 |
| `irac_detected_masked` | IRAC-detected=True | Yes | ~29 |
| `irac_detected_unmasked` | IRAC-detected=True | No | ~29 |
| `no_irac_footprint_masked` | IRAC-footprint=False | Yes | ~1800 |

### Key references

- Landy & Szalay 1993, ApJ 412, 64 — estimator
- Limber 1953, ApJ 117, 134 — angular-to-3D projection
- TreeCorr: Jarvis, Bernstein & Jain 2004
- OneCovariance: Reischke et al. 2023
- JWST LRDs: Matthee et al. 2024; Furtak et al. 2023
