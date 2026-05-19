"""
clustering.py
-------------
Power-law fit of the 2PACF, Limber inversion, and bias/halo-mass estimation
for the LRDs clustering analysis.

Methodology follows Zhuang et al. (2025, arXiv:2505.20393) — NEXUS paper —
which is the closest published analysis to ours.  The key equations are:

  w(θ) = A₀ θ^{-β}          [angular power law, β = 0.8, γ = β+1 = 1.8]

  A₀ = r₀^γ B(1/2,(γ-1)/2)  ∫ N²(z) χ(z)^{1-γ} (c/H(z)) dz
       ───────────────────────────────────────────────────────
                        [∫ N(z) dz]²

  b² = ξ₂₀(r₀) / [ξ_{m,20}(z=0) × D²(z_eff)]

  Halo mass via Sheth, Mo & Tormen (2001) bias–mass relation.

References
----------
- Limber (1953), ApJ, 117, 134
- Efstathiou et al. (1991), MNRAS, 252, 1P
- Zhuang et al. (2025, NEXUS), arXiv:2505.20393
- Sheth, Mo & Tormen (2001), MNRAS, 323, 1
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma as sc_gamma
from scipy.optimize import brentq
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
C_KMS = 299_792.458          # speed of light [km s⁻¹]
DELTA_C = 1.686              # critical overdensity for spherical collapse


# ---------------------------------------------------------------------------
# Cosmology helper
# ---------------------------------------------------------------------------

def make_cosmo(cosmo_cfg: dict) -> FlatLambdaCDM:
    """Build an astropy FlatLambdaCDM cosmology from the config dict.

    Parameters
    ----------
    cosmo_cfg : dict
        The ``cosmology`` sub-dict from ``config/analysis.yml``.

    Returns
    -------
    astropy.cosmology.FlatLambdaCDM
    """
    return FlatLambdaCDM(
        H0=cosmo_cfg["h"] * 100,
        Om0=cosmo_cfg["omega_m"],
        Ob0=cosmo_cfg["omega_b"],
        Tcmb0=2.7255,
    )


# ---------------------------------------------------------------------------
# Integral constraint
# ---------------------------------------------------------------------------

def integral_constraint_factor(theta_rad: np.ndarray, rr: np.ndarray, beta: float) -> float:
    """Compute the IC correction factor F such that C_IC = A₀ × F.

    The integral constraint (Groth & Peebles 1977) accounts for the finite
    survey area.  For a power-law model w(θ) = A₀ θ^{-β}:

        C_IC = A₀ × Σᵢ [RR(θᵢ) θᵢ^{-β}] / Σᵢ RR(θᵢ)

    So the observed correlation is:
        w_obs(θ) = A₀ (θ^{-β} - F)

    Parameters
    ----------
    theta_rad : array
        Angular bin centres in **radians**.
    rr : array
        Raw RR pair counts (not normalised).
    beta : float
        Power-law slope (β = 0.8 → γ = 1.8).

    Returns
    -------
    float
        F = Σ [RR θ^{-β}] / Σ RR
    """
    weights = rr / rr.sum()
    return float(np.sum(weights * theta_rad ** (-beta)))


# ---------------------------------------------------------------------------
# Chi-squared minimisation (analytic for linear model)
# ---------------------------------------------------------------------------

def fit_amplitude(
    theta_rad: np.ndarray,
    w_obs: np.ndarray,
    cov: np.ndarray,
    beta: float,
    ic_factor: float,
    valid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Analytically solve for A₀ by weighted least-squares.

    Model: w_model(θ) = A₀ × (θ^{-β} − F)   [F = ic_factor]

    This is linear in A₀, so the MLE is exact:

        A₀ = (M^T C⁻¹ w) / (M^T C⁻¹ M)
        σ²(A₀) = 1 / (M^T C⁻¹ M)

    Parameters
    ----------
    theta_rad, w_obs, cov : array
        Angular bins (rad), measured w(θ), covariance matrix.
    beta, ic_factor : float
        Power-law slope and pre-computed IC factor.
    valid : bool array, optional
        Mask of bins to include.  Default: all finite w_obs.

    Returns
    -------
    A0, sigma_A0 : float
        Best-fit amplitude and 1-σ uncertainty.
    """
    if valid is None:
        valid = np.isfinite(w_obs)

    theta_v = theta_rad[valid]
    w_v = w_obs[valid]
    cov_v = cov[np.ix_(valid, valid)]

    M = theta_v ** (-beta) - ic_factor     # model vector (linear in A₀)

    try:
        C_inv = np.linalg.inv(cov_v)
    except np.linalg.LinAlgError:
        C_inv = np.diag(1.0 / np.diag(cov_v))   # fallback: diagonal inverse

    MC = M @ C_inv
    A0 = float(MC @ w_v / (MC @ M))
    sigma_A0 = float(1.0 / np.sqrt(MC @ M))
    return A0, sigma_A0


def chi_squared(
    A0: float,
    theta_rad: np.ndarray,
    w_obs: np.ndarray,
    C_inv: np.ndarray,
    beta: float,
    ic_factor: float,
    valid: np.ndarray,
) -> float:
    """Evaluate χ² for a given A₀."""
    M = theta_rad[valid] ** (-beta) - ic_factor
    residual = w_obs[valid] - A0 * M
    return float(residual @ C_inv @ residual)


# ---------------------------------------------------------------------------
# Limber inversion → r₀
# ---------------------------------------------------------------------------

def _c_gamma(gamma: float) -> float:
    """Gamma-function prefactor B(1/2, (γ-1)/2) = Γ(1/2) Γ((γ-1)/2) / Γ(γ/2)."""
    return float(sc_gamma(0.5) * sc_gamma((gamma - 1) / 2) / sc_gamma(gamma / 2))


def limber_h_gamma(
    z_arr: np.ndarray,
    gamma: float,
    cosmo: FlatLambdaCDM,
    dz: float = 0.05,
) -> float:
    """Compute the Limber H_γ integral from a discrete redshift array.

    H_γ = ∫ [N(z)]² χ(z)^{1-γ} (H(z)/c) dz  /  [∫ N(z) dz]²

    where χ(z) is the comoving distance in Mpc and H(z)/c in Mpc⁻¹,
    giving H_γ in units of Mpc^{-γ} so that A₀ = C_γ r₀^γ H_γ is dimensionless.

    Parameters
    ----------
    z_arr : array
        Redshift values of all galaxies in the sample.
    gamma : float
        3D correlation function slope (γ = 1.8).
    cosmo : FlatLambdaCDM
        Astropy cosmology object.
    dz : float
        Histogram bin width for n(z) estimate.

    Returns
    -------
    H_gamma : float
        In units of Mpc^{-γ}.
    """
    z_min, z_max = z_arr.min(), z_arr.max()
    bins = np.arange(z_min, z_max + dz, dz)
    nz, edges = np.histogram(z_arr, bins=bins)
    z_mids = 0.5 * (edges[:-1] + edges[1:])

    mask = nz > 0
    z_m = z_mids[mask]
    n_m = nz[mask].astype(float) / dz    # dN/dz (count per unit z)

    chi_mpc = cosmo.comoving_distance(z_m).to(u.Mpc).value   # [Mpc]
    H_z = cosmo.H(z_m).to(u.km / u.s / u.Mpc).value         # [km/s/Mpc]
    # H(z)/c in Mpc^{-1}: correct factor for Limber integral to give Mpc^{-γ}
    H_over_c = H_z / C_KMS                                    # [Mpc^{-1}]

    try:                      # NumPy ≥2.0 renamed trapz → trapezoid
        _trapz = np.trapezoid
    except AttributeError:
        _trapz = np.trapz     # type: ignore[attr-defined]
    numerator = _trapz(n_m**2 * chi_mpc**(1 - gamma) * H_over_c, x=z_m)
    denominator = _trapz(n_m, x=z_m) ** 2

    return float(numerator / denominator)


def a0_to_r0(A0: float, gamma: float, H_gamma: float) -> float:
    """Convert the power-law amplitude A₀ to the 3D correlation length r₀.

    A₀ = C_γ × r₀^γ × H_γ   →   r₀ = (A₀ / (C_γ × H_γ))^{1/γ}

    Parameters
    ----------
    A0 : float
        Fitted angular amplitude (dimensionless; θ in radians).
    gamma : float
        3D slope (1.8).
    H_gamma : float
        Limber integral in Mpc^{-γ}.

    Returns
    -------
    r0 : float
        Comoving correlation length in Mpc.
    """
    Cgamma = _c_gamma(gamma)
    r0_gamma = A0 / (Cgamma * H_gamma)
    if r0_gamma <= 0:
        return np.nan
    return float(r0_gamma ** (1.0 / gamma))


# ---------------------------------------------------------------------------
# Integrated correlation function ξ₂₀  (Zhuang+25 Eq. 4–5)
# ---------------------------------------------------------------------------

def xi20(r0: float, gamma: float, r_min: float = 5.0, r_max: float = 20.0) -> float:
    """Integrated correlation function over [r_min, r_max] h⁻¹ cMpc.

    ξ₂₀ = 3 r₀^γ / [(3-γ) r_max³] × (r_max^{3-γ} − r_min^{3-γ})

    Parameters
    ----------
    r0, gamma : float
        Correlation length (h⁻¹ cMpc) and slope.
    r_min, r_max : float
        Integration limits in h⁻¹ cMpc  (default 5 and 20, Zhuang+25).

    Returns
    -------
    float
    """
    return (3 * r0**gamma / ((3 - gamma) * r_max**3)
            * (r_max**(3 - gamma) - r_min**(3 - gamma)))


def xi_matter_20(
    sigma8: float,
    cosmo: FlatLambdaCDM,
    gamma: float = 1.8,
    r_min: float = 5.0,
    r_max: float = 20.0,
    r0_matter_z0: float = 5.0,
) -> float:
    """Integrated matter correlation function ξ_{m,20} at z=0.

    Uses the power-law approximation ξ_m(r) ≈ (r/r₀_m)^{-γ} with
    r₀_m ≈ 5 h⁻¹ Mpc (Norberg et al. 2002; consistent with σ₈ ~ 0.81).

    This is the same approach as Zhuang+25, who use σ₈ = 0.84.

    Parameters
    ----------
    sigma8 : float
        σ₈ from the cosmological model.
    cosmo : FlatLambdaCDM
        (Currently unused; included for future CAMB-based version.)
    gamma : float
        Slope of the matter power spectrum in real space (1.8).
    r_min, r_max : float
        Integration limits [h⁻¹ cMpc].
    r0_matter_z0 : float
        Matter correlation length at z=0 [h⁻¹ cMpc].
        Default 5.0 h⁻¹ Mpc is the canonical value for γ=1.8, σ₈ ~ 0.8.

    Returns
    -------
    float
        ξ_{m,20} at z = 0.
    """
    # Scale r₀_m with σ₈ relative to the canonical value
    r0_m = r0_matter_z0 * (sigma8 / 0.80)  # very mild scaling
    return xi20(r0_m, gamma, r_min, r_max)


def growth_factor(z: float, cosmo: FlatLambdaCDM) -> float:
    """Linear growth factor D(z) normalised to D(0) = 1.

    Uses the approximation from Carroll, Press & Turner (1992) for
    flat ΛCDM:

        D(z) ∝ H(z) ∫_z^∞ (1+z') / H³(z') dz'

    Normalised so D(z=0) = 1.

    Parameters
    ----------
    z : float
        Redshift.
    cosmo : FlatLambdaCDM

    Returns
    -------
    float
    """
    def integrand(zp):
        Hp = cosmo.H(zp).to(u.km / u.s / u.Mpc).value
        return (1 + zp) / Hp**3

    from scipy.integrate import quad
    z_arr = np.linspace(z, 100.0, 2000)
    integral_z, _ = quad(integrand, z, 100.0, limit=500)
    integral_0, _ = quad(integrand, 0.0, 100.0, limit=500)

    Hz = cosmo.H(z).to(u.km / u.s / u.Mpc).value
    H0 = cosmo.H(0).to(u.km / u.s / u.Mpc).value

    return float((Hz / H0) * (integral_z / integral_0))


def linear_bias(
    r0_gal: float,
    z_eff: float,
    gamma: float,
    sigma8: float,
    cosmo: FlatLambdaCDM,
    r_min: float = 5.0,
    r_max: float = 20.0,
) -> float:
    """Estimate linear bias b from the galaxy correlation length.

    b² = ξ₂₀(r₀_gal) / [ξ_{m,20}(z=0) × D²(z_eff)]

    Parameters
    ----------
    r0_gal : float
        Galaxy correlation length [Mpc] (from Limber inversion).
        Converted internally to h⁻¹ cMpc for the ξ₂₀ formula.
    z_eff : float
        Effective redshift of the sample (median z).
    gamma : float
        Power-law slope (1.8).
    sigma8 : float
        σ₈ from cosmology config.
    cosmo : FlatLambdaCDM
    r_min, r_max : float
        Integration limits [h⁻¹ cMpc].

    Returns
    -------
    b : float
    """
    h = cosmo.H0.value / 100.0

    # Convert r₀ from Mpc → h⁻¹ cMpc
    r0_hinvMpc = r0_gal * h

    xi_gal = xi20(r0_hinvMpc, gamma, r_min, r_max)
    xi_m   = xi_matter_20(sigma8, cosmo, gamma, r_min, r_max)
    D_z    = growth_factor(z_eff, cosmo)

    b2 = xi_gal / (xi_m * D_z**2)
    if b2 <= 0:
        return np.nan
    return float(np.sqrt(b2))


# ---------------------------------------------------------------------------
# Sheth, Mo & Tormen (2001) halo bias → halo mass
# ---------------------------------------------------------------------------

def smt_bias(nu: float, a: float = 0.707, p: float = 0.3) -> float:
    """Sheth–Mo–Tormen (2001) halo bias as a function of peak height ν.

    b(ν) = 1 + (a ν² − 1)/δ_c + 2p / [δ_c (1 + (a ν²)^p)]

    Parameters
    ----------
    nu : float
        Peak height ν = δ_c / σ(M, z).
    a, p : float
        SMT parameters (default a=0.707, p=0.3).

    Returns
    -------
    b : float
    """
    anu2 = a * nu**2
    return 1.0 + (anu2 - 1.0) / DELTA_C + 2.0 * p / (DELTA_C * (1.0 + anu2**p))


def sigma_mass(
    log10_Mhalo: float,
    z: float,
    cosmo: FlatLambdaCDM,
    sigma8: float,
    ns: float = 0.965,
) -> float:
    """RMS density fluctuation σ(M, z) using the Eisenstein & Hu (1998)
    transfer function (via CAMB-independent analytic approximation).

    Uses the Press-Schechter mass-radius relation R = (3M / 4π ρ̄_m)^{1/3}
    and the approximate top-hat filter.

    For our purposes (rough halo mass estimate) we use the simple scaling:
        σ(M, z) ≈ σ₈ × D(z) × (M / M₈)^{−(n_s+3)/6}
    where M₈ = (4π/3) ρ̄_m (8 h⁻¹ Mpc)³.

    Parameters
    ----------
    log10_Mhalo : float
        log₁₀(M_halo / M_sun).
    z : float
        Redshift.
    cosmo : FlatLambdaCDM
    sigma8 : float
    ns : float
        Spectral index.

    Returns
    -------
    sigma : float
    """
    M = 10.0**log10_Mhalo                              # M_sun
    rho_m0 = cosmo.critical_density(0).to(u.M_sun / u.Mpc**3).value * cosmo.Om0
    h = cosmo.H0.value / 100.0

    # Mass enclosed in a sphere of radius 8 h⁻¹ Mpc
    R8_mpc = 8.0 / h
    M8 = (4.0 / 3.0) * np.pi * rho_m0 * R8_mpc**3    # M_sun

    D_z = growth_factor(z, cosmo)
    slope = -(ns + 3.0) / 6.0
    return sigma8 * D_z * (M / M8) ** slope


def halo_mass_from_bias(
    b_target: float,
    z: float,
    cosmo: FlatLambdaCDM,
    sigma8: float,
    ns: float = 0.965,
    log10_M_min: float = 10.0,
    log10_M_max: float = 15.0,
) -> float:
    """Invert the SMT bias relation to find M_halo given bias b.

    Parameters
    ----------
    b_target : float
        Linear bias.
    z : float
        Redshift.
    cosmo, sigma8, ns : ...
        Cosmological parameters.
    log10_M_min, log10_M_max : float
        Search range in log₁₀(M_halo/M_sun).

    Returns
    -------
    log10_Mhalo : float
        log₁₀(M_halo / M_sun), or NaN if no root found.
    """
    def objective(log10_M):
        sig = sigma_mass(log10_M, z, cosmo, sigma8, ns)
        nu = DELTA_C / sig
        return smt_bias(nu) - b_target

    try:
        # Check if a solution exists in the range
        f_min = objective(log10_M_min)
        f_max = objective(log10_M_max)
        if f_min * f_max > 0:
            return np.nan
        root = brentq(objective, log10_M_min, log10_M_max, xtol=0.01)
        return float(root)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# MCMC helpers
# ---------------------------------------------------------------------------

def log_likelihood(
    log10_A0: float,
    theta_rad: np.ndarray,
    w_obs: np.ndarray,
    C_inv: np.ndarray,
    beta: float,
    ic_factor: float,
    valid: np.ndarray,
) -> float:
    """Gaussian log-likelihood for the power-law model."""
    A0 = 10.0**log10_A0
    M = theta_rad[valid] ** (-beta) - ic_factor
    residual = w_obs[valid] - A0 * M
    return -0.5 * float(residual @ C_inv @ residual)


def log_prior(log10_A0: float, lo: float = -6.0, hi: float = 2.0) -> float:
    """Flat (uninformative) prior on log₁₀(A₀)."""
    if lo < log10_A0 < hi:
        return 0.0
    return -np.inf


def log_posterior(
    params: np.ndarray,
    theta_rad: np.ndarray,
    w_obs: np.ndarray,
    C_inv: np.ndarray,
    beta: float,
    ic_factor: float,
    valid: np.ndarray,
) -> float:
    log10_A0 = params[0]
    lp = log_prior(log10_A0)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(log10_A0, theta_rad, w_obs, C_inv, beta, ic_factor, valid)
