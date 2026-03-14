"""
Physical constants — CODATA 2018/2022 measured values.

These are MEASURED quantities, not derived from PAC. They serve as
validation targets for PAC-derived predictions.

All values in SI units unless noted.
"""

import math

# =============================================================================
# FUNDAMENTAL CONSTANTS (CODATA 2018)
# =============================================================================

SPEED_OF_LIGHT: float = 299_792_458.0
"""c = 299,792,458 m/s (exact by definition)."""
C = SPEED_OF_LIGHT

PLANCK_CONSTANT: float = 6.626_070_15e-34
"""h = 6.62607015 × 10⁻³⁴ J·s (exact by definition)."""
H = PLANCK_CONSTANT

HBAR: float = H / (2 * math.pi)
"""ℏ = h/(2π) = 1.054571817 × 10⁻³⁴ J·s."""

BOLTZMANN: float = 1.380_649e-23
"""k_B = 1.380649 × 10⁻²³ J/K (exact by definition)."""
K_B = BOLTZMANN

GRAVITATIONAL_CONSTANT: float = 6.674_30e-11
"""G = 6.67430 × 10⁻¹¹ m³/(kg·s²). Uncertainty: ±1.5 × 10⁻¹⁵ (22 ppm)."""
G = GRAVITATIONAL_CONSTANT
G_UNCERTAINTY: float = 1.5e-15
"""1σ uncertainty on G measurement."""

ELEMENTARY_CHARGE: float = 1.602_176_634e-19
"""e = 1.602176634 × 10⁻¹⁹ C (exact by definition)."""

VACUUM_PERMITTIVITY: float = 8.854_187_8128e-12
"""ε₀ = 8.8541878128 × 10⁻¹² F/m."""

VACUUM_PERMEABILITY: float = 1.256_637_062_12e-6
"""μ₀ = 1.25663706212 × 10⁻⁶ N/A²."""


# =============================================================================
# PARTICLE MASSES (CODATA 2018)
# =============================================================================

ELECTRON_MASS: float = 9.109_383_7015e-31
"""m_e = 9.1093837015 × 10⁻³¹ kg."""
M_E = ELECTRON_MASS

PROTON_MASS: float = 1.672_621_923_69e-27
"""m_p = 1.67262192369 × 10⁻²⁷ kg."""
M_P = PROTON_MASS

NEUTRON_MASS: float = 1.674_927_498_04e-27
"""m_n = 1.67492749804 × 10⁻²⁷ kg."""
M_N = NEUTRON_MASS

MUON_MASS: float = 1.883_531_627e-28
"""m_μ = 1.883531627 × 10⁻²⁸ kg."""
M_MU = MUON_MASS

TAU_MASS: float = 3.167_54e-27
"""m_τ = 3.16754 × 10⁻²⁷ kg."""
M_TAU = TAU_MASS

# Mass ratios (dimensionless, higher precision than individual masses)
MUON_ELECTRON_RATIO: float = 206.768_2830
"""m_μ/m_e = 206.7682830 (measured)."""

TAU_ELECTRON_RATIO: float = 3477.23
"""m_τ/m_e = 3477.23 (measured)."""

PROTON_ELECTRON_RATIO: float = 1836.152_673_43
"""m_p/m_e = 1836.15267343 (measured)."""


# =============================================================================
# PLANCK UNITS
# =============================================================================

PLANCK_MASS: float = math.sqrt(HBAR * C / G)
"""m_Pl = √(ℏc/G) ≈ 2.176 × 10⁻⁸ kg."""

PLANCK_LENGTH: float = math.sqrt(HBAR * G / C ** 3)
"""l_Pl = √(ℏG/c³) ≈ 1.616 × 10⁻³⁵ m."""

PLANCK_TIME: float = math.sqrt(HBAR * G / C ** 5)
"""t_Pl = √(ℏG/c⁵) ≈ 5.391 × 10⁻⁴⁴ s."""

PLANCK_ENERGY: float = PLANCK_MASS * C ** 2
"""E_Pl = m_Pl · c² ≈ 1.956 × 10⁹ J."""

PLANCK_TEMPERATURE: float = PLANCK_ENERGY / K_B
"""T_Pl = E_Pl/k_B ≈ 1.417 × 10³² K."""

PLANCK_DENSITY: float = C ** 5 / (HBAR * G ** 2)
"""ρ_Pl = c⁵/(ℏG²) ≈ 5.155 × 10⁹⁶ kg/m³."""


# =============================================================================
# GAUGE BOSON MASSES (PDG 2022)
# =============================================================================

W_BOSON_MASS_GEV: float = 80.377
"""M_W = 80.377 GeV/c²."""

Z_BOSON_MASS_GEV: float = 91.1876
"""M_Z = 91.1876 GeV/c²."""

HIGGS_MASS_GEV: float = 125.25
"""M_H = 125.25 GeV/c²."""


# =============================================================================
# COSMOLOGICAL (Planck 2018)
# =============================================================================

HUBBLE_CONSTANT: float = 67.36
"""H₀ = 67.36 km/s/Mpc (Planck 2018)."""
H0 = HUBBLE_CONSTANT
H0_UNCERTAINTY: float = 0.54

OMEGA_MATTER: float = 0.3153
"""Ω_m = 0.3153 (total matter fraction)."""

OMEGA_BARYON: float = 0.0493
"""Ω_b = 0.0493 (baryon fraction)."""

OMEGA_CDM: float = 0.2607
"""Ω_c = 0.2607 (cold dark matter fraction). PAC: F₃Ξ/F₆ = 0.2646 (0.15%)."""

OMEGA_LAMBDA: float = 0.6847
"""Ω_Λ = 0.6847 (dark energy fraction)."""
OMEGA_LAMBDA_UNCERTAINTY: float = 0.0073

VACUUM_ENERGY_DENSITY_LOG10: float = -122.95
"""log₁₀(ρ_Λ/ρ_Pl) ≈ -122.95 — the observed cosmological constant."""

CMB_TEMPERATURE: float = 2.7255
"""T_CMB = 2.7255 K."""

AGE_OF_UNIVERSE_GYR: float = 13.797
"""Age = 13.797 Gyr (Planck 2018)."""


# =============================================================================
# HIERARCHY RATIOS
# =============================================================================

PLANCK_PROTON_RATIO: float = PLANCK_MASS / M_P
"""m_Pl/m_p ≈ 1.301 × 10¹⁹ — the gravity hierarchy."""

EM_GRAVITY_RATIO: float = ELEMENTARY_CHARGE ** 2 / (4 * math.pi * VACUUM_PERMITTIVITY * G * M_P ** 2)
"""F_EM/F_grav for two protons ≈ 1.236 × 10³⁶."""
